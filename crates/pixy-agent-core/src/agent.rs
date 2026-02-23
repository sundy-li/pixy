use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use pixy_ai::{Message, Model, UserContent};
use tokio::sync::Notify;

use crate::agent_loop::{agent_loop, agent_loop_continue};
use crate::types::{
    AgentAbortController, AgentContext, AgentEvent, AgentLoopConfig, AgentMessage,
    AgentRetryConfig, AgentTool, ConvertToLlmFn, StreamFn,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueueMode {
    All,
    OneAtATime,
}

impl Default for QueueMode {
    fn default() -> Self {
        Self::OneAtATime
    }
}

#[derive(Clone)]
pub struct AgentConfig {
    pub system_prompt: String,
    pub model: Model,
    pub fallback_models: Vec<Model>,
    pub tools: Vec<AgentTool>,
    pub messages: Vec<AgentMessage>,
    pub convert_to_llm: ConvertToLlmFn,
    pub stream_fn: StreamFn,
    pub retry: AgentRetryConfig,
    pub steering_mode: QueueMode,
    pub follow_up_mode: QueueMode,
}

impl AgentConfig {
    pub fn new(system_prompt: String, model: Model, stream_fn: StreamFn) -> Self {
        Self {
            system_prompt,
            model,
            fallback_models: vec![],
            tools: vec![],
            messages: vec![],
            convert_to_llm: Arc::new(|messages: Vec<AgentMessage>| messages),
            stream_fn,
            retry: AgentRetryConfig::default(),
            steering_mode: QueueMode::OneAtATime,
            follow_up_mode: QueueMode::OneAtATime,
        }
    }
}

#[derive(Clone)]
pub struct AgentState {
    pub system_prompt: String,
    pub model: Model,
    pub tools: Vec<AgentTool>,
    pub messages: Vec<AgentMessage>,
    pub is_streaming: bool,
    pub stream_message: Option<AgentMessage>,
    pub pending_tool_calls: Vec<String>,
    pub error: Option<String>,
}

struct AgentInner {
    system_prompt: String,
    model: Model,
    fallback_models: Vec<Model>,
    tools: Vec<AgentTool>,
    messages: Vec<AgentMessage>,
    stream_message: Option<AgentMessage>,
    pending_tool_calls: HashSet<String>,
    error: Option<String>,
    steering_queue: Vec<AgentMessage>,
    follow_up_queue: Vec<AgentMessage>,
    steering_mode: QueueMode,
    follow_up_mode: QueueMode,
    retry: AgentRetryConfig,
    abort_controller: Option<AgentAbortController>,
}

#[derive(Clone)]
pub struct Agent {
    inner: Arc<Mutex<AgentInner>>,
    convert_to_llm: ConvertToLlmFn,
    stream_fn: StreamFn,
    is_running: Arc<AtomicBool>,
    idle_notify: Arc<Notify>,
}

impl Agent {
    pub fn new(config: AgentConfig) -> Self {
        Self {
            inner: Arc::new(Mutex::new(AgentInner {
                system_prompt: config.system_prompt,
                model: config.model,
                fallback_models: config.fallback_models,
                tools: config.tools,
                messages: config.messages,
                stream_message: None,
                pending_tool_calls: HashSet::new(),
                error: None,
                steering_queue: vec![],
                follow_up_queue: vec![],
                steering_mode: config.steering_mode,
                follow_up_mode: config.follow_up_mode,
                retry: config.retry,
                abort_controller: None,
            })),
            convert_to_llm: config.convert_to_llm,
            stream_fn: config.stream_fn,
            is_running: Arc::new(AtomicBool::new(false)),
            idle_notify: Arc::new(Notify::new()),
        }
    }

    pub fn state(&self) -> AgentState {
        let inner = self.inner.lock().expect("agent mutex poisoned");
        let mut pending_tool_calls = inner.pending_tool_calls.iter().cloned().collect::<Vec<_>>();
        pending_tool_calls.sort();

        AgentState {
            system_prompt: inner.system_prompt.clone(),
            model: inner.model.clone(),
            tools: inner.tools.clone(),
            messages: inner.messages.clone(),
            is_streaming: self.is_running.load(Ordering::SeqCst),
            stream_message: inner.stream_message.clone(),
            pending_tool_calls,
            error: inner.error.clone(),
        }
    }

    pub fn set_system_prompt(&self, system_prompt: String) {
        let mut inner = self.inner.lock().expect("agent mutex poisoned");
        inner.system_prompt = system_prompt;
    }

    pub fn set_model(&self, model: Model) {
        let mut inner = self.inner.lock().expect("agent mutex poisoned");
        inner.model = model;
    }

    pub fn set_fallback_models(&self, fallback_models: Vec<Model>) {
        let mut inner = self.inner.lock().expect("agent mutex poisoned");
        inner.fallback_models = fallback_models;
    }

    pub fn set_retry_config(&self, retry: AgentRetryConfig) {
        let mut inner = self.inner.lock().expect("agent mutex poisoned");
        inner.retry = retry;
    }

    pub fn set_tools(&self, tools: Vec<AgentTool>) {
        let mut inner = self.inner.lock().expect("agent mutex poisoned");
        inner.tools = tools;
    }

    pub fn replace_messages(&self, messages: Vec<AgentMessage>) {
        let mut inner = self.inner.lock().expect("agent mutex poisoned");
        inner.messages = messages;
    }

    pub fn append_message(&self, message: AgentMessage) {
        let mut inner = self.inner.lock().expect("agent mutex poisoned");
        inner.messages.push(message);
    }

    pub fn clear_messages(&self) {
        let mut inner = self.inner.lock().expect("agent mutex poisoned");
        inner.messages.clear();
    }

    pub fn set_steering_mode(&self, mode: QueueMode) {
        let mut inner = self.inner.lock().expect("agent mutex poisoned");
        inner.steering_mode = mode;
    }

    pub fn steering_mode(&self) -> QueueMode {
        let inner = self.inner.lock().expect("agent mutex poisoned");
        inner.steering_mode
    }

    pub fn set_follow_up_mode(&self, mode: QueueMode) {
        let mut inner = self.inner.lock().expect("agent mutex poisoned");
        inner.follow_up_mode = mode;
    }

    pub fn follow_up_mode(&self) -> QueueMode {
        let inner = self.inner.lock().expect("agent mutex poisoned");
        inner.follow_up_mode
    }

    pub fn steer(&self, message: AgentMessage) {
        let mut inner = self.inner.lock().expect("agent mutex poisoned");
        inner.steering_queue.push(message);
    }

    pub fn follow_up(&self, message: AgentMessage) {
        let mut inner = self.inner.lock().expect("agent mutex poisoned");
        inner.follow_up_queue.push(message);
    }

    pub fn clear_steering_queue(&self) {
        let mut inner = self.inner.lock().expect("agent mutex poisoned");
        inner.steering_queue.clear();
    }

    pub fn clear_follow_up_queue(&self) {
        let mut inner = self.inner.lock().expect("agent mutex poisoned");
        inner.follow_up_queue.clear();
    }

    pub fn clear_all_queues(&self) {
        let mut inner = self.inner.lock().expect("agent mutex poisoned");
        inner.steering_queue.clear();
        inner.follow_up_queue.clear();
    }

    pub fn has_queued_messages(&self) -> bool {
        let inner = self.inner.lock().expect("agent mutex poisoned");
        !inner.steering_queue.is_empty() || !inner.follow_up_queue.is_empty()
    }

    pub fn abort(&self) {
        let inner = self.inner.lock().expect("agent mutex poisoned");
        if let Some(controller) = inner.abort_controller.as_ref() {
            controller.abort();
        }
    }

    pub async fn wait_for_idle(&self) {
        loop {
            if !self.is_running.load(Ordering::SeqCst) {
                return;
            }
            self.idle_notify.notified().await;
        }
    }

    pub async fn prompt(&self, prompts: Vec<AgentMessage>) -> Result<Vec<AgentMessage>, String> {
        if prompts.is_empty() {
            return Err("Prompt messages cannot be empty".to_string());
        }
        self.run(Some(prompts), false).await
    }

    pub async fn prompt_text(&self, input: &str) -> Result<Vec<AgentMessage>, String> {
        let prompt = Message::User {
            content: UserContent::Text(input.to_string()),
            timestamp: now_millis(),
        };
        self.prompt(vec![prompt]).await
    }

    pub async fn continue_run(&self) -> Result<Vec<AgentMessage>, String> {
        enum ContinueMode {
            Context,
            Queued {
                messages: Vec<AgentMessage>,
                skip_initial_steering_poll: bool,
            },
        }

        let continue_mode = {
            let mut inner = self.inner.lock().expect("agent mutex poisoned");
            if inner.messages.is_empty() {
                return Err("No messages to continue from".to_string());
            }

            if matches!(inner.messages.last(), Some(Message::Assistant { .. })) {
                let steering_mode = inner.steering_mode;
                let steering = dequeue_messages(&mut inner.steering_queue, steering_mode);
                if !steering.is_empty() {
                    ContinueMode::Queued {
                        messages: steering,
                        skip_initial_steering_poll: true,
                    }
                } else {
                    let follow_up_mode = inner.follow_up_mode;
                    let follow_up = dequeue_messages(&mut inner.follow_up_queue, follow_up_mode);
                    if !follow_up.is_empty() {
                        ContinueMode::Queued {
                            messages: follow_up,
                            skip_initial_steering_poll: false,
                        }
                    } else {
                        return Err("Cannot continue from message role: assistant".to_string());
                    }
                }
            } else {
                ContinueMode::Context
            }
        };

        match continue_mode {
            ContinueMode::Context => self.run(None, false).await,
            ContinueMode::Queued {
                messages,
                skip_initial_steering_poll,
            } => self.run(Some(messages), skip_initial_steering_poll).await,
        }
    }

    async fn run(
        &self,
        prompts: Option<Vec<AgentMessage>>,
        skip_initial_steering_poll: bool,
    ) -> Result<Vec<AgentMessage>, String> {
        if self
            .is_running
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(
                "Agent is already processing. Wait for completion before prompting again."
                    .to_string(),
            );
        }

        let run_result = async {
            let controller = AgentAbortController::new();
            let signal = controller.signal();

            let (context, model, fallback_models, retry) = {
                let mut inner = self.inner.lock().expect("agent mutex poisoned");
                inner.error = None;
                inner.stream_message = None;
                inner.pending_tool_calls.clear();
                inner.abort_controller = Some(controller);

                (
                    AgentContext {
                        system_prompt: inner.system_prompt.clone(),
                        messages: inner.messages.clone(),
                        tools: inner.tools.clone(),
                    },
                    inner.model.clone(),
                    inner.fallback_models.clone(),
                    inner.retry.clone(),
                )
            };

            let steering_inner = Arc::clone(&self.inner);
            let follow_up_inner = Arc::clone(&self.inner);
            let skip_initial = Arc::new(AtomicBool::new(skip_initial_steering_poll));
            let skip_initial_ref = Arc::clone(&skip_initial);

            let get_steering_messages = Arc::new(move || {
                if skip_initial_ref.swap(false, Ordering::SeqCst) {
                    return vec![];
                }

                let mut inner = steering_inner.lock().expect("agent mutex poisoned");
                let mode = inner.steering_mode;
                dequeue_messages(&mut inner.steering_queue, mode)
            });

            let get_follow_up_messages = Arc::new(move || {
                let mut inner = follow_up_inner.lock().expect("agent mutex poisoned");
                let mode = inner.follow_up_mode;
                dequeue_messages(&mut inner.follow_up_queue, mode)
            });

            let config = AgentLoopConfig {
                model,
                fallback_models,
                convert_to_llm: self.convert_to_llm.clone(),
                stream_fn: self.stream_fn.clone(),
                retry,
                get_steering_messages: Some(get_steering_messages),
                get_follow_up_messages: Some(get_follow_up_messages),
            };

            let stream = match prompts {
                Some(prompts) => agent_loop(prompts, context, config, Some(signal)),
                None => agent_loop_continue(context, config, Some(signal)),
            };

            while let Some(event) = stream.next().await {
                self.apply_event(event);
            }

            stream
                .result()
                .await
                .ok_or_else(|| "Agent loop ended without a final result".to_string())
        }
        .await;

        {
            let mut inner = self.inner.lock().expect("agent mutex poisoned");
            inner.stream_message = None;
            inner.pending_tool_calls.clear();
            inner.abort_controller = None;
        }

        self.is_running.store(false, Ordering::SeqCst);
        self.idle_notify.notify_waiters();
        run_result
    }

    fn apply_event(&self, event: AgentEvent) {
        let mut inner = self.inner.lock().expect("agent mutex poisoned");

        match event {
            AgentEvent::MessageStart { message } => {
                inner.stream_message = Some(message);
            }
            AgentEvent::MessageUpdate { message, .. } => {
                inner.stream_message = Some(message);
            }
            AgentEvent::MessageEnd { message } => {
                inner.stream_message = None;
                inner.messages.push(message);
            }
            AgentEvent::ToolExecutionStart { tool_call_id, .. } => {
                inner.pending_tool_calls.insert(tool_call_id);
            }
            AgentEvent::ToolExecutionEnd { tool_call_id, .. } => {
                inner.pending_tool_calls.remove(&tool_call_id);
            }
            AgentEvent::TurnEnd { message, .. } => {
                if let Message::Assistant {
                    error_message: Some(error_message),
                    ..
                } = message
                {
                    inner.error = Some(error_message);
                }
            }
            AgentEvent::AgentStart
            | AgentEvent::AgentEnd { .. }
            | AgentEvent::TurnStart
            | AgentEvent::ToolExecutionUpdate { .. }
            | AgentEvent::RetryScheduled { .. }
            | AgentEvent::ModelFallback { .. }
            | AgentEvent::Metrics { .. } => {}
        }
    }
}

fn dequeue_messages(queue: &mut Vec<AgentMessage>, mode: QueueMode) -> Vec<AgentMessage> {
    match mode {
        QueueMode::All => std::mem::take(queue),
        QueueMode::OneAtATime => {
            if queue.is_empty() {
                vec![]
            } else {
                vec![queue.remove(0)]
            }
        }
    }
}

fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0)
}
