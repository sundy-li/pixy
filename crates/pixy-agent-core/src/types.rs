use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use pixy_ai::{
    AssistantMessageEvent, AssistantMessageEventStream, Context, Message, Model, PiAiError,
    SimpleStreamOptions, Tool, ToolResultContentBlock,
};
use serde_json::Value;
use tokio::sync::Notify;

pub type AgentMessage = Message;

pub trait StreamExecutor: Send + Sync {
    fn stream(
        &self,
        model: Model,
        context: Context,
        options: Option<SimpleStreamOptions>,
    ) -> Result<AssistantMessageEventStream, PiAiError>;
}

impl<F> StreamExecutor for F
where
    F: Fn(
            Model,
            Context,
            Option<SimpleStreamOptions>,
        ) -> Result<AssistantMessageEventStream, PiAiError>
        + Send
        + Sync
        + 'static,
{
    fn stream(
        &self,
        model: Model,
        context: Context,
        options: Option<SimpleStreamOptions>,
    ) -> Result<AssistantMessageEventStream, PiAiError> {
        (self)(model, context, options)
    }
}

pub type StreamFn = Arc<dyn StreamExecutor>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParentChildRunEvent {
    ChildRunStart {
        parent_session_id: String,
        child_session_file: String,
        task_id: String,
        subagent: String,
    },
    ChildRunEnd {
        parent_session_id: String,
        child_session_file: String,
        task_id: String,
        subagent: String,
        duration_ms: u64,
        summary: String,
    },
    ChildRunError {
        parent_session_id: String,
        child_session_file: String,
        task_id: String,
        subagent: String,
        error: String,
    },
}

impl ParentChildRunEvent {
    pub fn task_id(&self) -> &str {
        match self {
            Self::ChildRunStart { task_id, .. } => task_id,
            Self::ChildRunEnd { task_id, .. } => task_id,
            Self::ChildRunError { task_id, .. } => task_id,
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Self::ChildRunStart { .. } => "child_run_start",
            Self::ChildRunEnd { .. } => "child_run_end",
            Self::ChildRunError { .. } => "child_run_error",
        }
    }
}

pub type ParentChildRunEventSink = Arc<dyn Fn(ParentChildRunEvent) + Send + Sync>;

pub trait MessageConverter: Send + Sync {
    fn convert(&self, messages: Vec<AgentMessage>) -> Vec<Message>;
}

#[derive(Default)]
pub struct IdentityMessageConverter;

impl MessageConverter for IdentityMessageConverter {
    fn convert(&self, messages: Vec<AgentMessage>) -> Vec<Message> {
        messages
    }
}

impl<F> MessageConverter for F
where
    F: Fn(Vec<AgentMessage>) -> Vec<Message> + Send + Sync + 'static,
{
    fn convert(&self, messages: Vec<AgentMessage>) -> Vec<Message> {
        (self)(messages)
    }
}

pub type ConvertToLlmFn = Arc<dyn MessageConverter>;

pub trait MessageQueue: Send + Sync {
    fn poll(&self) -> Vec<AgentMessage>;
}

impl<F> MessageQueue for F
where
    F: Fn() -> Vec<AgentMessage> + Send + Sync + 'static,
{
    fn poll(&self) -> Vec<AgentMessage> {
        (self)()
    }
}

pub type MessageQueueFn = Arc<dyn MessageQueue>;

pub type ToolFuture = Pin<Box<dyn Future<Output = Result<AgentToolResult, PiAiError>> + Send>>;

#[async_trait]
pub trait AgentToolExecutor: Send + Sync {
    async fn execute(
        &self,
        tool_call_id: String,
        args: Value,
    ) -> Result<AgentToolResult, PiAiError>;
}

#[async_trait]
impl<F> AgentToolExecutor for F
where
    F: Fn(String, Value) -> ToolFuture + Send + Sync + 'static,
{
    async fn execute(
        &self,
        tool_call_id: String,
        args: Value,
    ) -> Result<AgentToolResult, PiAiError> {
        (self)(tool_call_id, args).await
    }
}

pub type AgentToolExecuteFn = Arc<dyn AgentToolExecutor>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AgentRetryConfig {
    pub max_attempts: usize,
    pub initial_backoff_ms: u64,
    pub max_backoff_ms: u64,
}

impl Default for AgentRetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_backoff_ms: 200,
            max_backoff_ms: 2_000,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AgentRunMetrics {
    pub assistant_request_count: usize,
    pub assistant_request_total_ms: u64,
    pub tool_execution_count: usize,
    pub tool_execution_total_ms: u64,
    pub retry_count: usize,
}

#[derive(Clone)]
pub struct AgentLoopConfig {
    pub model: Model,
    pub fallback_models: Vec<Model>,
    pub convert_to_llm: ConvertToLlmFn,
    pub stream_fn: StreamFn,
    pub retry: AgentRetryConfig,
    pub get_steering_messages: Option<MessageQueueFn>,
    pub get_follow_up_messages: Option<MessageQueueFn>,
}

#[derive(Clone)]
pub struct AgentTool {
    pub name: String,
    pub label: String,
    pub description: String,
    pub parameters: serde_json::Value,
    pub execute: AgentToolExecuteFn,
}

impl AgentTool {
    pub fn to_llm_tool(&self) -> Tool {
        Tool {
            name: self.name.clone(),
            description: self.description.clone(),
            parameters: self.parameters.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AgentToolResult {
    pub content: Vec<ToolResultContentBlock>,
    pub details: Value,
}

#[derive(Clone)]
pub struct AgentContext {
    pub system_prompt: String,
    pub messages: Vec<AgentMessage>,
    pub tools: Vec<AgentTool>,
}

#[derive(Clone)]
pub struct AgentAbortSignal {
    inner: Arc<AbortInner>,
}

struct AbortInner {
    aborted: AtomicBool,
    notify: Notify,
}

impl AgentAbortSignal {
    pub fn is_aborted(&self) -> bool {
        self.inner.aborted.load(Ordering::SeqCst)
    }

    pub async fn cancelled(&self) {
        if self.is_aborted() {
            return;
        }
        self.inner.notify.notified().await;
    }
}

pub struct AgentAbortController {
    signal: AgentAbortSignal,
}

impl AgentAbortController {
    pub fn new() -> Self {
        Self {
            signal: AgentAbortSignal {
                inner: Arc::new(AbortInner {
                    aborted: AtomicBool::new(false),
                    notify: Notify::new(),
                }),
            },
        }
    }

    pub fn signal(&self) -> AgentAbortSignal {
        self.signal.clone()
    }

    pub fn abort(&self) {
        self.signal.inner.aborted.store(true, Ordering::SeqCst);
        self.signal.inner.notify.notify_waiters();
    }
}

#[derive(Clone)]
pub enum AgentEvent {
    AgentStart,
    AgentEnd {
        messages: Vec<AgentMessage>,
    },
    TurnStart,
    TurnEnd {
        message: AgentMessage,
        tool_results: Vec<AgentMessage>,
    },
    MessageStart {
        message: AgentMessage,
    },
    MessageUpdate {
        message: AgentMessage,
        assistant_message_event: AssistantMessageEvent,
    },
    MessageEnd {
        message: AgentMessage,
    },
    ToolExecutionStart {
        tool_call_id: String,
        tool_name: String,
        args: Value,
    },
    ToolExecutionUpdate {
        tool_call_id: String,
        tool_name: String,
        args: Value,
        partial_result: Value,
    },
    ToolExecutionEnd {
        tool_call_id: String,
        tool_name: String,
        result: AgentToolResult,
        is_error: bool,
        duration_ms: u64,
    },
    RetryScheduled {
        attempt: usize,
        max_attempts: usize,
        delay_ms: u64,
        error: String,
    },
    ModelFallback {
        from_provider: String,
        from_model: String,
        to_provider: String,
        to_model: String,
    },
    Metrics {
        metrics: AgentRunMetrics,
    },
}

#[cfg(test)]
mod tests {
    use super::ParentChildRunEvent;

    #[test]
    fn parent_child_run_event_exposes_task_id_and_kind_for_all_variants() {
        let start = ParentChildRunEvent::ChildRunStart {
            parent_session_id: "parent".to_string(),
            child_session_file: "/tmp/child.jsonl".to_string(),
            task_id: "task-1".to_string(),
            subagent: "general".to_string(),
        };
        let end = ParentChildRunEvent::ChildRunEnd {
            parent_session_id: "parent".to_string(),
            child_session_file: "/tmp/child.jsonl".to_string(),
            task_id: "task-1".to_string(),
            subagent: "general".to_string(),
            duration_ms: 12,
            summary: "done".to_string(),
        };
        let error = ParentChildRunEvent::ChildRunError {
            parent_session_id: "parent".to_string(),
            child_session_file: "/tmp/child.jsonl".to_string(),
            task_id: "task-1".to_string(),
            subagent: "general".to_string(),
            error: "boom".to_string(),
        };

        assert_eq!(start.task_id(), "task-1");
        assert_eq!(start.kind(), "child_run_start");
        assert_eq!(end.task_id(), "task-1");
        assert_eq!(end.kind(), "child_run_end");
        assert_eq!(error.task_id(), "task-1");
        assert_eq!(error.kind(), "child_run_error");
    }
}
