use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use pi_ai::{
    AssistantMessageEvent, AssistantMessageEventStream, Context, Message, Model,
    SimpleStreamOptions, Tool, ToolResultContentBlock,
};
use serde_json::Value;
use tokio::sync::Notify;

pub type AgentMessage = Message;

pub type StreamFn = Arc<
    dyn Fn(
            Model,
            Context,
            Option<SimpleStreamOptions>,
        ) -> Result<AssistantMessageEventStream, String>
        + Send
        + Sync,
>;

pub type ConvertToLlmFn = Arc<dyn Fn(Vec<AgentMessage>) -> Vec<Message> + Send + Sync>;
pub type MessageQueueFn = Arc<dyn Fn() -> Vec<AgentMessage> + Send + Sync>;

pub type ToolFuture = Pin<Box<dyn Future<Output = Result<AgentToolResult, String>> + Send>>;

pub trait AgentToolExecutor: Send + Sync {
    fn execute(&self, tool_call_id: String, args: Value) -> ToolFuture;
}

impl<F> AgentToolExecutor for F
where
    F: Fn(String, Value) -> ToolFuture + Send + Sync + 'static,
{
    fn execute(&self, tool_call_id: String, args: Value) -> ToolFuture {
        (self)(tool_call_id, args)
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
