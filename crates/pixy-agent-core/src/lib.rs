//! Stateful agent loop built on top of `pi-ai`.

mod agent;
mod agent_loop;
mod types;

pub use agent::{Agent, AgentConfig, AgentState, QueueMode};
pub use agent_loop::{AgentLoopError, agent_loop, agent_loop_continue, try_agent_loop_continue};
pub use types::{
    AgentAbortController, AgentAbortSignal, AgentContext, AgentEvent, AgentLoopConfig,
    AgentMessage, AgentRetryConfig, AgentRunMetrics, AgentTool, AgentToolExecuteFn,
    AgentToolExecutor, AgentToolResult, ConvertToLlmFn, IdentityMessageConverter, MessageConverter,
    MessageQueue, MessageQueueFn, ParentChildRunEvent, ParentChildRunEventSink, StreamExecutor,
    StreamFn, ToolFuture,
};
