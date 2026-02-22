//! Stateful agent loop built on top of `pi-ai`.

mod agent;
mod agent_loop;
mod types;

pub use agent::{Agent, AgentConfig, AgentState, QueueMode};
pub use agent_loop::{agent_loop, agent_loop_continue};
pub use types::{
    AgentAbortController, AgentAbortSignal, AgentContext, AgentEvent, AgentLoopConfig,
    AgentMessage, AgentRetryConfig, AgentRunMetrics, AgentTool, AgentToolExecuteFn,
    AgentToolResult, ConvertToLlmFn, MessageQueueFn, StreamFn,
};
