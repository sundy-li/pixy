//! Core abstractions for provider-agnostic LLM streaming.

mod api_registry;
mod error;
mod event_stream;
mod providers;
mod stream;
mod types;
mod validation;

pub use api_registry::{
    ApiProvider, ApiProviderRef, ApiStreamFunction, ApiStreamSimpleFunction, ClosureApiProvider,
    clear_api_providers, get_api_provider, get_api_providers, register_api_provider,
    unregister_api_providers,
};
pub use error::{PiAiError, PiAiErrorCode};
pub use event_stream::{AssistantMessageEventStream, EventStream};
pub use providers::{register_builtin_api_providers, reset_api_providers};
pub use stream::{complete, complete_simple, stream, stream_simple};
pub use types::{
    Api, AssistantContentBlock, AssistantMessage, AssistantMessageEvent, Context, Cost, DoneReason,
    ErrorReason, Message, Model, Provider, SimpleStreamOptions, StopReason, StreamOptions,
    ThinkingLevel, Tool, ToolResultContentBlock, ToolResultMessage, Usage, UserContent,
    UserContentBlock, UserMessage,
};
pub use validation::{ToolCall, validate_tool_arguments, validate_tool_call};
