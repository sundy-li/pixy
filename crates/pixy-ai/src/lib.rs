//! Core abstractions for provider-agnostic LLM streaming.

mod api_registry;
mod error;
mod event_stream;
mod providers;
mod stream;
mod transport_retry;
mod types;
mod validation;

pub use api_registry::{
    ApiProvider, ApiProviderRef, ApiStreamFunction, ApiStreamSimpleFunction, ClosureApiProvider,
    clear_api_providers, get_api_provider, get_api_providers, register_api_provider,
    unregister_api_providers,
};
pub use error::{PiAiError, PiAiErrorCode};
pub use event_stream::{AssistantMessageEventStream, AssistantStreamWriter, EventStream};
pub use providers::{ReliableProvider, register_builtin_api_providers, reset_api_providers};
pub use stream::{complete, complete_simple, stream, stream_simple};
pub use transport_retry::{
    DEFAULT_TRANSPORT_RETRY_COUNT, set_transport_retry_count, transport_retry_count,
    transport_retry_count_with_override,
};
pub use types::{
    Api, AssistantContentBlock, AssistantMessage, AssistantMessageEvent, Context, Cost, DoneReason,
    ErrorReason, Message, Model, Provider, SimpleStreamOptions, StopReason, StreamOptions,
    ThinkingLevel, Tool, ToolResultContentBlock, ToolResultMessage, Usage, UserContent,
    UserContentBlock, UserMessage,
};
pub use validation::{ToolCall, validate_tool_arguments, validate_tool_call};
