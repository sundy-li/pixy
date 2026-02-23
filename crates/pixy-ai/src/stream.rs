use crate::AssistantMessageEventStream;
use crate::api_registry::StreamResult;
use crate::api_registry::get_api_provider;
use crate::providers::ensure_builtin_api_providers_registered;
use crate::types::{AssistantMessage, Context, Model, SimpleStreamOptions, StreamOptions};

fn resolve_provider(api: &str) -> StreamResult<crate::ApiProviderRef> {
    ensure_builtin_api_providers_registered();
    get_api_provider(api).ok_or_else(|| format!("No API provider registered for api: {api}"))
}

pub fn stream(
    model: Model,
    context: Context,
    options: Option<StreamOptions>,
) -> StreamResult<AssistantMessageEventStream> {
    let provider = resolve_provider(&model.api)?;
    provider.stream(model, context, options)
}

pub async fn complete(
    model: Model,
    context: Context,
    options: Option<StreamOptions>,
) -> StreamResult<AssistantMessage> {
    let event_stream = stream(model, context, options)?;
    event_stream
        .result()
        .await
        .ok_or_else(|| "Stream ended without terminal message".to_string())
}

pub fn stream_simple(
    model: Model,
    context: Context,
    options: Option<SimpleStreamOptions>,
) -> StreamResult<AssistantMessageEventStream> {
    let provider = resolve_provider(&model.api)?;
    provider.stream_simple(model, context, options)
}

pub async fn complete_simple(
    model: Model,
    context: Context,
    options: Option<SimpleStreamOptions>,
) -> StreamResult<AssistantMessage> {
    let event_stream = stream_simple(model, context, options)?;
    event_stream
        .result()
        .await
        .ok_or_else(|| "Stream ended without terminal message".to_string())
}
