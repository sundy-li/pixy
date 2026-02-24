use crate::api_registry::get_api_provider;
use crate::error::{PiAiError, PiAiErrorCode};
use crate::providers::ensure_builtin_api_providers_registered;
use crate::types::{
    AssistantMessage, Context, Cost, Model, SimpleStreamOptions, StopReason, StreamOptions, Usage,
};
use crate::{AssistantMessageEventStream, AssistantStreamWriter};

fn resolve_provider(api: &str) -> Result<crate::ApiProviderRef, PiAiError> {
    ensure_builtin_api_providers_registered();
    get_api_provider(api).ok_or_else(|| {
        PiAiError::new(
            PiAiErrorCode::ProviderProtocol,
            format!("No API provider registered for api: {api}"),
        )
    })
}

pub fn stream(
    model: Model,
    context: Context,
    options: Option<StreamOptions>,
) -> Result<AssistantMessageEventStream, PiAiError> {
    let provider = resolve_provider(&model.api)?;
    let stream = AssistantMessageEventStream::new();
    let writer = AssistantStreamWriter::new(stream.clone());
    let error_model = model.clone();
    spawn_provider_task(async move {
        let result = provider
            .stream(model, context, options, writer.stream())
            .await;
        if let Err(error) = result {
            writer.error(
                crate::types::ErrorReason::Error,
                transport_error_message(&error_model, error),
            );
        }
        writer.close();
    });
    Ok(stream)
}

pub async fn complete(
    model: Model,
    context: Context,
    options: Option<StreamOptions>,
) -> Result<AssistantMessage, PiAiError> {
    let event_stream = stream(model, context, options)?;
    event_stream.result().await.ok_or_else(|| {
        PiAiError::new(
            PiAiErrorCode::ProviderProtocol,
            "Stream ended without terminal message",
        )
    })
}

pub fn stream_simple(
    model: Model,
    context: Context,
    options: Option<SimpleStreamOptions>,
) -> Result<AssistantMessageEventStream, PiAiError> {
    let provider = resolve_provider(&model.api)?;
    let stream = AssistantMessageEventStream::new();
    let writer = AssistantStreamWriter::new(stream.clone());
    let error_model = model.clone();
    spawn_provider_task(async move {
        let result = provider
            .stream_simple(model, context, options, writer.stream())
            .await;
        if let Err(error) = result {
            writer.error(
                crate::types::ErrorReason::Error,
                transport_error_message(&error_model, error),
            );
        }
        writer.close();
    });
    Ok(stream)
}

pub async fn complete_simple(
    model: Model,
    context: Context,
    options: Option<SimpleStreamOptions>,
) -> Result<AssistantMessage, PiAiError> {
    let event_stream = stream_simple(model, context, options)?;
    event_stream.result().await.ok_or_else(|| {
        PiAiError::new(
            PiAiErrorCode::ProviderProtocol,
            "Stream ended without terminal message",
        )
    })
}

fn transport_error_message(model: &Model, error: PiAiError) -> AssistantMessage {
    AssistantMessage {
        role: "assistant".to_string(),
        content: vec![],
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        usage: Usage {
            input: 0,
            output: 0,
            cache_read: 0,
            cache_write: 0,
            total_tokens: 0,
            cost: Cost {
                input: 0.0,
                output: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
                total: 0.0,
            },
        },
        stop_reason: StopReason::Error,
        error_message: Some(error.as_compact_json()),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_millis() as i64)
            .unwrap_or(0),
    }
}

fn spawn_provider_task<F>(task: F)
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        handle.spawn(task);
        return;
    }

    std::thread::spawn(move || {
        if let Ok(runtime) = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            runtime.block_on(task);
        }
    });
}
