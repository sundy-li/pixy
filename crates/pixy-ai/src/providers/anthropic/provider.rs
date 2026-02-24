use std::env;
use std::sync::Arc;

use super::parser::apply_response_body;
use super::payload::build_anthropic_payload;
use crate::api_registry::{ApiProvider, ApiProviderFuture};
use crate::error::{PiAiError, PiAiErrorCode};
use crate::providers::common::{empty_assistant_message, join_url, shared_http_client};
use crate::types::{AssistantMessageEvent, Model, SimpleStreamOptions, StopReason, StreamOptions};
use crate::{ApiProviderRef, AssistantMessageEventStream};

struct AnthropicProvider;

impl ApiProvider for AnthropicProvider {
    fn api(&self) -> &str {
        "anthropic-messages"
    }

    fn stream(
        &self,
        model: Model,
        context: crate::types::Context,
        options: Option<StreamOptions>,
        stream: AssistantMessageEventStream,
    ) -> ApiProviderFuture {
        Box::pin(async move { run_anthropic(model, context, options, stream).await })
    }

    fn stream_simple(
        &self,
        model: Model,
        context: crate::types::Context,
        options: Option<SimpleStreamOptions>,
        stream: AssistantMessageEventStream,
    ) -> ApiProviderFuture {
        Box::pin(async move { run_simple_anthropic(model, context, options, stream).await })
    }
}

pub(crate) fn provider() -> ApiProviderRef {
    Arc::new(AnthropicProvider)
}

async fn run_anthropic(
    model: Model,
    context: crate::types::Context,
    options: Option<StreamOptions>,
    stream: AssistantMessageEventStream,
) -> Result<(), PiAiError> {
    let api_key = resolve_api_key(&model.provider, options.as_ref())?;
    let mut output = empty_assistant_message(&model);
    let payload = build_anthropic_payload(&model, &context, options.as_ref(), false);
    let endpoint = join_url(&model.base_url, "messages");
    let client = shared_http_client(&model.base_url);

    let execution = async {
        let mut request = client
            .post(endpoint.as_str())
            .header("x-api-key", api_key.as_str())
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json");

        if let Some(headers) = options.as_ref().and_then(|stream| stream.headers.as_ref()) {
            for (name, value) in headers {
                request = request.header(name, value);
            }
        }

        let response = request.json(&payload).send().await.map_err(|error| {
            PiAiError::new(
                PiAiErrorCode::ProviderTransport,
                format!("Anthropic transport failed: {error}"),
            )
        })?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "unable to read error body".to_string());
            return Err(PiAiError::new(
                PiAiErrorCode::ProviderHttp,
                format!("Anthropic HTTP {status}: {body}"),
            ));
        }

        let body = response.text().await.map_err(|error| {
            PiAiError::new(
                PiAiErrorCode::ProviderTransport,
                format!("Anthropic stream read failed: {error}"),
            )
        })?;

        stream.push(AssistantMessageEvent::Start {
            partial: output.clone(),
        });
        apply_response_body(&body, &mut output, &stream)
    }
    .await;

    if let Err(error) = execution {
        output.stop_reason = StopReason::Error;
        output.error_message = Some(error.as_compact_json());
        stream.push(AssistantMessageEvent::Error {
            reason: crate::types::ErrorReason::Error,
            error: output,
        });
    }

    Ok(())
}

async fn run_simple_anthropic(
    model: Model,
    context: crate::types::Context,
    options: Option<SimpleStreamOptions>,
    stream: AssistantMessageEventStream,
) -> Result<(), PiAiError> {
    let merged = options.map(|simple| {
        let mut stream = simple.stream;
        if simple.reasoning.is_some() && model.reasoning {
            let mut headers = stream.headers.unwrap_or_default();
            headers.insert("x-pi-thinking".to_string(), "enabled".to_string());
            stream.headers = Some(headers);
        }
        stream
    });

    run_anthropic(model, context, merged, stream).await
}

fn resolve_api_key(provider: &str, options: Option<&StreamOptions>) -> Result<String, PiAiError> {
    if let Some(api_key) = options.and_then(|options| options.api_key.clone()) {
        return Ok(api_key);
    }

    let provider_env = format!("{}_API_KEY", provider.to_uppercase().replace('-', "_"));
    if let Ok(value) = env::var(&provider_env) {
        if !value.trim().is_empty() {
            return Ok(value);
        }
    }

    if let Ok(value) = env::var("ANTHROPIC_API_KEY") {
        if !value.trim().is_empty() {
            return Ok(value);
        }
    }

    Err(PiAiError::new(
        PiAiErrorCode::ProviderAuthMissing,
        format!(
            "Missing API key for provider '{}'. Pass `StreamOptions.api_key` or set {} / ANTHROPIC_API_KEY.",
            provider, provider_env
        ),
    ))
}
