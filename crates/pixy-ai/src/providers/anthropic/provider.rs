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
    let payload = build_anthropic_payload(&model, &context, options.as_ref(), model.reasoning);
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::Duration;

    use crate::types::{Context, Cost, Message, UserContent};

    #[test]
    fn run_anthropic_request_enables_thinking_when_reasoning_is_enabled() {
        let response_body = r#"{
            "type":"message",
            "role":"assistant",
            "content":[{"type":"text","text":"ok"}],
            "stop_reason":"end_turn",
            "usage":{"input_tokens":1,"output_tokens":1}
        }"#
        .to_string();
        let (base_url, captured_body) = spawn_inspecting_server(response_body);

        let model = sample_model(base_url, true);
        let context = sample_context();
        let options = StreamOptions {
            api_key: Some("test-api-key".to_string()),
            ..StreamOptions::default()
        };

        let runtime = tokio::runtime::Runtime::new().expect("create runtime");
        runtime
            .block_on(run_anthropic(
                model,
                context,
                Some(options),
                AssistantMessageEventStream::new(),
            ))
            .expect("anthropic run succeeds");

        let request_body = captured_body
            .lock()
            .expect("capture lock")
            .clone()
            .expect("request body captured");
        assert!(
            request_body.contains("\"thinking\""),
            "expected request payload to include thinking block, got: {request_body}"
        );
    }

    fn sample_model(base_url: String, reasoning: bool) -> Model {
        Model {
            id: "claude-test".to_string(),
            name: "claude-test".to_string(),
            api: "anthropic-messages".to_string(),
            provider: "anthropic".to_string(),
            base_url,
            reasoning,
            reasoning_effort: None,
            input: vec!["text".to_string()],
            cost: Cost {
                input: 0.0,
                output: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
                total: 0.0,
            },
            context_window: 200_000,
            max_tokens: 8_192,
        }
    }

    fn sample_context() -> Context {
        Context {
            system_prompt: Some("You are a helpful assistant".to_string()),
            messages: vec![Message::User {
                content: UserContent::Text("hello".to_string()),
                timestamp: 1_700_000_000_000,
            }],
            tools: None,
        }
    }

    fn spawn_inspecting_server(response_body: String) -> (String, Arc<Mutex<Option<String>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let address = listener.local_addr().expect("server local addr");
        let captured_body = Arc::new(Mutex::new(None));
        let captured_body_thread = Arc::clone(&captured_body);

        thread::spawn(move || {
            if let Ok((mut socket, _)) = listener.accept() {
                socket
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .expect("set read timeout");

                let request = read_http_request(&mut socket);
                *captured_body_thread.lock().expect("capture lock") = request;

                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    response_body.len(),
                    response_body
                );
                socket
                    .write_all(response.as_bytes())
                    .expect("write response");
                let _ = socket.flush();
            }
        });

        (format!("http://{address}/v1"), captured_body)
    }

    fn read_http_request(socket: &mut std::net::TcpStream) -> Option<String> {
        let mut buffer = [0_u8; 16_384];
        let read_len = socket.read(&mut buffer).ok()?;
        if read_len == 0 {
            return None;
        }
        let request = String::from_utf8_lossy(&buffer[..read_len]).to_string();
        let body_start = request.find("\r\n\r\n")?;
        let body = request[(body_start + 4)..].to_string();
        Some(body)
    }
}
