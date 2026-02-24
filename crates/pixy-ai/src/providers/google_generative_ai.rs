use std::env;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use reqwest::RequestBuilder;
use serde_json::{Map, Value, json};

use super::common::{empty_assistant_message, join_url, shared_http_client};
use crate::api_registry::{ApiProvider, ApiProviderFuture};
use crate::error::{PiAiError, PiAiErrorCode};
use crate::types::{
    AssistantContentBlock, AssistantMessage, AssistantMessageEvent, Context, DoneReason, Message,
    Model, SimpleStreamOptions, StopReason, StreamOptions, Tool, ToolResultContentBlock, Usage,
    UserContent, UserContentBlock,
};
use crate::{ApiProviderRef, AssistantMessageEventStream};

const GOOGLE_GENERATIVE_AI_FALLBACK_ENVS: &[&str] = &["GOOGLE_API_KEY", "GEMINI_API_KEY"];
const DEFAULT_GOOGLE_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

static GOOGLE_TOOL_CALL_COUNTER: AtomicU64 = AtomicU64::new(0);

struct GoogleGenerativeAiProvider;

impl ApiProvider for GoogleGenerativeAiProvider {
    fn api(&self) -> &str {
        "google-generative-ai"
    }

    fn stream(
        &self,
        model: Model,
        context: Context,
        options: Option<StreamOptions>,
        stream: AssistantMessageEventStream,
    ) -> ApiProviderFuture {
        Box::pin(async move {
            run_google_with_mode(
                model,
                context,
                options,
                GoogleAuthMode::ApiKey,
                GOOGLE_GENERATIVE_AI_FALLBACK_ENVS,
                stream,
            )
            .await
        })
    }

    fn stream_simple(
        &self,
        model: Model,
        context: Context,
        options: Option<SimpleStreamOptions>,
        stream: AssistantMessageEventStream,
    ) -> ApiProviderFuture {
        Box::pin(async move {
            run_simple_google_with_mode(
                model,
                context,
                options,
                GoogleAuthMode::ApiKey,
                GOOGLE_GENERATIVE_AI_FALLBACK_ENVS,
                stream,
            )
            .await
        })
    }
}

pub(super) fn provider() -> ApiProviderRef {
    Arc::new(GoogleGenerativeAiProvider)
}

#[derive(Clone, Copy)]
pub(super) enum GoogleAuthMode {
    ApiKey,
    Bearer,
    Auto,
}

pub(super) async fn run_simple_google_with_mode(
    model: Model,
    context: Context,
    options: Option<SimpleStreamOptions>,
    auth_mode: GoogleAuthMode,
    fallback_envs: &[&str],
    stream: AssistantMessageEventStream,
) -> Result<(), PiAiError> {
    let merged = options.map(|simple| simple.stream);
    run_google_with_mode(model, context, merged, auth_mode, fallback_envs, stream).await
}

pub(super) async fn run_google_with_mode(
    model: Model,
    context: Context,
    options: Option<StreamOptions>,
    auth_mode: GoogleAuthMode,
    fallback_envs: &[&str],
    stream: AssistantMessageEventStream,
) -> Result<(), PiAiError> {
    let api_key = resolve_api_key(&model.provider, options.as_ref(), fallback_envs)?;

    let mut output = empty_assistant_message(&model);
    let payload = build_google_payload(&model, &context, options.as_ref());
    let endpoint = build_google_endpoint(&model);
    let client = shared_http_client(&model.base_url);

    let execution = async {
        let mut request = client
            .post(endpoint.as_str())
            .header("Content-Type", "application/json");
        request = apply_auth(request, &api_key, auth_mode);

        if let Some(headers) = options.as_ref().and_then(|stream| stream.headers.as_ref()) {
            for (name, value) in headers {
                request = request.header(name, value);
            }
        }
        let response = request.json(&payload).send().await.map_err(|error| {
            PiAiError::new(
                PiAiErrorCode::ProviderTransport,
                format!("Google transport failed: {error}"),
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
                format!("Google HTTP {status}: {body}"),
            ));
        }

        let body = response.text().await.map_err(|error| {
            PiAiError::new(
                PiAiErrorCode::ProviderTransport,
                format!("Google response read failed: {error}"),
            )
        })?;

        stream.push(AssistantMessageEvent::Start {
            partial: output.clone(),
        });

        let mut state = GoogleStreamState::default();
        let sse_events = parse_sse_data_events(&body);
        let parsed = if sse_events.is_empty() {
            let payload: Value = serde_json::from_str(&body).map_err(|error| {
                PiAiError::new(
                    PiAiErrorCode::ProviderProtocol,
                    format!("Invalid Google response JSON: {error}"),
                )
                .with_details(json!({ "bodyPrefix": truncate_for_details(&body, 800) }))
            })?;
            apply_google_payload(&payload, &mut output, &stream, &mut state)?
        } else {
            let mut handled_any = false;
            for data in sse_events {
                if data == "[DONE]" {
                    break;
                }
                let payload: Value = serde_json::from_str(&data).map_err(|error| {
                    PiAiError::new(
                        PiAiErrorCode::ProviderProtocol,
                        format!("Invalid Google SSE chunk JSON: {error}"),
                    )
                    .with_details(json!({ "chunk": data }))
                })?;
                handled_any |= apply_google_payload(&payload, &mut output, &stream, &mut state)?;
            }
            handled_any
        };

        close_current_google_block(&mut output, &stream, &mut state);

        if !parsed {
            return Err(PiAiError::new(
                PiAiErrorCode::ProviderProtocol,
                "Google response did not contain candidates/usage fields.",
            )
            .with_details(json!({
                "bodyPrefix": truncate_for_details(&body, 800),
            })));
        }

        if output
            .content
            .iter()
            .any(|block| matches!(block, AssistantContentBlock::ToolCall { .. }))
        {
            output.stop_reason = StopReason::ToolUse;
        }

        match map_done_reason(output.stop_reason.clone()) {
            Some(reason) => {
                stream.push(AssistantMessageEvent::Done {
                    reason,
                    message: output.clone(),
                });
            }
            None => {
                output.error_message = Some("Google returned an error stop reason".to_string());
                stream.push(AssistantMessageEvent::Error {
                    reason: crate::types::ErrorReason::Error,
                    error: output.clone(),
                });
            }
        }

        Ok(())
    }
    .await;

    if let Err(error) = execution {
        output.stop_reason = StopReason::Error;
        output.error_message = Some(error.message.clone());
        stream.push(AssistantMessageEvent::Error {
            reason: crate::types::ErrorReason::Error,
            error: output,
        });
    }

    Ok(())
}

#[derive(Default)]
struct GoogleStreamState {
    current_block: Option<GoogleCurrentBlock>,
}

struct GoogleCurrentBlock {
    index: usize,
    kind: GoogleContentKind,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum GoogleContentKind {
    Text,
    Thinking,
}

fn apply_google_payload(
    payload: &Value,
    output: &mut AssistantMessage,
    stream: &AssistantMessageEventStream,
    state: &mut GoogleStreamState,
) -> Result<bool, PiAiError> {
    let mut handled = false;

    if let Some(usage) = payload
        .get("usageMetadata")
        .or_else(|| payload.get("usage_metadata"))
    {
        update_usage_from_google(&mut output.usage, usage);
        handled = true;
    }

    let candidate = payload
        .get("candidates")
        .and_then(Value::as_array)
        .and_then(|candidates| candidates.first());

    if let Some(candidate) = candidate {
        handled = true;

        if let Some(parts) = candidate
            .get("content")
            .and_then(Value::as_object)
            .and_then(|content| content.get("parts"))
            .and_then(Value::as_array)
        {
            for part in parts {
                apply_google_part(part, output, stream, state)?;
            }
        }

        if let Some(finish_reason) = candidate
            .get("finishReason")
            .or_else(|| candidate.get("finish_reason"))
            .and_then(Value::as_str)
        {
            output.stop_reason = map_google_stop_reason(finish_reason);
        }
    }

    Ok(handled)
}

fn apply_google_part(
    part: &Value,
    output: &mut AssistantMessage,
    stream: &AssistantMessageEventStream,
    state: &mut GoogleStreamState,
) -> Result<(), PiAiError> {
    if let Some(text) = part.get("text").and_then(Value::as_str) {
        let kind = if part
            .get("thought")
            .or_else(|| part.get("isThought"))
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            GoogleContentKind::Thinking
        } else {
            GoogleContentKind::Text
        };

        let signature = part
            .get("thoughtSignature")
            .or_else(|| part.get("thought_signature"))
            .and_then(Value::as_str)
            .map(str::to_string);

        append_google_text_delta(output, stream, state, kind, text, signature);
    }

    if let Some(function_call) = part
        .get("functionCall")
        .or_else(|| part.get("function_call"))
        .and_then(Value::as_object)
    {
        close_current_google_block(output, stream, state);

        let name = function_call
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let id = function_call
            .get("id")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| {
                let sequence = GOOGLE_TOOL_CALL_COUNTER.fetch_add(1, Ordering::Relaxed) + 1;
                format!("google_tool_{sequence}")
            });
        let arguments = function_call
            .get("args")
            .or_else(|| function_call.get("arguments"))
            .cloned()
            .unwrap_or_else(|| Value::Object(Map::new()));
        let thought_signature = part
            .get("thoughtSignature")
            .or_else(|| part.get("thought_signature"))
            .and_then(Value::as_str)
            .map(str::to_string);

        let content_index = output.content.len();
        output.content.push(AssistantContentBlock::ToolCall {
            id: id.clone(),
            name: name.clone(),
            arguments: arguments.clone(),
            thought_signature: thought_signature.clone(),
        });

        stream.push(AssistantMessageEvent::ToolcallStart {
            content_index,
            partial: output.clone(),
        });
        stream.push(AssistantMessageEvent::ToolcallDelta {
            content_index,
            delta: arguments.to_string(),
            partial: output.clone(),
        });
        stream.push(AssistantMessageEvent::ToolcallEnd {
            content_index,
            tool_call: json!({
                "type": "toolCall",
                "id": id,
                "name": name,
                "arguments": arguments,
                "thoughtSignature": thought_signature,
            }),
            partial: output.clone(),
        });
    }

    Ok(())
}

fn append_google_text_delta(
    output: &mut AssistantMessage,
    stream: &AssistantMessageEventStream,
    state: &mut GoogleStreamState,
    kind: GoogleContentKind,
    delta: &str,
    signature: Option<String>,
) {
    let reopen = match state.current_block.as_ref() {
        Some(current) => current.kind != kind,
        None => true,
    };

    if reopen {
        close_current_google_block(output, stream, state);

        let content_index = output.content.len();
        match kind {
            GoogleContentKind::Text => {
                output.content.push(AssistantContentBlock::Text {
                    text: String::new(),
                    text_signature: signature.clone(),
                });
                stream.push(AssistantMessageEvent::TextStart {
                    content_index,
                    partial: output.clone(),
                });
            }
            GoogleContentKind::Thinking => {
                output.content.push(AssistantContentBlock::Thinking {
                    thinking: String::new(),
                    thinking_signature: signature.clone(),
                });
                stream.push(AssistantMessageEvent::ThinkingStart {
                    content_index,
                    partial: output.clone(),
                });
            }
        }

        state.current_block = Some(GoogleCurrentBlock {
            index: content_index,
            kind,
        });
    }

    let Some(current) = state.current_block.as_ref() else {
        return;
    };

    let content_index = current.index;
    match kind {
        GoogleContentKind::Text => {
            if let Some(AssistantContentBlock::Text {
                text,
                text_signature,
            }) = output.content.get_mut(content_index)
            {
                text.push_str(delta);
                if signature.is_some() {
                    *text_signature = signature;
                }
            }

            if !delta.is_empty() {
                stream.push(AssistantMessageEvent::TextDelta {
                    content_index,
                    delta: delta.to_string(),
                    partial: output.clone(),
                });
            }
        }
        GoogleContentKind::Thinking => {
            if let Some(AssistantContentBlock::Thinking {
                thinking,
                thinking_signature,
            }) = output.content.get_mut(content_index)
            {
                thinking.push_str(delta);
                if signature.is_some() {
                    *thinking_signature = signature;
                }
            }

            if !delta.is_empty() {
                stream.push(AssistantMessageEvent::ThinkingDelta {
                    content_index,
                    delta: delta.to_string(),
                    partial: output.clone(),
                });
            }
        }
    }
}

fn close_current_google_block(
    output: &mut AssistantMessage,
    stream: &AssistantMessageEventStream,
    state: &mut GoogleStreamState,
) {
    let Some(current) = state.current_block.take() else {
        return;
    };

    match current.kind {
        GoogleContentKind::Text => {
            stream.push(AssistantMessageEvent::TextEnd {
                content_index: current.index,
                content: extract_text_block(&output.content, current.index),
                partial: output.clone(),
            });
        }
        GoogleContentKind::Thinking => {
            stream.push(AssistantMessageEvent::ThinkingEnd {
                content_index: current.index,
                content: extract_thinking_block(&output.content, current.index),
                partial: output.clone(),
            });
        }
    }
}

fn build_google_payload(
    model: &Model,
    context: &Context,
    options: Option<&StreamOptions>,
) -> Value {
    let mut payload = json!({
        "contents": convert_messages(context),
    });

    if let Some(system_prompt) = &context.system_prompt {
        payload["systemInstruction"] = json!({
            "parts": [{
                "text": system_prompt,
            }],
        });
    }

    if let Some(tools) = &context.tools {
        payload["tools"] = convert_tools(tools);
    }

    let mut generation_config = Map::new();
    if let Some(temperature) = options.and_then(|opts| opts.temperature) {
        generation_config.insert("temperature".to_string(), json!(temperature));
    }
    if let Some(max_tokens) = options.and_then(|opts| opts.max_tokens) {
        generation_config.insert("maxOutputTokens".to_string(), json!(max_tokens));
    } else if model.max_tokens > 0 {
        generation_config.insert("maxOutputTokens".to_string(), json!(model.max_tokens));
    }
    if !generation_config.is_empty() {
        payload["generationConfig"] = Value::Object(generation_config);
    }

    payload
}

fn convert_messages(context: &Context) -> Vec<Value> {
    let mut messages = Vec::new();

    for message in &context.messages {
        match message {
            Message::User { content, .. } => {
                let parts = match content {
                    UserContent::Text(text) => vec![json!({ "text": text })],
                    UserContent::Blocks(blocks) => blocks
                        .iter()
                        .map(|block| match block {
                            UserContentBlock::Text { text, .. } => json!({
                                "text": text,
                            }),
                            UserContentBlock::Image { data, mime_type } => json!({
                                "inlineData": {
                                    "mimeType": mime_type,
                                    "data": data,
                                }
                            }),
                        })
                        .collect(),
                };

                if !parts.is_empty() {
                    messages.push(json!({
                        "role": "user",
                        "parts": parts,
                    }));
                }
            }
            Message::Assistant { content, .. } => {
                let parts: Vec<Value> = content
                    .iter()
                    .map(|block| match block {
                        AssistantContentBlock::Text {
                            text,
                            text_signature,
                        } => {
                            let mut part = json!({ "text": text });
                            if let Some(signature) = text_signature {
                                part["thoughtSignature"] = Value::String(signature.clone());
                            }
                            part
                        }
                        AssistantContentBlock::Thinking {
                            thinking,
                            thinking_signature,
                        } => {
                            let mut part = json!({
                                "text": thinking,
                                "thought": true,
                            });
                            if let Some(signature) = thinking_signature {
                                part["thoughtSignature"] = Value::String(signature.clone());
                            }
                            part
                        }
                        AssistantContentBlock::ToolCall {
                            id,
                            name,
                            arguments,
                            thought_signature,
                        } => {
                            let mut part = json!({
                                "functionCall": {
                                    "name": name,
                                    "args": arguments,
                                }
                            });
                            if !id.is_empty() {
                                part["functionCall"]["id"] = Value::String(id.clone());
                            }
                            if let Some(signature) = thought_signature {
                                part["thoughtSignature"] = Value::String(signature.clone());
                            }
                            part
                        }
                    })
                    .collect();

                if !parts.is_empty() {
                    messages.push(json!({
                        "role": "model",
                        "parts": parts,
                    }));
                }
            }
            Message::ToolResult {
                tool_call_id,
                tool_name,
                content,
                is_error,
                ..
            } => {
                let tool_part = json!({
                    "functionResponse": {
                        "name": tool_name,
                        "id": tool_call_id,
                        "response": convert_tool_result_response(content, *is_error),
                    }
                });

                let merged = if let Some(last) = messages.last_mut() {
                    let is_user = last.get("role").and_then(Value::as_str) == Some("user");
                    let has_function_response = last
                        .get("parts")
                        .and_then(Value::as_array)
                        .map(|parts| {
                            parts
                                .iter()
                                .any(|part| part.get("functionResponse").is_some())
                        })
                        .unwrap_or(false);

                    if is_user && has_function_response {
                        if let Some(parts) = last.get_mut("parts").and_then(Value::as_array_mut) {
                            parts.push(tool_part.clone());
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                } else {
                    false
                };

                if !merged {
                    messages.push(json!({
                        "role": "user",
                        "parts": [tool_part],
                    }));
                }
            }
        }
    }

    messages
}

fn convert_tool_result_response(content: &[ToolResultContentBlock], is_error: bool) -> Value {
    let text = content
        .iter()
        .filter_map(|block| match block {
            ToolResultContentBlock::Text { text, .. } => Some(text.as_str()),
            ToolResultContentBlock::Image { .. } => None,
        })
        .collect::<Vec<_>>()
        .join("\n");

    let value = if text.is_empty() {
        "(no text result)".to_string()
    } else {
        text
    };

    if is_error {
        json!({ "error": value })
    } else {
        json!({ "output": value })
    }
}

fn convert_tools(tools: &[Tool]) -> Value {
    json!([{
        "functionDeclarations": tools
            .iter()
            .map(|tool| {
                json!({
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                })
            })
            .collect::<Vec<_>>()
    }])
}

fn resolve_api_key(
    provider: &str,
    options: Option<&StreamOptions>,
    fallback_envs: &[&str],
) -> Result<String, PiAiError> {
    if let Some(api_key) = options.and_then(|opts| opts.api_key.clone()) {
        if !api_key.trim().is_empty() {
            return Ok(api_key);
        }
    }

    let provider_env = format!("{}_API_KEY", provider.to_uppercase().replace('-', "_"));
    if let Ok(value) = env::var(&provider_env) {
        if !value.trim().is_empty() {
            return Ok(value);
        }
    }

    for env_key in fallback_envs {
        if let Ok(value) = env::var(env_key) {
            if !value.trim().is_empty() {
                return Ok(value);
            }
        }
    }

    let mut env_hints = vec![provider_env];
    env_hints.extend(fallback_envs.iter().map(|value| value.to_string()));

    Err(PiAiError::new(
        PiAiErrorCode::ProviderAuthMissing,
        format!(
            "Missing API key for provider '{}'. Pass `StreamOptions.api_key` or set {}.",
            provider,
            env_hints.join(" / ")
        ),
    ))
}

fn parse_sse_data_events(body: &str) -> Vec<String> {
    let normalized = body.replace("\r\n", "\n");
    normalized
        .split("\n\n")
        .filter_map(|event| {
            let data = event
                .lines()
                .filter_map(|line| line.strip_prefix("data:").map(str::trim_start))
                .collect::<Vec<_>>()
                .join("\n");
            if data.is_empty() { None } else { Some(data) }
        })
        .collect()
}

fn apply_auth(request: RequestBuilder, token: &str, mode: GoogleAuthMode) -> RequestBuilder {
    match mode {
        GoogleAuthMode::ApiKey => request.header("x-goog-api-key", token),
        GoogleAuthMode::Bearer => request.header("Authorization", format!("Bearer {token}")),
        GoogleAuthMode::Auto => {
            if looks_like_google_api_key(token) {
                request.header("x-goog-api-key", token)
            } else {
                request.header("Authorization", format!("Bearer {token}"))
            }
        }
    }
}

fn looks_like_google_api_key(token: &str) -> bool {
    token.starts_with("AIza")
}

fn map_google_stop_reason(reason: &str) -> StopReason {
    match reason {
        "STOP" => StopReason::Stop,
        "MAX_TOKENS" => StopReason::Length,
        "MALFORMED_FUNCTION_CALL" | "UNEXPECTED_TOOL_CALL" => StopReason::Error,
        _ => StopReason::Stop,
    }
}

fn map_done_reason(reason: StopReason) -> Option<DoneReason> {
    match reason {
        StopReason::Stop => Some(DoneReason::Stop),
        StopReason::Length => Some(DoneReason::Length),
        StopReason::ToolUse => Some(DoneReason::ToolUse),
        StopReason::Error | StopReason::Aborted => None,
    }
}

fn update_usage_from_google(usage: &mut Usage, value: &Value) {
    let prompt_tokens = value
        .get("promptTokenCount")
        .or_else(|| value.get("prompt_token_count"))
        .and_then(Value::as_u64)
        .unwrap_or(usage.input + usage.cache_read);

    let output_tokens = value
        .get("candidatesTokenCount")
        .or_else(|| value.get("candidates_token_count"))
        .and_then(Value::as_u64)
        .unwrap_or(usage.output);

    let thoughts_tokens = value
        .get("thoughtsTokenCount")
        .or_else(|| value.get("thoughts_token_count"))
        .and_then(Value::as_u64)
        .unwrap_or(0);

    let cache_read = value
        .get("cachedContentTokenCount")
        .or_else(|| value.get("cached_content_token_count"))
        .and_then(Value::as_u64)
        .unwrap_or(usage.cache_read);

    let total_tokens = value
        .get("totalTokenCount")
        .or_else(|| value.get("total_token_count"))
        .and_then(Value::as_u64)
        .unwrap_or(prompt_tokens + output_tokens + thoughts_tokens);

    usage.input = prompt_tokens.saturating_sub(cache_read);
    usage.output = output_tokens + thoughts_tokens;
    usage.cache_read = cache_read;
    usage.cache_write = 0;
    usage.total_tokens = total_tokens.max(usage.input + usage.output + usage.cache_read);
}

fn build_google_endpoint(model: &Model) -> String {
    let base = if model.base_url.trim().is_empty() {
        DEFAULT_GOOGLE_BASE_URL
    } else {
        model.base_url.as_str()
    };
    let path = build_google_model_path(&model.id);
    join_url(base, &path)
}

fn build_google_model_path(model_id: &str) -> String {
    let trimmed = model_id.trim().trim_start_matches('/');
    let without_suffix = trimmed.strip_suffix(":generateContent").unwrap_or(trimmed);

    if without_suffix.starts_with("models/")
        || without_suffix.starts_with("projects/")
        || without_suffix.contains("/models/")
    {
        format!("{without_suffix}:generateContent")
    } else {
        format!("models/{without_suffix}:generateContent")
    }
}

fn extract_text_block(content: &[AssistantContentBlock], index: usize) -> String {
    content
        .get(index)
        .and_then(|block| match block {
            AssistantContentBlock::Text { text, .. } => Some(text.clone()),
            _ => None,
        })
        .unwrap_or_default()
}

fn extract_thinking_block(content: &[AssistantContentBlock], index: usize) -> String {
    content
        .get(index)
        .and_then(|block| match block {
            AssistantContentBlock::Thinking { thinking, .. } => Some(thinking.clone()),
            _ => None,
        })
        .unwrap_or_default()
}

fn truncate_for_details(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    if limit <= 3 {
        return ".".repeat(limit);
    }
    let prefix: String = text.chars().take(limit - 3).collect();
    format!("{prefix}...")
}
