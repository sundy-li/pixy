use std::env;
use std::sync::Arc;

use reqwest::blocking::Client;
use serde_json::{Map, Value, json};

use crate::api_registry::{ApiProvider, StreamResult};
use crate::error::{PiAiError, PiAiErrorCode};
use crate::types::{
    AssistantContentBlock, AssistantMessage, AssistantMessageEvent, Context, Cost, DoneReason,
    Message, Model, SimpleStreamOptions, StopReason, StreamOptions, Tool, ToolResultContentBlock,
    Usage, UserContent, UserContentBlock,
};
use crate::{ApiProviderRef, AssistantMessageEventStream};

const BEDROCK_FALLBACK_ENVS: &[&str] = &["AWS_BEARER_TOKEN_BEDROCK"];
const DEFAULT_BEDROCK_BASE_URL: &str = "https://bedrock-runtime.us-east-1.amazonaws.com";

struct BedrockConverseStreamProvider;

impl ApiProvider for BedrockConverseStreamProvider {
    fn api(&self) -> &str {
        "bedrock-converse-stream"
    }

    fn stream(
        &self,
        model: Model,
        context: Context,
        options: Option<StreamOptions>,
    ) -> Result<AssistantMessageEventStream, String> {
        stream_bedrock_converse_stream(model, context, options)
    }

    fn stream_simple(
        &self,
        model: Model,
        context: Context,
        options: Option<SimpleStreamOptions>,
    ) -> Result<AssistantMessageEventStream, String> {
        stream_simple_bedrock_converse_stream(model, context, options)
    }
}

pub(super) fn provider() -> ApiProviderRef {
    Arc::new(BedrockConverseStreamProvider)
}

pub fn stream_bedrock_converse_stream(
    model: Model,
    context: Context,
    options: Option<StreamOptions>,
) -> StreamResult<AssistantMessageEventStream> {
    let auth_token = resolve_auth_token(&model.provider, options.as_ref(), BEDROCK_FALLBACK_ENVS)
        .map_err(|error| error.as_compact_json())?;
    let stream = AssistantMessageEventStream::new();

    let mut output = empty_assistant_message(&model);
    let payload = build_bedrock_payload(&context, options.as_ref());
    let endpoint = build_bedrock_endpoint(&model);
    let client = Client::new();

    let execution = (|| -> Result<(), PiAiError> {
        let mut request = client
            .post(endpoint)
            .header("Content-Type", "application/json");

        if let Some(token) = auth_token.as_ref() {
            request = request.header("Authorization", format!("Bearer {token}"));
        }

        if let Some(headers) = options.as_ref().and_then(|stream| stream.headers.as_ref()) {
            for (name, value) in headers {
                request = request.header(name, value);
            }
        }

        let response = request.json(&payload).send().map_err(|error| {
            PiAiError::new(
                PiAiErrorCode::ProviderTransport,
                format!("Bedrock transport failed: {error}"),
            )
        })?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response
                .text()
                .unwrap_or_else(|_| "unable to read error body".to_string());
            return Err(PiAiError::new(
                PiAiErrorCode::ProviderHttp,
                format!("Bedrock HTTP {status}: {body}"),
            ));
        }

        let body = response.text().map_err(|error| {
            PiAiError::new(
                PiAiErrorCode::ProviderTransport,
                format!("Bedrock response read failed: {error}"),
            )
        })?;
        let parsed: Value = serde_json::from_str(&body).map_err(|error| {
            PiAiError::new(
                PiAiErrorCode::ProviderProtocol,
                format!("Invalid Bedrock response JSON: {error}"),
            )
            .with_details(json!({ "bodyPrefix": truncate_for_details(&body, 800) }))
        })?;

        stream.push(AssistantMessageEvent::Start {
            partial: output.clone(),
        });

        apply_bedrock_response(&parsed, &mut output, &stream)?;

        if output
            .content
            .iter()
            .any(|block| matches!(block, AssistantContentBlock::ToolCall { .. }))
        {
            output.stop_reason = StopReason::ToolUse;
        }

        match map_done_reason(output.stop_reason.clone()) {
            Some(reason) => stream.push(AssistantMessageEvent::Done {
                reason,
                message: output.clone(),
            }),
            None => {
                output.error_message = Some("Bedrock returned an error stop reason".to_string());
                stream.push(AssistantMessageEvent::Error {
                    reason: crate::types::ErrorReason::Error,
                    error: output.clone(),
                });
            }
        }

        Ok(())
    })();

    if let Err(error) = execution {
        output.stop_reason = StopReason::Error;
        output.error_message = Some(error.message.clone());
        stream.push(AssistantMessageEvent::Error {
            reason: crate::types::ErrorReason::Error,
            error: output,
        });
    }

    Ok(stream)
}

pub fn stream_simple_bedrock_converse_stream(
    model: Model,
    context: Context,
    options: Option<SimpleStreamOptions>,
) -> StreamResult<AssistantMessageEventStream> {
    let merged = options.map(|simple| simple.stream);
    stream_bedrock_converse_stream(model, context, merged)
}

fn build_bedrock_payload(context: &Context, options: Option<&StreamOptions>) -> Value {
    let mut payload = json!({
        "messages": convert_messages(context),
    });

    if let Some(system_prompt) = &context.system_prompt {
        payload["system"] = json!([{ "text": system_prompt }]);
    }

    if let Some(tools) = &context.tools {
        payload["toolConfig"] = convert_tools(tools);
    }

    let mut inference = Map::new();
    if let Some(temperature) = options.and_then(|opts| opts.temperature) {
        inference.insert("temperature".to_string(), json!(temperature));
    }
    if let Some(max_tokens) = options.and_then(|opts| opts.max_tokens) {
        inference.insert("maxTokens".to_string(), json!(max_tokens));
    }
    if !inference.is_empty() {
        payload["inferenceConfig"] = Value::Object(inference);
    }

    payload
}

fn convert_messages(context: &Context) -> Vec<Value> {
    let mut converted = Vec::new();
    let mut index = 0;

    while index < context.messages.len() {
        match &context.messages[index] {
            Message::User { content, .. } => {
                let blocks = match content {
                    UserContent::Text(text) => vec![json!({ "text": text })],
                    UserContent::Blocks(blocks) => blocks
                        .iter()
                        .filter_map(convert_user_block_to_bedrock)
                        .collect(),
                };
                if !blocks.is_empty() {
                    converted.push(json!({
                        "role": "user",
                        "content": blocks,
                    }));
                }
            }
            Message::Assistant { content, .. } => {
                let blocks: Vec<Value> = content
                    .iter()
                    .filter_map(|block| match block {
                        AssistantContentBlock::Text { text, .. } => Some(json!({ "text": text })),
                        AssistantContentBlock::Thinking { thinking, .. } => Some(json!({
                            "reasoningContent": {
                                "reasoningText": {
                                    "text": thinking
                                }
                            }
                        })),
                        AssistantContentBlock::ToolCall {
                            id,
                            name,
                            arguments,
                            ..
                        } => Some(json!({
                            "toolUse": {
                                "toolUseId": id,
                                "name": name,
                                "input": arguments,
                            }
                        })),
                    })
                    .collect();

                if !blocks.is_empty() {
                    converted.push(json!({
                        "role": "assistant",
                        "content": blocks,
                    }));
                }
            }
            Message::ToolResult { .. } => {
                let mut tool_results = Vec::new();
                let mut next = index;
                while next < context.messages.len() {
                    let Message::ToolResult {
                        tool_call_id,
                        content,
                        is_error,
                        ..
                    } = &context.messages[next]
                    else {
                        break;
                    };

                    tool_results.push(json!({
                        "toolResult": {
                            "toolUseId": tool_call_id,
                            "content": convert_tool_result_content(content),
                            "status": if *is_error { "error" } else { "success" },
                        }
                    }));

                    next += 1;
                }

                if !tool_results.is_empty() {
                    converted.push(json!({
                        "role": "user",
                        "content": tool_results,
                    }));
                }

                index = next.saturating_sub(1);
            }
        }

        index += 1;
    }

    converted
}

fn convert_user_block_to_bedrock(block: &UserContentBlock) -> Option<Value> {
    match block {
        UserContentBlock::Text { text, .. } => Some(json!({ "text": text })),
        UserContentBlock::Image { data, mime_type } => {
            let format = map_bedrock_image_format(mime_type)?;
            Some(json!({
                "image": {
                    "format": format,
                    "source": {
                        "bytes": data,
                    }
                }
            }))
        }
    }
}

fn map_bedrock_image_format(mime_type: &str) -> Option<&'static str> {
    match mime_type {
        "image/jpeg" | "image/jpg" => Some("jpeg"),
        "image/png" => Some("png"),
        "image/gif" => Some("gif"),
        "image/webp" => Some("webp"),
        _ => None,
    }
}

fn convert_tool_result_content(content: &[ToolResultContentBlock]) -> Value {
    let blocks: Vec<Value> = content
        .iter()
        .filter_map(|block| match block {
            ToolResultContentBlock::Text { text, .. } => Some(json!({ "text": text })),
            ToolResultContentBlock::Image { data, mime_type } => {
                map_bedrock_image_format(mime_type).map(|format| {
                    json!({
                        "image": {
                            "format": format,
                            "source": { "bytes": data },
                        }
                    })
                })
            }
        })
        .collect();

    if blocks.is_empty() {
        json!([{ "text": "(no text result)" }])
    } else {
        Value::Array(blocks)
    }
}

fn convert_tools(tools: &[Tool]) -> Value {
    json!({
        "tools": tools
            .iter()
            .map(|tool| {
                json!({
                    "toolSpec": {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": { "json": tool.parameters },
                    }
                })
            })
            .collect::<Vec<_>>(),
    })
}

fn apply_bedrock_response(
    parsed: &Value,
    output: &mut AssistantMessage,
    stream: &AssistantMessageEventStream,
) -> Result<(), PiAiError> {
    if let Some(stop_reason) = parsed
        .get("stopReason")
        .or_else(|| parsed.get("stop_reason"))
        .and_then(Value::as_str)
    {
        output.stop_reason = map_bedrock_stop_reason(stop_reason);
    }

    if let Some(usage) = parsed.get("usage") {
        update_usage_from_bedrock(&mut output.usage, usage);
    }

    let content_blocks = parsed
        .get("output")
        .and_then(Value::as_object)
        .and_then(|output| output.get("message"))
        .and_then(Value::as_object)
        .and_then(|message| message.get("content"))
        .and_then(Value::as_array)
        .ok_or_else(|| {
            PiAiError::new(
                PiAiErrorCode::ProviderProtocol,
                "Bedrock response missing `output.message.content` array",
            )
            .with_details(json!({ "payload": parsed }))
        })?;

    for block in content_blocks {
        if let Some(text) = block.get("text").and_then(Value::as_str) {
            let content_index = output.content.len();
            output.content.push(AssistantContentBlock::Text {
                text: text.to_string(),
                text_signature: None,
            });
            stream.push(AssistantMessageEvent::TextStart {
                content_index,
                partial: output.clone(),
            });
            if !text.is_empty() {
                stream.push(AssistantMessageEvent::TextDelta {
                    content_index,
                    delta: text.to_string(),
                    partial: output.clone(),
                });
            }
            stream.push(AssistantMessageEvent::TextEnd {
                content_index,
                content: text.to_string(),
                partial: output.clone(),
            });
        }

        if let Some(thinking) = block
            .get("reasoningContent")
            .and_then(Value::as_object)
            .and_then(|reasoning| reasoning.get("reasoningText"))
            .and_then(Value::as_object)
            .and_then(|reasoning| reasoning.get("text"))
            .and_then(Value::as_str)
        {
            let content_index = output.content.len();
            output.content.push(AssistantContentBlock::Thinking {
                thinking: thinking.to_string(),
                thinking_signature: None,
            });
            stream.push(AssistantMessageEvent::ThinkingStart {
                content_index,
                partial: output.clone(),
            });
            if !thinking.is_empty() {
                stream.push(AssistantMessageEvent::ThinkingDelta {
                    content_index,
                    delta: thinking.to_string(),
                    partial: output.clone(),
                });
            }
            stream.push(AssistantMessageEvent::ThinkingEnd {
                content_index,
                content: thinking.to_string(),
                partial: output.clone(),
            });
        }

        if let Some(tool_use) = block
            .get("toolUse")
            .or_else(|| block.get("tool_use"))
            .and_then(Value::as_object)
        {
            let id = tool_use
                .get("toolUseId")
                .or_else(|| tool_use.get("tool_use_id"))
                .or_else(|| tool_use.get("id"))
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let name = tool_use
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let arguments = tool_use
                .get("input")
                .cloned()
                .unwrap_or_else(|| Value::Object(Map::new()));

            let content_index = output.content.len();
            output.content.push(AssistantContentBlock::ToolCall {
                id: id.clone(),
                name: name.clone(),
                arguments: arguments.clone(),
                thought_signature: None,
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
                    "thoughtSignature": Value::Null,
                }),
                partial: output.clone(),
            });
        }
    }

    Ok(())
}

fn resolve_auth_token(
    provider: &str,
    options: Option<&StreamOptions>,
    fallback_envs: &[&str],
) -> Result<Option<String>, PiAiError> {
    if let Some(api_key) = options.and_then(|opts| opts.api_key.clone()) {
        if !api_key.trim().is_empty() {
            return Ok(Some(api_key));
        }
    }

    if let Some(headers) = options.and_then(|opts| opts.headers.as_ref()) {
        let has_auth = headers.iter().any(|(name, value)| {
            name.eq_ignore_ascii_case("authorization") && !value.trim().is_empty()
        });
        if has_auth {
            return Ok(None);
        }
    }

    let provider_env = format!("{}_API_KEY", provider.to_uppercase().replace('-', "_"));
    if let Ok(value) = env::var(&provider_env) {
        if !value.trim().is_empty() {
            return Ok(Some(value));
        }
    }

    for env_key in fallback_envs {
        if let Ok(value) = env::var(env_key) {
            if !value.trim().is_empty() {
                return Ok(Some(value));
            }
        }
    }

    let mut env_hints = vec![provider_env];
    env_hints.extend(fallback_envs.iter().map(|value| value.to_string()));

    Err(PiAiError::new(
        PiAiErrorCode::ProviderAuthMissing,
        format!(
            "Missing auth for provider '{}'. Pass `StreamOptions.api_key`, set {} or provide Authorization in `StreamOptions.headers`.",
            provider,
            env_hints.join(" / ")
        ),
    ))
}

fn map_bedrock_stop_reason(reason: &str) -> StopReason {
    match reason {
        "end_turn" | "stop_sequence" => StopReason::Stop,
        "max_tokens" | "model_context_window_exceeded" => StopReason::Length,
        "tool_use" => StopReason::ToolUse,
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

fn update_usage_from_bedrock(usage: &mut Usage, value: &Value) {
    let input_tokens = value
        .get("inputTokens")
        .or_else(|| value.get("input_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(usage.input);
    let output_tokens = value
        .get("outputTokens")
        .or_else(|| value.get("output_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(usage.output);
    let cache_read = value
        .get("cacheReadInputTokens")
        .or_else(|| value.get("cache_read_input_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(usage.cache_read);
    let cache_write = value
        .get("cacheWriteInputTokens")
        .or_else(|| value.get("cache_write_input_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(usage.cache_write);
    let total_tokens = value
        .get("totalTokens")
        .or_else(|| value.get("total_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(input_tokens + output_tokens + cache_read + cache_write);

    usage.input = input_tokens;
    usage.output = output_tokens;
    usage.cache_read = cache_read;
    usage.cache_write = cache_write;
    usage.total_tokens = total_tokens;
}

fn build_bedrock_endpoint(model: &Model) -> String {
    let base = if model.base_url.trim().is_empty() {
        DEFAULT_BEDROCK_BASE_URL
    } else {
        model.base_url.as_str()
    };

    if base.ends_with("/converse") || base.ends_with("/converse-stream") {
        return base.to_string();
    }

    let encoded_model_id = percent_encode_path_segment(model.id.as_str());
    join_url(base, &format!("model/{encoded_model_id}/converse"))
}

fn percent_encode_path_segment(input: &str) -> String {
    let mut encoded = String::new();
    for byte in input.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(byte as char)
            }
            _ => encoded.push_str(&format!("%{byte:02X}")),
        }
    }
    encoded
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

fn empty_assistant_message(model: &Model) -> AssistantMessage {
    AssistantMessage {
        role: "assistant".to_string(),
        content: Vec::new(),
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
        stop_reason: StopReason::Stop,
        error_message: None,
        timestamp: now_millis(),
    }
}

fn join_url(base_url: &str, path: &str) -> String {
    if base_url.ends_with('/') {
        format!("{base_url}{path}")
    } else {
        format!("{base_url}/{path}")
    }
}

fn now_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0)
}
