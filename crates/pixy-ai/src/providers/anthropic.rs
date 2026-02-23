use std::collections::HashMap;
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

struct AnthropicProvider;

impl ApiProvider for AnthropicProvider {
    fn api(&self) -> &str {
        "anthropic-messages"
    }

    fn stream(
        &self,
        model: Model,
        context: Context,
        options: Option<StreamOptions>,
    ) -> Result<AssistantMessageEventStream, String> {
        stream_anthropic(model, context, options)
    }

    fn stream_simple(
        &self,
        model: Model,
        context: Context,
        options: Option<SimpleStreamOptions>,
    ) -> Result<AssistantMessageEventStream, String> {
        stream_simple_anthropic(model, context, options)
    }
}

pub(super) fn provider() -> ApiProviderRef {
    Arc::new(AnthropicProvider)
}

pub fn stream_anthropic(
    model: Model,
    context: Context,
    options: Option<StreamOptions>,
) -> StreamResult<AssistantMessageEventStream> {
    let api_key = resolve_api_key(&model.provider, options.as_ref())
        .map_err(|error| error.as_compact_json())?;
    let stream = AssistantMessageEventStream::new();
    let mut output = empty_assistant_message(&model);
    let payload = build_anthropic_payload(&model, &context, options.as_ref(), false);
    let endpoint = join_url(&model.base_url, "messages");
    let client = Client::new();

    let execution = (|| -> Result<(), PiAiError> {
        let response = client
            .post(endpoint)
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&payload)
            .send()
            .map_err(|error| {
                PiAiError::new(
                    PiAiErrorCode::ProviderTransport,
                    format!("Anthropic transport failed: {error}"),
                )
            })?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response
                .text()
                .unwrap_or_else(|_| "unable to read error body".to_string());
            return Err(PiAiError::new(
                PiAiErrorCode::ProviderHttp,
                format!("Anthropic HTTP {status}: {body}"),
            ));
        }

        let body = response.text().map_err(|error| {
            PiAiError::new(
                PiAiErrorCode::ProviderTransport,
                format!("Anthropic stream read failed: {error}"),
            )
        })?;
        let events = parse_sse_data_events(&body);

        stream.push(AssistantMessageEvent::Start {
            partial: output.clone(),
        });

        if events.is_empty() {
            let parsed_non_stream =
                apply_non_stream_anthropic_response(&body, &mut output, &stream)?;
            if !parsed_non_stream {
                return Err(
                    PiAiError::new(
                        PiAiErrorCode::ProviderProtocol,
                        "Anthropic response did not contain SSE events or known non-stream message payload",
                    )
                    .with_details(json!({
                        "bodyPrefix": truncate_for_details(&body, 800),
                    })),
                );
            }

            let done_reason = map_done_reason(output.stop_reason.clone()).ok_or_else(|| {
                PiAiError::new(
                    PiAiErrorCode::ProviderProtocol,
                    "Anthropic response ended with non-terminal done reason",
                )
            })?;

            stream.push(AssistantMessageEvent::Done {
                reason: done_reason,
                message: output.clone(),
            });
            return Ok(());
        }

        #[derive(Clone, Debug)]
        enum BlockState {
            Text {
                content_index: usize,
            },
            Thinking {
                content_index: usize,
            },
            ToolCall {
                content_index: usize,
                partial_json: String,
            },
        }

        let mut block_states: HashMap<usize, BlockState> = HashMap::new();

        for data in events {
            let event: Value = serde_json::from_str(&data).map_err(|error| {
                PiAiError::new(
                    PiAiErrorCode::ProviderProtocol,
                    format!("Invalid Anthropic SSE event: {error}"),
                )
                .with_details(json!({ "event": data }))
            })?;
            let event_type = event.get("type").and_then(Value::as_str).ok_or_else(|| {
                PiAiError::new(
                    PiAiErrorCode::ProviderProtocol,
                    "Anthropic event missing `type` field",
                )
                .with_details(json!({ "event": event }))
            })?;

            match event_type {
                "message_start" => {
                    if let Some(usage) = event
                        .get("message")
                        .and_then(Value::as_object)
                        .and_then(|message| message.get("usage"))
                    {
                        update_usage_from_anthropic(&mut output.usage, usage);
                    }
                }
                "content_block_start" => {
                    let block_index = event
                        .get("index")
                        .and_then(Value::as_u64)
                        .map(|value| value as usize)
                        .ok_or_else(|| {
                            PiAiError::new(
                                PiAiErrorCode::ProviderProtocol,
                                "content_block_start missing `index`",
                            )
                            .with_details(json!({ "event": event }))
                        })?;
                    let block = event
                        .get("content_block")
                        .and_then(Value::as_object)
                        .ok_or_else(|| {
                            PiAiError::new(
                                PiAiErrorCode::ProviderProtocol,
                                "content_block_start missing `content_block`",
                            )
                            .with_details(json!({ "event": event }))
                        })?;

                    let block_type =
                        block.get("type").and_then(Value::as_str).ok_or_else(|| {
                            PiAiError::new(
                                PiAiErrorCode::ProviderProtocol,
                                "content_block_start content block missing `type`",
                            )
                            .with_details(json!({ "event": event }))
                        })?;

                    match block_type {
                        "text" => {
                            let content_index = output.content.len();
                            output.content.push(AssistantContentBlock::Text {
                                text: String::new(),
                                text_signature: None,
                            });
                            block_states.insert(block_index, BlockState::Text { content_index });
                            stream.push(AssistantMessageEvent::TextStart {
                                content_index,
                                partial: output.clone(),
                            });
                        }
                        "thinking" => {
                            let content_index = output.content.len();
                            output.content.push(AssistantContentBlock::Thinking {
                                thinking: String::new(),
                                thinking_signature: None,
                            });
                            block_states
                                .insert(block_index, BlockState::Thinking { content_index });
                            stream.push(AssistantMessageEvent::ThinkingStart {
                                content_index,
                                partial: output.clone(),
                            });
                        }
                        "tool_use" => {
                            let content_index = output.content.len();
                            output.content.push(AssistantContentBlock::ToolCall {
                                id: block
                                    .get("id")
                                    .and_then(Value::as_str)
                                    .unwrap_or_default()
                                    .to_string(),
                                name: block
                                    .get("name")
                                    .and_then(Value::as_str)
                                    .unwrap_or_default()
                                    .to_string(),
                                arguments: block.get("input").cloned().unwrap_or_else(|| json!({})),
                                thought_signature: None,
                            });
                            block_states.insert(
                                block_index,
                                BlockState::ToolCall {
                                    content_index,
                                    partial_json: String::new(),
                                },
                            );
                            stream.push(AssistantMessageEvent::ToolcallStart {
                                content_index,
                                partial: output.clone(),
                            });
                        }
                        _ => {}
                    }
                }
                "content_block_delta" => {
                    let block_index = event
                        .get("index")
                        .and_then(Value::as_u64)
                        .map(|value| value as usize)
                        .ok_or_else(|| {
                            PiAiError::new(
                                PiAiErrorCode::ProviderProtocol,
                                "content_block_delta missing `index`",
                            )
                            .with_details(json!({ "event": event }))
                        })?;
                    let delta = event
                        .get("delta")
                        .and_then(Value::as_object)
                        .ok_or_else(|| {
                            PiAiError::new(
                                PiAiErrorCode::ProviderProtocol,
                                "content_block_delta missing `delta` object",
                            )
                            .with_details(json!({ "event": event }))
                        })?;
                    let delta_type =
                        delta.get("type").and_then(Value::as_str).ok_or_else(|| {
                            PiAiError::new(
                                PiAiErrorCode::ProviderProtocol,
                                "content_block_delta missing delta `type`",
                            )
                            .with_details(json!({ "event": event }))
                        })?;

                    if let Some(state) = block_states.get_mut(&block_index) {
                        match (state, delta_type) {
                            (BlockState::Text { content_index }, "text_delta") => {
                                let text = delta
                                    .get("text")
                                    .and_then(Value::as_str)
                                    .unwrap_or_default()
                                    .to_string();
                                if let Some(AssistantContentBlock::Text { text: current, .. }) =
                                    output.content.get_mut(*content_index)
                                {
                                    current.push_str(&text);
                                }
                                stream.push(AssistantMessageEvent::TextDelta {
                                    content_index: *content_index,
                                    delta: text,
                                    partial: output.clone(),
                                });
                            }
                            (BlockState::Thinking { content_index }, "thinking_delta") => {
                                let text = delta
                                    .get("thinking")
                                    .and_then(Value::as_str)
                                    .unwrap_or_default()
                                    .to_string();
                                if let Some(AssistantContentBlock::Thinking { thinking, .. }) =
                                    output.content.get_mut(*content_index)
                                {
                                    thinking.push_str(&text);
                                }
                                stream.push(AssistantMessageEvent::ThinkingDelta {
                                    content_index: *content_index,
                                    delta: text,
                                    partial: output.clone(),
                                });
                            }
                            (
                                BlockState::ToolCall {
                                    content_index,
                                    partial_json,
                                },
                                "input_json_delta",
                            ) => {
                                let delta_text = delta
                                    .get("partial_json")
                                    .and_then(Value::as_str)
                                    .unwrap_or_default()
                                    .to_string();
                                partial_json.push_str(&delta_text);
                                if let Some(AssistantContentBlock::ToolCall { arguments, .. }) =
                                    output.content.get_mut(*content_index)
                                {
                                    *arguments = parse_partial_json(partial_json);
                                }
                                stream.push(AssistantMessageEvent::ToolcallDelta {
                                    content_index: *content_index,
                                    delta: delta_text,
                                    partial: output.clone(),
                                });
                            }
                            (BlockState::Thinking { content_index }, "signature_delta") => {
                                let signature = delta
                                    .get("signature")
                                    .and_then(Value::as_str)
                                    .unwrap_or_default()
                                    .to_string();
                                if let Some(AssistantContentBlock::Thinking {
                                    thinking_signature,
                                    ..
                                }) = output.content.get_mut(*content_index)
                                {
                                    let next = match thinking_signature.take() {
                                        Some(existing) => format!("{existing}{signature}"),
                                        None => signature,
                                    };
                                    *thinking_signature = Some(next);
                                }
                            }
                            _ => {}
                        }
                    }
                }
                "content_block_stop" => {
                    let block_index = event
                        .get("index")
                        .and_then(Value::as_u64)
                        .map(|value| value as usize)
                        .ok_or_else(|| {
                            PiAiError::new(
                                PiAiErrorCode::ProviderProtocol,
                                "content_block_stop missing `index`",
                            )
                            .with_details(json!({ "event": event }))
                        })?;

                    let Some(state) = block_states.remove(&block_index) else {
                        continue;
                    };

                    match state {
                        BlockState::Text { content_index } => {
                            stream.push(AssistantMessageEvent::TextEnd {
                                content_index,
                                content: extract_text_block(&output.content, content_index),
                                partial: output.clone(),
                            });
                        }
                        BlockState::Thinking { content_index } => {
                            stream.push(AssistantMessageEvent::ThinkingEnd {
                                content_index,
                                content: extract_thinking_block(&output.content, content_index),
                                partial: output.clone(),
                            });
                        }
                        BlockState::ToolCall { content_index, .. } => {
                            let tool_call =
                                output.content.get(content_index).cloned().ok_or_else(|| {
                                    PiAiError::new(
                                        PiAiErrorCode::ProviderProtocol,
                                        "Missing tool call content at content_block_stop",
                                    )
                                })?;
                            let AssistantContentBlock::ToolCall {
                                id,
                                name,
                                arguments,
                                thought_signature,
                            } = tool_call
                            else {
                                continue;
                            };
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
                    }
                }
                "message_delta" => {
                    if let Some(stop_reason) = event
                        .get("delta")
                        .and_then(Value::as_object)
                        .and_then(|delta| delta.get("stop_reason"))
                        .and_then(Value::as_str)
                    {
                        output.stop_reason = map_anthropic_stop_reason(stop_reason);
                    }
                    if let Some(usage) = event.get("usage") {
                        update_usage_from_anthropic(&mut output.usage, usage);
                    }
                }
                "message_stop" => {}
                "ping" => {}
                other => {
                    return Err(PiAiError::new(
                        PiAiErrorCode::ProviderProtocol,
                        format!("Unhandled Anthropic event type: {other}"),
                    ));
                }
            }
        }

        for state in block_states.into_values() {
            match state {
                BlockState::Text { content_index } => {
                    stream.push(AssistantMessageEvent::TextEnd {
                        content_index,
                        content: extract_text_block(&output.content, content_index),
                        partial: output.clone(),
                    });
                }
                BlockState::Thinking { content_index } => {
                    stream.push(AssistantMessageEvent::ThinkingEnd {
                        content_index,
                        content: extract_thinking_block(&output.content, content_index),
                        partial: output.clone(),
                    });
                }
                BlockState::ToolCall { content_index, .. } => {
                    let tool_call =
                        output.content.get(content_index).cloned().ok_or_else(|| {
                            PiAiError::new(
                                PiAiErrorCode::ProviderProtocol,
                                "Missing tool call content when finalizing Anthropic stream",
                            )
                        })?;
                    let AssistantContentBlock::ToolCall {
                        id,
                        name,
                        arguments,
                        thought_signature,
                    } = tool_call
                    else {
                        continue;
                    };
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
            }
        }

        let done_reason = map_done_reason(output.stop_reason.clone()).ok_or_else(|| {
            PiAiError::new(
                PiAiErrorCode::ProviderProtocol,
                "Anthropic response ended with non-terminal done reason",
            )
        })?;

        stream.push(AssistantMessageEvent::Done {
            reason: done_reason,
            message: output.clone(),
        });
        Ok(())
    })();

    if let Err(error) = execution {
        output.stop_reason = StopReason::Error;
        output.error_message = Some(error.as_compact_json());
        stream.push(AssistantMessageEvent::Error {
            reason: crate::types::ErrorReason::Error,
            error: output,
        });
    }

    Ok(stream)
}

pub fn stream_simple_anthropic(
    model: Model,
    context: Context,
    options: Option<SimpleStreamOptions>,
) -> StreamResult<AssistantMessageEventStream> {
    let merged = options.map(|simple| {
        let mut stream = simple.stream;
        if simple.reasoning.is_some() && model.reasoning {
            let mut headers = stream.headers.unwrap_or_default();
            headers.insert("x-pi-thinking".to_string(), "enabled".to_string());
            stream.headers = Some(headers);
        }
        stream
    });
    stream_anthropic(model, context, merged)
}

fn build_anthropic_payload(
    model: &Model,
    context: &Context,
    options: Option<&StreamOptions>,
    thinking_enabled: bool,
) -> Value {
    let mut payload = json!({
        "model": model.id,
        "stream": true,
        "messages": convert_messages(context),
        "max_tokens": options
            .and_then(|options| options.max_tokens)
            .unwrap_or((model.max_tokens / 3).max(256)),
    });

    if let Some(system_prompt) = &context.system_prompt {
        payload["system"] = Value::String(system_prompt.clone());
    }
    if let Some(tools) = &context.tools {
        payload["tools"] = convert_tools(tools);
    }
    if let Some(temperature) = options.and_then(|options| options.temperature) {
        payload["temperature"] = json!(temperature);
    }
    if thinking_enabled {
        payload["thinking"] = json!({
            "type": "enabled",
            "budget_tokens": 1024,
        });
    }

    payload
}

fn convert_messages(context: &Context) -> Vec<Value> {
    let mut messages = Vec::new();

    for message in &context.messages {
        match message {
            Message::User { content, .. } => match content {
                UserContent::Text(text) => messages.push(json!({
                    "role": "user",
                    "content": text,
                })),
                UserContent::Blocks(blocks) => {
                    let converted = blocks
                        .iter()
                        .map(convert_user_block_to_anthropic)
                        .collect::<Vec<_>>();
                    messages.push(json!({
                        "role": "user",
                        "content": converted,
                    }));
                }
            },
            Message::Assistant { content, .. } => {
                let converted = content
                    .iter()
                    .map(|block| match block {
                        AssistantContentBlock::Text { text, .. } => json!({
                            "type": "text",
                            "text": text,
                        }),
                        AssistantContentBlock::Thinking {
                            thinking,
                            thinking_signature,
                        } => {
                            let mut block = json!({
                                "type": "thinking",
                                "thinking": thinking,
                            });
                            if let Some(signature) = thinking_signature {
                                block["signature"] = Value::String(signature.clone());
                            }
                            block
                        }
                        AssistantContentBlock::ToolCall {
                            id,
                            name,
                            arguments,
                            ..
                        } => json!({
                            "type": "tool_use",
                            "id": id,
                            "name": name,
                            "input": arguments,
                        }),
                    })
                    .collect::<Vec<_>>();
                messages.push(json!({
                    "role": "assistant",
                    "content": converted,
                }));
            }
            Message::ToolResult {
                tool_call_id,
                content,
                is_error,
                ..
            } => {
                messages.push(json!({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": convert_tool_result_content(content),
                        "is_error": is_error,
                    }],
                }));
            }
        }
    }

    messages
}

fn convert_user_block_to_anthropic(block: &UserContentBlock) -> Value {
    match block {
        UserContentBlock::Text { text, .. } => json!({
            "type": "text",
            "text": text,
        }),
        UserContentBlock::Image { data, mime_type } => json!({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime_type,
                "data": data,
            },
        }),
    }
}

fn convert_tools(tools: &[Tool]) -> Value {
    Value::Array(
        tools
            .iter()
            .map(|tool| {
                json!({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters,
                })
            })
            .collect(),
    )
}

fn convert_tool_result_content(content: &[ToolResultContentBlock]) -> Value {
    let text_blocks = content
        .iter()
        .filter_map(|block| match block {
            ToolResultContentBlock::Text { text, .. } => Some(text.clone()),
            ToolResultContentBlock::Image { .. } => None,
        })
        .collect::<Vec<_>>();

    if text_blocks.is_empty() {
        Value::String("(no text result)".to_string())
    } else if text_blocks.len() == 1 {
        Value::String(text_blocks[0].clone())
    } else {
        Value::Array(
            text_blocks
                .into_iter()
                .map(|text| json!({ "type": "text", "text": text }))
                .collect(),
        )
    }
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

fn apply_non_stream_anthropic_response(
    body: &str,
    output: &mut AssistantMessage,
    stream: &AssistantMessageEventStream,
) -> Result<bool, PiAiError> {
    let payload: Value = match serde_json::from_str(body) {
        Ok(value) => value,
        Err(_) => return Ok(false),
    };

    if payload.get("type").and_then(Value::as_str) != Some("message") {
        return Ok(false);
    }

    if let Some(usage) = payload.get("usage") {
        update_usage_from_anthropic(&mut output.usage, usage);
    }
    if let Some(stop_reason) = payload.get("stop_reason").and_then(Value::as_str) {
        output.stop_reason = map_anthropic_stop_reason(stop_reason);
    }

    let content_blocks = payload
        .get("content")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            PiAiError::new(
                PiAiErrorCode::ProviderProtocol,
                "Anthropic non-stream response missing `content` array",
            )
            .with_details(json!({ "body": truncate_for_details(body, 800) }))
        })?;

    for block in content_blocks {
        let block_type = block
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();

        match block_type {
            "text" => {
                let text = block
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let content_index = output.content.len();
                output.content.push(AssistantContentBlock::Text {
                    text: text.clone(),
                    text_signature: None,
                });
                stream.push(AssistantMessageEvent::TextStart {
                    content_index,
                    partial: output.clone(),
                });
                if !text.is_empty() {
                    stream.push(AssistantMessageEvent::TextDelta {
                        content_index,
                        delta: text.clone(),
                        partial: output.clone(),
                    });
                }
                stream.push(AssistantMessageEvent::TextEnd {
                    content_index,
                    content: text,
                    partial: output.clone(),
                });
            }
            "thinking" => {
                let thinking = block
                    .get("thinking")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let signature = block
                    .get("signature")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                let content_index = output.content.len();
                output.content.push(AssistantContentBlock::Thinking {
                    thinking: thinking.clone(),
                    thinking_signature: signature,
                });
                stream.push(AssistantMessageEvent::ThinkingStart {
                    content_index,
                    partial: output.clone(),
                });
                if !thinking.is_empty() {
                    stream.push(AssistantMessageEvent::ThinkingDelta {
                        content_index,
                        delta: thinking.clone(),
                        partial: output.clone(),
                    });
                }
                stream.push(AssistantMessageEvent::ThinkingEnd {
                    content_index,
                    content: thinking,
                    partial: output.clone(),
                });
            }
            "tool_use" => {
                let id = block
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let name = block
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let arguments = block.get("input").cloned().unwrap_or_else(|| json!({}));
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
            _ => {}
        }
    }

    Ok(true)
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

fn map_anthropic_stop_reason(reason: &str) -> StopReason {
    match reason {
        "end_turn" | "stop_sequence" | "pause_turn" => StopReason::Stop,
        "max_tokens" => StopReason::Length,
        "tool_use" => StopReason::ToolUse,
        "refusal" | "sensitive" => StopReason::Error,
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

fn parse_partial_json(buffer: &str) -> Value {
    serde_json::from_str::<Value>(buffer).unwrap_or_else(|_| Value::Object(Map::new()))
}

fn update_usage_from_anthropic(usage: &mut Usage, value: &Value) {
    let input_tokens = value
        .get("input_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(usage.input);
    let output_tokens = value
        .get("output_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(usage.output);
    let cache_read = value
        .get("cache_read_input_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(usage.cache_read);
    let cache_write = value
        .get("cache_creation_input_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(usage.cache_write);

    usage.input = input_tokens;
    usage.output = output_tokens;
    usage.cache_read = cache_read;
    usage.cache_write = cache_write;
    usage.total_tokens = usage.input + usage.output + usage.cache_read + usage.cache_write;
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
