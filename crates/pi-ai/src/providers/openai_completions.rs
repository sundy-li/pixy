use std::collections::HashMap;
use std::env;
use std::sync::Arc;

use reqwest::blocking::Client;
use serde_json::{Map, Value, json};

use crate::api_registry::{ApiProvider, StreamResult};
use crate::error::{PiAiError, PiAiErrorCode};
use crate::types::{
    AssistantContentBlock, AssistantMessage, AssistantMessageEvent, Context, Cost, DoneReason,
    Message, Model, SimpleStreamOptions, StopReason, StreamOptions, Tool, Usage, UserContent,
    UserContentBlock,
};
use crate::{ApiProviderRef, AssistantMessageEventStream};

struct OpenAICompletionsProvider;

impl ApiProvider for OpenAICompletionsProvider {
    fn api(&self) -> &str {
        "openai-completions"
    }

    fn stream(
        &self,
        model: Model,
        context: Context,
        options: Option<StreamOptions>,
    ) -> Result<AssistantMessageEventStream, String> {
        stream_openai_completions(model, context, options)
    }

    fn stream_simple(
        &self,
        model: Model,
        context: Context,
        options: Option<SimpleStreamOptions>,
    ) -> Result<AssistantMessageEventStream, String> {
        stream_simple_openai_completions(model, context, options)
    }
}

pub(super) fn provider() -> ApiProviderRef {
    Arc::new(OpenAICompletionsProvider)
}

pub fn stream_openai_completions(
    model: Model,
    context: Context,
    options: Option<StreamOptions>,
) -> StreamResult<AssistantMessageEventStream> {
    let api_key = resolve_api_key(&model.provider, options.as_ref())
        .map_err(|error| error.as_compact_json())?;
    let stream = AssistantMessageEventStream::new();

    let mut output = empty_assistant_message(&model);
    let payload = build_openai_payload(&model, &context, options.as_ref());
    let endpoint = join_url(&model.base_url, "chat/completions");
    let client = Client::new();

    let execution = (|| -> Result<(), PiAiError> {
        let response = client
            .post(endpoint)
            .header("Authorization", format!("Bearer {api_key}"))
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .map_err(|error| {
                PiAiError::new(
                    PiAiErrorCode::ProviderTransport,
                    format!("OpenAI transport failed: {error}"),
                )
            })?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response
                .text()
                .unwrap_or_else(|_| "unable to read error body".to_string());
            return Err(PiAiError::new(
                PiAiErrorCode::ProviderHttp,
                format!("OpenAI HTTP {status}: {body}"),
            ));
        }

        let body = response.text().map_err(|error| {
            PiAiError::new(
                PiAiErrorCode::ProviderTransport,
                format!("OpenAI stream read failed: {error}"),
            )
        })?;

        stream.push(AssistantMessageEvent::Start {
            partial: output.clone(),
        });

        let mut text_block_index: Option<usize> = None;
        let mut tool_arg_buffers: HashMap<usize, String> = HashMap::new();
        let mut tool_block_indices: HashMap<usize, usize> = HashMap::new();

        for data in parse_sse_data_events(&body) {
            if data == "[DONE]" {
                break;
            }

            let chunk: Value = serde_json::from_str(&data).map_err(|error| {
                PiAiError::new(
                    PiAiErrorCode::ProviderProtocol,
                    format!("Invalid OpenAI chunk JSON: {error}"),
                )
                .with_details(json!({ "chunk": data }))
            })?;

            if let Some(usage_value) = chunk.get("usage") {
                update_usage_from_openai(&mut output.usage, usage_value);
            }

            let choice = chunk
                .get("choices")
                .and_then(Value::as_array)
                .and_then(|choices| choices.first())
                .ok_or_else(|| {
                    PiAiError::new(
                        PiAiErrorCode::ProviderProtocol,
                        "OpenAI chunk missing choices[0]",
                    )
                    .with_details(json!({ "chunk": chunk }))
                })?;

            if let Some(finish_reason) = choice.get("finish_reason").and_then(Value::as_str) {
                output.stop_reason = map_openai_stop_reason(finish_reason);
            }

            if let Some(content_delta) = choice
                .get("delta")
                .and_then(Value::as_object)
                .and_then(|delta| delta.get("content"))
                .and_then(Value::as_str)
            {
                if content_delta.is_empty() {
                    continue;
                }

                let idx = if let Some(idx) = text_block_index {
                    idx
                } else {
                    let new_index = output.content.len();
                    output.content.push(AssistantContentBlock::Text {
                        text: String::new(),
                        text_signature: None,
                    });
                    stream.push(AssistantMessageEvent::TextStart {
                        content_index: new_index,
                        partial: output.clone(),
                    });
                    text_block_index = Some(new_index);
                    new_index
                };

                if let Some(AssistantContentBlock::Text { text, .. }) = output.content.get_mut(idx)
                {
                    text.push_str(content_delta);
                }
                stream.push(AssistantMessageEvent::TextDelta {
                    content_index: idx,
                    delta: content_delta.to_string(),
                    partial: output.clone(),
                });
            }

            if let Some(tool_calls) = choice
                .get("delta")
                .and_then(Value::as_object)
                .and_then(|delta| delta.get("tool_calls"))
                .and_then(Value::as_array)
            {
                if let Some(text_idx) = text_block_index.take() {
                    let text = extract_text_block(&output.content, text_idx);
                    stream.push(AssistantMessageEvent::TextEnd {
                        content_index: text_idx,
                        content: text,
                        partial: output.clone(),
                    });
                }

                for tool_call in tool_calls {
                    let tool_provider_index = tool_call
                        .get("index")
                        .and_then(Value::as_u64)
                        .map(|value| value as usize)
                        .unwrap_or(0);

                    let content_index =
                        if let Some(existing) = tool_block_indices.get(&tool_provider_index) {
                            *existing
                        } else {
                            let new_index = output.content.len();
                            output.content.push(AssistantContentBlock::ToolCall {
                                id: String::new(),
                                name: String::new(),
                                arguments: json!({}),
                                thought_signature: None,
                            });
                            tool_arg_buffers.insert(tool_provider_index, String::new());
                            tool_block_indices.insert(tool_provider_index, new_index);
                            stream.push(AssistantMessageEvent::ToolcallStart {
                                content_index: new_index,
                                partial: output.clone(),
                            });
                            new_index
                        };

                    if let Some(AssistantContentBlock::ToolCall {
                        id,
                        name,
                        arguments,
                        ..
                    }) = output.content.get_mut(content_index)
                    {
                        if let Some(id_delta) = tool_call.get("id").and_then(Value::as_str) {
                            *id = id_delta.to_string();
                        }
                        if let Some(name_delta) = tool_call
                            .get("function")
                            .and_then(Value::as_object)
                            .and_then(|function| function.get("name"))
                            .and_then(Value::as_str)
                        {
                            *name = name_delta.to_string();
                        }

                        let arg_delta = tool_call
                            .get("function")
                            .and_then(Value::as_object)
                            .and_then(|function| function.get("arguments"))
                            .and_then(Value::as_str)
                            .unwrap_or("");

                        if !arg_delta.is_empty() {
                            if let Some(buffer) = tool_arg_buffers.get_mut(&tool_provider_index) {
                                buffer.push_str(arg_delta);
                                *arguments = parse_partial_json(buffer);
                            }
                        }

                        stream.push(AssistantMessageEvent::ToolcallDelta {
                            content_index,
                            delta: arg_delta.to_string(),
                            partial: output.clone(),
                        });
                    }
                }
            }
        }

        if let Some(text_idx) = text_block_index.take() {
            let text = extract_text_block(&output.content, text_idx);
            stream.push(AssistantMessageEvent::TextEnd {
                content_index: text_idx,
                content: text,
                partial: output.clone(),
            });
        }

        let mut ordered_tool_indices = tool_block_indices.values().copied().collect::<Vec<_>>();
        ordered_tool_indices.sort_unstable();
        ordered_tool_indices.dedup();

        for content_index in ordered_tool_indices {
            let tool_call = output.content.get(content_index).cloned().ok_or_else(|| {
                PiAiError::new(
                    PiAiErrorCode::ProviderProtocol,
                    "Missing tool call block when finishing OpenAI stream",
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

        let done_reason = map_done_reason(output.stop_reason.clone()).ok_or_else(|| {
            PiAiError::new(
                PiAiErrorCode::ProviderProtocol,
                "OpenAI response ended with non-terminal done reason",
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

pub fn stream_simple_openai_completions(
    model: Model,
    context: Context,
    options: Option<SimpleStreamOptions>,
) -> StreamResult<AssistantMessageEventStream> {
    let merged = options.map(|simple| {
        let mut stream = simple.stream;
        if simple.reasoning.is_some() && model.reasoning {
            let effort = match simple.reasoning {
                Some(crate::types::ThinkingLevel::Minimal) => "minimal",
                Some(crate::types::ThinkingLevel::Low) => "low",
                Some(crate::types::ThinkingLevel::Medium) => "medium",
                Some(crate::types::ThinkingLevel::High) => "high",
                Some(crate::types::ThinkingLevel::Xhigh) => "xhigh",
                None => "medium",
            };
            let mut headers = stream.headers.unwrap_or_default();
            headers.insert("x-pi-reasoning-effort".to_string(), effort.to_string());
            stream.headers = Some(headers);
        }
        stream
    });

    stream_openai_completions(model, context, merged)
}

fn build_openai_payload(
    model: &Model,
    context: &Context,
    options: Option<&StreamOptions>,
) -> Value {
    let mut payload = json!({
        "model": model.id,
        "stream": true,
        "messages": convert_messages(context),
    });

    if let Some(max_tokens) = options.and_then(|options| options.max_tokens) {
        payload["max_tokens"] = json!(max_tokens);
    }
    if let Some(temperature) = options.and_then(|options| options.temperature) {
        payload["temperature"] = json!(temperature);
    }
    if let Some(tools) = &context.tools {
        payload["tools"] = convert_tools(tools);
    }

    payload
}

fn convert_messages(context: &Context) -> Vec<Value> {
    let mut messages = Vec::new();

    if let Some(system_prompt) = &context.system_prompt {
        messages.push(json!({
            "role": "system",
            "content": system_prompt,
        }));
    }

    for message in &context.messages {
        match message {
            Message::User { content, .. } => match content {
                UserContent::Text(text) => {
                    messages.push(json!({
                        "role": "user",
                        "content": text,
                    }));
                }
                UserContent::Blocks(blocks) => {
                    let converted = blocks
                        .iter()
                        .filter_map(|block| match block {
                            UserContentBlock::Text { text, .. } => Some(json!({
                                "type": "text",
                                "text": text,
                            })),
                            UserContentBlock::Image { data, mime_type } => Some(json!({
                                "type": "image_url",
                                "image_url": {
                                    "url": format!("data:{mime_type};base64,{data}"),
                                }
                            })),
                        })
                        .collect::<Vec<_>>();
                    messages.push(json!({
                        "role": "user",
                        "content": converted,
                    }));
                }
            },
            Message::Assistant { content, .. } => {
                let mut text_parts = Vec::new();
                let mut tool_calls = Vec::new();
                for block in content {
                    match block {
                        AssistantContentBlock::Text { text, .. } => text_parts.push(text.clone()),
                        AssistantContentBlock::Thinking { thinking, .. } => {
                            text_parts.push(thinking.clone())
                        }
                        AssistantContentBlock::ToolCall {
                            id,
                            name,
                            arguments,
                            ..
                        } => {
                            tool_calls.push(json!({
                                "id": id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": arguments.to_string(),
                                }
                            }));
                        }
                    }
                }

                let mut assistant_message = json!({
                    "role": "assistant",
                    "content": if text_parts.is_empty() { Value::Null } else { Value::String(text_parts.join("\n")) },
                });
                if !tool_calls.is_empty() {
                    assistant_message["tool_calls"] = Value::Array(tool_calls);
                }
                messages.push(assistant_message);
            }
            Message::ToolResult {
                tool_call_id,
                tool_name,
                content,
                ..
            } => {
                let text = content
                    .iter()
                    .filter_map(|block| match block {
                        crate::types::ToolResultContentBlock::Text { text, .. } => {
                            Some(text.as_str())
                        }
                        crate::types::ToolResultContentBlock::Image { .. } => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": if text.is_empty() { "(no text result)" } else { &text },
                }));
            }
        }
    }

    messages
}

fn convert_tools(tools: &[Tool]) -> Value {
    Value::Array(
        tools
            .iter()
            .map(|tool| {
                json!({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    }
                })
            })
            .collect(),
    )
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

    if let Ok(value) = env::var("OPENAI_API_KEY") {
        if !value.trim().is_empty() {
            return Ok(value);
        }
    }

    Err(PiAiError::new(
        PiAiErrorCode::ProviderAuthMissing,
        format!(
            "Missing API key for provider '{}'. Pass `StreamOptions.api_key` or set {} / OPENAI_API_KEY.",
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

fn map_openai_stop_reason(reason: &str) -> StopReason {
    match reason {
        "stop" => StopReason::Stop,
        "length" => StopReason::Length,
        "function_call" | "tool_calls" => StopReason::ToolUse,
        "content_filter" => StopReason::Error,
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

fn update_usage_from_openai(usage: &mut Usage, value: &Value) {
    let prompt_tokens = value
        .get("prompt_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(usage.input);
    let completion_tokens = value
        .get("completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(usage.output);
    let cached_tokens = value
        .get("prompt_tokens_details")
        .and_then(Value::as_object)
        .and_then(|details| details.get("cached_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(usage.cache_read);

    usage.input = prompt_tokens.saturating_sub(cached_tokens);
    usage.output = completion_tokens;
    usage.cache_read = cached_tokens;
    usage.cache_write = 0;
    usage.total_tokens = usage.input + usage.output + usage.cache_read + usage.cache_write;
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
