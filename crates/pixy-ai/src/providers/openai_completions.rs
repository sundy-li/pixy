use std::collections::HashMap;
use std::env;
use std::io::{BufRead, BufReader, Read};
use std::sync::Arc;

use serde_json::{json, Map, Value};
use tracing::info;

use super::common::{empty_assistant_message, join_url, shared_http_client};
use crate::api_registry::{ApiProvider, ApiProviderFuture};
use crate::error::{PiAiError, PiAiErrorCode};
use crate::types::{
    AssistantContentBlock, AssistantMessageEvent, Context, DoneReason, Message, Model,
    SimpleStreamOptions, StopReason, StreamOptions, Tool, Usage, UserContent, UserContentBlock,
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
        stream: AssistantMessageEventStream,
    ) -> ApiProviderFuture {
        Box::pin(async move { run_openai_completions(model, context, options, stream).await })
    }

    fn stream_simple(
        &self,
        model: Model,
        context: Context,
        options: Option<SimpleStreamOptions>,
        stream: AssistantMessageEventStream,
    ) -> ApiProviderFuture {
        Box::pin(
            async move { run_simple_openai_completions(model, context, options, stream).await },
        )
    }
}

pub(super) fn provider() -> ApiProviderRef {
    Arc::new(OpenAICompletionsProvider)
}

pub async fn run_openai_completions(
    model: Model,
    context: Context,
    options: Option<StreamOptions>,
    stream: AssistantMessageEventStream,
) -> Result<(), PiAiError> {
    let api_key = resolve_api_key(&model.provider, options.as_ref())?;

    let mut output = empty_assistant_message(&model);
    let payload = build_openai_payload(&model, &context, options.as_ref());
    let endpoint = join_url(&model.base_url, "chat/completions");
    let client = shared_http_client(&model.base_url);

    info!("OpenAI completions payload: {}", payload);

    let execution = async {
        let mut request = client
            .post(endpoint.as_str())
            .header("Authorization", format!("Bearer {api_key}"))
            .header("Content-Type", "application/json");

        if let Some(headers) = options.as_ref().and_then(|stream| stream.headers.as_ref()) {
            for (name, value) in headers {
                request = request.header(name, value);
            }
        }
        let response = request.json(&payload).send().await.map_err(|error| {
            PiAiError::new(
                PiAiErrorCode::ProviderTransport,
                format!("OpenAI transport failed: {error}"),
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
                format!("OpenAI HTTP {status}: {body}"),
            ));
        }

        stream.push(AssistantMessageEvent::Start {
            partial: output.clone(),
        });

        let mut text_block_index: Option<usize> = None;
        let mut thinking_block_index: Option<usize> = None;
        let mut tool_arg_buffers: HashMap<usize, String> = HashMap::new();
        let mut tool_block_indices: HashMap<usize, usize> = HashMap::new();
        let body = response.text().await.map_err(|error| {
            PiAiError::new(
                PiAiErrorCode::ProviderTransport,
                format!("OpenAI stream read failed: {error}"),
            )
        })?;
        let mut reader = std::io::Cursor::new(body.into_bytes());
        process_sse_data_events(&mut reader, |data| {
            info!("OpenAI completions data: {}", data);
            if data == "[DONE]" {
                return Ok(true);
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

            let delta = choice.get("delta").and_then(Value::as_object);
            if let Some(reasoning_delta) = delta
                .and_then(|delta| delta.get("reasoning_content"))
                .and_then(Value::as_str)
            {
                let idx = if let Some(idx) = thinking_block_index {
                    idx
                } else {
                    let new_index = output.content.len();
                    output.content.push(AssistantContentBlock::Thinking {
                        thinking: String::new(),
                        thinking_signature: None,
                    });
                    stream.push(AssistantMessageEvent::ThinkingStart {
                        content_index: new_index,
                        partial: output.clone(),
                    });
                    thinking_block_index = Some(new_index);
                    new_index
                };

                let emitted_delta =
                    if let Some(AssistantContentBlock::Thinking { thinking, .. }) =
                        output.content.get_mut(idx)
                    {
                        merge_stream_delta(thinking, reasoning_delta)
                    } else {
                        None
                    };

                if let Some(delta_to_emit) = emitted_delta {
                    stream.push(AssistantMessageEvent::ThinkingDelta {
                        content_index: idx,
                        delta: delta_to_emit,
                        partial: output.clone(),
                    });
                }
            }

            if let Some(content_delta) = delta
                .and_then(|delta| delta.get("content"))
                .and_then(Value::as_str)
            {
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

                let emitted_delta = if let Some(AssistantContentBlock::Text { text, .. }) =
                    output.content.get_mut(idx)
                {
                    merge_stream_delta(text, content_delta)
                } else {
                    None
                };

                if let Some(delta_to_emit) = emitted_delta {
                    stream.push(AssistantMessageEvent::TextDelta {
                        content_index: idx,
                        delta: delta_to_emit,
                        partial: output.clone(),
                    });
                }
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
                if let Some(thinking_idx) = thinking_block_index.take() {
                    let thinking = extract_thinking_block(&output.content, thinking_idx);
                    stream.push(AssistantMessageEvent::ThinkingEnd {
                        content_index: thinking_idx,
                        content: thinking,
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

            Ok(false)
        })?;

        if let Some(text_idx) = text_block_index.take() {
            let text = extract_text_block(&output.content, text_idx);
            stream.push(AssistantMessageEvent::TextEnd {
                content_index: text_idx,
                content: text,
                partial: output.clone(),
            });
        }
        if let Some(thinking_idx) = thinking_block_index.take() {
            let thinking = extract_thinking_block(&output.content, thinking_idx);
            stream.push(AssistantMessageEvent::ThinkingEnd {
                content_index: thinking_idx,
                content: thinking,
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

pub async fn run_simple_openai_completions(
    model: Model,
    context: Context,
    options: Option<SimpleStreamOptions>,
    stream: AssistantMessageEventStream,
) -> Result<(), PiAiError> {
    let mut model = model;
    apply_simple_reasoning_to_model(&mut model, options.as_ref());
    let stream_options = options.map(|simple| simple.stream);
    run_openai_completions(model, context, stream_options, stream).await
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
    if model.reasoning {
        if let Some(effort) = model.reasoning_effort.as_ref() {
            payload["reasoning_effort"] = json!(thinking_level_to_effort(model, effort));
        }
    }

    payload
}

fn apply_simple_reasoning_to_model(model: &mut Model, options: Option<&SimpleStreamOptions>) {
    if !model.reasoning {
        return;
    }
    if let Some(reasoning) = options.and_then(|simple| simple.reasoning.clone()) {
        model.reasoning_effort = Some(reasoning);
    }
}

fn thinking_level_to_effort(model: &Model, level: &crate::types::ThinkingLevel) -> &'static str {
    let is_openai = model.name.to_lowercase().contains("openai");
    if !is_openai && level == &crate::types::ThinkingLevel::Xhigh {
        return "high";
    }

    match level {
        crate::types::ThinkingLevel::Minimal => "minimal",
        crate::types::ThinkingLevel::Low => "low",
        crate::types::ThinkingLevel::Medium => "medium",
        crate::types::ThinkingLevel::High => "high",
        crate::types::ThinkingLevel::Xhigh => "xhigh",
    }
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

fn process_sse_data_events<R, F>(reader: &mut R, mut on_data: F) -> Result<(), PiAiError>
where
    R: Read,
    F: FnMut(String) -> Result<bool, PiAiError>,
{
    let mut reader = BufReader::new(reader);
    let mut line = String::new();
    let mut data_lines: Vec<String> = Vec::new();

    let mut emit_event = |lines: &mut Vec<String>| -> Result<bool, PiAiError> {
        if lines.is_empty() {
            return Ok(false);
        }
        let data = lines.join("\n");
        lines.clear();
        on_data(data)
    };

    loop {
        line.clear();
        let bytes_read = reader.read_line(&mut line).map_err(|error| {
            PiAiError::new(
                PiAiErrorCode::ProviderTransport,
                format!("OpenAI stream read failed: {error}"),
            )
        })?;

        if bytes_read == 0 {
            let _ = emit_event(&mut data_lines)?;
            return Ok(());
        }

        let trimmed = line.trim_end_matches(&['\r', '\n'][..]);
        if trimmed.is_empty() {
            if emit_event(&mut data_lines)? {
                return Ok(());
            }
            continue;
        }

        if let Some(data) = trimmed.strip_prefix("data:") {
            data_lines.push(data.trim_start().to_string());
        }
    }
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

fn merge_stream_delta(current: &mut String, incoming: &str) -> Option<String> {
    if incoming.is_empty() {
        return None;
    }

    if current.is_empty() {
        current.push_str(incoming);
        return Some(incoming.to_string());
    }

    if incoming.starts_with(current.as_str()) {
        let appended = &incoming[current.len()..];
        if appended.is_empty() {
            return None;
        }
        *current = incoming.to_string();
        return Some(appended.to_string());
    }

    // Some providers may replay a long trailing segment on retry/resume boundaries.
    // Ignore only long tails to avoid dropping legitimate tiny repeats (e.g. "ha" + "ha").
    if incoming.len() >= 16 && current.ends_with(incoming) {
        return None;
    }

    current.push_str(incoming);
    Some(incoming.to_string())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Cost, ThinkingLevel};
    use std::io::{self, Read};

    fn sample_model() -> Model {
        Model {
            id: "gpt-o".to_string(),
            name: "gpt-o".to_string(),
            api: "openai-completions".to_string(),
            provider: "openai".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            reasoning: true,
            reasoning_effort: Some(ThinkingLevel::High),
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
            system_prompt: Some("You are helpful.".to_string()),
            messages: vec![Message::User {
                content: UserContent::Text("hello".to_string()),
                timestamp: 0,
            }],
            tools: None,
        }
    }

    #[test]
    fn openai_payload_includes_reasoning_effort_when_model_config_sets_it() {
        let model = sample_model();
        let context = sample_context();

        let payload = build_openai_payload(&model, &context, None);
        assert_eq!(payload["reasoning_effort"], "high");
    }

    #[test]
    fn simple_options_reasoning_overrides_model_reasoning_effort() {
        let mut model = sample_model();
        let options = SimpleStreamOptions {
            stream: StreamOptions::default(),
            reasoning: Some(ThinkingLevel::Low),
        };

        apply_simple_reasoning_to_model(&mut model, Some(&options));
        assert_eq!(model.reasoning_effort, Some(ThinkingLevel::Low));
    }

    #[test]
    fn process_sse_data_events_stops_reading_after_done_event() {
        struct FailAfterDoneReader {
            chunks: Vec<&'static [u8]>,
            cursor: usize,
        }

        impl Read for FailAfterDoneReader {
            fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
                if self.cursor >= self.chunks.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::BrokenPipe,
                        "reader should not be polled after [DONE]",
                    ));
                }

                let chunk = self.chunks[self.cursor];
                self.cursor += 1;
                if chunk.len() > buf.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "chunk larger than read buffer",
                    ));
                }
                buf[..chunk.len()].copy_from_slice(chunk);
                Ok(chunk.len())
            }
        }

        let mut reader = FailAfterDoneReader {
            chunks: vec![b"data: first\n\n", b"data: [DONE]\n\n"],
            cursor: 0,
        };
        let mut events = Vec::new();

        let result = process_sse_data_events(&mut reader, |data| {
            let is_done = data == "[DONE]";
            events.push(data);
            Ok(is_done)
        });

        assert!(result.is_ok(), "unexpected parse error: {result:?}");
        assert_eq!(events, vec!["first".to_string(), "[DONE]".to_string()]);
    }

    #[test]
    fn process_sse_data_events_handles_split_chunks() {
        struct ChunkedReader {
            chunks: Vec<&'static [u8]>,
            cursor: usize,
        }

        impl Read for ChunkedReader {
            fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
                if self.cursor >= self.chunks.len() {
                    return Ok(0);
                }

                let chunk = self.chunks[self.cursor];
                self.cursor += 1;
                if chunk.len() > buf.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "chunk larger than read buffer",
                    ));
                }
                buf[..chunk.len()].copy_from_slice(chunk);
                Ok(chunk.len())
            }
        }

        let mut reader = ChunkedReader {
            chunks: vec![
                b"data: {\"choices\":[{\"delta\":{\"content\":\"Hel",
                b"lo\"}}]}\n",
                b"\n",
                b"data: [DONE]\n",
                b"\n",
            ],
            cursor: 0,
        };

        let mut events = Vec::new();
        let result = process_sse_data_events(&mut reader, |data| {
            let is_done = data == "[DONE]";
            events.push(data);
            Ok(is_done)
        });

        assert!(result.is_ok(), "unexpected parse error: {result:?}");
        assert_eq!(
            events,
            vec![
                "{\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}".to_string(),
                "[DONE]".to_string()
            ]
        );
    }

    #[test]
    fn merge_stream_delta_supports_incremental_and_snapshot_chunks() {
        let mut value = String::new();

        assert_eq!(
            merge_stream_delta(&mut value, "Ana"),
            Some("Ana".to_string())
        );
        assert_eq!(value, "Ana");

        assert_eq!(
            merge_stream_delta(&mut value, "Analyzing"),
            Some("lyzing".to_string())
        );
        assert_eq!(value, "Analyzing");

        assert_eq!(merge_stream_delta(&mut value, "Analyzing"), None);
        assert_eq!(value, "Analyzing");

        assert_eq!(
            merge_stream_delta(&mut value, " code path"),
            Some(" code path".to_string())
        );
        assert_eq!(value, "Analyzing code path");
    }

    #[test]
    fn openai_client_is_reused_across_requests() {
        let first = shared_http_client("https://api.openai.com/v1");
        let second = shared_http_client("https://api.openai.com/v1");
        assert!(std::ptr::eq(first, second));
    }
}
