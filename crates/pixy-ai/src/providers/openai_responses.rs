use serde_json::{Map, Value, json};
use std::collections::HashMap;
use std::env;
use std::io::{BufRead, BufReader, Read};
use tracing::info;

use super::common::{empty_assistant_message, join_url, shared_http_client};
use crate::AssistantMessageEventStream;
use crate::error::{PiAiError, PiAiErrorCode};
use crate::types::{
    AssistantContentBlock, AssistantMessage, AssistantMessageEvent, Context, DoneReason, Message,
    Model, SimpleStreamOptions, StopReason, StreamOptions, Tool, Usage, UserContent,
    UserContentBlock,
};

pub async fn run_openai_responses(
    model: Model,
    context: Context,
    options: Option<StreamOptions>,
    stream: AssistantMessageEventStream,
) -> Result<(), PiAiError> {
    let api_key = resolve_api_key(&model.provider, options.as_ref())?;

    let mut output = empty_assistant_message(&model);
    let payload = build_openai_responses_payload(&model, &context, options.as_ref());
    let endpoint = join_url(&model.base_url, "responses");
    let client = shared_http_client(&model.base_url);

    let execution = async {
        let mut request = client
            .post(endpoint.as_str())
            .header("Authorization", format!("Bearer {api_key}"))
            .header("Content-Type", "application/json");

        if let Some(headers) = options.as_ref().and_then(|opts| opts.headers.as_ref()) {
            for (key, value) in headers {
                request = request.header(key, value);
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

        let mut text_block_indices: HashMap<String, usize> = HashMap::new();
        let mut tool_block_indices: HashMap<String, usize> = HashMap::new();
        let mut tool_arg_buffers: HashMap<String, String> = HashMap::new();

        let body = response.text().await.map_err(|error| {
            PiAiError::new(
                PiAiErrorCode::ProviderTransport,
                format!("OpenAI responses read failed: {error}"),
            )
        })?;
        let mut reader = std::io::Cursor::new(body.into_bytes());
        process_sse_data_events(&mut reader, |data| {
            handle_openai_responses_event(
                data,
                &mut output,
                &stream,
                &mut text_block_indices,
                &mut tool_block_indices,
                &mut tool_arg_buffers,
            )
        })?;

        if output.content.is_empty() && output.stop_reason == StopReason::Stop {
            return Err(PiAiError::new(
                PiAiErrorCode::ProviderProtocol,
                "OpenAI responses ended without output content",
            ));
        }

        let done_reason = map_done_reason(output.stop_reason.clone()).ok_or_else(|| {
            PiAiError::new(
                PiAiErrorCode::ProviderProtocol,
                "OpenAI responses ended with non-terminal done reason",
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
        if is_provider_http_404(&error) {
            return Err(error);
        }
        output.stop_reason = StopReason::Error;
        output.error_message = Some(error.as_compact_json());
        stream.push(AssistantMessageEvent::Error {
            reason: crate::types::ErrorReason::Error,
            error: output,
        });
    }

    Ok(())
}

pub async fn run_simple_openai_responses(
    model: Model,
    context: Context,
    options: Option<SimpleStreamOptions>,
    stream: AssistantMessageEventStream,
) -> Result<(), PiAiError> {
    let mut model = model;
    apply_simple_reasoning_to_model(&mut model, options.as_ref());
    let stream_options = options.map(|simple| simple.stream);
    run_openai_responses(model, context, stream_options, stream).await
}

fn handle_openai_responses_event(
    data: String,
    output: &mut AssistantMessage,
    stream: &AssistantMessageEventStream,
    text_block_indices: &mut HashMap<String, usize>,
    tool_block_indices: &mut HashMap<String, usize>,
    tool_arg_buffers: &mut HashMap<String, String>,
) -> Result<bool, PiAiError> {
    info!("OpenAI responses data: {}", data);
    if data == "[DONE]" {
        return Ok(true);
    }

    let event: Value = serde_json::from_str(&data).map_err(|error| {
        PiAiError::new(
            PiAiErrorCode::ProviderProtocol,
            format!("Invalid OpenAI responses event JSON: {error}"),
        )
        .with_details(json!({ "event": data }))
    })?;

    let Some(event_type) = event.get("type").and_then(Value::as_str) else {
        return Ok(false);
    };

    match event_type {
        "response.output_item.added" => {
            let Some(item) = event.get("item").and_then(Value::as_object) else {
                return Ok(false);
            };
            let Some(item_type) = item.get("type").and_then(Value::as_str) else {
                return Ok(false);
            };
            match item_type {
                "message" => {
                    let Some(item_id) = item.get("id").and_then(Value::as_str) else {
                        return Ok(false);
                    };
                    let content_index = output.content.len();
                    output.content.push(AssistantContentBlock::Text {
                        text: String::new(),
                        text_signature: None,
                    });
                    text_block_indices.insert(item_id.to_string(), content_index);
                    stream.push(AssistantMessageEvent::TextStart {
                        content_index,
                        partial: output.clone(),
                    });
                }
                "function_call" => {
                    let item_id = item.get("id").and_then(Value::as_str).unwrap_or_default();
                    let call_id = item
                        .get("call_id")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    let tool_name = item.get("name").and_then(Value::as_str).unwrap_or_default();
                    if call_id.is_empty() || tool_name.is_empty() {
                        return Ok(false);
                    }
                    let tool_key = tool_item_key(call_id, item_id);
                    let content_index = output.content.len();
                    output.content.push(AssistantContentBlock::ToolCall {
                        id: tool_call_compound_id(call_id, item_id),
                        name: tool_name.to_string(),
                        arguments: Value::Object(Map::new()),
                        thought_signature: None,
                    });

                    let initial_arguments = item
                        .get("arguments")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    tool_arg_buffers.insert(tool_key.clone(), initial_arguments.clone());
                    if let Some(AssistantContentBlock::ToolCall { arguments, .. }) =
                        output.content.get_mut(content_index)
                    {
                        if !initial_arguments.is_empty() {
                            *arguments = parse_partial_json(&initial_arguments);
                        }
                    }
                    tool_block_indices.insert(tool_key, content_index);
                    stream.push(AssistantMessageEvent::ToolcallStart {
                        content_index,
                        partial: output.clone(),
                    });
                }
                _ => {}
            }
        }
        "response.output_text.delta" | "response.refusal.delta" => {
            let Some(item_id) = event.get("item_id").and_then(Value::as_str) else {
                return Ok(false);
            };
            let Some(delta) = event.get("delta").and_then(Value::as_str) else {
                return Ok(false);
            };
            if delta.is_empty() {
                return Ok(false);
            }
            let Some(content_index) = text_block_indices.get(item_id).copied() else {
                return Ok(false);
            };
            if let Some(AssistantContentBlock::Text { text, .. }) =
                output.content.get_mut(content_index)
            {
                text.push_str(delta);
            }
            stream.push(AssistantMessageEvent::TextDelta {
                content_index,
                delta: delta.to_string(),
                partial: output.clone(),
            });
        }
        "response.function_call_arguments.delta" => {
            let Some(tool_key) = resolve_tool_key_for_arg_event(&event, tool_block_indices) else {
                return Ok(false);
            };
            let Some(delta) = event.get("delta").and_then(Value::as_str) else {
                return Ok(false);
            };
            let Some(content_index) = tool_block_indices.get(&tool_key).copied() else {
                return Ok(false);
            };
            let buffer = tool_arg_buffers.entry(tool_key).or_default();
            buffer.push_str(delta);
            if let Some(AssistantContentBlock::ToolCall { arguments, .. }) =
                output.content.get_mut(content_index)
            {
                *arguments = parse_partial_json(buffer);
            }
            stream.push(AssistantMessageEvent::ToolcallDelta {
                content_index,
                delta: delta.to_string(),
                partial: output.clone(),
            });
        }
        "response.function_call_arguments.done" => {
            let Some(tool_key) = resolve_tool_key_for_arg_event(&event, tool_block_indices) else {
                return Ok(false);
            };
            let Some(arguments) = event.get("arguments").and_then(Value::as_str) else {
                return Ok(false);
            };
            if let Some(content_index) = tool_block_indices.get(&tool_key).copied() {
                tool_arg_buffers.insert(tool_key, arguments.to_string());
                if let Some(AssistantContentBlock::ToolCall {
                    arguments: arg_json,
                    ..
                }) = output.content.get_mut(content_index)
                {
                    *arg_json = parse_partial_json(arguments);
                }
            }
        }
        "response.output_item.done" => {
            let Some(item) = event.get("item").and_then(Value::as_object) else {
                return Ok(false);
            };
            let Some(item_type) = item.get("type").and_then(Value::as_str) else {
                return Ok(false);
            };
            match item_type {
                "message" => {
                    let Some(item_id) = item.get("id").and_then(Value::as_str) else {
                        return Ok(false);
                    };
                    let Some(content_index) = text_block_indices.remove(item_id) else {
                        return Ok(false);
                    };
                    if let Some(AssistantContentBlock::Text {
                        text,
                        text_signature,
                    }) = output.content.get_mut(content_index)
                    {
                        if text.is_empty() {
                            *text = item
                                .get("content")
                                .and_then(Value::as_array)
                                .map(|parts| {
                                    parts
                                        .iter()
                                        .filter_map(|part| {
                                            let part_type = part
                                                .get("type")
                                                .and_then(Value::as_str)
                                                .unwrap_or_default();
                                            match part_type {
                                                "output_text" => {
                                                    part.get("text").and_then(Value::as_str)
                                                }
                                                "refusal" => {
                                                    part.get("refusal").and_then(Value::as_str)
                                                }
                                                _ => None,
                                            }
                                        })
                                        .collect::<Vec<_>>()
                                        .join("")
                                })
                                .unwrap_or_default();
                        }
                        *text_signature =
                            item.get("id").and_then(Value::as_str).map(str::to_string);
                        let content = text.clone();
                        stream.push(AssistantMessageEvent::TextEnd {
                            content_index,
                            content,
                            partial: output.clone(),
                        });
                    }
                }
                "function_call" => {
                    let item_id = item.get("id").and_then(Value::as_str).unwrap_or_default();
                    let call_id = item
                        .get("call_id")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    if call_id.is_empty() {
                        return Ok(false);
                    }
                    let tool_key = tool_item_key(call_id, item_id);
                    let Some(content_index) = tool_block_indices.remove(&tool_key) else {
                        return Ok(false);
                    };
                    let parsed_arguments = if let Some(buffer) = tool_arg_buffers.remove(&tool_key)
                    {
                        parse_partial_json(&buffer)
                    } else if let Some(arguments) = item.get("arguments").and_then(Value::as_str) {
                        parse_partial_json(arguments)
                    } else {
                        Value::Object(Map::new())
                    };

                    let mut tool_call_json = Value::Null;
                    if let Some(AssistantContentBlock::ToolCall {
                        id,
                        name,
                        arguments,
                        thought_signature,
                    }) = output.content.get_mut(content_index)
                    {
                        *arguments = parsed_arguments.clone();
                        tool_call_json = json!({
                            "type": "toolCall",
                            "id": id,
                            "name": name,
                            "arguments": parsed_arguments,
                            "thoughtSignature": thought_signature,
                        });
                    }
                    stream.push(AssistantMessageEvent::ToolcallEnd {
                        content_index,
                        tool_call: tool_call_json,
                        partial: output.clone(),
                    });
                }
                _ => {}
            }
        }
        "response.completed" => {
            let Some(response_value) = event.get("response") else {
                return Ok(false);
            };
            if let Some(usage_value) = response_value.get("usage") {
                update_usage_from_openai_responses(&mut output.usage, usage_value);
            }
            let status = response_value
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("completed");
            output.stop_reason = map_responses_stop_reason(status);
            if output
                .content
                .iter()
                .any(|block| matches!(block, AssistantContentBlock::ToolCall { .. }))
                && output.stop_reason == StopReason::Stop
            {
                output.stop_reason = StopReason::ToolUse;
            }
        }
        "error" => {
            let message = event
                .get("message")
                .and_then(Value::as_str)
                .or_else(|| {
                    event
                        .get("error")
                        .and_then(Value::as_object)
                        .and_then(|error| error.get("message"))
                        .and_then(Value::as_str)
                })
                .unwrap_or("Unknown OpenAI responses error");
            return Err(PiAiError::new(
                PiAiErrorCode::ProviderProtocol,
                format!("OpenAI responses error: {message}"),
            ));
        }
        "response.failed" => {
            return Err(PiAiError::new(
                PiAiErrorCode::ProviderProtocol,
                "OpenAI responses failed",
            ));
        }
        _ => {}
    }

    Ok(false)
}

fn build_openai_responses_payload(
    model: &Model,
    context: &Context,
    options: Option<&StreamOptions>,
) -> Value {
    let mut payload = json!({
        "model": model.id,
        "store": false,
        "stream": true,
        "instructions": context
            .system_prompt
            .clone()
            .unwrap_or_else(|| "You are a helpful assistant.".to_string()),
        "input": convert_responses_messages(context),
        "tool_choice": "auto",
        "parallel_tool_calls": true,
    });

    if let Some(max_tokens) = options.and_then(|options| options.max_tokens) {
        payload["max_output_tokens"] = json!(max_tokens);
    }
    if let Some(temperature) = options.and_then(|options| options.temperature) {
        payload["temperature"] = json!(temperature);
    }
    if let Some(tools) = &context.tools {
        payload["tools"] = convert_responses_tools(tools);
    }
    if model.reasoning {
        let effort = model
            .reasoning_effort
            .as_ref()
            .unwrap_or(&crate::types::ThinkingLevel::Medium);
        payload["reasoning"] = json!({
            "effort": thinking_level_to_effort(model, effort),
        });
    }

    info!(
        target: "pixy_ai::providers::openai_responses",
        provider = %model.provider,
        model = %model.id,
        payload = %payload,
        "build_openai_responses_payload"
    );

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
    let is_openai = model.name.to_lowercase().contains("openai")
        || model.provider.to_lowercase().contains("openai")
        || model.id.to_lowercase().starts_with("gpt-");
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

fn convert_responses_messages(context: &Context) -> Value {
    let mut messages = Vec::new();
    let mut synthetic_message_index = 0usize;

    for message in &context.messages {
        match message {
            Message::User { content, .. } => match content {
                UserContent::Text(text) => messages.push(json!({
                    "role": "user",
                    "content": [{
                        "type": "input_text",
                        "text": text,
                    }],
                })),
                UserContent::Blocks(blocks) => {
                    let converted = blocks
                        .iter()
                        .filter_map(|block| match block {
                            UserContentBlock::Text { text, .. } => Some(json!({
                                "type": "input_text",
                                "text": text,
                            })),
                            UserContentBlock::Image { data, mime_type } => Some(json!({
                                "type": "input_image",
                                "detail": "auto",
                                "image_url": format!("data:{mime_type};base64,{data}"),
                            })),
                        })
                        .collect::<Vec<_>>();
                    if !converted.is_empty() {
                        messages.push(json!({
                            "role": "user",
                            "content": converted,
                        }));
                    }
                }
            },
            Message::Assistant { content, .. } => {
                for block in content {
                    match block {
                        AssistantContentBlock::Text {
                            text,
                            text_signature,
                        } => {
                            synthetic_message_index = synthetic_message_index.saturating_add(1);
                            messages.push(json!({
                                "type": "message",
                                "id": text_signature
                                    .clone()
                                    .unwrap_or_else(|| format!("msg_{synthetic_message_index}")),
                                "role": "assistant",
                                "status": "completed",
                                "content": [{
                                    "type": "output_text",
                                    "text": text,
                                    "annotations": [],
                                }],
                            }));
                        }
                        AssistantContentBlock::Thinking { .. } => {}
                        AssistantContentBlock::ToolCall {
                            id,
                            name,
                            arguments,
                            ..
                        } => {
                            let (call_id, item_id) = split_tool_call_id(id);
                            let mut payload = json!({
                                "type": "function_call",
                                "call_id": call_id,
                                "name": name,
                                "arguments": arguments.to_string(),
                            });
                            if !item_id.is_empty() {
                                payload["id"] = json!(item_id);
                            }
                            messages.push(payload);
                        }
                    }
                }
            }
            Message::ToolResult {
                tool_call_id,
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
                let (call_id, _) = split_tool_call_id(tool_call_id);
                messages.push(json!({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": if text.is_empty() { "(no text result)" } else { text.as_str() },
                }));
            }
        }
    }

    Value::Array(messages)
}

fn convert_responses_tools(tools: &[Tool]) -> Value {
    Value::Array(
        tools
            .iter()
            .map(|tool| {
                json!({
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                })
            })
            .collect(),
    )
}

fn split_tool_call_id(id: &str) -> (String, String) {
    if let Some((call_id, item_id)) = id.split_once('|') {
        return (call_id.to_string(), item_id.to_string());
    }
    (id.to_string(), String::new())
}

fn tool_item_key(call_id: &str, item_id: &str) -> String {
    if item_id.is_empty() {
        call_id.to_string()
    } else {
        format!("{call_id}|{item_id}")
    }
}

fn tool_call_compound_id(call_id: &str, item_id: &str) -> String {
    if item_id.is_empty() {
        call_id.to_string()
    } else {
        format!("{call_id}|{item_id}")
    }
}

fn tool_key_by_item_or_event(item_id: &str, event: &Value) -> String {
    if item_id.contains('|') {
        return item_id.to_string();
    }

    if let Some(call_id) = event.get("call_id").and_then(Value::as_str) {
        return tool_item_key(call_id, item_id);
    }

    item_id.to_string()
}

fn resolve_tool_key_for_arg_event(
    event: &Value,
    tool_block_indices: &HashMap<String, usize>,
) -> Option<String> {
    if let Some(item_id) = event.get("item_id").and_then(Value::as_str) {
        let candidate = tool_key_by_item_or_event(item_id, event);
        if tool_block_indices.contains_key(&candidate) {
            return Some(candidate);
        }

        if !candidate.contains('|') {
            let suffix = format!("|{candidate}");
            let mut matches = tool_block_indices
                .keys()
                .filter(|key| key.ends_with(&suffix))
                .cloned();
            if let Some(first_match) = matches.next() {
                if matches.next().is_none() {
                    return Some(first_match);
                }
            }
        }
    }

    if let Some(call_id) = event.get("call_id").and_then(Value::as_str) {
        if tool_block_indices.contains_key(call_id) {
            return Some(call_id.to_string());
        }

        let prefix = format!("{call_id}|");
        let mut matches = tool_block_indices
            .keys()
            .filter(|key| key.starts_with(&prefix))
            .cloned();
        if let Some(first_match) = matches.next() {
            if matches.next().is_none() {
                return Some(first_match);
            }
        }
    }

    if tool_block_indices.len() == 1 {
        return tool_block_indices.keys().next().cloned();
    }

    None
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

fn map_responses_stop_reason(status: &str) -> StopReason {
    match status {
        "completed" => StopReason::Stop,
        "incomplete" => StopReason::Length,
        "failed" | "cancelled" => StopReason::Error,
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

fn update_usage_from_openai_responses(usage: &mut Usage, value: &Value) {
    let input_tokens = value
        .get("input_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(usage.input);
    let output_tokens = value
        .get("output_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(usage.output);
    let cached_tokens = value
        .get("input_tokens_details")
        .and_then(Value::as_object)
        .and_then(|details| details.get("cached_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(usage.cache_read);
    let total_tokens = value
        .get("total_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(input_tokens + output_tokens);

    usage.input = input_tokens.saturating_sub(cached_tokens);
    usage.output = output_tokens;
    usage.cache_read = cached_tokens;
    usage.cache_write = 0;
    usage.total_tokens = total_tokens;
}

fn is_provider_http_404(error: &PiAiError) -> bool {
    error.code == PiAiErrorCode::ProviderHttp && error.message.contains("HTTP 404")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Cost, ThinkingLevel};
    use std::io::{self, Read};

    fn sample_model() -> Model {
        Model {
            id: "gpt-5.3-codex".to_string(),
            name: "gpt-5.3-codex".to_string(),
            api: "openai-responses".to_string(),
            provider: "openai".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            reasoning: true,
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
            system_prompt: Some("You are helpful.".to_string()),
            messages: vec![Message::User {
                content: UserContent::Text("hello".to_string()),
                timestamp: 0,
            }],
            tools: None,
        }
    }

    #[test]
    fn openai_responses_payload_injects_reasoning_with_default_medium() {
        let model = sample_model();
        let context = sample_context();

        let payload = build_openai_responses_payload(&model, &context, None);
        assert_eq!(payload["reasoning"]["effort"], "medium");
    }

    #[test]
    fn openai_responses_payload_uses_configured_reasoning_effort() {
        let mut model = sample_model();
        model.reasoning_effort = Some(ThinkingLevel::High);
        let context = sample_context();

        let payload = build_openai_responses_payload(&model, &context, None);
        assert_eq!(payload["reasoning"]["effort"], "high");
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
                b"data: {\"type\":\"response.completed\",\"response\":",
                b"{\"status\":\"completed\"}}\n",
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
                "{\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}"
                    .to_string(),
                "[DONE]".to_string()
            ]
        );
    }

    #[test]
    fn openai_client_is_reused_across_requests() {
        let first = shared_http_client("https://api.openai.com/v1");
        let second = shared_http_client("https://api.openai.com/v1");
        assert!(std::ptr::eq(first, second));
    }
}
