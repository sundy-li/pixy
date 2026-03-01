use serde_json::{json, Value};

use crate::types::{
    AssistantContentBlock, Context, Message, Model, StreamOptions, Tool, ToolResultContentBlock,
    UserContent, UserContentBlock,
};

pub(super) fn build_anthropic_payload(
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
