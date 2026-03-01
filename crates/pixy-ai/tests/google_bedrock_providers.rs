use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;
use std::time::Duration;

use pixy_ai::{
    stream, AssistantContentBlock, Context, Cost, Message, Model, StopReason, StreamOptions, Tool,
    UserContent,
};
use serde_json::json;

fn spawn_json_server(body: String) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind local test server");
    let address = listener.local_addr().expect("server local addr");
    thread::spawn(move || {
        if let Ok((mut socket, _)) = listener.accept() {
            socket
                .set_read_timeout(Some(Duration::from_secs(2)))
                .expect("set read timeout");
            let mut buffer = [0_u8; 8192];
            let _ = socket.read(&mut buffer);

            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            socket
                .write_all(response.as_bytes())
                .expect("write response");
            let _ = socket.flush();
        }
    });

    format!("http://{address}")
}

fn sample_model(api: &str, provider: &str, base_url: String) -> Model {
    Model {
        id: "test-model".to_string(),
        name: "Test Model".to_string(),
        api: api.to_string(),
        provider: provider.to_string(),
        base_url,
        reasoning: true,
        reasoning_effort: None,
        input: vec!["text".to_string(), "image".to_string()],
        cost: Cost {
            input: 0.0,
            output: 0.0,
            cache_read: 0.0,
            cache_write: 0.0,
            total: 0.0,
        },
        context_window: 128_000,
        max_tokens: 8_192,
    }
}

fn sample_context() -> Context {
    Context {
        system_prompt: Some("You are a file assistant".to_string()),
        messages: vec![Message::User {
            content: UserContent::Text("List files".to_string()),
            timestamp: 1_700_000_000_000,
        }],
        tools: Some(vec![Tool {
            name: "read".to_string(),
            description: "Read file".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" }
                },
                "required": ["path"]
            }),
        }]),
    }
}

fn collect_text(content: &[AssistantContentBlock]) -> String {
    content
        .iter()
        .filter_map(|block| match block {
            AssistantContentBlock::Text { text, .. } => Some(text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

#[test]
fn google_provider_parses_text_tool_and_usage() {
    let body = json!({
        "candidates": [{
            "content": {
                "parts": [
                    { "text": "Checking files." },
                    {
                        "functionCall": {
                            "id": "call_read_1",
                            "name": "read",
                            "args": { "path": "README.md" }
                        }
                    }
                ]
            },
            "finishReason": "STOP"
        }],
        "usageMetadata": {
            "promptTokenCount": 12,
            "candidatesTokenCount": 5,
            "totalTokenCount": 17
        }
    })
    .to_string();

    let model = sample_model(
        "google-generative-ai",
        "google-generative-ai",
        spawn_json_server(body),
    );
    let event_stream = stream(
        model,
        sample_context(),
        Some(StreamOptions {
            api_key: Some("AIza-test-key".to_string()),
            temperature: None,
            max_tokens: None,
            headers: None,
            transport_retry_count: None,
        }),
    )
    .expect("stream should start");

    let runtime = tokio::runtime::Runtime::new().expect("create runtime");
    let message = runtime
        .block_on(event_stream.result())
        .expect("stream should produce final message");

    assert_eq!(message.stop_reason, StopReason::ToolUse);
    assert_eq!(collect_text(&message.content), "Checking files.");
    assert_eq!(message.usage.input, 12);
    assert_eq!(message.usage.output, 5);

    let tool_call = message
        .content
        .iter()
        .find_map(|block| match block {
            AssistantContentBlock::ToolCall {
                id,
                name,
                arguments,
                ..
            } => Some((id.clone(), name.clone(), arguments.clone())),
            _ => None,
        })
        .expect("tool call block should exist");
    assert_eq!(tool_call.0, "call_read_1");
    assert_eq!(tool_call.1, "read");
    assert_eq!(tool_call.2, json!({ "path": "README.md" }));
}

#[test]
fn bedrock_provider_parses_text_tool_and_usage() {
    let body = json!({
        "output": {
            "message": {
                "content": [
                    { "text": "Checking files." },
                    {
                        "toolUse": {
                            "toolUseId": "call_read_1",
                            "name": "read",
                            "input": { "path": "README.md" }
                        }
                    }
                ]
            }
        },
        "stopReason": "tool_use",
        "usage": {
            "inputTokens": 12,
            "outputTokens": 5,
            "totalTokens": 17
        }
    })
    .to_string();

    let model = sample_model(
        "bedrock-converse-stream",
        "amazon-bedrock",
        spawn_json_server(body),
    );
    let event_stream = stream(
        model,
        sample_context(),
        Some(StreamOptions {
            api_key: Some("bedrock-token".to_string()),
            temperature: None,
            max_tokens: None,
            headers: None,
            transport_retry_count: None,
        }),
    )
    .expect("stream should start");

    let runtime = tokio::runtime::Runtime::new().expect("create runtime");
    let message = runtime
        .block_on(event_stream.result())
        .expect("stream should produce final message");

    assert_eq!(message.stop_reason, StopReason::ToolUse);
    assert_eq!(collect_text(&message.content), "Checking files.");
    assert_eq!(message.usage.input, 12);
    assert_eq!(message.usage.output, 5);

    let tool_call = message
        .content
        .iter()
        .find_map(|block| match block {
            AssistantContentBlock::ToolCall {
                id,
                name,
                arguments,
                ..
            } => Some((id.clone(), name.clone(), arguments.clone())),
            _ => None,
        })
        .expect("tool call block should exist");
    assert_eq!(tool_call.0, "call_read_1");
    assert_eq!(tool_call.1, "read");
    assert_eq!(tool_call.2, json!({ "path": "README.md" }));
}
