use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::thread;
use std::time::Duration;

use pi_ai::{
    AssistantContentBlock, Context, Cost, Message, Model, StreamOptions, Tool, UserContent, stream,
};
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Debug, Deserialize)]
struct Fixture {
    provider: String,
    chunks: Vec<Value>,
    expected: Expected,
}

#[derive(Debug, Deserialize)]
struct Expected {
    #[serde(rename = "stopReason")]
    stop_reason: String,
    text: String,
    #[serde(rename = "toolCall")]
    tool_call: ExpectedToolCall,
    usage: ExpectedUsage,
}

#[derive(Debug, Deserialize)]
struct ExpectedToolCall {
    id: String,
    name: String,
    arguments: Value,
}

#[derive(Debug, Deserialize)]
struct ExpectedUsage {
    input: u64,
    output: u64,
}

fn sample_model(api: &str, base_url: String) -> Model {
    Model {
        id: "test-model".to_string(),
        name: "Test Model".to_string(),
        api: api.to_string(),
        provider: if api == "anthropic-messages" {
            "anthropic".to_string()
        } else {
            "openai".to_string()
        },
        base_url,
        reasoning: true,
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

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/fixtures/m1-provider")
        .join(name)
}

fn read_fixture(name: &str) -> Fixture {
    let data = fs::read_to_string(fixture_path(name)).expect("read fixture file");
    serde_json::from_str(&data).expect("parse fixture json")
}

fn spawn_sse_server(body: String) -> String {
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
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.as_bytes().len(),
                body
            );
            socket
                .write_all(response.as_bytes())
                .expect("write response");
            let _ = socket.flush();
        }
    });

    format!("http://{address}/v1")
}

fn sse_body(chunks: &[Value], append_done: bool) -> String {
    let mut body = String::new();
    for chunk in chunks {
        body.push_str("data: ");
        body.push_str(&serde_json::to_string(chunk).expect("serialize chunk"));
        body.push_str("\n\n");
    }
    if append_done {
        body.push_str("data: [DONE]\n\n");
    }
    body
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
fn openai_fixture_matches_expected_contract() {
    let fixture = read_fixture("openai-tooluse.json");
    let base_url = spawn_sse_server(sse_body(&fixture.chunks, true));

    let model = sample_model(&fixture.provider, base_url);
    let context = sample_context();
    let event_stream = stream(
        model,
        context,
        Some(StreamOptions {
            api_key: Some("test-key".to_string()),
            temperature: None,
            max_tokens: None,
            headers: None,
        }),
    )
    .expect("stream should start");

    let runtime = tokio::runtime::Runtime::new().expect("create runtime");
    let message = runtime
        .block_on(event_stream.result())
        .expect("stream should produce final message");

    let stop_reason = serde_json::to_value(&message.stop_reason).expect("serialize stop reason");
    assert_eq!(stop_reason, json!(fixture.expected.stop_reason));
    assert_eq!(collect_text(&message.content), fixture.expected.text);

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

    assert_eq!(tool_call.0, fixture.expected.tool_call.id);
    assert_eq!(tool_call.1, fixture.expected.tool_call.name);
    assert_eq!(tool_call.2, fixture.expected.tool_call.arguments);
    assert_eq!(message.usage.input, fixture.expected.usage.input);
    assert_eq!(message.usage.output, fixture.expected.usage.output);
}

#[test]
fn anthropic_fixture_matches_expected_contract() {
    let fixture = read_fixture("anthropic-tooluse.json");
    let base_url = spawn_sse_server(sse_body(&fixture.chunks, false));

    let model = sample_model(&fixture.provider, base_url);
    let context = sample_context();
    let event_stream = stream(
        model,
        context,
        Some(StreamOptions {
            api_key: Some("test-key".to_string()),
            temperature: None,
            max_tokens: None,
            headers: None,
        }),
    )
    .expect("stream should start");

    let runtime = tokio::runtime::Runtime::new().expect("create runtime");
    let message = runtime
        .block_on(event_stream.result())
        .expect("stream should produce final message");

    let stop_reason = serde_json::to_value(&message.stop_reason).expect("serialize stop reason");
    assert_eq!(stop_reason, json!(fixture.expected.stop_reason));
    assert_eq!(collect_text(&message.content), fixture.expected.text);

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

    assert_eq!(tool_call.0, fixture.expected.tool_call.id);
    assert_eq!(tool_call.1, fixture.expected.tool_call.name);
    assert_eq!(tool_call.2, fixture.expected.tool_call.arguments);
    assert_eq!(message.usage.input, fixture.expected.usage.input);
    assert_eq!(message.usage.output, fixture.expected.usage.output);
}
