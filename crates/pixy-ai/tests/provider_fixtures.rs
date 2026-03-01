use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use pixy_ai::{
    stream, AssistantContentBlock, AssistantMessageEvent, Context, Cost, Message, Model,
    StreamOptions, Tool, UserContent,
};
use serde::Deserialize;
use serde_json::{json, Value};

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

fn fixture_path(name: &str) -> PathBuf {
    let crate_fixture_relative = PathBuf::from("crates")
        .join("pixy-ai")
        .join("tests")
        .join("fixtures")
        .join("m1-provider")
        .join(name);

    // Prefer runtime lookup from current working directory upward when running
    // from workspace root or nested workspace paths.
    if let Ok(cwd) = std::env::current_dir() {
        for ancestor in cwd.ancestors() {
            let candidate = ancestor.join(&crate_fixture_relative);
            if candidate.is_file() {
                return candidate;
            }
        }
    }

    // Fallback to compile-time manifest-dir based location inside the crate.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("m1-provider")
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

fn spawn_openai_fallback_server(
    completions_body: String,
) -> (String, Arc<AtomicUsize>, Arc<AtomicUsize>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind local test server");
    listener
        .set_nonblocking(true)
        .expect("set nonblocking accept");
    let address = listener.local_addr().expect("server local addr");
    let responses_hits = Arc::new(AtomicUsize::new(0));
    let completions_hits = Arc::new(AtomicUsize::new(0));
    let responses_hits_thread = Arc::clone(&responses_hits);
    let completions_hits_thread = Arc::clone(&completions_hits);

    thread::spawn(move || {
        let mut idle_ticks = 0usize;
        loop {
            match listener.accept() {
                Ok((mut socket, _)) => {
                    idle_ticks = 0;
                    socket
                        .set_read_timeout(Some(Duration::from_secs(2)))
                        .expect("set read timeout");
                    let mut buffer = [0_u8; 16384];
                    let read_len = socket.read(&mut buffer).unwrap_or(0);
                    let request = String::from_utf8_lossy(&buffer[..read_len]);
                    let path = request
                        .lines()
                        .next()
                        .and_then(|line| line.split_whitespace().nth(1))
                        .unwrap_or("/");

                    if path.ends_with("/responses") {
                        responses_hits_thread.fetch_add(1, Ordering::SeqCst);
                        let body =
                            r#"{"error":{"message":"Not found","type":"invalid_request_error"}}"#;
                        let response = format!(
                            "HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                            body.len(),
                            body
                        );
                        socket
                            .write_all(response.as_bytes())
                            .expect("write 404 response");
                    } else if path.ends_with("/chat/completions") {
                        completions_hits_thread.fetch_add(1, Ordering::SeqCst);
                        let response = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                            completions_body.len(),
                            completions_body
                        );
                        socket
                            .write_all(response.as_bytes())
                            .expect("write completions response");
                    } else {
                        let body = r#"{"error":"unknown path"}"#;
                        let response = format!(
                            "HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                            body.len(),
                            body
                        );
                        socket
                            .write_all(response.as_bytes())
                            .expect("write unknown-path response");
                    }
                    let _ = socket.flush();
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    idle_ticks += 1;
                    if idle_ticks > 300 {
                        break;
                    }
                    thread::sleep(Duration::from_millis(10));
                }
                Err(_) => break,
            }
        }
    });

    (
        format!("http://{address}/v1"),
        responses_hits,
        completions_hits,
    )
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

fn collect_thinking(content: &[AssistantContentBlock]) -> String {
    content
        .iter()
        .filter_map(|block| match block {
            AssistantContentBlock::Thinking { thinking, .. } => Some(thinking.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

#[test]
fn fixture_files_resolve_from_crate_test_fixtures() {
    let path = fixture_path("openai-tooluse.json");
    let expected_suffix = PathBuf::from("tests")
        .join("fixtures")
        .join("m1-provider")
        .join("openai-tooluse.json");
    assert!(
        path.ends_with(&expected_suffix),
        "fixture path should end with {:?}, got {}",
        expected_suffix,
        path.display()
    );
    assert!(
        path.is_file(),
        "fixture file should exist: {}",
        path.display()
    );
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
            transport_retry_count: None,
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
fn openai_completions_reasoning_content_maps_to_thinking_blocks() {
    let chunks = vec![
        json!({
            "choices": [{
                "index": 0,
                "delta": { "reasoning_content": "I should inspect files first." },
                "finish_reason": null
            }]
        }),
        json!({
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }),
    ];
    let base_url = spawn_sse_server(sse_body(&chunks, true));
    let model = sample_model("openai-completions", base_url);
    let context = sample_context();
    let event_stream = stream(
        model,
        context,
        Some(StreamOptions {
            api_key: Some("test-key".to_string()),
            temperature: None,
            max_tokens: None,
            headers: None,
            transport_retry_count: None,
        }),
    )
    .expect("stream should start");

    let runtime = tokio::runtime::Runtime::new().expect("create runtime");
    let (message, saw_thinking_delta) = runtime.block_on(async {
        let mut saw_thinking_delta = false;
        while let Some(event) = event_stream.next().await {
            if matches!(event, AssistantMessageEvent::ThinkingDelta { .. }) {
                saw_thinking_delta = true;
            }
        }

        let message = event_stream
            .result()
            .await
            .expect("stream should produce final message");
        (message, saw_thinking_delta)
    });

    assert!(saw_thinking_delta, "expected ThinkingDelta event");
    assert_eq!(collect_text(&message.content), "");
    assert_eq!(
        collect_thinking(&message.content),
        "I should inspect files first."
    );
}

#[test]
fn openai_completions_reasoning_snapshot_chunks_do_not_duplicate_thinking() {
    let chunks = vec![
        json!({
            "choices": [{
                "index": 0,
                "delta": { "reasoning_content": "**Analyzing code path**" },
                "finish_reason": null
            }]
        }),
        json!({
            "choices": [{
                "index": 0,
                "delta": { "reasoning_content": "**Analyzing code path**\nNeed to inspect files." },
                "finish_reason": null
            }]
        }),
        json!({
            "choices": [{
                "index": 0,
                "delta": { "reasoning_content": "**Analyzing code path**\nNeed to inspect files." },
                "finish_reason": null
            }]
        }),
        json!({
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }),
    ];
    let base_url = spawn_sse_server(sse_body(&chunks, true));
    let model = sample_model("openai-completions", base_url);
    let context = sample_context();
    let event_stream = stream(
        model,
        context,
        Some(StreamOptions {
            api_key: Some("test-key".to_string()),
            temperature: None,
            max_tokens: None,
            headers: None,
            transport_retry_count: None,
        }),
    )
    .expect("stream should start");

    let runtime = tokio::runtime::Runtime::new().expect("create runtime");
    let (message, thinking_deltas) = runtime.block_on(async {
        let mut thinking_deltas = Vec::new();
        while let Some(event) = event_stream.next().await {
            if let AssistantMessageEvent::ThinkingDelta { delta, .. } = event {
                thinking_deltas.push(delta);
            }
        }

        let message = event_stream
            .result()
            .await
            .expect("stream should produce final message");
        (message, thinking_deltas)
    });

    assert_eq!(
        collect_thinking(&message.content),
        "**Analyzing code path**\nNeed to inspect files."
    );
    assert_eq!(
        thinking_deltas,
        vec![
            "**Analyzing code path**".to_string(),
            "\nNeed to inspect files.".to_string()
        ]
    );
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
            transport_retry_count: None,
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
fn openai_responses_stream_events_are_parsed_to_text_and_tool_call() {
    let chunks = vec![
        json!({
            "type": "response.output_item.added",
            "item": {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "in_progress",
                "content": [],
            },
        }),
        json!({
            "type": "response.content_part.added",
            "part": {
                "type": "output_text",
                "text": "",
            },
        }),
        json!({
            "type": "response.output_text.delta",
            "delta": "Need to inspect the file first. ",
        }),
        json!({
            "type": "response.output_text.delta",
            "delta": "I will call read.",
        }),
        json!({
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Need to inspect the file first. I will call read.",
                    }
                ],
            },
        }),
        json!({
            "type": "response.output_item.added",
            "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "read",
                "arguments": "",
            },
        }),
        json!({
            "type": "response.function_call_arguments.delta",
            "delta": "{\"path\":\"/tmp/snake_v3/index.html\"}",
        }),
        json!({
            "type": "response.function_call_arguments.done",
            "arguments": "{\"path\":\"/tmp/snake_v3/index.html\"}",
        }),
        json!({
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "read",
                "arguments": "{\"path\":\"/tmp/snake_v3/index.html\"}",
            },
        }),
        json!({
            "type": "response.completed",
            "response": {
                "status": "completed",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "total_tokens": 120,
                    "input_tokens_details": {
                        "cached_tokens": 10,
                    },
                },
            },
        }),
    ];
    let base_url = spawn_sse_server(sse_body(&chunks, false));
    let model = sample_model("openai-responses", base_url);
    let context = sample_context();
    let event_stream = stream(
        model,
        context,
        Some(StreamOptions {
            api_key: Some("test-key".to_string()),
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

    assert_eq!(
        collect_text(&message.content),
        "Need to inspect the file first. I will call read."
    );
    assert_eq!(message.stop_reason, pixy_ai::StopReason::ToolUse);

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

    assert_eq!(tool_call.0, "call_1|fc_1");
    assert_eq!(tool_call.1, "read");
    assert_eq!(tool_call.2, json!({"path":"/tmp/snake_v3/index.html"}));
    assert_eq!(message.usage.input, 90);
    assert_eq!(message.usage.output, 20);
    assert_eq!(message.usage.cache_read, 10);
    assert_eq!(message.usage.total_tokens, 120);
}

#[test]
fn openai_responses_tool_arguments_resolve_when_arg_events_only_have_item_id() {
    let chunks = vec![
        json!({
            "type": "response.output_item.added",
            "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "read",
                "arguments": "",
            },
        }),
        json!({
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_1",
            "delta": "{\"path\":\"/tmp/snake_v3/index.html\"}",
        }),
        json!({
            "type": "response.function_call_arguments.done",
            "item_id": "fc_1",
            "arguments": "{\"path\":\"/tmp/snake_v3/index.html\"}",
        }),
        json!({
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "read",
                "arguments": "{\"path\":\"/tmp/snake_v3/index.html\"}",
            },
        }),
        json!({
            "type": "response.completed",
            "response": {
                "status": "completed",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 3,
                    "total_tokens": 13,
                },
            },
        }),
    ];

    let base_url = spawn_sse_server(sse_body(&chunks, false));
    let model = sample_model("openai-responses", base_url);
    let context = sample_context();
    let event_stream = stream(
        model,
        context,
        Some(StreamOptions {
            api_key: Some("test-key".to_string()),
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

    let tool_call = message
        .content
        .iter()
        .find_map(|block| match block {
            AssistantContentBlock::ToolCall { arguments, .. } => Some(arguments.clone()),
            _ => None,
        })
        .expect("tool call block should exist");

    assert_eq!(tool_call, json!({"path":"/tmp/snake_v3/index.html"}));
}

#[test]
fn openai_responses_404_falls_back_to_completions_and_caches_base_url() {
    let completions_chunks = vec![
        json!({
            "id": "cmpl-1",
            "choices": [{
                "index": 0,
                "delta": { "content": "fallback answer" },
                "finish_reason": Value::Null,
            }],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 3,
                "total_tokens": 15,
            },
        }),
        json!({
            "id": "cmpl-1",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
        }),
    ];
    let completions_body = sse_body(&completions_chunks, true);
    let (base_url, responses_hits, completions_hits) =
        spawn_openai_fallback_server(completions_body);
    let model = sample_model("openai-responses", base_url);

    let runtime = tokio::runtime::Runtime::new().expect("create runtime");
    for _ in 0..2 {
        let event_stream = stream(
            model.clone(),
            sample_context(),
            Some(StreamOptions {
                api_key: Some("test-key".to_string()),
                temperature: None,
                max_tokens: None,
                headers: None,
                transport_retry_count: None,
            }),
        )
        .expect("stream should start");

        let message = runtime
            .block_on(event_stream.result())
            .expect("stream should produce final message");
        assert_eq!(collect_text(&message.content), "fallback answer");
        assert_eq!(message.api, "openai-completions");
    }

    assert_eq!(responses_hits.load(Ordering::SeqCst), 1);
    assert_eq!(completions_hits.load(Ordering::SeqCst), 2);
}
