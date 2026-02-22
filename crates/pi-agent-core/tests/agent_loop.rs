use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use pi_agent_core::{
    AgentAbortController, AgentContext, AgentEvent, AgentLoopConfig, AgentMessage,
    AgentRetryConfig, AgentTool, AgentToolResult, agent_loop, agent_loop_continue,
};
use pi_ai::{
    AssistantContentBlock, AssistantMessage, AssistantMessageEvent, AssistantMessageEventStream,
    Context, Cost, DoneReason, Message, Model, StopReason, ToolResultContentBlock, Usage,
    UserContent,
};
use serde_json::{Value, json};
use tokio::time::sleep;

fn sample_usage() -> Usage {
    Usage {
        input: 10,
        output: 5,
        cache_read: 0,
        cache_write: 0,
        total_tokens: 15,
        cost: Cost {
            input: 0.01,
            output: 0.02,
            cache_read: 0.0,
            cache_write: 0.0,
            total: 0.03,
        },
    }
}

fn sample_model(api: &str) -> Model {
    Model {
        id: "test-model".to_string(),
        name: "Test Model".to_string(),
        api: api.to_string(),
        provider: "test".to_string(),
        base_url: "http://localhost".to_string(),
        reasoning: false,
        input: vec!["text".to_string()],
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

fn user_message(text: &str, ts: i64) -> AgentMessage {
    Message::User {
        content: UserContent::Text(text.to_string()),
        timestamp: ts,
    }
}

fn assistant_message(
    blocks: Vec<AssistantContentBlock>,
    stop_reason: StopReason,
    ts: i64,
) -> AssistantMessage {
    AssistantMessage {
        role: "assistant".to_string(),
        content: blocks,
        api: "test-api".to_string(),
        provider: "test".to_string(),
        model: "test-model".to_string(),
        usage: sample_usage(),
        stop_reason,
        error_message: None,
        timestamp: ts,
    }
}

fn done_stream(message: AssistantMessage, reason: DoneReason) -> AssistantMessageEventStream {
    let stream = AssistantMessageEventStream::new();
    stream.push(AssistantMessageEvent::Start {
        partial: message.clone(),
    });
    stream.push(AssistantMessageEvent::Done { reason, message });
    stream
}

async fn collect_events_and_result(
    stream: pi_ai::EventStream<AgentEvent, Vec<AgentMessage>>,
) -> (Vec<AgentEvent>, Vec<AgentMessage>) {
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }

    let result = stream
        .result()
        .await
        .expect("agent stream should produce result");
    (events, result)
}

#[tokio::test]
async fn agent_loop_emits_basic_lifecycle_events() {
    let prompts = vec![user_message("hello", 1_700_000_000_000)];
    let context = AgentContext {
        system_prompt: "You are helpful".to_string(),
        messages: vec![],
        tools: vec![],
    };

    let stream_fn = Arc::new(
        |_model: Model, _context: Context, _options: Option<pi_ai::SimpleStreamOptions>| {
            let message = assistant_message(
                vec![AssistantContentBlock::Text {
                    text: "world".to_string(),
                    text_signature: None,
                }],
                StopReason::Stop,
                1_700_000_000_010,
            );
            Ok(done_stream(message, DoneReason::Stop))
        },
    );

    let config = AgentLoopConfig {
        model: sample_model("test-api"),
        fallback_models: vec![],
        convert_to_llm: Arc::new(|messages| messages),
        stream_fn,
        retry: AgentRetryConfig::default(),
        get_steering_messages: None,
        get_follow_up_messages: None,
    };

    let stream = agent_loop(prompts, context, config, None);
    let (events, result) = collect_events_and_result(stream).await;

    let event_types: Vec<&'static str> = events
        .iter()
        .map(|e| match e {
            AgentEvent::AgentStart => "agent_start",
            AgentEvent::AgentEnd { .. } => "agent_end",
            AgentEvent::TurnStart => "turn_start",
            AgentEvent::TurnEnd { .. } => "turn_end",
            AgentEvent::MessageStart { .. } => "message_start",
            AgentEvent::MessageUpdate { .. } => "message_update",
            AgentEvent::MessageEnd { .. } => "message_end",
            AgentEvent::ToolExecutionStart { .. } => "tool_execution_start",
            AgentEvent::ToolExecutionUpdate { .. } => "tool_execution_update",
            AgentEvent::ToolExecutionEnd { .. } => "tool_execution_end",
            AgentEvent::RetryScheduled { .. } => "retry_scheduled",
            AgentEvent::ModelFallback { .. } => "model_fallback",
            AgentEvent::Metrics { .. } => "metrics",
        })
        .collect();

    assert_eq!(
        event_types,
        vec![
            "agent_start",
            "turn_start",
            "message_start",
            "message_end",
            "message_start",
            "message_end",
            "turn_end",
            "metrics",
            "agent_end"
        ]
    );

    assert_eq!(result.len(), 2, "result should include prompt + assistant");
}

#[tokio::test]
async fn agent_loop_executes_tool_calls_and_continues() {
    let call_count = Arc::new(AtomicUsize::new(0));
    let stream_fn_calls = call_count.clone();

    let stream_fn = Arc::new(
        move |_model: Model, _context: Context, _options: Option<pi_ai::SimpleStreamOptions>| {
            let index = stream_fn_calls.fetch_add(1, Ordering::SeqCst);
            if index == 0 {
                let tool_call_msg = assistant_message(
                    vec![AssistantContentBlock::ToolCall {
                        id: "call_1".to_string(),
                        name: "read_file".to_string(),
                        arguments: json!({"path": "README.md"}),
                        thought_signature: None,
                    }],
                    StopReason::ToolUse,
                    1_700_000_000_020,
                );
                Ok(done_stream(tool_call_msg, DoneReason::ToolUse))
            } else {
                let final_msg = assistant_message(
                    vec![AssistantContentBlock::Text {
                        text: "done".to_string(),
                        text_signature: None,
                    }],
                    StopReason::Stop,
                    1_700_000_000_030,
                );
                Ok(done_stream(final_msg, DoneReason::Stop))
            }
        },
    );

    let tool = AgentTool {
        name: "read_file".to_string(),
        label: "Read File".to_string(),
        description: "Read a file from disk".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" }
            },
            "required": ["path"],
            "additionalProperties": false
        }),
        execute: Arc::new(
            |_tool_call_id: String,
             _args: Value|
             -> Pin<Box<dyn Future<Output = Result<AgentToolResult, String>> + Send>> {
                Box::pin(async move {
                    Ok(AgentToolResult {
                        content: vec![ToolResultContentBlock::Text {
                            text: "file-content".to_string(),
                            text_signature: None,
                        }],
                        details: json!({"bytes": 12}),
                    })
                })
            },
        ),
    };

    let prompts = vec![user_message("read file", 1_700_000_000_000)];
    let context = AgentContext {
        system_prompt: "You are helpful".to_string(),
        messages: vec![],
        tools: vec![tool],
    };

    let config = AgentLoopConfig {
        model: sample_model("test-api"),
        fallback_models: vec![],
        convert_to_llm: Arc::new(|messages| messages),
        stream_fn,
        retry: AgentRetryConfig::default(),
        get_steering_messages: None,
        get_follow_up_messages: None,
    };

    let stream = agent_loop(prompts, context, config, None);
    let (events, result) = collect_events_and_result(stream).await;

    let turn_starts = events
        .iter()
        .filter(|e| matches!(e, AgentEvent::TurnStart))
        .count();
    let tool_starts = events
        .iter()
        .filter(|e| matches!(e, AgentEvent::ToolExecutionStart { .. }))
        .count();
    let tool_ends = events
        .iter()
        .filter(|e| matches!(e, AgentEvent::ToolExecutionEnd { .. }))
        .count();

    assert_eq!(turn_starts, 2, "tool use should trigger a second turn");
    assert_eq!(tool_starts, 1);
    assert_eq!(tool_ends, 1);
    assert_eq!(
        result.len(),
        4,
        "prompt + assistant(toolcall) + toolResult + assistant"
    );
}

#[tokio::test]
async fn agent_loop_continue_reuses_existing_context_messages() {
    let observed_message_count = Arc::new(AtomicUsize::new(0));
    let observed_message_count_in_stream = observed_message_count.clone();

    let stream_fn = Arc::new(
        move |_model: Model, context: Context, _options: Option<pi_ai::SimpleStreamOptions>| {
            observed_message_count_in_stream.store(context.messages.len(), Ordering::SeqCst);
            let message = assistant_message(
                vec![AssistantContentBlock::Text {
                    text: "continued".to_string(),
                    text_signature: None,
                }],
                StopReason::Stop,
                1_700_000_000_010,
            );
            Ok(done_stream(message, DoneReason::Stop))
        },
    );

    let context = AgentContext {
        system_prompt: "You are helpful".to_string(),
        messages: vec![user_message("history", 1_700_000_000_000)],
        tools: vec![],
    };
    let config = AgentLoopConfig {
        model: sample_model("test-api"),
        fallback_models: vec![],
        convert_to_llm: Arc::new(|messages| messages),
        stream_fn,
        retry: AgentRetryConfig::default(),
        get_steering_messages: None,
        get_follow_up_messages: None,
    };

    let stream = agent_loop_continue(context, config, None);
    let (_events, result) = collect_events_and_result(stream).await;

    assert_eq!(observed_message_count.load(Ordering::SeqCst), 1);
    assert_eq!(result.len(), 1, "continue should only return new messages");
}

#[tokio::test]
async fn agent_loop_skips_remaining_tool_calls_when_steering_arrives() {
    let stream_fn_calls = Arc::new(AtomicUsize::new(0));
    let stream_fn_calls_in_stream = stream_fn_calls.clone();

    let stream_fn = Arc::new(
        move |_model: Model, _context: Context, _options: Option<pi_ai::SimpleStreamOptions>| {
            let index = stream_fn_calls_in_stream.fetch_add(1, Ordering::SeqCst);
            if index == 0 {
                let tool_call_msg = assistant_message(
                    vec![
                        AssistantContentBlock::ToolCall {
                            id: "call_1".to_string(),
                            name: "first_tool".to_string(),
                            arguments: json!({"value": 1}),
                            thought_signature: None,
                        },
                        AssistantContentBlock::ToolCall {
                            id: "call_2".to_string(),
                            name: "second_tool".to_string(),
                            arguments: json!({"value": 2}),
                            thought_signature: None,
                        },
                    ],
                    StopReason::ToolUse,
                    1_700_000_000_020,
                );
                Ok(done_stream(tool_call_msg, DoneReason::ToolUse))
            } else {
                let final_msg = assistant_message(
                    vec![AssistantContentBlock::Text {
                        text: "done after steering".to_string(),
                        text_signature: None,
                    }],
                    StopReason::Stop,
                    1_700_000_000_030,
                );
                Ok(done_stream(final_msg, DoneReason::Stop))
            }
        },
    );

    let make_tool =
        |name: &'static str| AgentTool {
            name: name.to_string(),
            label: name.to_string(),
            description: format!("Execute tool {name}"),
            parameters: json!({
                "type": "object",
                "properties": {
                    "value": { "type": "integer" }
                },
                "required": ["value"],
                "additionalProperties": false
            }),
            execute:
                Arc::new(
                    move |_tool_call_id: String,
                          _args: Value|
                          -> Pin<
                        Box<dyn Future<Output = Result<AgentToolResult, String>> + Send>,
                    > {
                        let tool_name = name.to_string();
                        Box::pin(async move {
                            Ok(AgentToolResult {
                                content: vec![ToolResultContentBlock::Text {
                                    text: format!("{tool_name}-ok"),
                                    text_signature: None,
                                }],
                                details: json!({}),
                            })
                        })
                    },
                ),
        };

    let steering_polls = Arc::new(AtomicUsize::new(0));
    let steering_polls_in_callback = steering_polls.clone();
    let get_steering_messages = Arc::new(move || {
        let poll_index = steering_polls_in_callback.fetch_add(1, Ordering::SeqCst);
        if poll_index == 1 {
            vec![user_message("interrupt", 1_700_000_000_025)]
        } else {
            vec![]
        }
    });

    let prompts = vec![user_message("run tools", 1_700_000_000_000)];
    let context = AgentContext {
        system_prompt: "You are helpful".to_string(),
        messages: vec![],
        tools: vec![make_tool("first_tool"), make_tool("second_tool")],
    };

    let config = AgentLoopConfig {
        model: sample_model("test-api"),
        fallback_models: vec![],
        convert_to_llm: Arc::new(|messages| messages),
        stream_fn,
        retry: AgentRetryConfig::default(),
        get_steering_messages: Some(get_steering_messages),
        get_follow_up_messages: None,
    };

    let stream = agent_loop(prompts, context, config, None);
    let (_events, result) = collect_events_and_result(stream).await;

    let mut tool_results = result.iter().filter_map(|message| match message {
        Message::ToolResult {
            is_error, content, ..
        } => Some((*is_error, content.clone())),
        _ => None,
    });

    let first = tool_results.next().expect("first tool result should exist");
    assert!(!first.0, "first tool call should execute normally");

    let second = tool_results
        .next()
        .expect("second tool result should exist");
    assert!(
        second.0,
        "second tool call should be skipped as error result"
    );
    assert!(
        matches!(
            second.1.first(),
            Some(ToolResultContentBlock::Text { text, .. }) if text == "Skipped due to queued user message."
        ),
        "skipped tool call should contain steering skip reason"
    );

    assert!(
        result.iter().any(|message| {
            matches!(
                message,
                Message::User {
                    content: UserContent::Text(text),
                    ..
                } if text == "interrupt"
            )
        }),
        "steering message should be injected into the context"
    );
}

#[tokio::test]
async fn agent_loop_processes_follow_up_messages_after_turn_completion() {
    let stream_fn_calls = Arc::new(AtomicUsize::new(0));
    let stream_fn_calls_in_stream = stream_fn_calls.clone();

    let stream_fn = Arc::new(
        move |_model: Model, _context: Context, _options: Option<pi_ai::SimpleStreamOptions>| {
            let index = stream_fn_calls_in_stream.fetch_add(1, Ordering::SeqCst);
            let text = if index == 0 { "first" } else { "second" };
            let message = assistant_message(
                vec![AssistantContentBlock::Text {
                    text: text.to_string(),
                    text_signature: None,
                }],
                StopReason::Stop,
                1_700_000_000_010 + index as i64,
            );
            Ok(done_stream(message, DoneReason::Stop))
        },
    );

    let follow_up_polls = Arc::new(AtomicUsize::new(0));
    let follow_up_polls_in_callback = follow_up_polls.clone();
    let get_follow_up_messages = Arc::new(move || {
        if follow_up_polls_in_callback.fetch_add(1, Ordering::SeqCst) == 0 {
            vec![user_message("follow-up", 1_700_000_000_020)]
        } else {
            vec![]
        }
    });

    let prompts = vec![user_message("hello", 1_700_000_000_000)];
    let context = AgentContext {
        system_prompt: "You are helpful".to_string(),
        messages: vec![],
        tools: vec![],
    };

    let config = AgentLoopConfig {
        model: sample_model("test-api"),
        fallback_models: vec![],
        convert_to_llm: Arc::new(|messages| messages),
        stream_fn,
        retry: AgentRetryConfig::default(),
        get_steering_messages: None,
        get_follow_up_messages: Some(get_follow_up_messages),
    };

    let stream = agent_loop(prompts, context, config, None);
    let (events, result) = collect_events_and_result(stream).await;

    let turn_starts = events
        .iter()
        .filter(|event| matches!(event, AgentEvent::TurnStart))
        .count();
    assert_eq!(
        turn_starts, 2,
        "follow-up message should trigger a new turn"
    );

    assert!(
        result.iter().any(|message| {
            matches!(
                message,
                Message::User {
                    content: UserContent::Text(text),
                    ..
                } if text == "follow-up"
            )
        }),
        "follow-up message should be part of produced messages"
    );
}

#[tokio::test]
async fn agent_loop_abort_signal_interrupts_assistant_streaming() {
    let stream_fn = Arc::new(
        |_model: Model, _context: Context, _options: Option<pi_ai::SimpleStreamOptions>| {
            let stream = AssistantMessageEventStream::new();
            let partial = assistant_message(
                vec![AssistantContentBlock::Text {
                    text: "partial".to_string(),
                    text_signature: None,
                }],
                StopReason::Stop,
                1_700_000_000_010,
            );
            stream.push(AssistantMessageEvent::Start { partial });
            Ok(stream)
        },
    );

    let prompts = vec![user_message("hello", 1_700_000_000_000)];
    let context = AgentContext {
        system_prompt: "You are helpful".to_string(),
        messages: vec![],
        tools: vec![],
    };
    let config = AgentLoopConfig {
        model: sample_model("test-api"),
        fallback_models: vec![],
        convert_to_llm: Arc::new(|messages| messages),
        stream_fn,
        retry: AgentRetryConfig::default(),
        get_steering_messages: None,
        get_follow_up_messages: None,
    };

    let controller = AgentAbortController::new();
    let signal = controller.signal();
    let stream = agent_loop(prompts, context, config, Some(signal));

    tokio::spawn(async move {
        sleep(Duration::from_millis(20)).await;
        controller.abort();
    });

    let (_events, result) = collect_events_and_result(stream).await;
    let assistant = result
        .iter()
        .find_map(|message| match message {
            Message::Assistant { stop_reason, .. } => Some(stop_reason),
            _ => None,
        })
        .expect("assistant message should exist");

    assert_eq!(*assistant, StopReason::Aborted);
}

#[tokio::test]
async fn agent_loop_abort_signal_interrupts_tool_execution() {
    let stream_fn = Arc::new(
        |_model: Model, _context: Context, _options: Option<pi_ai::SimpleStreamOptions>| {
            let message = assistant_message(
                vec![AssistantContentBlock::ToolCall {
                    id: "call_1".to_string(),
                    name: "long_tool".to_string(),
                    arguments: json!({"value": 1}),
                    thought_signature: None,
                }],
                StopReason::ToolUse,
                1_700_000_000_010,
            );
            Ok(done_stream(message, DoneReason::ToolUse))
        },
    );

    let tool = AgentTool {
        name: "long_tool".to_string(),
        label: "Long Tool".to_string(),
        description: "Long running tool".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "value": { "type": "integer" }
            },
            "required": ["value"],
            "additionalProperties": false
        }),
        execute: Arc::new(
            |_tool_call_id: String,
             _args: Value|
             -> Pin<Box<dyn Future<Output = Result<AgentToolResult, String>> + Send>> {
                Box::pin(async move {
                    sleep(Duration::from_millis(250)).await;
                    Ok(AgentToolResult {
                        content: vec![ToolResultContentBlock::Text {
                            text: "done".to_string(),
                            text_signature: None,
                        }],
                        details: json!({}),
                    })
                })
            },
        ),
    };

    let prompts = vec![user_message("run", 1_700_000_000_000)];
    let context = AgentContext {
        system_prompt: "You are helpful".to_string(),
        messages: vec![],
        tools: vec![tool],
    };
    let config = AgentLoopConfig {
        model: sample_model("test-api"),
        fallback_models: vec![],
        convert_to_llm: Arc::new(|messages| messages),
        stream_fn,
        retry: AgentRetryConfig::default(),
        get_steering_messages: None,
        get_follow_up_messages: None,
    };

    let controller = AgentAbortController::new();
    let signal = controller.signal();
    let stream = agent_loop(prompts, context, config, Some(signal));

    tokio::spawn(async move {
        sleep(Duration::from_millis(20)).await;
        controller.abort();
    });

    let (_events, result) = collect_events_and_result(stream).await;
    let tool_result = result
        .iter()
        .find_map(|message| match message {
            Message::ToolResult {
                content, is_error, ..
            } => Some((content.clone(), *is_error)),
            _ => None,
        })
        .expect("tool result should exist");

    assert!(
        tool_result.1,
        "aborted tool execution should be marked as error"
    );
    assert!(
        matches!(
            tool_result.0.first(),
            Some(ToolResultContentBlock::Text { text, .. }) if text == "Tool execution aborted"
        ),
        "tool result should contain abort reason"
    );
}

#[tokio::test]
async fn agent_loop_retries_stream_creation_errors_with_backoff() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let attempts_in_fn = attempts.clone();
    let stream_fn = Arc::new(
        move |_model: Model, _context: Context, _options: Option<pi_ai::SimpleStreamOptions>| {
            let attempt = attempts_in_fn.fetch_add(1, Ordering::SeqCst);
            if attempt == 0 {
                return Err("temporary upstream failure".to_string());
            }
            let message = assistant_message(
                vec![AssistantContentBlock::Text {
                    text: "retry success".to_string(),
                    text_signature: None,
                }],
                StopReason::Stop,
                1_700_000_000_010 + attempt as i64,
            );
            Ok(done_stream(message, DoneReason::Stop))
        },
    );

    let config = AgentLoopConfig {
        model: sample_model("test-api"),
        fallback_models: vec![],
        convert_to_llm: Arc::new(|messages| messages),
        stream_fn,
        retry: AgentRetryConfig {
            max_attempts: 2,
            initial_backoff_ms: 1,
            max_backoff_ms: 1,
        },
        get_steering_messages: None,
        get_follow_up_messages: None,
    };

    let prompts = vec![user_message("hello", 1_700_000_000_000)];
    let context = AgentContext {
        system_prompt: "You are helpful".to_string(),
        messages: vec![],
        tools: vec![],
    };
    let stream = agent_loop(prompts, context, config, None);
    let (events, result) = collect_events_and_result(stream).await;

    assert_eq!(attempts.load(Ordering::SeqCst), 2, "should retry once");
    assert!(
        events.iter().any(|event| {
            matches!(
                event,
                AgentEvent::RetryScheduled {
                    attempt: 1,
                    max_attempts: 2,
                    ..
                }
            )
        }),
        "retry scheduling event should be emitted"
    );
    assert!(
        result.iter().any(|message| {
            matches!(
                message,
                Message::Assistant { content, .. }
                    if content.iter().any(|block| matches!(
                        block,
                        AssistantContentBlock::Text { text, .. } if text == "retry success"
                    ))
            )
        }),
        "final assistant response should come from retry attempt"
    );
}

#[tokio::test]
async fn agent_loop_retries_with_model_fallback() {
    let attempted_models = Arc::new(Mutex::new(Vec::<String>::new()));
    let attempted_models_in_fn = attempted_models.clone();
    let stream_fn = Arc::new(
        move |model: Model, _context: Context, _options: Option<pi_ai::SimpleStreamOptions>| {
            attempted_models_in_fn
                .lock()
                .expect("attempted models lock")
                .push(model.id.clone());
            if model.id == "primary" {
                return Err("primary model unavailable".to_string());
            }

            let mut message = assistant_message(
                vec![AssistantContentBlock::Text {
                    text: "fallback success".to_string(),
                    text_signature: None,
                }],
                StopReason::Stop,
                1_700_000_000_020,
            );
            message.model = model.id;
            message.provider = model.provider;
            message.api = model.api;
            Ok(done_stream(message, DoneReason::Stop))
        },
    );

    let mut primary = sample_model("test-api");
    primary.id = "primary".to_string();
    primary.name = "Primary".to_string();
    let mut fallback = sample_model("test-api");
    fallback.id = "fallback".to_string();
    fallback.name = "Fallback".to_string();

    let config = AgentLoopConfig {
        model: primary,
        fallback_models: vec![fallback],
        convert_to_llm: Arc::new(|messages| messages),
        stream_fn,
        retry: AgentRetryConfig {
            max_attempts: 2,
            initial_backoff_ms: 0,
            max_backoff_ms: 0,
        },
        get_steering_messages: None,
        get_follow_up_messages: None,
    };

    let prompts = vec![user_message("hello", 1_700_000_000_000)];
    let context = AgentContext {
        system_prompt: "You are helpful".to_string(),
        messages: vec![],
        tools: vec![],
    };
    let stream = agent_loop(prompts, context, config, None);
    let (events, result) = collect_events_and_result(stream).await;

    let attempted_models = attempted_models
        .lock()
        .expect("attempted models lock")
        .clone();
    assert_eq!(attempted_models, vec!["primary", "fallback"]);
    assert!(
        events.iter().any(|event| {
            matches!(
                event,
                AgentEvent::ModelFallback {
                    from_model,
                    to_model,
                    ..
                } if from_model == "primary" && to_model == "fallback"
            )
        }),
        "fallback event should be emitted"
    );
    assert!(
        result.iter().any(|message| {
            matches!(
                message,
                Message::Assistant { model, .. } if model == "fallback"
            )
        }),
        "assistant response should be produced by fallback model"
    );
}

#[tokio::test]
async fn agent_loop_emits_metrics_with_retry_and_tool_duration() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let attempts_in_fn = attempts.clone();
    let stream_fn = Arc::new(
        move |_model: Model, _context: Context, _options: Option<pi_ai::SimpleStreamOptions>| {
            let attempt = attempts_in_fn.fetch_add(1, Ordering::SeqCst);
            match attempt {
                0 => Err("temporary upstream failure".to_string()),
                1 => {
                    let tool_call_msg = assistant_message(
                        vec![AssistantContentBlock::ToolCall {
                            id: "call_1".to_string(),
                            name: "measure_tool".to_string(),
                            arguments: json!({}),
                            thought_signature: None,
                        }],
                        StopReason::ToolUse,
                        1_700_000_000_020,
                    );
                    Ok(done_stream(tool_call_msg, DoneReason::ToolUse))
                }
                _ => {
                    let final_msg = assistant_message(
                        vec![AssistantContentBlock::Text {
                            text: "done".to_string(),
                            text_signature: None,
                        }],
                        StopReason::Stop,
                        1_700_000_000_030,
                    );
                    Ok(done_stream(final_msg, DoneReason::Stop))
                }
            }
        },
    );

    let tool = AgentTool {
        name: "measure_tool".to_string(),
        label: "Measure Tool".to_string(),
        description: "Sleep a little and return".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false
        }),
        execute: Arc::new(
            |_tool_call_id: String,
             _args: Value|
             -> Pin<Box<dyn Future<Output = Result<AgentToolResult, String>> + Send>> {
                Box::pin(async move {
                    sleep(Duration::from_millis(5)).await;
                    Ok(AgentToolResult {
                        content: vec![ToolResultContentBlock::Text {
                            text: "ok".to_string(),
                            text_signature: None,
                        }],
                        details: json!({}),
                    })
                })
            },
        ),
    };

    let config = AgentLoopConfig {
        model: sample_model("test-api"),
        fallback_models: vec![],
        convert_to_llm: Arc::new(|messages| messages),
        stream_fn,
        retry: AgentRetryConfig {
            max_attempts: 2,
            initial_backoff_ms: 0,
            max_backoff_ms: 0,
        },
        get_steering_messages: None,
        get_follow_up_messages: None,
    };
    let prompts = vec![user_message("hello", 1_700_000_000_000)];
    let context = AgentContext {
        system_prompt: "You are helpful".to_string(),
        messages: vec![],
        tools: vec![tool],
    };

    let stream = agent_loop(prompts, context, config, None);
    let (events, _result) = collect_events_and_result(stream).await;

    let tool_duration_ms = events
        .iter()
        .find_map(|event| match event {
            AgentEvent::ToolExecutionEnd { duration_ms, .. } => Some(*duration_ms),
            _ => None,
        })
        .expect("tool execution end should be emitted");
    assert!(
        tool_duration_ms > 0,
        "tool duration should be recorded in milliseconds"
    );

    let metrics = events
        .iter()
        .find_map(|event| match event {
            AgentEvent::Metrics { metrics } => Some(metrics.clone()),
            _ => None,
        })
        .expect("metrics event should be emitted");
    assert_eq!(metrics.assistant_request_count, 2);
    assert_eq!(metrics.tool_execution_count, 1);
    assert_eq!(metrics.retry_count, 1);
    assert!(
        metrics.tool_execution_total_ms >= tool_duration_ms,
        "aggregate tool duration should include tool end duration"
    );
}
