use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use pixy_agent_core::ParentChildRunEvent;
use pixy_ai::{
    AssistantContentBlock, AssistantMessage, AssistantMessageEvent, AssistantMessageEventStream,
    Context, Cost, DoneReason, Message, Model, StopReason, Usage,
};
use pixy_coding_agent::{
    create_task_tool, AgentSession, AgentSessionConfig, ChildSessionStore, DefaultSubAgentRegistry,
    DispatchPolicyConfig, DispatchPolicyRule, MultiAgentPluginRuntime, PolicyRuleEffect,
    SessionManager, SubAgentMode, SubAgentResolver, SubAgentSpec, TaskDispatcher,
    TaskDispatcherConfig,
};
use serde_json::json;
use tempfile::tempdir;
use tokio::sync::Mutex;

fn sample_model() -> Model {
    Model {
        id: "test-model".to_string(),
        name: "Test Model".to_string(),
        api: "openai-responses".to_string(),
        provider: "openai".to_string(),
        base_url: "http://localhost".to_string(),
        reasoning: false,
        reasoning_effort: None,
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

fn sample_usage() -> Usage {
    Usage {
        input: 1,
        output: 1,
        cache_read: 0,
        cache_write: 0,
        total_tokens: 2,
        cost: Cost {
            input: 0.0,
            output: 0.0,
            cache_read: 0.0,
            cache_write: 0.0,
            total: 0.0,
        },
    }
}

fn assistant_message(
    content: Vec<AssistantContentBlock>,
    stop_reason: StopReason,
) -> AssistantMessage {
    AssistantMessage {
        role: "assistant".to_string(),
        content,
        api: "openai-responses".to_string(),
        provider: "openai".to_string(),
        model: "test-model".to_string(),
        usage: sample_usage(),
        stop_reason,
        error_message: None,
        timestamp: 1,
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

fn has_tool_result_after_latest_user(context: &Context) -> bool {
    for message in context.messages.iter().rev() {
        match message {
            Message::User { .. } => return false,
            Message::ToolResult { .. } => return true,
            Message::Assistant { .. } => {}
        }
    }
    false
}

fn registry() -> Arc<dyn SubAgentResolver> {
    let built = DefaultSubAgentRegistry::builder()
        .register_builtin(SubAgentSpec {
            name: "general".to_string(),
            description: "General helper".to_string(),
            mode: SubAgentMode::SubAgent,
        })
        .expect("register general")
        .build();
    Arc::new(built)
}

#[tokio::test]
async fn parent_tool_call_roundtrips_into_child_and_back() {
    let dir = tempdir().expect("tempdir");
    let stream_calls = Arc::new(AtomicUsize::new(0));
    let stream_calls_clone = stream_calls.clone();

    let stream_fn = Arc::new(
        move |_model: Model, context: Context, _options: Option<pixy_ai::SimpleStreamOptions>| {
            let _ = stream_calls_clone.fetch_add(1, Ordering::SeqCst);
            let is_child = context
                .system_prompt
                .as_deref()
                .unwrap_or_default()
                .contains("<subagent_context>");
            if is_child {
                let message = assistant_message(
                    vec![AssistantContentBlock::Text {
                        text: "child finished delegated task".to_string(),
                        text_signature: None,
                    }],
                    StopReason::Stop,
                );
                return Ok(done_stream(message, DoneReason::Stop));
            }

            if has_tool_result_after_latest_user(&context) {
                let message = assistant_message(
                    vec![AssistantContentBlock::Text {
                        text: "parent received task result".to_string(),
                        text_signature: None,
                    }],
                    StopReason::Stop,
                );
                return Ok(done_stream(message, DoneReason::Stop));
            }

            let message = assistant_message(
                vec![AssistantContentBlock::ToolCall {
                    id: "task-call-1".to_string(),
                    name: "task".to_string(),
                    arguments: json!({
                        "subagent_type": "general",
                        "prompt": "delegate this",
                        "task_id": "task-123"
                    }),
                    thought_signature: None,
                }],
                StopReason::ToolUse,
            );
            Ok(done_stream(message, DoneReason::ToolUse))
        },
    );

    let store = Arc::new(Mutex::new(ChildSessionStore::new("parent-session")));
    let dispatcher = Arc::new(TaskDispatcher::new(TaskDispatcherConfig {
        cwd: dir.path().to_path_buf(),
        parent_session_id: "parent-session".to_string(),
        parent_session_dir: dir.path().to_path_buf(),
        model: sample_model(),
        system_prompt: "You are parent".to_string(),
        stream_fn: stream_fn.clone(),
        child_tools: vec![],
        subagent_registry: registry(),
        session_store: store.clone(),
        dispatch_policy: DispatchPolicyConfig::default(),
        plugin_runtime: Arc::new(MultiAgentPluginRuntime::default()),
        lifecycle_event_sink: None,
    }));
    let task_tool = create_task_tool(dispatcher);

    let mut session = AgentSession::new(
        SessionManager::create(
            dir.path().to_str().expect("utf-8 cwd"),
            dir.path().join("sessions"),
        )
        .expect("create session"),
        AgentSessionConfig {
            model: sample_model(),
            system_prompt: "You are parent".to_string(),
            stream_fn,
            tools: vec![task_tool],
        },
    );

    let produced = session
        .prompt("run delegated flow")
        .await
        .expect("prompt succeeds");
    assert!(produced.iter().any(|message| {
        matches!(
            message,
            Message::ToolResult {
                tool_name,
                is_error,
                ..
            } if tool_name == "task" && !is_error
        )
    }));
    assert!(produced.iter().any(|message| {
        matches!(
            message,
            Message::Assistant { content, .. }
            if content.iter().any(|block| matches!(
                block,
                AssistantContentBlock::Text { text, .. } if text.contains("parent received task result")
            ))
        )
    }));
}

#[tokio::test]
async fn repeated_task_id_reuses_child_session_history() {
    let dir = tempdir().expect("tempdir");
    let stream_fn = Arc::new(
        move |_model: Model, context: Context, _options: Option<pixy_ai::SimpleStreamOptions>| {
            let is_child = context
                .system_prompt
                .as_deref()
                .unwrap_or_default()
                .contains("<subagent_context>");
            if is_child {
                let message = assistant_message(
                    vec![AssistantContentBlock::Text {
                        text: "child turn complete".to_string(),
                        text_signature: None,
                    }],
                    StopReason::Stop,
                );
                return Ok(done_stream(message, DoneReason::Stop));
            }

            if has_tool_result_after_latest_user(&context) {
                let message = assistant_message(
                    vec![AssistantContentBlock::Text {
                        text: "parent completed".to_string(),
                        text_signature: None,
                    }],
                    StopReason::Stop,
                );
                return Ok(done_stream(message, DoneReason::Stop));
            }

            let message = assistant_message(
                vec![AssistantContentBlock::ToolCall {
                    id: "task-call-1".to_string(),
                    name: "task".to_string(),
                    arguments: json!({
                        "subagent_type": "general",
                        "prompt": "delegate this",
                        "task_id": "task-123"
                    }),
                    thought_signature: None,
                }],
                StopReason::ToolUse,
            );
            Ok(done_stream(message, DoneReason::ToolUse))
        },
    );

    let store = Arc::new(Mutex::new(ChildSessionStore::new("parent-session")));
    let dispatcher = Arc::new(TaskDispatcher::new(TaskDispatcherConfig {
        cwd: dir.path().to_path_buf(),
        parent_session_id: "parent-session".to_string(),
        parent_session_dir: dir.path().to_path_buf(),
        model: sample_model(),
        system_prompt: "You are parent".to_string(),
        stream_fn: stream_fn.clone(),
        child_tools: vec![],
        subagent_registry: registry(),
        session_store: store.clone(),
        dispatch_policy: DispatchPolicyConfig::default(),
        plugin_runtime: Arc::new(MultiAgentPluginRuntime::default()),
        lifecycle_event_sink: None,
    }));
    let task_tool = create_task_tool(dispatcher);

    let mut session = AgentSession::new(
        SessionManager::create(
            dir.path().to_str().expect("utf-8 cwd"),
            dir.path().join("sessions"),
        )
        .expect("create session"),
        AgentSessionConfig {
            model: sample_model(),
            system_prompt: "You are parent".to_string(),
            stream_fn,
            tools: vec![task_tool],
        },
    );

    session.prompt("first run").await.expect("first run");
    session.prompt("second run").await.expect("second run");

    let child_path = store
        .lock()
        .await
        .resolve("task-123")
        .expect("task id should map to child session");
    let child = SessionManager::load(&child_path).expect("load child session");
    let child_context = child.build_session_context();
    assert!(
        child_context.messages.len() >= 4,
        "reused task id should accumulate multiple turns in the same child session"
    );
}

#[tokio::test]
async fn policy_fallback_routes_unknown_subagent_to_configured_default() {
    let dir = tempdir().expect("tempdir");
    let stream_fn = Arc::new(
        move |_model: Model, context: Context, _options: Option<pixy_ai::SimpleStreamOptions>| {
            let is_child = context
                .system_prompt
                .as_deref()
                .unwrap_or_default()
                .contains("<subagent_context>");
            if is_child {
                let message = assistant_message(
                    vec![AssistantContentBlock::Text {
                        text: "child turn complete".to_string(),
                        text_signature: None,
                    }],
                    StopReason::Stop,
                );
                return Ok(done_stream(message, DoneReason::Stop));
            }

            if has_tool_result_after_latest_user(&context) {
                let message = assistant_message(
                    vec![AssistantContentBlock::Text {
                        text: "parent completed".to_string(),
                        text_signature: None,
                    }],
                    StopReason::Stop,
                );
                return Ok(done_stream(message, DoneReason::Stop));
            }

            let message = assistant_message(
                vec![AssistantContentBlock::ToolCall {
                    id: "task-call-1".to_string(),
                    name: "task".to_string(),
                    arguments: json!({
                        "subagent_type": "unknown-subagent",
                        "prompt": "delegate this"
                    }),
                    thought_signature: None,
                }],
                StopReason::ToolUse,
            );
            Ok(done_stream(message, DoneReason::ToolUse))
        },
    );

    let store = Arc::new(Mutex::new(ChildSessionStore::new("parent-session")));
    let dispatcher = Arc::new(TaskDispatcher::new(TaskDispatcherConfig {
        cwd: dir.path().to_path_buf(),
        parent_session_id: "parent-session".to_string(),
        parent_session_dir: dir.path().to_path_buf(),
        model: sample_model(),
        system_prompt: "You are parent".to_string(),
        stream_fn: stream_fn.clone(),
        child_tools: vec![],
        subagent_registry: registry(),
        session_store: store,
        dispatch_policy: DispatchPolicyConfig {
            fallback_subagent: Some("general".to_string()),
            rules: vec![],
        },
        plugin_runtime: Arc::new(MultiAgentPluginRuntime::default()),
        lifecycle_event_sink: None,
    }));
    let task_tool = create_task_tool(dispatcher);

    let mut session = AgentSession::new(
        SessionManager::create(
            dir.path().to_str().expect("utf-8 cwd"),
            dir.path().join("sessions"),
        )
        .expect("create session"),
        AgentSessionConfig {
            model: sample_model(),
            system_prompt: "You are parent".to_string(),
            stream_fn,
            tools: vec![task_tool],
        },
    );

    let produced = session
        .prompt("fallback path")
        .await
        .expect("prompt succeeds");
    let tool_result = produced
        .iter()
        .find_map(|message| match message {
            Message::ToolResult {
                tool_name,
                details,
                is_error,
                ..
            } if tool_name == "task" && !is_error => details.clone(),
            _ => None,
        })
        .expect("task tool result should exist");
    assert_eq!(tool_result["resolved_subagent"], "general");
    assert_eq!(tool_result["routing_hint_applied"], true);
}

#[tokio::test]
async fn policy_block_returns_explicit_blocked_error_details() {
    let dir = tempdir().expect("tempdir");
    let stream_fn = Arc::new(
        move |_model: Model, context: Context, _options: Option<pixy_ai::SimpleStreamOptions>| {
            if has_tool_result_after_latest_user(&context) {
                let message = assistant_message(
                    vec![AssistantContentBlock::Text {
                        text: "parent done".to_string(),
                        text_signature: None,
                    }],
                    StopReason::Stop,
                );
                return Ok(done_stream(message, DoneReason::Stop));
            }

            let message = assistant_message(
                vec![AssistantContentBlock::ToolCall {
                    id: "task-call-1".to_string(),
                    name: "task".to_string(),
                    arguments: json!({
                        "subagent_type": "general",
                        "prompt": "delegate this"
                    }),
                    thought_signature: None,
                }],
                StopReason::ToolUse,
            );
            Ok(done_stream(message, DoneReason::ToolUse))
        },
    );

    let dispatcher = Arc::new(TaskDispatcher::new(TaskDispatcherConfig {
        cwd: dir.path().to_path_buf(),
        parent_session_id: "parent-session".to_string(),
        parent_session_dir: dir.path().to_path_buf(),
        model: sample_model(),
        system_prompt: "You are parent".to_string(),
        stream_fn: stream_fn.clone(),
        child_tools: vec![],
        subagent_registry: registry(),
        session_store: Arc::new(Mutex::new(ChildSessionStore::new("parent-session"))),
        dispatch_policy: DispatchPolicyConfig {
            fallback_subagent: None,
            rules: vec![DispatchPolicyRule {
                subagent: "general".to_string(),
                tool: "task".to_string(),
                effect: PolicyRuleEffect::Deny,
            }],
        },
        plugin_runtime: Arc::new(MultiAgentPluginRuntime::default()),
        lifecycle_event_sink: None,
    }));
    let task_tool = create_task_tool(dispatcher);

    let mut session = AgentSession::new(
        SessionManager::create(
            dir.path().to_str().expect("utf-8 cwd"),
            dir.path().join("sessions"),
        )
        .expect("create session"),
        AgentSessionConfig {
            model: sample_model(),
            system_prompt: "You are parent".to_string(),
            stream_fn,
            tools: vec![task_tool],
        },
    );

    let produced = session
        .prompt("policy block")
        .await
        .expect("prompt succeeds");
    let tool_error_details = produced
        .iter()
        .find_map(|message| match message {
            Message::ToolResult {
                tool_name,
                details,
                is_error,
                ..
            } if tool_name == "task" && *is_error => details.clone(),
            _ => None,
        })
        .expect("blocked task tool error should exist");
    assert_eq!(
        tool_error_details["error"]["details"]["kind"],
        "task_dispatch_blocked"
    );
}

#[tokio::test]
async fn lifecycle_events_emit_child_run_start_and_end_with_task_id_correlation() {
    let dir = tempdir().expect("tempdir");
    let events = Arc::new(std::sync::Mutex::new(Vec::<ParentChildRunEvent>::new()));
    let events_for_sink = events.clone();
    let stream_fn = Arc::new(
        move |_model: Model, context: Context, _options: Option<pixy_ai::SimpleStreamOptions>| {
            let is_child = context
                .system_prompt
                .as_deref()
                .unwrap_or_default()
                .contains("<subagent_context>");
            if is_child {
                let message = assistant_message(
                    vec![AssistantContentBlock::Text {
                        text: "child done".to_string(),
                        text_signature: None,
                    }],
                    StopReason::Stop,
                );
                return Ok(done_stream(message, DoneReason::Stop));
            }
            if has_tool_result_after_latest_user(&context) {
                let message = assistant_message(
                    vec![AssistantContentBlock::Text {
                        text: "parent done".to_string(),
                        text_signature: None,
                    }],
                    StopReason::Stop,
                );
                return Ok(done_stream(message, DoneReason::Stop));
            }
            let message = assistant_message(
                vec![AssistantContentBlock::ToolCall {
                    id: "task-call-event-1".to_string(),
                    name: "task".to_string(),
                    arguments: json!({
                        "subagent_type": "general",
                        "prompt": "delegate this",
                        "task_id": "task-event-1"
                    }),
                    thought_signature: None,
                }],
                StopReason::ToolUse,
            );
            Ok(done_stream(message, DoneReason::ToolUse))
        },
    );
    let dispatcher = Arc::new(TaskDispatcher::new(TaskDispatcherConfig {
        cwd: dir.path().to_path_buf(),
        parent_session_id: "parent-session".to_string(),
        parent_session_dir: dir.path().to_path_buf(),
        model: sample_model(),
        system_prompt: "You are parent".to_string(),
        stream_fn: stream_fn.clone(),
        child_tools: vec![],
        subagent_registry: registry(),
        session_store: Arc::new(Mutex::new(ChildSessionStore::new("parent-session"))),
        dispatch_policy: DispatchPolicyConfig::default(),
        plugin_runtime: Arc::new(MultiAgentPluginRuntime::default()),
        lifecycle_event_sink: Some(Arc::new(move |event| {
            events_for_sink.lock().expect("lock events").push(event);
        })),
    }));
    let task_tool = create_task_tool(dispatcher);

    let mut session = AgentSession::new(
        SessionManager::create(
            dir.path().to_str().expect("utf-8 cwd"),
            dir.path().join("sessions"),
        )
        .expect("create session"),
        AgentSessionConfig {
            model: sample_model(),
            system_prompt: "You are parent".to_string(),
            stream_fn,
            tools: vec![task_tool],
        },
    );

    session
        .prompt("trigger child run")
        .await
        .expect("prompt succeeds");

    let events = events.lock().expect("lock events");
    assert!(
        events
            .iter()
            .any(|event| matches!(event, ParentChildRunEvent::ChildRunStart { task_id, .. } if task_id == "task-event-1"))
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, ParentChildRunEvent::ChildRunEnd { task_id, .. } if task_id == "task-event-1"))
    );
}

#[tokio::test]
async fn lifecycle_events_emit_child_run_error_with_task_id_correlation() {
    let dir = tempdir().expect("tempdir");
    let events = Arc::new(std::sync::Mutex::new(Vec::<ParentChildRunEvent>::new()));
    let events_for_sink = events.clone();
    let stream_fn = Arc::new(
        move |_model: Model, context: Context, _options: Option<pixy_ai::SimpleStreamOptions>| {
            let is_child = context
                .system_prompt
                .as_deref()
                .unwrap_or_default()
                .contains("<subagent_context>");
            if is_child {
                return Err(pixy_ai::PiAiError::new(
                    pixy_ai::PiAiErrorCode::ProviderTransport,
                    "simulated child failure",
                ));
            }
            if has_tool_result_after_latest_user(&context) {
                let message = assistant_message(
                    vec![AssistantContentBlock::Text {
                        text: "parent done".to_string(),
                        text_signature: None,
                    }],
                    StopReason::Stop,
                );
                return Ok(done_stream(message, DoneReason::Stop));
            }
            let message = assistant_message(
                vec![AssistantContentBlock::ToolCall {
                    id: "task-call-event-2".to_string(),
                    name: "task".to_string(),
                    arguments: json!({
                        "subagent_type": "general",
                        "prompt": "delegate this",
                        "task_id": "task-event-2"
                    }),
                    thought_signature: None,
                }],
                StopReason::ToolUse,
            );
            Ok(done_stream(message, DoneReason::ToolUse))
        },
    );
    let dispatcher = Arc::new(TaskDispatcher::new(TaskDispatcherConfig {
        cwd: dir.path().to_path_buf(),
        parent_session_id: "parent-session".to_string(),
        parent_session_dir: dir.path().to_path_buf(),
        model: sample_model(),
        system_prompt: "You are parent".to_string(),
        stream_fn: stream_fn.clone(),
        child_tools: vec![],
        subagent_registry: registry(),
        session_store: Arc::new(Mutex::new(ChildSessionStore::new("parent-session"))),
        dispatch_policy: DispatchPolicyConfig::default(),
        plugin_runtime: Arc::new(MultiAgentPluginRuntime::default()),
        lifecycle_event_sink: Some(Arc::new(move |event| {
            events_for_sink.lock().expect("lock events").push(event);
        })),
    }));
    let task_tool = create_task_tool(dispatcher);

    let mut session = AgentSession::new(
        SessionManager::create(
            dir.path().to_str().expect("utf-8 cwd"),
            dir.path().join("sessions"),
        )
        .expect("create session"),
        AgentSessionConfig {
            model: sample_model(),
            system_prompt: "You are parent".to_string(),
            stream_fn,
            tools: vec![task_tool],
        },
    );

    session
        .prompt("trigger child failure")
        .await
        .expect("prompt succeeds");

    let events = events.lock().expect("lock events");
    assert!(
        events
            .iter()
            .any(|event| matches!(event, ParentChildRunEvent::ChildRunStart { task_id, .. } if task_id == "task-event-2"))
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, ParentChildRunEvent::ChildRunError { task_id, .. } if task_id == "task-event-2"))
    );
}

#[tokio::test]
async fn single_agent_mode_keeps_existing_flow_without_task_tool() {
    let dir = tempdir().expect("tempdir");

    let stream_fn = Arc::new(
        move |_model: Model, _context: Context, _options: Option<pixy_ai::SimpleStreamOptions>| {
            let message = assistant_message(
                vec![AssistantContentBlock::Text {
                    text: "plain response".to_string(),
                    text_signature: None,
                }],
                StopReason::Stop,
            );
            Ok(done_stream(message, DoneReason::Stop))
        },
    );

    let mut session = AgentSession::new(
        SessionManager::create(
            dir.path().to_str().expect("utf-8 cwd"),
            dir.path().join("sessions"),
        )
        .expect("create session"),
        AgentSessionConfig {
            model: sample_model(),
            system_prompt: "You are parent".to_string(),
            stream_fn,
            tools: vec![],
        },
    );

    let produced = session.prompt("hello").await.expect("prompt succeeds");
    assert!(!produced
        .iter()
        .any(|message| matches!(message, Message::ToolResult { .. })));
    assert!(produced.iter().any(|message| {
        matches!(
            message,
            Message::Assistant { content, .. }
            if content.iter().any(|block| matches!(
                block,
                AssistantContentBlock::Text { text, .. } if text == "plain response"
            ))
        )
    }));
}
