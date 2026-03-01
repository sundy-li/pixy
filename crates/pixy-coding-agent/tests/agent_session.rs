use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use pixy_ai::{
    AssistantContentBlock, AssistantMessage, AssistantMessageEvent, AssistantMessageEventStream,
    Context, Cost, DoneReason, ErrorReason, Message, Model, StopReason, ToolResultContentBlock,
    Usage,
};
use pixy_coding_agent::{
    create_coding_tools, AgentSession, AgentSessionConfig, AgentSessionStreamUpdate,
    AutoCompactionConfig, SessionManager, COMPACTION_SUMMARY_PREFIX,
};
use serde_json::json;
use tempfile::tempdir;

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

fn assistant_error_message(error_message: &str, ts: i64, input_tokens: u64) -> AssistantMessage {
    AssistantMessage {
        role: "assistant".to_string(),
        content: vec![AssistantContentBlock::Text {
            text: String::new(),
            text_signature: None,
        }],
        api: "test-api".to_string(),
        provider: "test".to_string(),
        model: "test-model".to_string(),
        usage: Usage {
            input: input_tokens,
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
        stop_reason: StopReason::Error,
        error_message: Some(error_message.to_string()),
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

fn error_stream(message: AssistantMessage) -> AssistantMessageEventStream {
    let stream = AssistantMessageEventStream::new();
    stream.push(AssistantMessageEvent::Start {
        partial: message.clone(),
    });
    stream.push(AssistantMessageEvent::Error {
        reason: ErrorReason::Error,
        error: message,
    });
    stream
}

fn text_delta_stream(deltas: &[&str], final_text: &str, ts: i64) -> AssistantMessageEventStream {
    let partial = assistant_message(
        vec![AssistantContentBlock::Text {
            text: String::new(),
            text_signature: None,
        }],
        StopReason::Stop,
        ts,
    );
    let final_message = assistant_message(
        vec![AssistantContentBlock::Text {
            text: final_text.to_string(),
            text_signature: None,
        }],
        StopReason::Stop,
        ts,
    );

    let stream = AssistantMessageEventStream::new();
    stream.push(AssistantMessageEvent::Start {
        partial: partial.clone(),
    });
    stream.push(AssistantMessageEvent::TextStart {
        content_index: 0,
        partial: partial.clone(),
    });
    for delta in deltas {
        stream.push(AssistantMessageEvent::TextDelta {
            content_index: 0,
            delta: (*delta).to_string(),
            partial: partial.clone(),
        });
    }
    stream.push(AssistantMessageEvent::Done {
        reason: DoneReason::Stop,
        message: final_message,
    });
    stream
}

fn text_delta_stream_with_final_thinking(
    deltas: &[&str],
    final_text: &str,
    final_thinking: &str,
    ts: i64,
) -> AssistantMessageEventStream {
    let partial = assistant_message(
        vec![AssistantContentBlock::Text {
            text: String::new(),
            text_signature: None,
        }],
        StopReason::Stop,
        ts,
    );
    let final_message = assistant_message(
        vec![
            AssistantContentBlock::Text {
                text: final_text.to_string(),
                text_signature: None,
            },
            AssistantContentBlock::Thinking {
                thinking: final_thinking.to_string(),
                thinking_signature: None,
            },
        ],
        StopReason::Stop,
        ts,
    );

    let stream = AssistantMessageEventStream::new();
    stream.push(AssistantMessageEvent::Start {
        partial: partial.clone(),
    });
    stream.push(AssistantMessageEvent::TextStart {
        content_index: 0,
        partial: partial.clone(),
    });
    for delta in deltas {
        stream.push(AssistantMessageEvent::TextDelta {
            content_index: 0,
            delta: (*delta).to_string(),
            partial: partial.clone(),
        });
    }
    stream.push(AssistantMessageEvent::Done {
        reason: DoneReason::Stop,
        message: final_message,
    });
    stream
}

fn thinking_delta_stream(
    deltas: &[&str],
    final_thinking: &str,
    ts: i64,
) -> AssistantMessageEventStream {
    let partial = assistant_message(
        vec![AssistantContentBlock::Thinking {
            thinking: String::new(),
            thinking_signature: None,
        }],
        StopReason::Stop,
        ts,
    );
    let final_message = assistant_message(
        vec![AssistantContentBlock::Thinking {
            thinking: final_thinking.to_string(),
            thinking_signature: None,
        }],
        StopReason::Stop,
        ts,
    );

    let stream = AssistantMessageEventStream::new();
    stream.push(AssistantMessageEvent::Start {
        partial: partial.clone(),
    });
    stream.push(AssistantMessageEvent::ThinkingStart {
        content_index: 0,
        partial: partial.clone(),
    });
    for delta in deltas {
        stream.push(AssistantMessageEvent::ThinkingDelta {
            content_index: 0,
            delta: (*delta).to_string(),
            partial: partial.clone(),
        });
    }
    stream.push(AssistantMessageEvent::Done {
        reason: DoneReason::Stop,
        message: final_message,
    });
    stream
}

fn thinking_snapshot_stream(
    snapshots: &[&str],
    final_thinking: &str,
    ts: i64,
) -> AssistantMessageEventStream {
    let start_partial = assistant_message(
        vec![AssistantContentBlock::Thinking {
            thinking: String::new(),
            thinking_signature: None,
        }],
        StopReason::Stop,
        ts,
    );
    let final_message = assistant_message(
        vec![AssistantContentBlock::Thinking {
            thinking: final_thinking.to_string(),
            thinking_signature: None,
        }],
        StopReason::Stop,
        ts,
    );

    let stream = AssistantMessageEventStream::new();
    stream.push(AssistantMessageEvent::Start {
        partial: start_partial.clone(),
    });
    stream.push(AssistantMessageEvent::ThinkingStart {
        content_index: 0,
        partial: start_partial,
    });

    for snapshot in snapshots {
        let partial = assistant_message(
            vec![AssistantContentBlock::Thinking {
                thinking: (*snapshot).to_string(),
                thinking_signature: None,
            }],
            StopReason::Stop,
            ts,
        );
        stream.push(AssistantMessageEvent::ThinkingDelta {
            content_index: 0,
            delta: (*snapshot).to_string(),
            partial,
        });
    }

    stream.push(AssistantMessageEvent::Done {
        reason: DoneReason::Stop,
        message: final_message,
    });
    stream
}

fn is_summary_request(context: &Context) -> bool {
    context.messages.iter().any(|message| {
        matches!(
            message,
            Message::User {
                content: pixy_ai::UserContent::Text(text),
                ..
            } if text.contains("<conversation>") && text.contains("Summarize the conversation above")
        )
    })
}

#[tokio::test]
async fn agent_session_prompt_runs_tool_loop_and_persists_messages() {
    let dir = tempdir().expect("tempdir");
    std::fs::write(dir.path().join("note.txt"), "hello from file").expect("seed note");

    let stream_call_count = Arc::new(AtomicUsize::new(0));
    let stream_call_count_in_fn = stream_call_count.clone();
    let stream_fn = Arc::new(
        move |_model: Model, context: Context, _options: Option<pixy_ai::SimpleStreamOptions>| {
            let call_index = stream_call_count_in_fn.fetch_add(1, Ordering::SeqCst);
            if call_index == 0 {
                let msg = assistant_message(
                    vec![AssistantContentBlock::ToolCall {
                        id: "tool-1".to_string(),
                        name: "read".to_string(),
                        arguments: json!({"path":"note.txt"}),
                        thought_signature: None,
                    }],
                    StopReason::ToolUse,
                    1_700_000_000_010,
                );
                Ok(done_stream(msg, DoneReason::ToolUse))
            } else {
                let had_tool_result = context.messages.iter().any(|message| {
                    matches!(
                        message,
                        Message::ToolResult {
                            tool_name,
                            content,
                            ..
                        } if tool_name == "read"
                            && content.iter().any(|block| matches!(
                                block,
                                ToolResultContentBlock::Text { text, .. }
                                if text.contains("hello from file")
                            ))
                    )
                });
                assert!(
                    had_tool_result,
                    "second turn should receive read tool result"
                );

                let msg = assistant_message(
                    vec![AssistantContentBlock::Text {
                        text: "all done".to_string(),
                        text_signature: None,
                    }],
                    StopReason::Stop,
                    1_700_000_000_020,
                );
                Ok(done_stream(msg, DoneReason::Stop))
            }
        },
    );

    let manager = SessionManager::create(
        dir.path().to_str().expect("cwd utf-8"),
        dir.path().join("sessions"),
    )
    .expect("create session manager");
    let tools = create_coding_tools(dir.path());
    let config = AgentSessionConfig {
        model: sample_model("test-api"),
        system_prompt: "You are helpful".to_string(),
        stream_fn,
        tools,
    };
    let mut session = AgentSession::new(manager, config);

    let produced = session
        .prompt("please read note")
        .await
        .expect("prompt succeeds");
    assert_eq!(
        produced.len(),
        4,
        "user + assistant(toolcall) + toolresult + assistant"
    );

    let context = session.build_session_context();
    assert_eq!(
        context.messages.len(),
        4,
        "messages should persist to session"
    );

    let file = session.session_file().expect("session file");
    let lines = std::fs::read_to_string(file)
        .expect("read file")
        .lines()
        .count();
    assert_eq!(lines, 5, "header + 4 message entries");
    assert_eq!(stream_call_count.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn agent_session_continue_run_after_reload_uses_history_and_persists() {
    let dir = tempdir().expect("tempdir");
    let session_dir = dir.path().join("sessions");

    let first_stream_fn = Arc::new(
        move |_model: Model, _context: Context, _options: Option<pixy_ai::SimpleStreamOptions>| {
            let msg = assistant_message(
                vec![AssistantContentBlock::Text {
                    text: "first answer".to_string(),
                    text_signature: None,
                }],
                StopReason::Stop,
                1_700_000_000_010,
            );
            Ok(done_stream(msg, DoneReason::Stop))
        },
    );

    let first_manager =
        SessionManager::create(dir.path().to_str().expect("cwd utf-8"), &session_dir)
            .expect("create manager");
    let first_config = AgentSessionConfig {
        model: sample_model("test-api"),
        system_prompt: "You are helpful".to_string(),
        stream_fn: first_stream_fn,
        tools: create_coding_tools(dir.path()),
    };
    let mut first_session = AgentSession::new(first_manager, first_config);
    first_session
        .prompt("first prompt")
        .await
        .expect("first prompt succeeds");

    let session_file = first_session.session_file().expect("session file").clone();
    drop(first_session);

    let observed_context_len = Arc::new(AtomicUsize::new(0));
    let observed_context_len_in_fn = observed_context_len.clone();
    let second_stream_fn = Arc::new(
        move |_model: Model, context: Context, _options: Option<pixy_ai::SimpleStreamOptions>| {
            observed_context_len_in_fn.store(context.messages.len(), Ordering::SeqCst);
            let had_first_prompt = context.messages.iter().any(|message| {
                matches!(
                    message,
                    Message::User {
                        content: pixy_ai::UserContent::Text(text),
                        ..
                    } if text == "first prompt"
                )
            });
            let had_first_answer = context.messages.iter().any(|message| {
                matches!(
                    message,
                    Message::Assistant { content, .. }
                        if content.iter().any(|block| matches!(
                            block,
                            AssistantContentBlock::Text { text, .. } if text == "first answer"
                        ))
                )
            });
            assert!(
                had_first_prompt,
                "continue_run should see persisted user message"
            );
            assert!(
                had_first_answer,
                "continue_run should see persisted assistant message"
            );

            let msg = assistant_message(
                vec![AssistantContentBlock::Text {
                    text: "continued answer".to_string(),
                    text_signature: None,
                }],
                StopReason::Stop,
                1_700_000_000_020,
            );
            Ok(done_stream(msg, DoneReason::Stop))
        },
    );

    let loaded_manager = SessionManager::load(&session_file).expect("load session manager");
    let second_config = AgentSessionConfig {
        model: sample_model("test-api"),
        system_prompt: "You are helpful".to_string(),
        stream_fn: second_stream_fn,
        tools: create_coding_tools(dir.path()),
    };
    let mut second_session = AgentSession::new(loaded_manager, second_config);
    let produced = second_session
        .continue_run()
        .await
        .expect("continue_run succeeds");

    assert_eq!(observed_context_len.load(Ordering::SeqCst), 2);
    assert_eq!(
        produced.len(),
        1,
        "continue_run should produce only new assistant message"
    );
    let context = second_session.build_session_context();
    assert_eq!(
        context.messages.len(),
        3,
        "reloaded session should append continued message"
    );

    let lines = std::fs::read_to_string(session_file)
        .expect("read session file")
        .lines()
        .count();
    assert_eq!(lines, 4, "header + three message entries");
}

#[tokio::test]
async fn agent_session_branch_then_continue_uses_branched_path_only() {
    let dir = tempdir().expect("tempdir");
    let session_dir = dir.path().join("sessions");

    let prompt_call_count = Arc::new(AtomicUsize::new(0));
    let prompt_call_count_in_fn = prompt_call_count.clone();
    let prompt_stream_fn = Arc::new(
        move |_model: Model, _context: Context, _options: Option<pixy_ai::SimpleStreamOptions>| {
            let idx = prompt_call_count_in_fn.fetch_add(1, Ordering::SeqCst);
            let text = if idx == 0 {
                "first answer"
            } else {
                "second answer"
            };
            let ts = if idx == 0 {
                1_700_000_000_010
            } else {
                1_700_000_000_020
            };
            let msg = assistant_message(
                vec![AssistantContentBlock::Text {
                    text: text.to_string(),
                    text_signature: None,
                }],
                StopReason::Stop,
                ts,
            );
            Ok(done_stream(msg, DoneReason::Stop))
        },
    );

    let first_manager =
        SessionManager::create(dir.path().to_str().expect("cwd utf-8"), &session_dir)
            .expect("create manager");
    let first_config = AgentSessionConfig {
        model: sample_model("test-api"),
        system_prompt: "You are helpful".to_string(),
        stream_fn: prompt_stream_fn,
        tools: create_coding_tools(dir.path()),
    };
    let mut first_session = AgentSession::new(first_manager, first_config);
    first_session
        .prompt("first prompt")
        .await
        .expect("first prompt succeeds");
    first_session
        .prompt("second prompt")
        .await
        .expect("second prompt succeeds");

    let session_file = first_session.session_file().expect("session file").clone();
    drop(first_session);

    let raw = std::fs::read_to_string(&session_file).expect("read session file");
    let lines: Vec<&str> = raw.lines().collect();
    assert!(lines.len() >= 3, "header + at least two messages expected");
    let first_user_entry: serde_json::Value =
        serde_json::from_str(lines[1]).expect("first entry json");
    let first_user_id = first_user_entry["id"]
        .as_str()
        .expect("first entry id")
        .to_string();

    let observed_context_len = Arc::new(AtomicUsize::new(0));
    let observed_context_len_in_fn = observed_context_len.clone();
    let continue_stream_fn = Arc::new(
        move |_model: Model, context: Context, _options: Option<pixy_ai::SimpleStreamOptions>| {
            observed_context_len_in_fn.store(context.messages.len(), Ordering::SeqCst);
            let has_first_prompt = context.messages.iter().any(|message| {
                matches!(
                    message,
                    Message::User {
                        content: pixy_ai::UserContent::Text(text),
                        ..
                    } if text == "first prompt"
                )
            });
            let has_second_prompt = context.messages.iter().any(|message| {
                matches!(
                    message,
                    Message::User {
                        content: pixy_ai::UserContent::Text(text),
                        ..
                    } if text == "second prompt"
                )
            });
            assert!(
                has_first_prompt,
                "branched context should include first prompt"
            );
            assert!(
                !has_second_prompt,
                "branched context should not include sibling-branch second prompt"
            );
            let has_branch_summary = context.messages.iter().any(|message| {
                matches!(
                    message,
                    Message::User {
                        content: pixy_ai::UserContent::Blocks(blocks),
                        ..
                    } if blocks.iter().any(|block| matches!(
                        block,
                        pixy_ai::UserContentBlock::Text { text, .. }
                            if text.contains("summary from second branch")
                    ))
                )
            });
            assert!(
                has_branch_summary,
                "branched context should include branch summary message"
            );

            let msg = assistant_message(
                vec![AssistantContentBlock::Text {
                    text: "after branch".to_string(),
                    text_signature: None,
                }],
                StopReason::Stop,
                1_700_000_000_030,
            );
            Ok(done_stream(msg, DoneReason::Stop))
        },
    );

    let mut loaded_manager = SessionManager::load(&session_file).expect("load session manager");
    loaded_manager
        .branch_with_summary(Some(&first_user_id), "summary from second branch")
        .expect("branch with summary");
    let second_config = AgentSessionConfig {
        model: sample_model("test-api"),
        system_prompt: "You are helpful".to_string(),
        stream_fn: continue_stream_fn,
        tools: create_coding_tools(dir.path()),
    };
    let mut second_session = AgentSession::new(loaded_manager, second_config);
    let produced = second_session
        .continue_run()
        .await
        .expect("continue_run succeeds");

    assert_eq!(observed_context_len.load(Ordering::SeqCst), 2);
    assert_eq!(produced.len(), 1);
    let context = second_session.build_session_context();
    assert_eq!(
        context.messages.len(),
        3,
        "first prompt + branch summary + branched continuation"
    );
}

#[tokio::test]
async fn agent_session_compact_then_continue_uses_compacted_context() {
    let dir = tempdir().expect("tempdir");
    let session_dir = dir.path().join("sessions");

    let stream_call_count = Arc::new(AtomicUsize::new(0));
    let stream_call_count_in_fn = stream_call_count.clone();
    let observed_context_len = Arc::new(AtomicUsize::new(0));
    let observed_context_len_in_fn = observed_context_len.clone();
    let stream_fn = Arc::new(
        move |_model: Model, context: Context, _options: Option<pixy_ai::SimpleStreamOptions>| {
            let call_index = stream_call_count_in_fn.fetch_add(1, Ordering::SeqCst);
            if call_index == 0 {
                let msg = assistant_message(
                    vec![AssistantContentBlock::Text {
                        text: "first answer".to_string(),
                        text_signature: None,
                    }],
                    StopReason::Stop,
                    1_700_000_000_010,
                );
                return Ok(done_stream(msg, DoneReason::Stop));
            }
            if call_index == 1 {
                let msg = assistant_message(
                    vec![AssistantContentBlock::Text {
                        text: "second answer".to_string(),
                        text_signature: None,
                    }],
                    StopReason::Stop,
                    1_700_000_000_020,
                );
                return Ok(done_stream(msg, DoneReason::Stop));
            }

            observed_context_len_in_fn.store(context.messages.len(), Ordering::SeqCst);
            let has_first_prompt = context.messages.iter().any(|message| {
                matches!(
                    message,
                    Message::User {
                        content: pixy_ai::UserContent::Text(text),
                        ..
                    } if text == "first prompt"
                )
            });
            let has_first_answer = context.messages.iter().any(|message| {
                matches!(
                    message,
                    Message::Assistant { content, .. }
                        if content.iter().any(|block| matches!(
                            block,
                            AssistantContentBlock::Text { text, .. } if text == "first answer"
                        ))
                )
            });
            let has_second_prompt = context.messages.iter().any(|message| {
                matches!(
                    message,
                    Message::User {
                        content: pixy_ai::UserContent::Text(text),
                        ..
                    } if text == "second prompt"
                )
            });
            let has_second_answer = context.messages.iter().any(|message| {
                matches!(
                    message,
                    Message::Assistant { content, .. }
                        if content.iter().any(|block| matches!(
                            block,
                            AssistantContentBlock::Text { text, .. } if text == "second answer"
                        ))
                )
            });
            let has_compaction_summary = context.messages.iter().any(|message| {
                matches!(
                    message,
                    Message::User {
                        content: pixy_ai::UserContent::Blocks(blocks),
                        ..
                    } if blocks.iter().any(|block| matches!(
                        block,
                        pixy_ai::UserContentBlock::Text { text, .. }
                            if text.contains(COMPACTION_SUMMARY_PREFIX)
                                && text.contains("compaction recap")
                    ))
                )
            });

            assert!(
                !has_first_prompt,
                "compaction should remove messages before kept point"
            );
            assert!(
                has_first_answer,
                "kept message should remain after compaction"
            );
            assert!(
                has_second_prompt,
                "post-kept messages should remain after compaction"
            );
            assert!(
                has_second_answer,
                "post-kept messages should remain after compaction"
            );
            assert!(
                has_compaction_summary,
                "compaction summary should be injected"
            );

            let msg = assistant_message(
                vec![AssistantContentBlock::Text {
                    text: "continued after compact".to_string(),
                    text_signature: None,
                }],
                StopReason::Stop,
                1_700_000_000_030,
            );
            Ok(done_stream(msg, DoneReason::Stop))
        },
    );

    let manager = SessionManager::create(dir.path().to_str().expect("cwd utf-8"), &session_dir)
        .expect("create manager");
    let config = AgentSessionConfig {
        model: sample_model("test-api"),
        system_prompt: "You are helpful".to_string(),
        stream_fn,
        tools: create_coding_tools(dir.path()),
    };
    let mut session = AgentSession::new(manager, config);

    session
        .prompt("first prompt")
        .await
        .expect("first prompt succeeds");
    session
        .prompt("second prompt")
        .await
        .expect("second prompt succeeds");

    let session_file = session.session_file().expect("session file").clone();
    let content_before_compact = std::fs::read_to_string(&session_file).expect("read session file");
    let lines_before_compact: Vec<&str> = content_before_compact.lines().collect();
    let first_assistant_entry: serde_json::Value =
        serde_json::from_str(lines_before_compact[2]).expect("assistant entry json");
    let first_kept_entry_id = first_assistant_entry["id"]
        .as_str()
        .expect("assistant entry id")
        .to_string();

    let compaction_id = session
        .compact("compaction recap", Some(&first_kept_entry_id), 42_000)
        .expect("compact succeeds");
    assert!(!compaction_id.is_empty());

    let content_after_compact = std::fs::read_to_string(&session_file).expect("read session file");
    let lines_after_compact: Vec<&str> = content_after_compact.lines().collect();
    let compaction_entry: serde_json::Value =
        serde_json::from_str(lines_after_compact.last().expect("compaction line"))
            .expect("compaction entry json");
    assert_eq!(compaction_entry["type"], "compaction");
    assert_eq!(
        compaction_entry["firstKeptEntryId"],
        serde_json::Value::String(first_kept_entry_id)
    );
    assert_eq!(
        compaction_entry["tokensBefore"],
        serde_json::Value::Number(42_000_u64.into())
    );

    let produced = session
        .continue_run()
        .await
        .expect("continue after compact succeeds");
    assert_eq!(produced.len(), 1);
    assert_eq!(observed_context_len.load(Ordering::SeqCst), 4);

    let context = session.build_session_context();
    assert_eq!(
        context.messages.len(),
        5,
        "summary + kept segment + post-compaction + continued assistant"
    );
}

#[tokio::test]
async fn agent_session_compact_keep_recent_uses_recent_count_cut_point() {
    let dir = tempdir().expect("tempdir");
    let session_dir = dir.path().join("sessions");

    let stream_call_count = Arc::new(AtomicUsize::new(0));
    let stream_call_count_in_fn = stream_call_count.clone();
    let stream_fn = Arc::new(
        move |_model: Model, _context: Context, _options: Option<pixy_ai::SimpleStreamOptions>| {
            let call_index = stream_call_count_in_fn.fetch_add(1, Ordering::SeqCst);
            let text = if call_index == 0 {
                "first answer"
            } else {
                "second answer"
            };
            let timestamp = if call_index == 0 {
                1_700_000_000_010
            } else {
                1_700_000_000_020
            };
            let msg = assistant_message(
                vec![AssistantContentBlock::Text {
                    text: text.to_string(),
                    text_signature: None,
                }],
                StopReason::Stop,
                timestamp,
            );
            Ok(done_stream(msg, DoneReason::Stop))
        },
    );

    let manager = SessionManager::create(dir.path().to_str().expect("cwd utf-8"), &session_dir)
        .expect("create manager");
    let config = AgentSessionConfig {
        model: sample_model("test-api"),
        system_prompt: "You are helpful".to_string(),
        stream_fn,
        tools: create_coding_tools(dir.path()),
    };
    let mut session = AgentSession::new(manager, config);

    session
        .prompt("first prompt")
        .await
        .expect("first prompt succeeds");
    session
        .prompt("second prompt")
        .await
        .expect("second prompt succeeds");

    let compaction_id = session
        .compact_keep_recent("keep recent recap", 2, 7_000)
        .expect("compact keep recent succeeds")
        .expect("compaction should happen");
    assert!(!compaction_id.is_empty());

    let context = session.build_session_context();
    assert_eq!(
        context.messages.len(),
        3,
        "summary + second turn user/assistant"
    );

    let has_first_prompt = context.messages.iter().any(|message| {
        matches!(
            message,
            Message::User {
                content: pixy_ai::UserContent::Text(text),
                ..
            } if text == "first prompt"
        )
    });
    assert!(!has_first_prompt, "first turn should be compacted away");

    let has_second_prompt = context.messages.iter().any(|message| {
        matches!(
            message,
            Message::User {
                content: pixy_ai::UserContent::Text(text),
                ..
            } if text == "second prompt"
        )
    });
    assert!(has_second_prompt, "latest user prompt should be kept");
}

#[tokio::test]
async fn agent_session_auto_compaction_triggers_when_threshold_exceeded() {
    let dir = tempdir().expect("tempdir");
    let session_dir = dir.path().join("sessions");

    let stream_fn = Arc::new(
        move |_model: Model, context: Context, _options: Option<pixy_ai::SimpleStreamOptions>| {
            if is_summary_request(&context) {
                let summary = assistant_message(
                    vec![AssistantContentBlock::Text {
                        text: "LLM summary: captured first prompt".to_string(),
                        text_signature: None,
                    }],
                    StopReason::Stop,
                    1_700_000_000_011,
                );
                return Ok(done_stream(summary, DoneReason::Stop));
            }

            let answer = assistant_message(
                vec![AssistantContentBlock::Text {
                    text: "answer before auto compact".to_string(),
                    text_signature: None,
                }],
                StopReason::Stop,
                1_700_000_000_010,
            );
            Ok(done_stream(answer, DoneReason::Stop))
        },
    );

    let mut model = sample_model("test-api");
    model.context_window = 20;

    let manager = SessionManager::create(dir.path().to_str().expect("cwd utf-8"), &session_dir)
        .expect("create manager");
    let config = AgentSessionConfig {
        model,
        system_prompt: "You are helpful".to_string(),
        stream_fn,
        tools: create_coding_tools(dir.path()),
    };
    let mut session = AgentSession::new(manager, config);
    session.set_auto_compaction_config(AutoCompactionConfig {
        enabled: true,
        reserve_tokens: 10,
        keep_recent_messages: 1,
        max_summary_chars: 800,
    });

    session
        .prompt("first prompt")
        .await
        .expect("prompt succeeds with auto compaction");

    let session_file = session.session_file().expect("session file").clone();
    let content = std::fs::read_to_string(&session_file).expect("read session file");
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 4, "header + user + assistant + compaction");

    let assistant_entry: serde_json::Value =
        serde_json::from_str(lines[2]).expect("assistant entry json");
    let assistant_id = assistant_entry["id"]
        .as_str()
        .expect("assistant entry id")
        .to_string();

    let compaction_entry: serde_json::Value =
        serde_json::from_str(lines[3]).expect("compaction entry json");
    assert_eq!(compaction_entry["type"], "compaction");
    assert_eq!(
        compaction_entry["firstKeptEntryId"],
        serde_json::Value::String(assistant_id)
    );
    assert_eq!(
        compaction_entry["tokensBefore"],
        serde_json::Value::Number(15_u64.into()),
        "sample assistant usage total tokens is 15"
    );

    let context = session.build_session_context();
    assert_eq!(context.messages.len(), 2, "summary + kept assistant");

    let has_llm_summary = context.messages.iter().any(|message| {
        matches!(
            message,
            Message::User {
                content: pixy_ai::UserContent::Blocks(blocks),
                ..
            } if blocks.iter().any(|block| matches!(
                block,
                pixy_ai::UserContentBlock::Text { text, .. }
                    if text.contains(COMPACTION_SUMMARY_PREFIX)
                        && text.contains("LLM summary: captured first prompt")
            ))
        )
    });
    assert!(
        has_llm_summary,
        "auto compaction should use LLM-generated summary when model call succeeds"
    );
}

#[tokio::test]
async fn agent_session_auto_compaction_falls_back_to_rule_summary_when_llm_summary_fails() {
    let dir = tempdir().expect("tempdir");
    let session_dir = dir.path().join("sessions");

    let stream_fn = Arc::new(
        move |_model: Model, context: Context, _options: Option<pixy_ai::SimpleStreamOptions>| {
            if is_summary_request(&context) {
                let error = assistant_error_message("summary request failed", 1_700_000_000_011, 0);
                return Ok(error_stream(error));
            }

            let answer = assistant_message(
                vec![AssistantContentBlock::Text {
                    text: "answer before auto compact".to_string(),
                    text_signature: None,
                }],
                StopReason::Stop,
                1_700_000_000_010,
            );
            Ok(done_stream(answer, DoneReason::Stop))
        },
    );

    let mut model = sample_model("test-api");
    model.context_window = 20;

    let manager = SessionManager::create(dir.path().to_str().expect("cwd utf-8"), &session_dir)
        .expect("create manager");
    let config = AgentSessionConfig {
        model,
        system_prompt: "You are helpful".to_string(),
        stream_fn,
        tools: create_coding_tools(dir.path()),
    };
    let mut session = AgentSession::new(manager, config);
    session.set_auto_compaction_config(AutoCompactionConfig {
        enabled: true,
        reserve_tokens: 10,
        keep_recent_messages: 1,
        max_summary_chars: 800,
    });

    session
        .prompt("first prompt")
        .await
        .expect("prompt succeeds with fallback compaction");

    let context = session.build_session_context();
    let has_fallback_summary = context.messages.iter().any(|message| {
        matches!(
            message,
            Message::User {
                content: pixy_ai::UserContent::Blocks(blocks),
                ..
            } if blocks.iter().any(|block| matches!(
                block,
                pixy_ai::UserContentBlock::Text { text, .. }
                    if text.contains(COMPACTION_SUMMARY_PREFIX)
                        && text.contains("first prompt")
            ))
        )
    });
    assert!(
        has_fallback_summary,
        "fallback summary should include compacted prompt text when LLM summarization fails"
    );
}

#[tokio::test]
async fn agent_session_overflow_triggers_auto_compaction_and_retry() {
    let dir = tempdir().expect("tempdir");
    let session_dir = dir.path().join("sessions");

    let stream_call_count = Arc::new(AtomicUsize::new(0));
    let stream_call_count_in_fn = stream_call_count.clone();
    let stream_fn = Arc::new(
        move |_model: Model, context: Context, _options: Option<pixy_ai::SimpleStreamOptions>| {
            let call_index = stream_call_count_in_fn.fetch_add(1, Ordering::SeqCst);
            match call_index {
                0 => {
                    let warmup = assistant_message(
                        vec![AssistantContentBlock::Text {
                            text: "warmup answer".to_string(),
                            text_signature: None,
                        }],
                        StopReason::Stop,
                        1_700_000_000_010,
                    );
                    Ok(done_stream(warmup, DoneReason::Stop))
                }
                1 => {
                    let overflow = assistant_error_message(
                        "prompt is too long: 250000 tokens > 200000 maximum",
                        1_700_000_000_020,
                        250_000,
                    );
                    Ok(error_stream(overflow))
                }
                2 => {
                    assert!(
                        is_summary_request(&context),
                        "overflow recovery should request compaction summary"
                    );
                    let summary = assistant_message(
                        vec![AssistantContentBlock::Text {
                            text: "overflow compaction summary".to_string(),
                            text_signature: None,
                        }],
                        StopReason::Stop,
                        1_700_000_000_030,
                    );
                    Ok(done_stream(summary, DoneReason::Stop))
                }
                _ => {
                    let has_overflow_error_in_context = context.messages.iter().any(|message| {
                        matches!(
                            message,
                            Message::Assistant {
                                stop_reason: StopReason::Error,
                                ..
                            }
                        )
                    });
                    assert!(
                        !has_overflow_error_in_context,
                        "overflow retry should not carry previous assistant error in context"
                    );
                    let recovered = assistant_message(
                        vec![AssistantContentBlock::Text {
                            text: "recovered response".to_string(),
                            text_signature: None,
                        }],
                        StopReason::Stop,
                        1_700_000_000_040,
                    );
                    Ok(done_stream(recovered, DoneReason::Stop))
                }
            }
        },
    );

    let mut model = sample_model("test-api");
    model.context_window = 200_000;

    let manager = SessionManager::create(dir.path().to_str().expect("cwd utf-8"), &session_dir)
        .expect("create manager");
    let config = AgentSessionConfig {
        model,
        system_prompt: "You are helpful".to_string(),
        stream_fn,
        tools: create_coding_tools(dir.path()),
    };
    let mut session = AgentSession::new(manager, config);
    session.set_auto_compaction_config(AutoCompactionConfig {
        enabled: true,
        reserve_tokens: 10,
        keep_recent_messages: 1,
        max_summary_chars: 800,
    });

    session.prompt("warmup").await.expect("warmup succeeds");
    let produced = session
        .prompt("trigger overflow")
        .await
        .expect("overflow path should auto-compact and retry");

    let has_recovered_response = produced.iter().any(|message| {
        matches!(
            message,
            Message::Assistant { content, .. }
                if content.iter().any(|block| matches!(
                    block,
                    AssistantContentBlock::Text { text, .. } if text == "recovered response"
                ))
        )
    });
    assert!(
        has_recovered_response,
        "prompt should include assistant response from automatic retry"
    );

    let context = session.build_session_context();
    let has_error_assistant = context.messages.iter().any(|message| {
        matches!(
            message,
            Message::Assistant {
                stop_reason: StopReason::Error,
                ..
            }
        )
    });
    assert!(
        !has_error_assistant,
        "active context should branch away from overflow error message after retry"
    );

    assert!(
        stream_call_count.load(Ordering::SeqCst) >= 4,
        "expected calls: warmup, overflow, summary, retry"
    );
}

#[tokio::test]
async fn agent_session_prompt_streaming_emits_text_delta_updates() {
    let dir = tempdir().expect("tempdir");
    let session_dir = dir.path().join("sessions");

    let stream_fn = Arc::new(
        move |_model: Model, _context: Context, _options: Option<pixy_ai::SimpleStreamOptions>| {
            Ok(text_delta_stream(
                &["hello", " world"],
                "hello world",
                1_700_000_000_010,
            ))
        },
    );

    let manager = SessionManager::create(dir.path().to_str().expect("cwd utf-8"), &session_dir)
        .expect("create manager");
    let config = AgentSessionConfig {
        model: sample_model("test-api"),
        system_prompt: "You are helpful".to_string(),
        stream_fn,
        tools: create_coding_tools(dir.path()),
    };
    let mut session = AgentSession::new(manager, config);

    let mut updates = vec![];
    let produced = session
        .prompt_streaming("hi", |update| updates.push(update))
        .await
        .expect("prompt streaming succeeds");

    assert_eq!(
        updates,
        vec![
            AgentSessionStreamUpdate::AssistantTextDelta("hello".to_string()),
            AgentSessionStreamUpdate::AssistantTextDelta(" world".to_string()),
            AgentSessionStreamUpdate::AssistantLine(String::new()),
        ]
    );
    assert_eq!(produced.len(), 2, "user + assistant");
}

#[tokio::test]
async fn agent_session_prompt_streaming_hides_read_tool_result_text() {
    let dir = tempdir().expect("tempdir");
    let session_dir = dir.path().join("sessions");
    std::fs::write(dir.path().join("note.txt"), "hello from file").expect("seed note");

    let stream_call_count = Arc::new(AtomicUsize::new(0));
    let stream_call_count_in_fn = stream_call_count.clone();
    let stream_fn = Arc::new(
        move |_model: Model, _context: Context, _options: Option<pixy_ai::SimpleStreamOptions>| {
            let call_index = stream_call_count_in_fn.fetch_add(1, Ordering::SeqCst);
            if call_index == 0 {
                let msg = assistant_message(
                    vec![AssistantContentBlock::ToolCall {
                        id: "tool-read-1".to_string(),
                        name: "read".to_string(),
                        arguments: json!({"path":"note.txt"}),
                        thought_signature: None,
                    }],
                    StopReason::ToolUse,
                    1_700_000_000_010,
                );
                Ok(done_stream(msg, DoneReason::ToolUse))
            } else {
                let msg = assistant_message(
                    vec![AssistantContentBlock::Text {
                        text: "done".to_string(),
                        text_signature: None,
                    }],
                    StopReason::Stop,
                    1_700_000_000_020,
                );
                Ok(done_stream(msg, DoneReason::Stop))
            }
        },
    );

    let manager = SessionManager::create(dir.path().to_str().expect("cwd utf-8"), &session_dir)
        .expect("create manager");
    let config = AgentSessionConfig {
        model: sample_model("test-api"),
        system_prompt: "You are helpful".to_string(),
        stream_fn,
        tools: create_coding_tools(dir.path()),
    };
    let mut session = AgentSession::new(manager, config);

    let mut updates = vec![];
    session
        .prompt_streaming("please read note", |update| updates.push(update))
        .await
        .expect("prompt streaming succeeds");

    assert!(
        updates
            .iter()
            .any(|update| matches!(update, AgentSessionStreamUpdate::ToolLine(line) if line == "• Ran read note.txt")),
        "streaming updates should still include read tool run line"
    );
    assert!(
        updates
            .iter()
            .all(|update| !matches!(update, AgentSessionStreamUpdate::ToolLine(line) if line.contains("hello from file"))),
        "streaming updates should not include read tool output text"
    );
}

#[tokio::test]
async fn agent_session_continue_run_streaming_emits_updates() {
    let dir = tempdir().expect("tempdir");
    let session_dir = dir.path().join("sessions");

    let stream_call_count = Arc::new(AtomicUsize::new(0));
    let stream_call_count_in_fn = stream_call_count.clone();
    let stream_fn = Arc::new(
        move |_model: Model, _context: Context, _options: Option<pixy_ai::SimpleStreamOptions>| {
            let call_index = stream_call_count_in_fn.fetch_add(1, Ordering::SeqCst);
            if call_index == 0 {
                let seed = assistant_message(
                    vec![AssistantContentBlock::Text {
                        text: "seed answer".to_string(),
                        text_signature: None,
                    }],
                    StopReason::Stop,
                    1_700_000_000_010,
                );
                Ok(done_stream(seed, DoneReason::Stop))
            } else {
                Ok(text_delta_stream(
                    &["cont", "inue"],
                    "continue",
                    1_700_000_000_020,
                ))
            }
        },
    );

    let manager = SessionManager::create(dir.path().to_str().expect("cwd utf-8"), &session_dir)
        .expect("create manager");
    let config = AgentSessionConfig {
        model: sample_model("test-api"),
        system_prompt: "You are helpful".to_string(),
        stream_fn,
        tools: create_coding_tools(dir.path()),
    };
    let mut session = AgentSession::new(manager, config);
    session.prompt("seed").await.expect("seed prompt succeeds");

    let mut updates = vec![];
    let produced = session
        .continue_run_streaming(|update| updates.push(update))
        .await
        .expect("continue streaming succeeds");

    assert_eq!(
        updates,
        vec![
            AgentSessionStreamUpdate::AssistantTextDelta("cont".to_string()),
            AgentSessionStreamUpdate::AssistantTextDelta("inue".to_string()),
            AgentSessionStreamUpdate::AssistantLine(String::new()),
        ]
    );
    assert_eq!(produced.len(), 1, "continue should only append assistant");
}

#[tokio::test]
async fn agent_session_prompt_streaming_updates_thinking_without_duplicate_append() {
    let dir = tempdir().expect("tempdir");
    let session_dir = dir.path().join("sessions");
    let stream_fn = Arc::new(
        move |_model: Model, _context: Context, _options: Option<pixy_ai::SimpleStreamOptions>| {
            Ok(thinking_delta_stream(
                &["Analy", "zing"],
                "Analyzing",
                1_700_000_000_030,
            ))
        },
    );

    let manager = SessionManager::create(dir.path().to_str().expect("cwd utf-8"), &session_dir)
        .expect("create manager");
    let config = AgentSessionConfig {
        model: sample_model("test-api"),
        system_prompt: "You are helpful".to_string(),
        stream_fn,
        tools: create_coding_tools(dir.path()),
    };
    let mut session = AgentSession::new(manager, config);

    let mut updates = vec![];
    let produced = session
        .prompt_streaming("hi", |update| updates.push(update))
        .await
        .expect("prompt streaming succeeds");

    assert_eq!(
        updates,
        vec![
            AgentSessionStreamUpdate::AssistantLine("[thinking] Analy".to_string()),
            AgentSessionStreamUpdate::AssistantLine("[thinking] Analyzing".to_string()),
        ]
    );
    assert_eq!(produced.len(), 2, "user + assistant");
}

#[tokio::test]
async fn agent_session_prompt_streaming_keeps_final_thinking_when_only_text_delta_streamed() {
    let dir = tempdir().expect("tempdir");
    let session_dir = dir.path().join("sessions");
    let stream_fn = Arc::new(
        move |_model: Model, _context: Context, _options: Option<pixy_ai::SimpleStreamOptions>| {
            Ok(text_delta_stream_with_final_thinking(
                &["hello", " world"],
                "hello world",
                "Need to inspect repository structure first.",
                1_700_000_000_030,
            ))
        },
    );

    let manager = SessionManager::create(dir.path().to_str().expect("cwd utf-8"), &session_dir)
        .expect("create manager");
    let config = AgentSessionConfig {
        model: sample_model("test-api"),
        system_prompt: "You are helpful".to_string(),
        stream_fn,
        tools: create_coding_tools(dir.path()),
    };
    let mut session = AgentSession::new(manager, config);

    let mut updates = vec![];
    let produced = session
        .prompt_streaming("hi", |update| updates.push(update))
        .await
        .expect("prompt streaming succeeds");

    assert_eq!(
        updates,
        vec![
            AgentSessionStreamUpdate::AssistantTextDelta("hello".to_string()),
            AgentSessionStreamUpdate::AssistantTextDelta(" world".to_string()),
            AgentSessionStreamUpdate::AssistantLine(String::new()),
            AgentSessionStreamUpdate::AssistantLine(
                "[thinking] Need to inspect repository structure first.".to_string(),
            ),
        ]
    );
    assert_eq!(produced.len(), 2, "user + assistant");
}

#[tokio::test]
async fn agent_session_prompt_streaming_thinking_snapshot_deltas_do_not_duplicate() {
    let dir = tempdir().expect("tempdir");
    let session_dir = dir.path().join("sessions");
    let stream_fn = Arc::new(
        move |_model: Model, _context: Context, _options: Option<pixy_ai::SimpleStreamOptions>| {
            Ok(thinking_snapshot_stream(
                &[
                    "**Analyzing directory settings**",
                    "**Analyzing directory settings**",
                ],
                "**Analyzing directory settings**",
                1_700_000_000_031,
            ))
        },
    );

    let manager = SessionManager::create(dir.path().to_str().expect("cwd utf-8"), &session_dir)
        .expect("create manager");
    let config = AgentSessionConfig {
        model: sample_model("test-api"),
        system_prompt: "You are helpful".to_string(),
        stream_fn,
        tools: create_coding_tools(dir.path()),
    };
    let mut session = AgentSession::new(manager, config);

    let mut updates = vec![];
    let produced = session
        .prompt_streaming("hi", |update| updates.push(update))
        .await
        .expect("prompt streaming succeeds");

    assert_eq!(
        updates,
        vec![AgentSessionStreamUpdate::AssistantLine(
            "[thinking] **Analyzing directory settings**".to_string()
        )]
    );
    assert_eq!(produced.len(), 2, "user + assistant");
}

#[tokio::test]
async fn agent_session_cycle_model_switches_and_persists_model_change_entries() {
    let dir = tempdir().expect("tempdir");
    let session_dir = dir.path().join("sessions");
    let stream_fn = Arc::new(
        move |_model: Model, _context: Context, _options: Option<pixy_ai::SimpleStreamOptions>| {
            let msg = assistant_message(
                vec![AssistantContentBlock::Text {
                    text: "ok".to_string(),
                    text_signature: None,
                }],
                StopReason::Stop,
                1_700_000_000_010,
            );
            Ok(done_stream(msg, DoneReason::Stop))
        },
    );

    let manager = SessionManager::create(dir.path().to_str().expect("cwd utf-8"), &session_dir)
        .expect("create manager");
    let mut model_a = sample_model("test-api");
    model_a.id = "model-a".to_string();
    let mut model_b = sample_model("test-api");
    model_b.id = "model-b".to_string();
    let config = AgentSessionConfig {
        model: model_a.clone(),
        system_prompt: "You are helpful".to_string(),
        stream_fn,
        tools: create_coding_tools(dir.path()),
    };
    let mut session = AgentSession::new(manager, config);
    session.set_model_catalog(vec![model_a.clone(), model_b.clone()]);

    assert_eq!(session.current_model().id, "model-a");
    let switched_forward = session
        .cycle_model_forward()
        .expect("cycle forward should succeed")
        .expect("should switch to next model");
    assert_eq!(switched_forward.id, "model-b");
    assert_eq!(session.current_model().id, "model-b");

    let switched_backward = session
        .cycle_model_backward()
        .expect("cycle backward should succeed")
        .expect("should switch to previous model");
    assert_eq!(switched_backward.id, "model-a");
    assert_eq!(session.current_model().id, "model-a");

    let session_file = session.session_file().expect("session file");
    let lines = std::fs::read_to_string(session_file)
        .expect("read session file")
        .lines()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();
    assert_eq!(lines.len(), 3, "header + two model change entries");

    let first_change: serde_json::Value =
        serde_json::from_str(&lines[1]).expect("first model change json");
    let second_change: serde_json::Value =
        serde_json::from_str(&lines[2]).expect("second model change json");
    assert_eq!(first_change["type"], "model_change");
    assert_eq!(first_change["modelId"], "model-b");
    assert_eq!(second_change["type"], "model_change");
    assert_eq!(second_change["modelId"], "model-a");
}

#[tokio::test]
async fn agent_session_select_model_maps_to_next_catalog_entry() {
    let dir = tempdir().expect("tempdir");
    let session_dir = dir.path().join("sessions");
    let stream_fn = Arc::new(
        move |_model: Model, _context: Context, _options: Option<pixy_ai::SimpleStreamOptions>| {
            let msg = assistant_message(
                vec![AssistantContentBlock::Text {
                    text: "ok".to_string(),
                    text_signature: None,
                }],
                StopReason::Stop,
                1_700_000_000_010,
            );
            Ok(done_stream(msg, DoneReason::Stop))
        },
    );

    let manager = SessionManager::create(dir.path().to_str().expect("cwd utf-8"), &session_dir)
        .expect("create manager");
    let mut model_a = sample_model("test-api");
    model_a.id = "model-a".to_string();
    let mut model_b = sample_model("test-api");
    model_b.id = "model-b".to_string();
    let config = AgentSessionConfig {
        model: model_a.clone(),
        system_prompt: "You are helpful".to_string(),
        stream_fn,
        tools: create_coding_tools(dir.path()),
    };
    let mut session = AgentSession::new(manager, config);
    session.set_model_catalog(vec![model_a, model_b]);

    let selected = session
        .select_model()
        .expect("select model should succeed")
        .expect("select should switch model");
    assert_eq!(selected.id, "model-b");
    assert_eq!(session.current_model().id, "model-b");
}
