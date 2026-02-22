use std::fs;

use pi_ai::{Message, StopReason, UserContent};
use pi_coding_agent::{
    BRANCH_SUMMARY_PREFIX, BRANCH_SUMMARY_SUFFIX, COMPACTION_SUMMARY_PREFIX,
    COMPACTION_SUMMARY_SUFFIX, CURRENT_SESSION_VERSION, SessionManager,
};
use serde_json::{Value, json};
use tempfile::tempdir;

fn user_message(text: &str, ts: i64) -> Message {
    Message::User {
        content: UserContent::Text(text.to_string()),
        timestamp: ts,
    }
}

fn assistant_message(text: &str, ts: i64) -> Message {
    assistant_message_with_stop_reason(text, ts, StopReason::Stop, None)
}

fn assistant_message_with_stop_reason(
    text: &str,
    ts: i64,
    stop_reason: StopReason,
    error_message: Option<&str>,
) -> Message {
    Message::Assistant {
        content: vec![pi_ai::AssistantContentBlock::Text {
            text: text.to_string(),
            text_signature: None,
        }],
        api: "openai-completions".to_string(),
        provider: "openai".to_string(),
        model: "gpt-4o-mini".to_string(),
        usage: pi_ai::Usage {
            input: 1,
            output: 1,
            cache_read: 0,
            cache_write: 0,
            total_tokens: 2,
            cost: pi_ai::Cost {
                input: 0.0,
                output: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
                total: 0.0,
            },
        },
        stop_reason,
        error_message: error_message.map(ToOwned::to_owned),
        timestamp: ts,
    }
}

#[test]
fn session_manager_persists_header_and_messages_as_jsonl() {
    let dir = tempdir().expect("tempdir");
    let mut manager = SessionManager::create("/repo", dir.path()).expect("create session manager");

    manager
        .append_message(user_message("hello", 1_700_000_000_000))
        .expect("append user message");

    let file_path = manager.session_file().expect("session file path");
    let content = fs::read_to_string(file_path).expect("read session file");
    let lines: Vec<&str> = content.lines().collect();

    assert_eq!(lines.len(), 2, "header + one message entry expected");

    let header: Value = serde_json::from_str(lines[0]).expect("header json");
    assert_eq!(header["type"], "session");
    assert_eq!(header["version"], CURRENT_SESSION_VERSION);

    let message: Value = serde_json::from_str(lines[1]).expect("message json");
    assert_eq!(message["type"], "message");
    assert_eq!(message["parentId"], Value::Null);
}

#[test]
fn session_manager_builds_context_in_append_order_with_parent_chain() {
    let dir = tempdir().expect("tempdir");
    let mut manager = SessionManager::create("/repo", dir.path()).expect("create session manager");

    let first_id = manager
        .append_message(user_message("hello", 1_700_000_000_000))
        .expect("append user");
    manager
        .append_message(assistant_message("world", 1_700_000_000_010))
        .expect("append assistant");

    let context = manager.build_session_context();
    assert_eq!(context.messages.len(), 2);

    let file_path = manager.session_file().expect("session file path");
    let content = fs::read_to_string(file_path).expect("read session file");
    let lines: Vec<&str> = content.lines().collect();
    let second_entry: Value = serde_json::from_str(lines[2]).expect("second message entry");
    assert_eq!(second_entry["parentId"], Value::String(first_id));
}

#[test]
fn session_manager_branch_changes_leaf_and_context_path() {
    let dir = tempdir().expect("tempdir");
    let mut manager = SessionManager::create("/repo", dir.path()).expect("create session manager");

    let first_id = manager
        .append_message(user_message("root", 1_700_000_000_000))
        .expect("append first");
    manager
        .append_message(assistant_message("main-1", 1_700_000_000_010))
        .expect("append second");
    manager
        .append_message(user_message("main-2", 1_700_000_000_020))
        .expect("append third");

    manager.branch(&first_id).expect("branch to first");
    manager
        .append_message(assistant_message("branch-1", 1_700_000_000_030))
        .expect("append branch message");

    let context = manager.build_session_context();
    assert_eq!(
        context.messages.len(),
        2,
        "context should follow current leaf path only"
    );
    match &context.messages[0] {
        Message::User { content, .. } => match content {
            UserContent::Text(text) => assert_eq!(text, "root"),
            _ => panic!("unexpected first message content"),
        },
        _ => panic!("unexpected first message role"),
    }
    match &context.messages[1] {
        Message::Assistant { content, .. } => match &content[0] {
            pi_ai::AssistantContentBlock::Text { text, .. } => assert_eq!(text, "branch-1"),
            _ => panic!("unexpected assistant content"),
        },
        _ => panic!("unexpected second message role"),
    }
}

#[test]
fn session_manager_load_restores_state_and_appends_with_new_id() {
    let dir = tempdir().expect("tempdir");
    let mut manager = SessionManager::create("/repo", dir.path()).expect("create session manager");

    manager
        .append_message(user_message("first", 1_700_000_000_000))
        .expect("append first");
    let second_id = manager
        .append_message(assistant_message("second", 1_700_000_000_010))
        .expect("append second");
    let file_path = manager.session_file().expect("session file path").clone();

    let mut loaded = SessionManager::load(&file_path).expect("load session manager");
    let third_id = loaded
        .append_message(user_message("third", 1_700_000_000_020))
        .expect("append third");
    assert_ne!(
        third_id, second_id,
        "loaded manager must continue id sequence"
    );

    let content = fs::read_to_string(&file_path).expect("read session file");
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 4, "header + three message entries expected");

    let third_entry: Value = serde_json::from_str(lines[3]).expect("third message entry");
    assert_eq!(third_entry["parentId"], Value::String(second_id));
}

#[test]
fn session_manager_branch_with_summary_adds_summary_message_on_target_branch() {
    let dir = tempdir().expect("tempdir");
    let mut manager = SessionManager::create("/repo", dir.path()).expect("create session manager");

    let first_id = manager
        .append_message(user_message("root", 1_700_000_000_000))
        .expect("append first");
    manager
        .append_message(assistant_message("main-1", 1_700_000_000_010))
        .expect("append second");
    manager
        .append_message(user_message("main-2", 1_700_000_000_020))
        .expect("append third");

    manager
        .branch_with_summary(Some(&first_id), "branch recap")
        .expect("branch with summary");

    let file_path = manager.session_file().expect("session file path");
    let content = fs::read_to_string(file_path).expect("read session file");
    let lines: Vec<&str> = content.lines().collect();
    let summary_entry: Value =
        serde_json::from_str(lines.last().expect("summary line")).expect("summary entry json");
    assert_eq!(summary_entry["type"], "branch_summary");
    assert_eq!(summary_entry["fromId"], Value::String(first_id.clone()));
    assert_eq!(summary_entry["parentId"], Value::String(first_id.clone()));
    assert_eq!(
        summary_entry["summary"],
        Value::String("branch recap".to_string())
    );

    let context = manager.build_session_context();
    assert_eq!(context.messages.len(), 2);
    match &context.messages[1] {
        Message::User { content, .. } => match content {
            UserContent::Blocks(blocks) => {
                let text = blocks
                    .iter()
                    .filter_map(|block| {
                        if let pi_ai::UserContentBlock::Text { text, .. } = block {
                            Some(text.clone())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("");
                assert_eq!(
                    text,
                    format!("{BRANCH_SUMMARY_PREFIX}branch recap{BRANCH_SUMMARY_SUFFIX}")
                );
            }
            _ => panic!("summary should be user blocks"),
        },
        _ => panic!("summary should be user message"),
    }
}

#[test]
fn session_manager_compaction_entry_persists_and_builds_compacted_context() {
    let dir = tempdir().expect("tempdir");
    let mut manager = SessionManager::create("/repo", dir.path()).expect("create session manager");

    manager
        .append_message(user_message("m1", 1_700_000_000_000))
        .expect("append first");
    let second_id = manager
        .append_message(assistant_message("m2", 1_700_000_000_010))
        .expect("append second");
    manager
        .append_message(user_message("m3", 1_700_000_000_020))
        .expect("append third");
    manager
        .append_compaction("compact recap", Some(&second_id), 50_000)
        .expect("append compaction");
    manager
        .append_message(assistant_message("m4", 1_700_000_000_030))
        .expect("append fourth");

    let file_path = manager.session_file().expect("session file path");
    let content = fs::read_to_string(file_path).expect("read session file");
    let lines: Vec<&str> = content.lines().collect();

    let compaction_entry: Value = serde_json::from_str(lines[4]).expect("compaction entry json");
    assert_eq!(compaction_entry["type"], "compaction");
    assert_eq!(
        compaction_entry["summary"],
        Value::String("compact recap".to_string())
    );
    assert_eq!(
        compaction_entry["firstKeptEntryId"],
        Value::String(second_id.clone())
    );
    assert_eq!(
        compaction_entry["tokensBefore"],
        Value::Number(50_000_u64.into())
    );

    let context = manager.build_session_context();
    assert_eq!(
        context.messages.len(),
        4,
        "summary + kept segment + post-compaction"
    );

    match &context.messages[0] {
        Message::User { content, .. } => match content {
            UserContent::Blocks(blocks) => {
                let text = blocks
                    .iter()
                    .filter_map(|block| {
                        if let pi_ai::UserContentBlock::Text { text, .. } = block {
                            Some(text.clone())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("");
                assert_eq!(
                    text,
                    format!("{COMPACTION_SUMMARY_PREFIX}compact recap{COMPACTION_SUMMARY_SUFFIX}")
                );
            }
            _ => panic!("compaction summary should be user blocks"),
        },
        _ => panic!("first context message should be compaction summary"),
    }

    match &context.messages[1] {
        Message::Assistant { content, .. } => match &content[0] {
            pi_ai::AssistantContentBlock::Text { text, .. } => assert_eq!(text, "m2"),
            _ => panic!("unexpected second context message content"),
        },
        _ => panic!("second context message should be assistant m2"),
    }

    match &context.messages[2] {
        Message::User { content, .. } => match content {
            UserContent::Text(text) => assert_eq!(text, "m3"),
            _ => panic!("unexpected third context message content"),
        },
        _ => panic!("third context message should be user m3"),
    }

    match &context.messages[3] {
        Message::Assistant { content, .. } => match &content[0] {
            pi_ai::AssistantContentBlock::Text { text, .. } => assert_eq!(text, "m4"),
            _ => panic!("unexpected fourth context message content"),
        },
        _ => panic!("fourth context message should be assistant m4"),
    }

    let root_present = context.messages.iter().any(|message| {
        matches!(
            message,
            Message::User {
                content: UserContent::Text(text),
                ..
            } if text == "m1"
        )
    });
    assert!(!root_present, "m1 should be compacted away");
}

#[test]
fn session_manager_first_kept_entry_id_for_recent_messages_respects_context_order() {
    let dir = tempdir().expect("tempdir");
    let mut manager = SessionManager::create("/repo", dir.path()).expect("create session manager");

    let first_id = manager
        .append_message(user_message("m1", 1_700_000_000_000))
        .expect("append first");
    manager
        .append_message(assistant_message("m2", 1_700_000_000_010))
        .expect("append second");
    let third_id = manager
        .append_message(user_message("m3", 1_700_000_000_020))
        .expect("append third");
    let fourth_id = manager
        .append_message(assistant_message("m4", 1_700_000_000_030))
        .expect("append fourth");

    let keep_two = manager
        .first_kept_entry_id_for_recent_messages(2)
        .expect("first kept id for keep=2");
    assert_eq!(keep_two, third_id);

    let keep_one = manager
        .first_kept_entry_id_for_recent_messages(1)
        .expect("first kept id for keep=1");
    assert_eq!(keep_one, fourth_id);

    let keep_all = manager.first_kept_entry_id_for_recent_messages(4);
    assert!(
        keep_all.is_none(),
        "keeping full context should not compact"
    );

    let keep_none = manager
        .first_kept_entry_id_for_recent_messages(0)
        .expect("first kept id for keep=0");
    assert_eq!(keep_none, first_id);
}

#[test]
fn session_manager_load_accepts_pi_mono_extended_entries_and_uses_custom_message_in_context() {
    let dir = tempdir().expect("tempdir");
    let file_path = dir.path().join("session.jsonl");

    let entries = [
        json!({
            "type": "session",
            "id": "session-1",
            "timestamp": "2026-02-22T10:00:00.000Z",
            "cwd": "/repo"
        }),
        json!({
            "type": "message",
            "id": "00000001",
            "parentId": Value::Null,
            "timestamp": "2026-02-22T10:00:01.000Z",
            "message": user_message("hello", 1_700_000_000_000)
        }),
        json!({
            "type": "thinking_level_change",
            "id": "00000002",
            "parentId": "00000001",
            "timestamp": "2026-02-22T10:00:02.000Z",
            "thinkingLevel": "high"
        }),
        json!({
            "type": "model_change",
            "id": "00000003",
            "parentId": "00000002",
            "timestamp": "2026-02-22T10:00:03.000Z",
            "provider": "anthropic",
            "modelId": "claude-opus-4-6"
        }),
        json!({
            "type": "custom",
            "id": "00000004",
            "parentId": "00000003",
            "timestamp": "2026-02-22T10:00:04.000Z",
            "customType": "ext_state",
            "data": { "k": "v" }
        }),
        json!({
            "type": "label",
            "id": "00000005",
            "parentId": "00000004",
            "timestamp": "2026-02-22T10:00:05.000Z",
            "targetId": "00000001",
            "label": "bookmark"
        }),
        json!({
            "type": "session_info",
            "id": "00000006",
            "parentId": "00000005",
            "timestamp": "2026-02-22T10:00:06.000Z",
            "name": "demo session"
        }),
        json!({
            "type": "custom_message",
            "id": "00000007",
            "parentId": "00000006",
            "timestamp": "2026-02-22T10:00:07.000Z",
            "customType": "ext_message",
            "content": "custom context payload",
            "display": true,
            "details": { "source": "test" }
        }),
        json!({
            "type": "branch_summary",
            "id": "00000008",
            "parentId": "00000007",
            "timestamp": "2026-02-22T10:00:08.000Z",
            "fromId": "00000001",
            "summary": "branch recap",
            "details": { "extra": true },
            "fromHook": true
        }),
        json!({
            "type": "compaction",
            "id": "00000009",
            "parentId": "00000008",
            "timestamp": "2026-02-22T10:00:09.000Z",
            "summary": "compact recap",
            "firstKeptEntryId": "00000007",
            "tokensBefore": 12_000,
            "details": { "readFiles": ["a.rs"] },
            "fromHook": true
        }),
        json!({
            "type": "message",
            "id": "0000000a",
            "parentId": "00000009",
            "timestamp": "2026-02-22T10:00:10.000Z",
            "message": assistant_message("post compact", 1_700_000_000_020)
        }),
    ];

    let serialized = entries
        .iter()
        .map(serde_json::to_string)
        .collect::<Result<Vec<_>, _>>()
        .expect("serialize entries")
        .join("\n");
    fs::write(&file_path, format!("{serialized}\n")).expect("write session file");

    let manager = SessionManager::load(&file_path).expect("load manager");
    let context = manager.build_session_context();
    assert_eq!(
        context.messages.len(),
        4,
        "compaction summary + kept custom_message + kept branch_summary + post-compaction message"
    );

    match &context.messages[1] {
        Message::User { content, timestamp } => {
            assert_eq!(
                *timestamp,
                chrono::DateTime::parse_from_rfc3339("2026-02-22T10:00:07.000Z")
                    .expect("parse ts")
                    .timestamp_millis()
            );
            match content {
                UserContent::Text(text) => assert_eq!(text, "custom context payload"),
                _ => panic!("custom_message should deserialize as text"),
            }
        }
        _ => panic!("kept custom_message should map to user message"),
    }
}

#[test]
fn session_manager_extended_appenders_and_rewind_error_leaf_work() {
    let dir = tempdir().expect("tempdir");
    let mut manager = SessionManager::create("/repo", dir.path()).expect("create session manager");

    let root_id = manager
        .append_message(user_message("root", 1_700_000_000_000))
        .expect("append root user");
    manager
        .append_thinking_level_change("medium")
        .expect("append thinking level");
    manager
        .append_model_change("anthropic", "claude-opus-4-6")
        .expect("append model change");
    manager
        .append_custom_entry("ext_state", Some(json!({ "phase": 1 })))
        .expect("append custom entry");
    manager
        .append_label(&root_id, Some("root label"))
        .expect("append label");
    manager
        .append_session_info(Some("my session"))
        .expect("append session info");
    manager
        .append_custom_message_entry(
            "ext_message",
            UserContent::Text("custom context".to_string()),
            true,
            Some(json!({ "k": "v" })),
        )
        .expect("append custom message");

    manager
        .append_message(assistant_message_with_stop_reason(
            "overflow error",
            1_700_000_000_010,
            StopReason::Error,
            Some("prompt is too long"),
        ))
        .expect("append assistant error");

    assert!(
        manager.rewind_leaf_if_last_assistant_error(),
        "rewind should move leaf to parent when leaf is assistant error"
    );
    manager
        .append_message(assistant_message("retry success", 1_700_000_000_020))
        .expect("append retry assistant");

    let context = manager.build_session_context();
    let has_error_assistant = context.messages.iter().any(|message| {
        matches!(
            message,
            Message::Assistant { stop_reason, .. } if *stop_reason == StopReason::Error
        )
    });
    assert!(
        !has_error_assistant,
        "rewound branch should exclude previous assistant error from active context"
    );

    let has_custom_context = context.messages.iter().any(|message| {
        matches!(
            message,
            Message::User {
                content: UserContent::Text(text),
                ..
            } if text == "custom context"
        )
    });
    assert!(
        has_custom_context,
        "custom_message entry should be in context"
    );

    let file_path = manager.session_file().expect("session file path");
    let file_text = fs::read_to_string(file_path).expect("read session file");
    assert!(file_text.contains("\"type\":\"thinking_level_change\""));
    assert!(file_text.contains("\"type\":\"model_change\""));
    assert!(file_text.contains("\"type\":\"custom\""));
    assert!(file_text.contains("\"type\":\"custom_message\""));
    assert!(file_text.contains("\"type\":\"label\""));
    assert!(file_text.contains("\"type\":\"session_info\""));
}
