use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use pixy_agent_core::StreamFn;
use pixy_ai::{Cost, Message, Model, UserContent};
use pixy_coding_agent::{AgentSession, AgentSessionConfig, SessionManager};

fn sample_model() -> Model {
    Model {
        id: "test-model".to_string(),
        name: "Test Model".to_string(),
        api: "openai-completions".to_string(),
        provider: "openai".to_string(),
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

fn sample_stream_fn() -> StreamFn {
    Arc::new(|_model, _context, _options| Err("unused in resume tests".to_string()))
}

fn create_session_with_user_message(
    session_dir: &Path,
    cwd: &Path,
    text: &str,
) -> Result<SessionManager, String> {
    let cwd_text = cwd
        .to_str()
        .ok_or_else(|| format!("cwd is not valid UTF-8: {}", cwd.display()))?;
    let mut manager = SessionManager::create(cwd_text, session_dir)?;
    manager.append_message(Message::User {
        content: UserContent::Text(text.to_string()),
        timestamp: 1_700_000_000_000,
    })?;
    Ok(manager)
}

#[test]
fn resume_without_argument_loads_latest_history_session() {
    let dir = tempfile::tempdir().expect("tempdir");
    let session_dir = dir.path().join("sessions");

    let manager_old = create_session_with_user_message(&session_dir, dir.path(), "old session")
        .expect("create old session");
    let old_file = manager_old
        .session_file()
        .expect("old session path")
        .clone();

    std::thread::sleep(Duration::from_millis(2));

    let manager_current =
        create_session_with_user_message(&session_dir, dir.path(), "current session")
            .expect("create current session");

    let mut session = AgentSession::new(
        manager_current,
        AgentSessionConfig {
            model: sample_model(),
            system_prompt: "test".to_string(),
            stream_fn: sample_stream_fn(),
            tools: vec![],
        },
    );

    let resumed = session.resume(None).expect("resume should succeed");
    assert_eq!(resumed, old_file);
    let context = session.build_session_context();
    assert!(
        context.messages.iter().any(|message| {
            matches!(
                message,
                Message::User {
                    content: UserContent::Text(text),
                    ..
                } if text == "old session"
            )
        }),
        "resumed context should come from historical session"
    );
}

#[test]
fn resume_with_file_name_loads_target_session() {
    let dir = tempfile::tempdir().expect("tempdir");
    let session_dir = dir.path().join("sessions");

    let manager_target =
        create_session_with_user_message(&session_dir, dir.path(), "target session")
            .expect("create target session");
    let target_file = manager_target
        .session_file()
        .expect("target session path")
        .clone();
    let target_name = target_file
        .file_name()
        .and_then(|name| name.to_str())
        .expect("target file name")
        .to_string();

    std::thread::sleep(Duration::from_millis(2));

    let manager_current =
        create_session_with_user_message(&session_dir, dir.path(), "current session")
            .expect("create current session");
    let mut session = AgentSession::new(
        manager_current,
        AgentSessionConfig {
            model: sample_model(),
            system_prompt: "test".to_string(),
            stream_fn: sample_stream_fn(),
            tools: vec![],
        },
    );

    let resumed = session
        .resume(Some(&target_name))
        .expect("resume by file name should succeed");
    assert_eq!(resumed, target_file);
}
