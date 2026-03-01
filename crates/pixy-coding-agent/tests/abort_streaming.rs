use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use pixy_agent_core::AgentAbortController;
use pixy_ai::{
    AssistantContentBlock, AssistantMessage, AssistantMessageEvent, AssistantMessageEventStream,
    Context, Cost, DoneReason, Message, Model, StopReason, Usage,
};
use pixy_coding_agent::{
    AgentSession, AgentSessionConfig, AgentSessionStreamUpdate, SessionManager,
};
use tempfile::tempdir;

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

fn sample_model() -> Model {
    Model {
        id: "test-model".to_string(),
        name: "Test Model".to_string(),
        api: "test-api".to_string(),
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
    content: Vec<AssistantContentBlock>,
    stop_reason: StopReason,
    timestamp: i64,
) -> AssistantMessage {
    AssistantMessage {
        role: "assistant".to_string(),
        content,
        api: "test-api".to_string(),
        provider: "test".to_string(),
        model: "test-model".to_string(),
        usage: sample_usage(),
        stop_reason,
        error_message: None,
        timestamp,
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

fn hanging_stream(partial: AssistantMessage) -> AssistantMessageEventStream {
    let stream = AssistantMessageEventStream::new();
    stream.push(AssistantMessageEvent::Start { partial });
    stream
}

#[tokio::test]
async fn prompt_streaming_with_abort_returns_aborted_assistant() {
    let dir = tempdir().expect("tempdir");
    let stream_fn = Arc::new(
        move |_model: Model, _context: Context, _options: Option<pixy_ai::SimpleStreamOptions>| {
            let partial = assistant_message(
                vec![AssistantContentBlock::Text {
                    text: String::new(),
                    text_signature: None,
                }],
                StopReason::Stop,
                1_700_000_000_100,
            );
            Ok(hanging_stream(partial))
        },
    );

    let manager = SessionManager::create(
        dir.path().to_str().expect("cwd utf-8"),
        dir.path().join("sessions"),
    )
    .expect("create session manager");
    let config = AgentSessionConfig {
        model: sample_model(),
        system_prompt: "You are helpful".to_string(),
        stream_fn,
        tools: vec![],
    };
    let mut session = AgentSession::new(manager, config);

    let controller = AgentAbortController::new();
    let signal = controller.signal();
    let abort_task = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(30)).await;
        controller.abort();
    });

    let mut updates = Vec::new();
    let produced = session
        .prompt_streaming_with_abort("hi", Some(signal), |update| updates.push(update))
        .await
        .expect("prompt succeeds with aborted assistant");
    abort_task.await.expect("abort task completes");

    assert!(produced.iter().any(|message| matches!(
        message,
        Message::Assistant {
            stop_reason: StopReason::Aborted,
            error_message: Some(error),
            ..
        } if error == "Request was aborted"
    )));
    assert!(updates.contains(&AgentSessionStreamUpdate::AssistantLine(
        "[assistant_aborted] Request was aborted".to_string()
    )));
}

#[tokio::test]
async fn continue_streaming_with_abort_returns_aborted_assistant() {
    let dir = tempdir().expect("tempdir");
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_in_fn = calls.clone();
    let stream_fn = Arc::new(
        move |_model: Model, _context: Context, _options: Option<pixy_ai::SimpleStreamOptions>| {
            let call = calls_in_fn.fetch_add(1, Ordering::SeqCst);
            if call == 0 {
                let message = assistant_message(
                    vec![AssistantContentBlock::Text {
                        text: "ready".to_string(),
                        text_signature: None,
                    }],
                    StopReason::Stop,
                    1_700_000_000_200,
                );
                return Ok(done_stream(message, DoneReason::Stop));
            }

            let partial = assistant_message(
                vec![AssistantContentBlock::Text {
                    text: String::new(),
                    text_signature: None,
                }],
                StopReason::Stop,
                1_700_000_000_210,
            );
            Ok(hanging_stream(partial))
        },
    );

    let manager = SessionManager::create(
        dir.path().to_str().expect("cwd utf-8"),
        dir.path().join("sessions"),
    )
    .expect("create session manager");
    let config = AgentSessionConfig {
        model: sample_model(),
        system_prompt: "You are helpful".to_string(),
        stream_fn,
        tools: vec![],
    };
    let mut session = AgentSession::new(manager, config);
    let _ = session.prompt("hello").await.expect("seed prompt succeeds");

    let controller = AgentAbortController::new();
    let signal = controller.signal();
    let abort_task = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(30)).await;
        controller.abort();
    });

    let produced = session
        .continue_run_streaming_with_abort(Some(signal), |_| {})
        .await
        .expect("continue succeeds with aborted assistant");
    abort_task.await.expect("abort task completes");

    assert!(produced.iter().any(|message| matches!(
        message,
        Message::Assistant {
            stop_reason: StopReason::Aborted,
            ..
        }
    )));
}
