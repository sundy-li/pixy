use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use pixy_agent_core::{Agent, AgentConfig, AgentMessage, QueueMode};
use pixy_ai::{
    AssistantContentBlock, AssistantMessage, AssistantMessageEvent, AssistantMessageEventStream,
    Context, Cost, DoneReason, Message, Model, StopReason, Usage, UserContent,
};
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

fn user_message(text: &str, ts: i64) -> AgentMessage {
    Message::User {
        content: UserContent::Text(text.to_string()),
        timestamp: ts,
    }
}

fn assistant_message(text: &str, ts: i64) -> AssistantMessage {
    AssistantMessage {
        role: "assistant".to_string(),
        content: vec![AssistantContentBlock::Text {
            text: text.to_string(),
            text_signature: None,
        }],
        api: "test-api".to_string(),
        provider: "test".to_string(),
        model: "test-model".to_string(),
        usage: sample_usage(),
        stop_reason: StopReason::Stop,
        error_message: None,
        timestamp: ts,
    }
}

fn done_stream(message: AssistantMessage) -> AssistantMessageEventStream {
    let stream = AssistantMessageEventStream::new();
    stream.push(AssistantMessageEvent::Start {
        partial: message.clone(),
    });
    stream.push(AssistantMessageEvent::Done {
        reason: DoneReason::Stop,
        message,
    });
    stream
}

fn user_texts(messages: &[AgentMessage]) -> Vec<String> {
    messages
        .iter()
        .filter_map(|message| match message {
            Message::User {
                content: UserContent::Text(text),
                ..
            } => Some(text.clone()),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn agent_continue_consumes_follow_up_queue_one_at_a_time_by_default() {
    let stream_calls = Arc::new(AtomicUsize::new(0));
    let stream_calls_in_fn = stream_calls.clone();
    let stream_fn = Arc::new(
        move |_model: Model, _context: Context, _options: Option<pixy_ai::SimpleStreamOptions>| {
            let index = stream_calls_in_fn.fetch_add(1, Ordering::SeqCst);
            Ok(done_stream(assistant_message(
                &format!("ack-{index}"),
                1_700_000_000_010 + index as i64,
            )))
        },
    );

    let config = AgentConfig::new(
        "You are helpful".to_string(),
        sample_model("test-api"),
        stream_fn,
    );
    let agent = Agent::new(config);

    let _ = agent
        .prompt_text("hello")
        .await
        .expect("initial prompt should succeed");
    agent.follow_up(user_message("follow-up-1", 1_700_000_000_100));
    agent.follow_up(user_message("follow-up-2", 1_700_000_000_200));

    let continue_result = agent.continue_run().await.expect("continue should succeed");

    assert_eq!(
        user_texts(&continue_result),
        vec!["follow-up-1", "follow-up-2"]
    );
    let continue_error = agent
        .continue_run()
        .await
        .expect_err("no queued messages should make continue fail");
    assert_eq!(
        continue_error,
        "Cannot continue from message role: assistant"
    );
    assert!(!agent.has_queued_messages());
}

#[tokio::test]
async fn agent_follow_up_mode_all_drains_all_follow_ups_in_single_turn() {
    let stream_fn = Arc::new(
        |_model: Model, _context: Context, _options: Option<pixy_ai::SimpleStreamOptions>| {
            Ok(done_stream(assistant_message("ok", 1_700_000_000_010)))
        },
    );

    let config = AgentConfig::new(
        "You are helpful".to_string(),
        sample_model("test-api"),
        stream_fn,
    );
    let agent = Agent::new(config);
    agent.set_follow_up_mode(QueueMode::All);

    let _ = agent
        .prompt_text("hello")
        .await
        .expect("initial prompt should succeed");
    agent.follow_up(user_message("follow-up-1", 1_700_000_000_100));
    agent.follow_up(user_message("follow-up-2", 1_700_000_000_200));

    let continue_result = agent.continue_run().await.expect("continue should succeed");
    assert_eq!(
        user_texts(&continue_result),
        vec!["follow-up-1", "follow-up-2"],
        "all queued follow-up messages should be consumed in one turn"
    );
    assert!(!agent.has_queued_messages());
}

#[tokio::test]
async fn agent_continue_prioritizes_steering_messages_over_follow_up_messages() {
    let stream_fn = Arc::new(
        |_model: Model, _context: Context, _options: Option<pixy_ai::SimpleStreamOptions>| {
            Ok(done_stream(assistant_message("ok", 1_700_000_000_010)))
        },
    );

    let config = AgentConfig::new(
        "You are helpful".to_string(),
        sample_model("test-api"),
        stream_fn,
    );
    let agent = Agent::new(config);

    let _ = agent
        .prompt_text("hello")
        .await
        .expect("initial prompt should succeed");
    agent.steer(user_message("steer-now", 1_700_000_000_100));
    agent.follow_up(user_message("follow-later", 1_700_000_000_200));

    let continue_result = agent.continue_run().await.expect("continue should succeed");

    assert_eq!(
        user_texts(&continue_result),
        vec!["steer-now", "follow-later"]
    );
    let continue_error = agent
        .continue_run()
        .await
        .expect_err("queues are already drained by previous continue");
    assert_eq!(
        continue_error,
        "Cannot continue from message role: assistant"
    );
}

#[tokio::test]
async fn agent_abort_interrupts_running_prompt_and_wait_for_idle_unblocks() {
    let stream_fn = Arc::new(
        |_model: Model, _context: Context, _options: Option<pixy_ai::SimpleStreamOptions>| {
            let stream = AssistantMessageEventStream::new();
            stream.push(AssistantMessageEvent::Start {
                partial: assistant_message("partial", 1_700_000_000_010),
            });
            Ok(stream)
        },
    );

    let config = AgentConfig::new(
        "You are helpful".to_string(),
        sample_model("test-api"),
        stream_fn,
    );
    let agent = Agent::new(config);
    let running_agent = agent.clone();

    let prompt_handle = tokio::spawn(async move { running_agent.prompt_text("hello").await });

    for _ in 0..30 {
        if agent.state().is_streaming {
            break;
        }
        sleep(Duration::from_millis(10)).await;
    }
    assert!(
        agent.state().is_streaming,
        "agent should be streaming before abort"
    );

    agent.abort();
    agent.wait_for_idle().await;

    let produced = prompt_handle
        .await
        .expect("prompt task should join")
        .expect("prompt should resolve after abort");
    assert!(
        matches!(
            produced.last(),
            Some(Message::Assistant {
                stop_reason: StopReason::Aborted,
                ..
            })
        ),
        "final assistant message should be aborted"
    );
    assert!(
        !agent.state().is_streaming,
        "wait_for_idle should observe idle state"
    );
}
