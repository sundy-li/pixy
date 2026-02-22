use std::time::Duration;

use pi_ai::{
    AssistantContentBlock, AssistantMessage, AssistantMessageEvent, AssistantMessageEventStream,
    Cost, DoneReason, StopReason, Usage,
};
use serde_json::json;
use tokio::time::timeout;

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

fn sample_message(stop_reason: StopReason) -> AssistantMessage {
    AssistantMessage {
        role: "assistant".to_string(),
        content: vec![AssistantContentBlock::Text {
            text: "hello".to_string(),
            text_signature: None,
        }],
        api: "openai-completions".to_string(),
        provider: "openai".to_string(),
        model: "gpt-4o-mini".to_string(),
        usage: sample_usage(),
        stop_reason,
        error_message: None,
        timestamp: 1_700_000_000_000,
    }
}

#[test]
fn assistant_message_serializes_camel_case_fields() {
    let message = sample_message(StopReason::Stop);
    let value = serde_json::to_value(&message).expect("message should serialize");

    assert_eq!(value["role"], "assistant");
    assert_eq!(value["stopReason"], "stop");
    assert!(value.get("stop_reason").is_none());
    assert!(value.get("errorMessage").is_none());
}

#[tokio::test]
async fn assistant_event_stream_result_returns_done_message() {
    let stream = AssistantMessageEventStream::new();
    let partial = sample_message(StopReason::Stop);
    let final_message = sample_message(StopReason::Stop);

    stream.push(AssistantMessageEvent::Start {
        partial: partial.clone(),
    });
    stream.push(AssistantMessageEvent::Done {
        reason: DoneReason::Stop,
        message: final_message.clone(),
    });

    let result = timeout(Duration::from_millis(100), stream.result())
        .await
        .expect("result should resolve")
        .expect("result should be available");
    assert_eq!(result.stop_reason, StopReason::Stop);

    let first = stream.next().await.expect("first event");
    let second = stream.next().await.expect("second event");
    let third = stream.next().await;

    assert!(matches!(first, AssistantMessageEvent::Start { .. }));
    assert!(matches!(second, AssistantMessageEvent::Done { .. }));
    assert!(
        third.is_none(),
        "stream should be closed after terminal event"
    );
}

#[tokio::test]
async fn assistant_event_stream_ignores_push_after_terminal_event() {
    let stream = AssistantMessageEventStream::new();
    let message = sample_message(StopReason::Stop);

    stream.push(AssistantMessageEvent::Done {
        reason: DoneReason::Stop,
        message: message.clone(),
    });
    stream.push(AssistantMessageEvent::TextDelta {
        content_index: 0,
        delta: "ignored".to_string(),
        partial: message.clone(),
    });

    let first = stream
        .next()
        .await
        .expect("terminal event should be present");
    let second = stream.next().await;

    assert!(matches!(first, AssistantMessageEvent::Done { .. }));
    assert!(second.is_none(), "post-terminal pushes must be ignored");
}

#[test]
fn assistant_event_type_serialization_matches_protocol_literals() {
    let message = sample_message(StopReason::Stop);
    let event = AssistantMessageEvent::TextDelta {
        content_index: 0,
        delta: "abc".to_string(),
        partial: message,
    };

    let value = serde_json::to_value(event).expect("event should serialize");
    assert_eq!(value["type"], "text_delta");
    assert_eq!(value["contentIndex"], json!(0));
}
