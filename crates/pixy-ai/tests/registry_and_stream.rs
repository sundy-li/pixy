use std::sync::Arc;
use std::sync::{Mutex, OnceLock};

use pixy_ai::{
    AssistantContentBlock, AssistantMessage, AssistantMessageEvent, AssistantMessageEventStream,
    ClosureApiProvider, Context, Cost, DoneReason, Message, Model, StopReason, Usage, UserContent,
    clear_api_providers, complete, complete_simple, register_api_provider, stream, stream_simple,
    unregister_api_providers,
};

fn sample_usage() -> Usage {
    Usage {
        input: 10,
        output: 2,
        cache_read: 0,
        cache_write: 0,
        total_tokens: 12,
        cost: Cost {
            input: 0.01,
            output: 0.02,
            cache_read: 0.0,
            cache_write: 0.0,
            total: 0.03,
        },
    }
}

fn sample_assistant(stop_reason: StopReason, text: &str) -> AssistantMessage {
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
        stop_reason,
        error_message: None,
        timestamp: 1_700_000_000_000,
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

fn sample_context() -> Context {
    Context {
        system_prompt: Some("You are a test assistant.".to_string()),
        messages: vec![Message::User {
            content: UserContent::Text("hi".to_string()),
            timestamp: 1_700_000_000_000,
        }],
        tools: None,
    }
}

fn done_stream(text: &str) -> AssistantMessageEventStream {
    let stream = AssistantMessageEventStream::new();
    stream.push(AssistantMessageEvent::Done {
        reason: DoneReason::Stop,
        message: sample_assistant(StopReason::Stop, text),
    });
    stream
}

fn registry_guard() -> std::sync::MutexGuard<'static, ()> {
    static TEST_GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    TEST_GUARD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .expect("test guard lock poisoned")
}

#[tokio::test]
async fn stream_and_complete_use_registered_provider() {
    let _guard = registry_guard();
    clear_api_providers();

    register_api_provider(
        Arc::new(ClosureApiProvider {
            api: "test-api".to_string(),
            stream: Arc::new(|_, _, _| Ok(done_stream("from-stream"))),
            stream_simple: Arc::new(|_, _, _| Ok(done_stream("from-simple-stream"))),
        }),
        Some("test-source".to_string()),
    );

    let model = sample_model("test-api");
    let context = sample_context();

    let streamed = stream(model.clone(), context.clone(), None).expect("stream should resolve");
    let streamed_result = streamed
        .result()
        .await
        .expect("stream should return final message");
    assert_eq!(
        streamed_result.content,
        vec![AssistantContentBlock::Text {
            text: "from-stream".to_string(),
            text_signature: None
        }]
    );

    let completed = complete(model.clone(), context.clone(), None)
        .await
        .expect("complete should resolve");
    assert_eq!(
        completed.content,
        vec![AssistantContentBlock::Text {
            text: "from-stream".to_string(),
            text_signature: None
        }]
    );

    let simple_streamed =
        stream_simple(model.clone(), context.clone(), None).expect("stream_simple should resolve");
    let simple_streamed_result = simple_streamed
        .result()
        .await
        .expect("stream_simple should return final message");
    assert_eq!(
        simple_streamed_result.content,
        vec![AssistantContentBlock::Text {
            text: "from-simple-stream".to_string(),
            text_signature: None
        }]
    );

    let simple_completed = complete_simple(model, context, None)
        .await
        .expect("complete_simple should resolve");
    assert_eq!(
        simple_completed.content,
        vec![AssistantContentBlock::Text {
            text: "from-simple-stream".to_string(),
            text_signature: None
        }]
    );
}

#[tokio::test]
async fn stream_fails_when_provider_is_missing() {
    let _guard = registry_guard();
    clear_api_providers();
    let result = stream(sample_model("missing-api"), sample_context(), None);
    assert!(result.is_err(), "missing provider should return error");
}

#[test]
fn unregister_api_providers_removes_only_matching_source_id() {
    let _guard = registry_guard();
    clear_api_providers();
    register_api_provider(
        Arc::new(ClosureApiProvider {
            api: "api-a".to_string(),
            stream: Arc::new(|_, _, _| Ok(done_stream("a"))),
            stream_simple: Arc::new(|_, _, _| Ok(done_stream("a"))),
        }),
        Some("source-a".to_string()),
    );
    register_api_provider(
        Arc::new(ClosureApiProvider {
            api: "api-b".to_string(),
            stream: Arc::new(|_, _, _| Ok(done_stream("b"))),
            stream_simple: Arc::new(|_, _, _| Ok(done_stream("b"))),
        }),
        Some("source-b".to_string()),
    );

    unregister_api_providers("source-a");

    assert!(stream(sample_model("api-a"), sample_context(), None).is_err());
    assert!(stream(sample_model("api-b"), sample_context(), None).is_ok());
}
