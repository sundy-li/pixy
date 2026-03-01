use std::sync::Arc;
use std::sync::{Mutex, OnceLock};

use pixy_ai::{
    clear_api_providers, complete, complete_simple, register_api_provider, stream, stream_simple,
    unregister_api_providers, AssistantContentBlock, AssistantMessage, AssistantMessageEvent,
    AssistantMessageEventStream, ClosureApiProvider, Context, Cost, DoneReason, Message, Model,
    SimpleStreamOptions, StopReason, StreamOptions, Usage, UserContent,
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

fn emit_done(stream: &AssistantMessageEventStream, text: &str) {
    stream.push(AssistantMessageEvent::Done {
        reason: DoneReason::Stop,
        message: sample_assistant(StopReason::Stop, text),
    });
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
            stream: Arc::new(|_, _, _, stream| {
                Box::pin(async move {
                    emit_done(&stream, "from-stream");
                    Ok(())
                })
            }),
            stream_simple: Arc::new(|_, _, _, stream| {
                Box::pin(async move {
                    emit_done(&stream, "from-simple-stream");
                    Ok(())
                })
            }),
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

#[tokio::test]
async fn unregister_api_providers_removes_only_matching_source_id() {
    let _guard = registry_guard();
    clear_api_providers();
    register_api_provider(
        Arc::new(ClosureApiProvider {
            api: "api-a".to_string(),
            stream: Arc::new(|_, _, _, stream| {
                Box::pin(async move {
                    emit_done(&stream, "a");
                    Ok(())
                })
            }),
            stream_simple: Arc::new(|_, _, _, stream| {
                Box::pin(async move {
                    emit_done(&stream, "a");
                    Ok(())
                })
            }),
        }),
        Some("source-a".to_string()),
    );
    register_api_provider(
        Arc::new(ClosureApiProvider {
            api: "api-b".to_string(),
            stream: Arc::new(|_, _, _, stream| {
                Box::pin(async move {
                    emit_done(&stream, "b");
                    Ok(())
                })
            }),
            stream_simple: Arc::new(|_, _, _, stream| {
                Box::pin(async move {
                    emit_done(&stream, "b");
                    Ok(())
                })
            }),
        }),
        Some("source-b".to_string()),
    );

    unregister_api_providers("source-a");

    assert!(stream(sample_model("api-a"), sample_context(), None).is_err());
    assert!(stream(sample_model("api-b"), sample_context(), None).is_ok());
}

#[tokio::test]
async fn stream_forwarding_preserves_transport_retry_override() {
    let _guard = registry_guard();
    clear_api_providers();
    let seen_stream_retry = Arc::new(Mutex::new(Vec::new()));
    let seen_simple_retry = Arc::new(Mutex::new(Vec::new()));
    let seen_stream_retry_for_provider = Arc::clone(&seen_stream_retry);
    let seen_simple_retry_for_provider = Arc::clone(&seen_simple_retry);

    register_api_provider(
        Arc::new(ClosureApiProvider {
            api: "test-api".to_string(),
            stream: Arc::new(move |_, _, options, stream| {
                seen_stream_retry_for_provider
                    .lock()
                    .expect("stream retry lock poisoned")
                    .push(options.and_then(|stream_options| stream_options.transport_retry_count));
                Box::pin(async move {
                    emit_done(&stream, "stream");
                    Ok(())
                })
            }),
            stream_simple: Arc::new(move |_, _, options, stream| {
                seen_simple_retry_for_provider
                    .lock()
                    .expect("simple retry lock poisoned")
                    .push(
                        options
                            .and_then(|simple_options| simple_options.stream.transport_retry_count),
                    );
                Box::pin(async move {
                    emit_done(&stream, "simple");
                    Ok(())
                })
            }),
        }),
        Some("test-source".to_string()),
    );

    let model = sample_model("test-api");
    let context = sample_context();
    let stream_events = stream(
        model.clone(),
        context.clone(),
        Some(StreamOptions {
            api_key: None,
            temperature: None,
            max_tokens: None,
            headers: None,
            transport_retry_count: Some(7),
        }),
    )
    .expect("stream should resolve");
    let _ = stream_events.result().await.expect("stream should finish");

    let simple_stream_events = stream_simple(
        model,
        context,
        Some(SimpleStreamOptions {
            stream: StreamOptions {
                api_key: None,
                temperature: None,
                max_tokens: None,
                headers: None,
                transport_retry_count: Some(3),
            },
            reasoning: None,
        }),
    )
    .expect("stream_simple should resolve");
    let _ = simple_stream_events
        .result()
        .await
        .expect("stream_simple should finish");

    assert_eq!(
        seen_stream_retry
            .lock()
            .expect("stream retry lock poisoned")
            .as_slice(),
        &[Some(7)]
    );
    assert_eq!(
        seen_simple_retry
            .lock()
            .expect("simple retry lock poisoned")
            .as_slice(),
        &[Some(3)]
    );
}
