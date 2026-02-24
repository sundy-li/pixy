use std::sync::Arc;

#[cfg(not(test))]
use tokio::time::{Duration, sleep};

use crate::AssistantMessageEventStream;
use crate::api_registry::{ApiProvider, ApiProviderFuture};
use crate::error::{PiAiError, PiAiErrorCode};
use crate::transport_retry::{DEFAULT_TRANSPORT_RETRY_COUNT, transport_retry_count};
use crate::types::{
    AssistantMessage, AssistantMessageEvent, Context, Model, SimpleStreamOptions, StopReason,
    StreamOptions,
};

pub struct ReliableProvider {
    inner: Arc<dyn ApiProvider>,
    max_retries: u32,
    base_backoff_ms: u64,
}

impl ReliableProvider {
    pub fn wrap(inner: Arc<dyn ApiProvider>) -> Self {
        Self {
            inner,
            max_retries: DEFAULT_TRANSPORT_RETRY_COUNT as u32,
            base_backoff_ms: 1_000,
        }
    }

    pub fn max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    pub fn base_backoff_ms(mut self, base_backoff_ms: u64) -> Self {
        self.base_backoff_ms = base_backoff_ms;
        self
    }

    fn resolve_retry_count(&self, request_override: Option<usize>) -> u32 {
        let request_override = request_override.map(|value| value.min(u32::MAX as usize) as u32);
        request_override.unwrap_or_else(|| {
            if self.max_retries == DEFAULT_TRANSPORT_RETRY_COUNT as u32 {
                transport_retry_count().min(u32::MAX as usize) as u32
            } else {
                self.max_retries
            }
        })
    }
}

impl ApiProvider for ReliableProvider {
    fn api(&self) -> &str {
        self.inner.api()
    }

    fn stream(
        &self,
        model: Model,
        context: Context,
        options: Option<StreamOptions>,
        stream: AssistantMessageEventStream,
    ) -> ApiProviderFuture {
        let inner = self.inner.clone();
        let provider_api = self.api().to_string();
        let max_retries = self.resolve_retry_count(
            options
                .as_ref()
                .and_then(|stream_options| stream_options.transport_retry_count),
        );
        let base_backoff_ms = self.base_backoff_ms;
        Box::pin(async move {
            run_with_retry(
                provider_api,
                max_retries,
                base_backoff_ms,
                stream,
                move |attempt_stream| {
                    let inner = inner.clone();
                    let model = model.clone();
                    let context = context.clone();
                    let options = options.clone();
                    async move { inner.stream(model, context, options, attempt_stream).await }
                },
            )
            .await
        })
    }

    fn stream_simple(
        &self,
        model: Model,
        context: Context,
        options: Option<SimpleStreamOptions>,
        stream: AssistantMessageEventStream,
    ) -> ApiProviderFuture {
        let inner = self.inner.clone();
        let provider_api = self.api().to_string();
        let max_retries = self.resolve_retry_count(
            options
                .as_ref()
                .and_then(|simple_options| simple_options.stream.transport_retry_count),
        );
        let base_backoff_ms = self.base_backoff_ms;
        Box::pin(async move {
            run_with_retry(
                provider_api,
                max_retries,
                base_backoff_ms,
                stream,
                move |attempt_stream| {
                    let inner = inner.clone();
                    let model = model.clone();
                    let context = context.clone();
                    let options = options.clone();
                    async move {
                        inner
                            .stream_simple(model, context, options, attempt_stream)
                            .await
                    }
                },
            )
            .await
        })
    }
}

enum AttemptStatus {
    Success,
    Failure {
        error: PiAiError,
        retryable: bool,
        terminal_emitted: bool,
    },
}

async fn run_with_retry<F, Fut>(
    provider_api: String,
    max_retries: u32,
    base_backoff_ms: u64,
    output_stream: AssistantMessageEventStream,
    mut operation: F,
) -> Result<(), PiAiError>
where
    F: FnMut(AssistantMessageEventStream) -> Fut,
    Fut: std::future::Future<Output = Result<(), PiAiError>>,
{
    let mut retries_used = 0u32;

    loop {
        let attempt_stream = AssistantMessageEventStream::new();
        let attempt_result = operation(attempt_stream.clone()).await;
        attempt_stream.end(None);
        let attempt_events = drain_events(&attempt_stream).await;

        match classify_attempt(&provider_api, &attempt_result, &attempt_events) {
            AttemptStatus::Success => {
                replay_events(&output_stream, attempt_events);
                return Ok(());
            }
            AttemptStatus::Failure {
                error,
                retryable,
                terminal_emitted,
            } => {
                if retryable && retries_used < max_retries {
                    sleep_backoff(base_backoff_ms, retries_used).await;
                    retries_used += 1;
                    continue;
                }

                if terminal_emitted {
                    replay_events(&output_stream, attempt_events);
                    return Ok(());
                }
                return Err(error);
            }
        }
    }
}

fn classify_attempt(
    provider_api: &str,
    attempt_result: &Result<(), PiAiError>,
    attempt_events: &[AssistantMessageEvent],
) -> AttemptStatus {
    if let Some(terminal) = attempt_events.iter().rev().find(|event| {
        matches!(
            event,
            AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. }
        )
    }) {
        return match terminal {
            AssistantMessageEvent::Done { .. } => AttemptStatus::Success,
            AssistantMessageEvent::Error { error, .. } => {
                let parsed = parse_error_message(error).unwrap_or_else(|| {
                    PiAiError::new(
                        PiAiErrorCode::ProviderProtocol,
                        format!(
                            "Provider '{provider_api}' emitted an error event without structured error_message"
                        ),
                    )
                });
                AttemptStatus::Failure {
                    retryable: parsed.code == PiAiErrorCode::ProviderTransport,
                    error: parsed,
                    terminal_emitted: true,
                }
            }
            _ => unreachable!(),
        };
    }

    if let Err(error) = attempt_result {
        return AttemptStatus::Failure {
            retryable: error.code == PiAiErrorCode::ProviderTransport,
            error: error.clone(),
            terminal_emitted: false,
        };
    }

    AttemptStatus::Failure {
        retryable: false,
        error: PiAiError::new(
            PiAiErrorCode::ProviderProtocol,
            format!("Provider '{provider_api}' returned without a terminal event"),
        ),
        terminal_emitted: false,
    }
}

fn parse_error_message(error: &AssistantMessage) -> Option<PiAiError> {
    if error.stop_reason != StopReason::Error {
        return None;
    }
    error
        .error_message
        .as_deref()
        .and_then(|value| serde_json::from_str::<PiAiError>(value).ok())
}

async fn drain_events(stream: &AssistantMessageEventStream) -> Vec<AssistantMessageEvent> {
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }
    events
}

fn replay_events(output_stream: &AssistantMessageEventStream, events: Vec<AssistantMessageEvent>) {
    for event in events {
        output_stream.push(event);
    }
}

#[cfg(not(test))]
async fn sleep_backoff(base_backoff_ms: u64, retry_index: u32) {
    let shift = retry_index.min(63);
    let multiplier = 1u64.checked_shl(shift).unwrap_or(u64::MAX);
    let delay_ms = base_backoff_ms.saturating_mul(multiplier);
    sleep(Duration::from_millis(delay_ms)).await;
}

#[cfg(test)]
async fn sleep_backoff(_base_backoff_ms: u64, _retry_index: u32) {}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::api_registry::ApiProviderFuture;
    use crate::types::{
        AssistantContentBlock, Cost, DoneReason, ErrorReason, Message, Usage, UserContent,
    };

    #[derive(Clone)]
    struct TestProvider {
        api: &'static str,
        attempts: Arc<AtomicUsize>,
        behavior:
            Arc<dyn Fn(usize, AssistantMessageEventStream) -> ApiProviderFuture + Send + Sync>,
    }

    impl ApiProvider for TestProvider {
        fn api(&self) -> &str {
            self.api
        }

        fn stream(
            &self,
            _model: Model,
            _context: Context,
            _options: Option<StreamOptions>,
            stream: AssistantMessageEventStream,
        ) -> ApiProviderFuture {
            let attempt = self.attempts.fetch_add(1, Ordering::SeqCst);
            (self.behavior)(attempt, stream)
        }

        fn stream_simple(
            &self,
            _model: Model,
            _context: Context,
            _options: Option<SimpleStreamOptions>,
            stream: AssistantMessageEventStream,
        ) -> ApiProviderFuture {
            let attempt = self.attempts.fetch_add(1, Ordering::SeqCst);
            (self.behavior)(attempt, stream)
        }
    }

    #[tokio::test]
    async fn reliable_provider_retries_transport_errors_and_discards_partial_attempts() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let provider = Arc::new(TestProvider {
            api: "test",
            attempts: attempts.clone(),
            behavior: Arc::new(|attempt, stream| {
                Box::pin(async move {
                    if attempt == 0 {
                        let mut message = assistant_message(
                            StopReason::Error,
                            Some(transport_error_json("fail-0")),
                        );
                        message.content.push(AssistantContentBlock::Text {
                            text: "partial-failed".to_string(),
                            text_signature: None,
                        });
                        stream.push(AssistantMessageEvent::Start {
                            partial: message.clone(),
                        });
                        stream.push(AssistantMessageEvent::Error {
                            reason: ErrorReason::Error,
                            error: message,
                        });
                        return Ok(());
                    }

                    let mut partial = assistant_message(StopReason::Stop, None);
                    partial.content.push(AssistantContentBlock::Text {
                        text: "ok".to_string(),
                        text_signature: None,
                    });
                    stream.push(AssistantMessageEvent::Start {
                        partial: partial.clone(),
                    });
                    stream.push(AssistantMessageEvent::Done {
                        reason: DoneReason::Stop,
                        message: partial,
                    });
                    Ok(())
                })
            }),
        });
        let reliable = ReliableProvider::wrap(provider)
            .max_retries(1)
            .base_backoff_ms(0);

        let out = AssistantMessageEventStream::new();
        let result = reliable
            .stream(sample_model("test"), sample_context(), None, out.clone())
            .await;
        out.end(None);
        let events = drain_events(&out).await;

        assert!(result.is_ok());
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        assert!(
            events
                .iter()
                .any(|event| matches!(event, AssistantMessageEvent::Done { .. }))
        );
        assert!(events.iter().all(|event| match event {
            AssistantMessageEvent::Start { partial } => !partial
                .content
                .iter()
                .any(|block| matches!(block, AssistantContentBlock::Text { text, .. } if text == "partial-failed")),
            _ => true,
        }));
    }

    #[tokio::test]
    async fn reliable_provider_does_not_retry_non_transport_errors() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let provider = Arc::new(TestProvider {
            api: "test",
            attempts: attempts.clone(),
            behavior: Arc::new(|_, stream| {
                Box::pin(async move {
                    let message = assistant_message(
                        StopReason::Error,
                        Some(
                            PiAiError::new(PiAiErrorCode::ProviderProtocol, "bad payload")
                                .as_compact_json(),
                        ),
                    );
                    stream.push(AssistantMessageEvent::Error {
                        reason: ErrorReason::Error,
                        error: message,
                    });
                    Ok(())
                })
            }),
        });
        let reliable = ReliableProvider::wrap(provider)
            .max_retries(3)
            .base_backoff_ms(0);

        let out = AssistantMessageEventStream::new();
        let result = reliable
            .stream(sample_model("test"), sample_context(), None, out.clone())
            .await;
        out.end(None);

        assert!(result.is_ok());
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn reliable_provider_honors_request_retry_override() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let provider = Arc::new(TestProvider {
            api: "test",
            attempts: attempts.clone(),
            behavior: Arc::new(|_, stream| {
                Box::pin(async move {
                    let message =
                        assistant_message(StopReason::Error, Some(transport_error_json("fail")));
                    stream.push(AssistantMessageEvent::Error {
                        reason: ErrorReason::Error,
                        error: message,
                    });
                    Ok(())
                })
            }),
        });
        let reliable = ReliableProvider::wrap(provider)
            .max_retries(5)
            .base_backoff_ms(0);

        let out = AssistantMessageEventStream::new();
        let options = StreamOptions {
            transport_retry_count: Some(0),
            ..StreamOptions::default()
        };
        let result = reliable
            .stream(
                sample_model("test"),
                sample_context(),
                Some(options),
                out.clone(),
            )
            .await;
        out.end(None);

        assert!(result.is_ok());
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }

    fn transport_error_json(message: &str) -> String {
        PiAiError::new(PiAiErrorCode::ProviderTransport, message).as_compact_json()
    }

    fn assistant_message(
        stop_reason: StopReason,
        error_message: Option<String>,
    ) -> AssistantMessage {
        AssistantMessage {
            role: "assistant".to_string(),
            content: vec![],
            api: "test".to_string(),
            provider: "test-provider".to_string(),
            model: "test-model".to_string(),
            usage: Usage {
                input: 0,
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
            stop_reason,
            error_message,
            timestamp: 0,
        }
    }

    fn sample_model(api: &str) -> Model {
        Model {
            id: "model".to_string(),
            name: "Model".to_string(),
            api: api.to_string(),
            provider: "provider".to_string(),
            base_url: "https://example.com".to_string(),
            reasoning: false,
            reasoning_effort: None,
            input: vec![],
            cost: Cost {
                input: 0.0,
                output: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
                total: 0.0,
            },
            context_window: 128_000,
            max_tokens: 4_096,
        }
    }

    fn sample_context() -> Context {
        Context {
            system_prompt: None,
            messages: vec![Message::User {
                content: UserContent::Text("hi".to_string()),
                timestamp: 0,
            }],
            tools: None,
        }
    }
}
