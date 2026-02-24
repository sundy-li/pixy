use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(not(test))]
use std::time::Duration;

pub const DEFAULT_TRANSPORT_RETRY_COUNT: usize = 5;

static TRANSPORT_RETRY_COUNT: AtomicUsize = AtomicUsize::new(DEFAULT_TRANSPORT_RETRY_COUNT);

pub fn set_transport_retry_count(retries: usize) {
    TRANSPORT_RETRY_COUNT.store(retries, Ordering::SeqCst);
}

pub fn transport_retry_count() -> usize {
    TRANSPORT_RETRY_COUNT.load(Ordering::SeqCst)
}

pub fn transport_retry_count_with_override(request_override: Option<usize>) -> usize {
    resolve_transport_retry_count(request_override, transport_retry_count())
}

fn resolve_transport_retry_count(request_override: Option<usize>, runtime_default: usize) -> usize {
    request_override.unwrap_or(runtime_default)
}

#[allow(dead_code)]
pub(crate) async fn retry_transport_operation_async<T, E, F, Fut>(
    retries: usize,
    mut operation: F,
) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
{
    let mut remaining_retries = retries;
    loop {
        match operation().await {
            Ok(result) => return Ok(result),
            Err(error) => {
                if remaining_retries == 0 {
                    return Err(error);
                }
                remaining_retries = remaining_retries.saturating_sub(1);
                sleep_retry_interval_async().await;
            }
        }
    }
}

#[allow(dead_code)]
async fn sleep_retry_interval_async() {
    #[cfg(not(test))]
    {
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[tokio::test]
    async fn transport_retry_async_retries_until_success() {
        let attempts = AtomicUsize::new(0);
        let retry_count = 3usize;

        let result = retry_transport_operation_async(retry_count, || {
            let current = attempts.fetch_add(1, Ordering::SeqCst);
            async move {
                if current < retry_count {
                    Err("transport")
                } else {
                    Ok("ok")
                }
            }
        })
        .await;

        assert!(result.is_ok());
        assert_eq!(attempts.load(Ordering::SeqCst), retry_count + 1);
    }

    #[tokio::test]
    async fn transport_retry_async_returns_last_error_after_exhaustion() {
        let attempts = AtomicUsize::new(0);
        let retry_count = 2usize;

        let result: Result<(), &'static str> = retry_transport_operation_async(retry_count, || {
            attempts.fetch_add(1, Ordering::SeqCst);
            async { Err("transport") }
        })
        .await;

        assert_eq!(result, Err("transport"));
        assert_eq!(attempts.load(Ordering::SeqCst), retry_count + 1);
    }

    #[test]
    fn resolve_transport_retry_count_prefers_request_override() {
        assert_eq!(resolve_transport_retry_count(Some(2), 8), 2);
    }

    #[test]
    fn resolve_transport_retry_count_uses_runtime_default_when_no_override() {
        assert_eq!(resolve_transport_retry_count(None, 7), 7);
    }

    #[test]
    fn resolve_transport_retry_count_supports_constant_default_fallback() {
        assert_eq!(
            resolve_transport_retry_count(None, DEFAULT_TRANSPORT_RETRY_COUNT),
            DEFAULT_TRANSPORT_RETRY_COUNT
        );
    }
}
