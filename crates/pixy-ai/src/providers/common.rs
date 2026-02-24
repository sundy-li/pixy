use std::sync::OnceLock;

use reqwest::Client;

use crate::types::{AssistantMessage, Cost, Model, StopReason, Usage};

pub(super) fn empty_assistant_message(model: &Model) -> AssistantMessage {
    AssistantMessage {
        role: "assistant".to_string(),
        content: Vec::new(),
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
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
        stop_reason: StopReason::Stop,
        error_message: None,
        timestamp: now_millis(),
    }
}

pub(super) fn join_url(base_url: &str, path: &str) -> String {
    if base_url.ends_with('/') {
        format!("{base_url}{path}")
    } else {
        format!("{base_url}/{path}")
    }
}

pub(super) fn shared_http_client(base_url: &str) -> &'static Client {
    static DEFAULT_CLIENT: OnceLock<Client> = OnceLock::new();
    static LOOPBACK_CLIENT: OnceLock<Client> = OnceLock::new();

    if is_loopback_base_url(base_url) {
        LOOPBACK_CLIENT.get_or_init(|| {
            Client::builder()
                .no_proxy()
                .build()
                .unwrap_or_else(|_| Client::new())
        })
    } else {
        DEFAULT_CLIENT.get_or_init(Client::new)
    }
}

pub(super) fn is_loopback_base_url(base_url: &str) -> bool {
    let Ok(url) = reqwest::Url::parse(base_url) else {
        return false;
    };
    let Some(host) = url.host_str() else {
        return false;
    };
    host.eq_ignore_ascii_case("localhost") || host == "127.0.0.1" || host == "::1"
}

pub(super) fn now_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0)
}
