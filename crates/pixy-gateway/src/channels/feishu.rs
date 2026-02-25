use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::routing::post;
use axum::{Json, Router};
use reqwest::{Client, Proxy};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::{Mutex, mpsc};
use tokio::time::Instant;

use crate::channels::{Channel, ChannelFuture, SessionDispatcher};
use crate::config::FeishuChannelConfig;

const FEISHU_MESSAGE_TYPE_TEXT: &str = "text";
const FEISHU_CHAT_TYPE_PRIVATE: &str = "p2p";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeishuInboundMessage {
    pub message_id: String,
    pub chat_id: String,
    pub user_id: String,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct FeishuWebhookBinding {
    pub channel_name: String,
    pub verification_token: String,
    pub sender: mpsc::UnboundedSender<FeishuInboundMessage>,
}

pub struct FeishuChannel {
    name: String,
    client: FeishuClient,
    poll_interval: Duration,
    allowed_user_ids: HashSet<String>,
    next_poll_at: Instant,
    receiver: mpsc::UnboundedReceiver<FeishuInboundMessage>,
}

struct FeishuClient {
    client: Client,
    api_base: String,
    app_id: String,
    app_secret: String,
    token_cache: Mutex<Option<CachedTenantToken>>,
}

#[derive(Debug, Clone)]
struct CachedTenantToken {
    value: String,
    expires_at: Instant,
}

#[derive(Debug, Clone)]
struct FeishuWebhookState {
    bindings: Arc<HashMap<String, FeishuWebhookBinding>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum FeishuWebhookParseResult {
    Challenge {
        value: String,
    },
    Inbound {
        message: Option<FeishuInboundMessage>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum FeishuWebhookError {
    Unauthorized(String),
    BadRequest(String),
}

#[derive(Debug, Deserialize)]
struct FeishuWebhookEnvelope {
    #[serde(default)]
    token: Option<String>,
    #[serde(default, rename = "type")]
    kind: Option<String>,
    #[serde(default)]
    challenge: Option<String>,
    #[serde(default)]
    header: Option<FeishuWebhookHeader>,
    #[serde(default)]
    event: Option<FeishuWebhookEvent>,
}

#[derive(Debug, Deserialize)]
struct FeishuWebhookHeader {
    #[serde(default)]
    token: Option<String>,
    #[serde(default)]
    event_type: Option<String>,
}

#[derive(Debug, Deserialize)]
struct FeishuWebhookEvent {
    #[serde(default)]
    sender: Option<FeishuWebhookSender>,
    #[serde(default)]
    message: Option<FeishuWebhookMessage>,
}

#[derive(Debug, Deserialize)]
struct FeishuWebhookSender {
    #[serde(default)]
    sender_id: Option<FeishuWebhookSenderId>,
}

#[derive(Debug, Deserialize)]
struct FeishuWebhookSenderId {
    #[serde(default)]
    open_id: Option<String>,
    #[serde(default)]
    user_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct FeishuWebhookMessage {
    #[serde(default)]
    message_id: Option<String>,
    #[serde(default)]
    chat_id: Option<String>,
    #[serde(default)]
    chat_type: Option<String>,
    #[serde(default)]
    message_type: Option<String>,
    #[serde(default)]
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct FeishuMessageTextContent {
    #[serde(default)]
    text: Option<String>,
}

#[derive(Debug, Serialize)]
struct FeishuTenantAccessTokenRequest<'a> {
    app_id: &'a str,
    app_secret: &'a str,
}

#[derive(Debug, Deserialize)]
struct FeishuTenantAccessTokenResponse {
    #[serde(default)]
    code: i32,
    #[serde(default)]
    msg: Option<String>,
    #[serde(default)]
    tenant_access_token: Option<String>,
    #[serde(default)]
    expire: Option<u64>,
}

#[derive(Debug, Serialize)]
struct FeishuSendTextRequest {
    receive_id: String,
    msg_type: &'static str,
    content: String,
}

#[derive(Debug, Deserialize)]
struct FeishuApiStatusResponse {
    #[serde(default)]
    code: i32,
    #[serde(default)]
    msg: Option<String>,
}

fn build_send_text_request(chat_id: &str, text: &str) -> FeishuSendTextRequest {
    FeishuSendTextRequest {
        receive_id: chat_id.to_string(),
        msg_type: FEISHU_MESSAGE_TYPE_TEXT,
        content: serde_json::json!({ "text": text }).to_string(),
    }
}

fn parse_feishu_webhook_payload(
    payload: Value,
    expected_token: &str,
) -> Result<FeishuWebhookParseResult, FeishuWebhookError> {
    let envelope: FeishuWebhookEnvelope = serde_json::from_value(payload).map_err(|error| {
        FeishuWebhookError::BadRequest(format!("decode feishu payload failed: {error}"))
    })?;
    let token = envelope
        .header
        .as_ref()
        .and_then(|header| header.token.as_deref())
        .or(envelope.token.as_deref())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            FeishuWebhookError::Unauthorized("missing verification token".to_string())
        })?;
    if token != expected_token {
        return Err(FeishuWebhookError::Unauthorized(
            "invalid verification token".to_string(),
        ));
    }

    if envelope
        .kind
        .as_deref()
        .is_some_and(|kind| kind.eq_ignore_ascii_case("url_verification"))
    {
        let challenge = envelope.challenge.ok_or_else(|| {
            FeishuWebhookError::BadRequest("missing challenge in url_verification".to_string())
        })?;
        return Ok(FeishuWebhookParseResult::Challenge { value: challenge });
    }

    let event_type = envelope
        .header
        .as_ref()
        .and_then(|header| header.event_type.as_deref())
        .map(str::trim)
        .unwrap_or_default();
    if !event_type.eq_ignore_ascii_case("im.message.receive_v1") {
        return Ok(FeishuWebhookParseResult::Inbound { message: None });
    }

    let event = envelope
        .event
        .ok_or_else(|| FeishuWebhookError::BadRequest("missing event payload".to_string()))?;
    let message = event
        .message
        .ok_or_else(|| FeishuWebhookError::BadRequest("missing event.message".to_string()))?;
    if !message
        .chat_type
        .as_deref()
        .unwrap_or_default()
        .eq_ignore_ascii_case(FEISHU_CHAT_TYPE_PRIVATE)
    {
        return Ok(FeishuWebhookParseResult::Inbound { message: None });
    }
    if !message
        .message_type
        .as_deref()
        .unwrap_or_default()
        .eq_ignore_ascii_case(FEISHU_MESSAGE_TYPE_TEXT)
    {
        return Ok(FeishuWebhookParseResult::Inbound { message: None });
    }

    let text = message
        .content
        .as_deref()
        .and_then(parse_feishu_text_content)
        .ok_or_else(|| {
            FeishuWebhookError::BadRequest("event message has empty text content".to_string())
        })?;
    let user_id = event
        .sender
        .as_ref()
        .and_then(|sender| sender.sender_id.as_ref())
        .and_then(|sender_id| {
            sender_id
                .open_id
                .as_deref()
                .or(sender_id.user_id.as_deref())
                .map(str::trim)
        })
        .filter(|value| !value.is_empty())
        .ok_or_else(|| FeishuWebhookError::BadRequest("missing sender user id".to_string()))?;
    let message_id = message
        .message_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| FeishuWebhookError::BadRequest("missing message_id".to_string()))?;
    let chat_id = message
        .chat_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| FeishuWebhookError::BadRequest("missing chat_id".to_string()))?;

    Ok(FeishuWebhookParseResult::Inbound {
        message: Some(FeishuInboundMessage {
            message_id: message_id.to_string(),
            chat_id: chat_id.to_string(),
            user_id: user_id.to_string(),
            text,
        }),
    })
}

fn parse_feishu_text_content(raw: &str) -> Option<String> {
    let parsed: FeishuMessageTextContent = serde_json::from_str(raw).ok()?;
    parsed
        .text
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

async fn handle_feishu_webhook(
    State(state): State<FeishuWebhookState>,
    Path(channel_name): Path<String>,
    Json(payload): Json<Value>,
) -> (StatusCode, Json<Value>) {
    let Some(binding) = state.bindings.get(&channel_name) else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "code": 404,
                "msg": format!("unknown feishu channel '{channel_name}'"),
            })),
        );
    };

    match parse_feishu_webhook_payload(payload, &binding.verification_token) {
        Ok(FeishuWebhookParseResult::Challenge { value }) => (
            StatusCode::OK,
            Json(serde_json::json!({ "challenge": value })),
        ),
        Ok(FeishuWebhookParseResult::Inbound { message }) => {
            if let Some(message) = message
                && binding.sender.send(message).is_err()
            {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "code": 500,
                        "msg": format!("channel '{}' receiver dropped", binding.channel_name),
                    })),
                );
            }
            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "code": 0,
                    "msg": "ok",
                })),
            )
        }
        Err(FeishuWebhookError::Unauthorized(message)) => (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({ "code": 401, "msg": message })),
        ),
        Err(FeishuWebhookError::BadRequest(message)) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "code": 400, "msg": message })),
        ),
    }
}

impl FeishuChannel {
    pub fn new(
        config: FeishuChannelConfig,
        request_timeout: Duration,
    ) -> Result<(Self, FeishuWebhookBinding), String> {
        let (sender, receiver) = mpsc::unbounded_channel();
        let channel_name = config.name.clone();
        let binding = FeishuWebhookBinding {
            channel_name,
            verification_token: config.verification_token.clone(),
            sender,
        };
        Ok((
            Self {
                name: config.name,
                client: FeishuClient::new(
                    config.app_id,
                    config.app_secret,
                    config.proxy_url,
                    request_timeout,
                )?,
                poll_interval: config.poll_interval,
                allowed_user_ids: config.allowed_user_ids.into_iter().collect(),
                next_poll_at: Instant::now(),
                receiver,
            },
            binding,
        ))
    }
}

impl Channel for FeishuChannel {
    fn name(&self) -> &str {
        &self.name
    }

    fn time_until_next_poll(&self, now: Instant) -> Duration {
        if !self.receiver.is_empty() || self.next_poll_at <= now {
            Duration::from_millis(0)
        } else {
            self.next_poll_at - now
        }
    }

    fn poll_if_due<'a>(
        &'a mut self,
        dispatcher: &'a mut dyn SessionDispatcher,
    ) -> ChannelFuture<'a> {
        Box::pin(async move {
            let now = Instant::now();
            if self.next_poll_at > now && self.receiver.is_empty() {
                return Ok(());
            }
            self.next_poll_at = now + self.poll_interval;

            loop {
                let inbound = match self.receiver.try_recv() {
                    Ok(inbound) => inbound,
                    Err(mpsc::error::TryRecvError::Empty) => break,
                    Err(mpsc::error::TryRecvError::Disconnected) => break,
                };
                if !self.allowed_user_ids.contains(&inbound.user_id) {
                    continue;
                }

                let reply = match dispatcher
                    .dispatch_text(&self.name, &inbound.user_id, &inbound.text)
                    .await
                {
                    Ok(text) => text,
                    Err(error) => {
                        eprintln!(
                            "warning: route '{}:{}' failed: {error}",
                            self.name, inbound.user_id
                        );
                        "Sorry, I hit an internal error while processing your message.".to_string()
                    }
                };

                self.client
                    .send_text_message(&inbound.chat_id, &reply)
                    .await?;
            }
            Ok(())
        })
    }
}

impl FeishuClient {
    fn new(
        app_id: String,
        app_secret: String,
        proxy_url: Option<String>,
        request_timeout: Duration,
    ) -> Result<Self, String> {
        let mut builder = Client::builder().timeout(request_timeout);
        if let Some(proxy_url) = proxy_url
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            let proxy = Proxy::all(proxy_url)
                .map_err(|error| format!("build feishu proxy failed: {error}"))?;
            builder = builder.proxy(proxy);
        }
        let client = builder
            .build()
            .map_err(|error| format!("build feishu client failed: {error}"))?;
        Ok(Self {
            client,
            api_base: "https://open.feishu.cn/open-apis".to_string(),
            app_id,
            app_secret,
            token_cache: Mutex::new(None),
        })
    }

    async fn send_text_message(&self, chat_id: &str, text: &str) -> Result<(), String> {
        let token = self.tenant_access_token().await?;
        let url = format!("{}/im/v1/messages?receive_id_type=chat_id", self.api_base);
        let payload = build_send_text_request(chat_id, text);
        let response = self
            .client
            .post(url)
            .bearer_auth(token)
            .json(&payload)
            .send()
            .await
            .map_err(|error| format!("feishu send message request failed: {error}"))?;
        let parsed = response
            .json::<FeishuApiStatusResponse>()
            .await
            .map_err(|error| format!("feishu send message decode failed: {error}"))?;
        if parsed.code == 0 {
            Ok(())
        } else {
            Err(parsed
                .msg
                .unwrap_or_else(|| format!("feishu send message failed with code {}", parsed.code)))
        }
    }

    async fn tenant_access_token(&self) -> Result<String, String> {
        {
            let cache = self.token_cache.lock().await;
            if let Some(cache) = cache.as_ref()
                && cache.expires_at > Instant::now()
            {
                return Ok(cache.value.clone());
            }
        }

        let payload = FeishuTenantAccessTokenRequest {
            app_id: &self.app_id,
            app_secret: &self.app_secret,
        };
        let url = format!("{}/auth/v3/tenant_access_token/internal", self.api_base);
        let response = self
            .client
            .post(url)
            .json(&payload)
            .send()
            .await
            .map_err(|error| format!("feishu tenant_access_token request failed: {error}"))?;
        let parsed = response
            .json::<FeishuTenantAccessTokenResponse>()
            .await
            .map_err(|error| format!("feishu tenant_access_token decode failed: {error}"))?;
        if parsed.code != 0 {
            return Err(parsed.msg.unwrap_or_else(|| {
                format!(
                    "feishu tenant_access_token failed with code {}",
                    parsed.code
                )
            }));
        }
        let token = parsed
            .tenant_access_token
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| "feishu tenant_access_token is empty".to_string())?;
        let expire_seconds = parsed.expire.unwrap_or(7200).saturating_sub(60).max(1);
        let cached = CachedTenantToken {
            value: token.clone(),
            expires_at: Instant::now() + Duration::from_secs(expire_seconds),
        };
        *self.token_cache.lock().await = Some(cached);
        Ok(token)
    }
}

pub fn build_feishu_webhook_router(bindings: Vec<FeishuWebhookBinding>) -> Router {
    let mut binding_map = HashMap::new();
    for binding in bindings {
        binding_map.insert(binding.channel_name.clone(), binding);
    }
    let state = FeishuWebhookState {
        bindings: Arc::new(binding_map),
    };
    Router::new()
        .route(
            "/webhook/feishu/{channel_name}",
            post(handle_feishu_webhook),
        )
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    #[test]
    fn parse_feishu_webhook_payload_supports_url_verification() {
        let payload = serde_json::json!({
            "token": "verify-token",
            "type": "url_verification",
            "challenge": "challenge-value"
        });

        let parsed = parse_feishu_webhook_payload(payload, "verify-token")
            .expect("url verification payload should parse");
        assert_eq!(
            parsed,
            FeishuWebhookParseResult::Challenge {
                value: "challenge-value".to_string()
            }
        );
    }

    #[test]
    fn parse_feishu_webhook_payload_extracts_private_text_message() {
        let payload = serde_json::json!({
            "schema": "2.0",
            "header": {
                "token": "verify-token",
                "event_type": "im.message.receive_v1"
            },
            "event": {
                "sender": {
                    "sender_id": {
                        "open_id": "ou_abc"
                    }
                },
                "message": {
                    "message_id": "om_1",
                    "chat_id": "oc_1",
                    "chat_type": "p2p",
                    "message_type": "text",
                    "content": "{\"text\":\"hello pixy\"}"
                }
            }
        });

        let parsed = parse_feishu_webhook_payload(payload, "verify-token")
            .expect("private text payload should parse");
        assert_eq!(
            parsed,
            FeishuWebhookParseResult::Inbound {
                message: Some(FeishuInboundMessage {
                    message_id: "om_1".to_string(),
                    chat_id: "oc_1".to_string(),
                    user_id: "ou_abc".to_string(),
                    text: "hello pixy".to_string(),
                })
            }
        );
    }

    #[test]
    fn parse_feishu_webhook_payload_rejects_invalid_token() {
        let payload = serde_json::json!({
            "token": "wrong-token",
            "type": "url_verification",
            "challenge": "challenge-value"
        });

        let error = parse_feishu_webhook_payload(payload, "verify-token")
            .expect_err("invalid token should be rejected");
        assert!(
            matches!(error, FeishuWebhookError::Unauthorized(_)),
            "token mismatch should return unauthorized error"
        );
    }

    #[test]
    fn build_send_text_request_serializes_as_feishu_text_message() {
        let request = build_send_text_request("oc_1", "hello");
        let value = serde_json::to_value(request).expect("request should serialize");
        assert_eq!(
            value,
            serde_json::json!({
                "receive_id": "oc_1",
                "msg_type": "text",
                "content": "{\"text\":\"hello\"}"
            })
        );
    }

    #[tokio::test]
    async fn feishu_webhook_router_routes_message_to_channel_queue() {
        let (sender, mut receiver) = mpsc::unbounded_channel();
        let app = build_feishu_webhook_router(vec![FeishuWebhookBinding {
            channel_name: "feishu-main".to_string(),
            verification_token: "verify-token".to_string(),
            sender,
        }]);
        let payload = serde_json::json!({
            "schema": "2.0",
            "header": {
                "token": "verify-token",
                "event_type": "im.message.receive_v1"
            },
            "event": {
                "sender": {
                    "sender_id": {
                        "open_id": "ou_abc"
                    }
                },
                "message": {
                    "message_id": "om_1",
                    "chat_id": "oc_1",
                    "chat_type": "p2p",
                    "message_type": "text",
                    "content": "{\"text\":\"hello pixy\"}"
                }
            }
        });
        let request = Request::builder()
            .method("POST")
            .uri("/webhook/feishu/feishu-main")
            .header("content-type", "application/json")
            .body(Body::from(payload.to_string()))
            .expect("request should build");

        let response = app
            .oneshot(request)
            .await
            .expect("router should accept feishu webhook request");
        assert_eq!(response.status(), StatusCode::OK);

        let inbound = receiver
            .try_recv()
            .expect("channel queue should receive parsed inbound message");
        assert_eq!(inbound.user_id, "ou_abc");
        assert_eq!(inbound.chat_id, "oc_1");
        assert_eq!(inbound.text, "hello pixy");
    }
}
