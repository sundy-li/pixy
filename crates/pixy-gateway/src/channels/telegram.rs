use std::collections::HashSet;
use std::time::Duration;

use reqwest::{Client, Proxy};
use serde::{Deserialize, Serialize};
use tokio::time::Instant;

use crate::channels::{Channel, ChannelFuture, SessionDispatcher};
use crate::config::TelegramChannelConfig;

const TELEGRAM_MAX_TEXT_CHARS: usize = 4_000;
const TELEGRAM_TYPING_ACTION: &str = "typing";
const TELEGRAM_TYPING_REFRESH_INTERVAL: Duration = Duration::from_secs(4);

#[derive(Debug, Clone)]
pub struct TelegramClient {
    client: Client,
    api_base: String,
    bot_token: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TelegramInboundMessage {
    pub update_id: i64,
    pub message_id: i64,
    pub chat_id: i64,
    pub user_id: String,
    pub text: String,
}

pub struct TelegramChannel {
    name: String,
    client: TelegramClient,
    poll_interval: Duration,
    update_limit: u8,
    allowed_user_ids: HashSet<String>,
    next_poll_at: Instant,
    offset: Option<i64>,
}

#[derive(Debug, Deserialize)]
pub struct TelegramUpdatesResponse {
    pub ok: bool,
    #[serde(default)]
    pub result: Vec<TelegramUpdate>,
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TelegramApiStatusResponse {
    ok: bool,
    #[serde(default)]
    description: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TelegramUpdate {
    pub update_id: i64,
    #[serde(default)]
    pub message: Option<TelegramMessage>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TelegramMessage {
    pub message_id: i64,
    pub chat: TelegramChat,
    #[serde(default)]
    pub from: Option<TelegramUser>,
    #[serde(default)]
    pub text: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TelegramChat {
    pub id: i64,
    #[serde(rename = "type")]
    pub kind: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TelegramUser {
    pub id: i64,
    #[serde(default)]
    pub is_bot: Option<bool>,
}

#[derive(Debug, Serialize)]
struct GetUpdatesRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    offset: Option<i64>,
    limit: u8,
}

#[derive(Debug, Serialize)]
struct SendMessageRequest<'a> {
    chat_id: i64,
    text: &'a str,
}

#[derive(Debug, Serialize)]
struct SendChatActionRequest<'a> {
    chat_id: i64,
    action: &'a str,
}

fn build_chat_action_request<'a>(chat_id: i64, action: &'a str) -> SendChatActionRequest<'a> {
    SendChatActionRequest { chat_id, action }
}

impl TelegramChannel {
    pub fn new(config: TelegramChannelConfig, request_timeout: Duration) -> Result<Self, String> {
        Ok(Self {
            name: config.name,
            client: TelegramClient::new(config.bot_token, config.proxy_url, request_timeout)?,
            poll_interval: config.poll_interval,
            update_limit: config.update_limit,
            allowed_user_ids: config.allowed_user_ids.into_iter().collect(),
            next_poll_at: Instant::now(),
            offset: None,
        })
    }

    async fn dispatch_with_typing(
        &self,
        dispatcher: &mut dyn SessionDispatcher,
        inbound: &TelegramInboundMessage,
    ) -> Result<String, String> {
        if let Err(error) = self.client.send_typing_action(inbound.chat_id).await {
            eprintln!(
                "warning: channel '{}' failed to send typing action for route '{}:{}': {error}",
                self.name, self.name, inbound.user_id
            );
        }

        let dispatch = dispatcher.dispatch_text(&self.name, &inbound.user_id, &inbound.text);
        tokio::pin!(dispatch);

        loop {
            tokio::select! {
                result = &mut dispatch => return result,
                _ = tokio::time::sleep(TELEGRAM_TYPING_REFRESH_INTERVAL) => {
                    if let Err(error) = self.client.send_typing_action(inbound.chat_id).await {
                        eprintln!(
                            "warning: channel '{}' failed to refresh typing action for route '{}:{}': {error}",
                            self.name, self.name, inbound.user_id
                        );
                    }
                }
            }
        }
    }
}

impl Channel for TelegramChannel {
    fn name(&self) -> &str {
        &self.name
    }

    fn time_until_next_poll(&self, now: Instant) -> Duration {
        if self.next_poll_at <= now {
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
            if self.next_poll_at > now {
                return Ok(());
            }
            self.next_poll_at = now + self.poll_interval;

            let updates = self
                .client
                .get_updates(self.offset, self.update_limit)
                .await?;
            for update in updates {
                self.offset = Some(update.update_id + 1);
                let Some(inbound) = extract_private_text_message(&update, &self.allowed_user_ids)
                else {
                    continue;
                };

                let reply = match self.dispatch_with_typing(dispatcher, &inbound).await {
                    Ok(text) => text,
                    Err(error) => {
                        eprintln!(
                            "warning: route '{}:{}' failed: {error}",
                            self.name, inbound.user_id
                        );
                        "Sorry, I hit an internal error while processing your message.".to_string()
                    }
                };

                for chunk in split_telegram_message(&reply, TELEGRAM_MAX_TEXT_CHARS) {
                    self.client.send_message(inbound.chat_id, &chunk).await?;
                }
            }
            Ok(())
        })
    }
}

impl TelegramClient {
    pub fn new(
        bot_token: String,
        proxy_url: Option<String>,
        request_timeout: Duration,
    ) -> Result<Self, String> {
        let mut builder = Client::builder().timeout(request_timeout);
        if let Some(proxy_url) = proxy_url
            .as_deref()
            .map(str::trim)
            .filter(|v| !v.is_empty())
        {
            let proxy = Proxy::all(proxy_url)
                .map_err(|error| format!("build telegram proxy failed: {error}"))?;
            builder = builder.proxy(proxy);
        }
        let client = builder
            .build()
            .map_err(|error| format!("build telegram client failed: {error}"))?;
        Ok(Self {
            client,
            api_base: "https://api.telegram.org".to_string(),
            bot_token,
        })
    }

    pub async fn get_updates(
        &self,
        offset: Option<i64>,
        limit: u8,
    ) -> Result<Vec<TelegramUpdate>, String> {
        let request = GetUpdatesRequest { offset, limit };
        let url = format!("{}/bot{}/getUpdates", self.api_base, self.bot_token);
        let response = self
            .client
            .post(url)
            .json(&request)
            .send()
            .await
            .map_err(|error| format!("telegram getUpdates request failed: {error}"))?;
        let parsed = response
            .json::<TelegramUpdatesResponse>()
            .await
            .map_err(|error| format!("telegram getUpdates decode failed: {error}"))?;
        if parsed.ok {
            Ok(parsed.result)
        } else {
            Err(parsed
                .description
                .unwrap_or_else(|| "telegram getUpdates returned ok=false".to_string()))
        }
    }

    pub async fn send_message(&self, chat_id: i64, text: &str) -> Result<(), String> {
        let url = format!("{}/bot{}/sendMessage", self.api_base, self.bot_token);
        let request = SendMessageRequest { chat_id, text };
        let response = self
            .client
            .post(url)
            .json(&request)
            .send()
            .await
            .map_err(|error| format!("telegram sendMessage request failed: {error}"))?;
        let parsed = response
            .json::<TelegramApiStatusResponse>()
            .await
            .map_err(|error| format!("telegram sendMessage decode failed: {error}"))?;
        if parsed.ok {
            Ok(())
        } else {
            Err(parsed
                .description
                .unwrap_or_else(|| "telegram sendMessage returned ok=false".to_string()))
        }
    }

    pub async fn send_typing_action(&self, chat_id: i64) -> Result<(), String> {
        self.send_chat_action(chat_id, TELEGRAM_TYPING_ACTION).await
    }

    async fn send_chat_action(&self, chat_id: i64, action: &str) -> Result<(), String> {
        let url = format!("{}/bot{}/sendChatAction", self.api_base, self.bot_token);
        let request = build_chat_action_request(chat_id, action);
        let response = self
            .client
            .post(url)
            .json(&request)
            .send()
            .await
            .map_err(|error| format!("telegram sendChatAction request failed: {error}"))?;
        let parsed = response
            .json::<TelegramApiStatusResponse>()
            .await
            .map_err(|error| format!("telegram sendChatAction decode failed: {error}"))?;
        if parsed.ok {
            Ok(())
        } else {
            Err(parsed
                .description
                .unwrap_or_else(|| "telegram sendChatAction returned ok=false".to_string()))
        }
    }

    pub fn base_url(&self) -> &str {
        &self.api_base
    }

    pub fn bot_token(&self) -> &str {
        &self.bot_token
    }

    pub fn http_client(&self) -> &Client {
        &self.client
    }
}

pub fn extract_private_text_message(
    update: &TelegramUpdate,
    allowed_user_ids: &HashSet<String>,
) -> Option<TelegramInboundMessage> {
    let message = update.message.as_ref()?;
    if !message.chat.kind.eq_ignore_ascii_case("private") {
        return None;
    }
    let from = message.from.as_ref()?;
    if from.is_bot == Some(true) {
        return None;
    }
    let text = message
        .text
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())?;
    let user_id = from.id.to_string();
    if !allowed_user_ids.contains(&user_id) {
        return None;
    }

    Some(TelegramInboundMessage {
        update_id: update.update_id,
        message_id: message.message_id,
        chat_id: message.chat.id,
        user_id,
        text: text.to_string(),
    })
}

pub fn split_telegram_message(text: &str, max_chars: usize) -> Vec<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return vec![];
    }
    if max_chars == 0 {
        return vec![trimmed.to_string()];
    }
    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut current_chars = 0_usize;
    for ch in trimmed.chars() {
        if current_chars == max_chars {
            chunks.push(std::mem::take(&mut current));
            current_chars = 0;
        }
        current.push(ch);
        current_chars += 1;
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_private_text_message_accepts_allowed_private_chat() {
        let update = TelegramUpdate {
            update_id: 42,
            message: Some(TelegramMessage {
                message_id: 7,
                chat: TelegramChat {
                    id: 555,
                    kind: "private".to_string(),
                },
                from: Some(TelegramUser {
                    id: 10001,
                    is_bot: Some(false),
                }),
                text: Some("hello pixy".to_string()),
            }),
        };

        let allowed = HashSet::from(["10001".to_string()]);
        let inbound = extract_private_text_message(&update, &allowed)
            .expect("allowed private text update should be accepted");
        assert_eq!(inbound.update_id, 42);
        assert_eq!(inbound.message_id, 7);
        assert_eq!(inbound.chat_id, 555);
        assert_eq!(inbound.user_id, "10001");
        assert_eq!(inbound.text, "hello pixy");
    }

    #[test]
    fn extract_private_text_message_rejects_group_and_disallowed_users() {
        let group_update = TelegramUpdate {
            update_id: 1,
            message: Some(TelegramMessage {
                message_id: 1,
                chat: TelegramChat {
                    id: 1,
                    kind: "group".to_string(),
                },
                from: Some(TelegramUser {
                    id: 10001,
                    is_bot: Some(false),
                }),
                text: Some("should ignore".to_string()),
            }),
        };
        let disallowed_update = TelegramUpdate {
            update_id: 2,
            message: Some(TelegramMessage {
                message_id: 2,
                chat: TelegramChat {
                    id: 2,
                    kind: "private".to_string(),
                },
                from: Some(TelegramUser {
                    id: 99999,
                    is_bot: Some(false),
                }),
                text: Some("should ignore".to_string()),
            }),
        };

        let allowed = HashSet::from(["10001".to_string()]);
        assert!(
            extract_private_text_message(&group_update, &allowed).is_none(),
            "group chat update should be ignored"
        );
        assert!(
            extract_private_text_message(&disallowed_update, &allowed).is_none(),
            "disallowed user should be ignored"
        );
    }

    #[test]
    fn split_telegram_message_splits_long_text_without_losing_chars() {
        let text = "abcdefghij";
        let chunks = split_telegram_message(text, 4);
        assert_eq!(chunks, vec!["abcd", "efgh", "ij"]);
        assert_eq!(chunks.concat(), text);
    }

    #[test]
    fn build_chat_action_request_serializes_typing_action() {
        let request = build_chat_action_request(5566, TELEGRAM_TYPING_ACTION);
        let value = serde_json::to_value(request).expect("request should serialize");
        assert_eq!(
            value,
            serde_json::json!({
                "chat_id": 5566,
                "action": "typing"
            })
        );
    }
}
