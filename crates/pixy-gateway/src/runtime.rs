use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use chrono::{Datelike, Local};
use pixy_ai::{AssistantContentBlock, Message, Model, SimpleStreamOptions, StopReason};
use pixy_coding_agent::{AgentSession, AgentSessionConfig, SessionManager, create_coding_tools};
use tokio::task::JoinHandle;
use tokio::time::Instant;

use crate::DEFAULT_PROMPT_INTRO;
use crate::channels::feishu::{FeishuChannel, FeishuWebhookBinding, build_feishu_webhook_router};
use crate::channels::telegram::TelegramChannel;
use crate::channels::{Channel, DispatchFuture, SessionDispatcher};
use crate::config::{GatewayChannelConfig, GatewayConfig};

const NEW_SESSION_COMMAND_REPLY: &str = "Started a new session. Send your next message.";

pub struct SessionRouter {
    cwd: PathBuf,
    session_root: PathBuf,
    model: Model,
    api_key: Option<String>,
    sessions: HashMap<String, AgentSession>,
}

impl SessionRouter {
    pub fn new(cwd: PathBuf, session_root: PathBuf, model: Model, api_key: Option<String>) -> Self {
        Self {
            cwd,
            session_root,
            model,
            api_key,
            sessions: HashMap::new(),
        }
    }

    pub async fn process_text_message(
        &mut self,
        channel_name: &str,
        user_id: &str,
        text: &str,
    ) -> Result<String, String> {
        let key = session_key(channel_name, user_id);
        if is_new_session_command(text) {
            let session = create_gateway_session(
                &self.cwd,
                &self.session_root,
                channel_name,
                user_id,
                &self.model,
                self.api_key.clone(),
                false,
            )?;
            self.sessions.insert(key, session);
            return Ok(NEW_SESSION_COMMAND_REPLY.to_string());
        }

        if !self.sessions.contains_key(&key) {
            let session = create_gateway_session(
                &self.cwd,
                &self.session_root,
                channel_name,
                user_id,
                &self.model,
                self.api_key.clone(),
                true,
            )?;
            self.sessions.insert(key.clone(), session);
        }

        let session = self
            .sessions
            .get_mut(&key)
            .ok_or_else(|| format!("gateway route session '{key}' was not initialized"))?;
        let produced = session.prompt(text).await?;
        Ok(extract_assistant_reply(&produced))
    }
}

impl SessionDispatcher for SessionRouter {
    fn dispatch_text<'a>(
        &'a mut self,
        channel_name: &'a str,
        user_id: &'a str,
        text: &'a str,
    ) -> DispatchFuture<'a> {
        Box::pin(async move { self.process_text_message(channel_name, user_id, text).await })
    }
}

pub async fn serve_gateway(config: GatewayConfig) -> Result<(), String> {
    let GatewayConfig {
        enabled,
        bind_addr,
        request_timeout,
        transport_retry_count: _,
        model,
        api_key,
        channels,
    } = config;

    if !enabled {
        return Ok(());
    }

    let cwd = std::env::current_dir().map_err(|error| format!("read cwd failed: {error}"))?;
    let session_root = default_session_root();
    for line in startup_log_lines(
        &cwd,
        &session_root,
        &bind_addr,
        request_timeout,
        &model,
        &channels,
    ) {
        println!("{line}");
    }
    let mut router = SessionRouter::new(cwd, session_root, model, api_key);
    let BuiltChannels {
        mut channels,
        feishu_webhook_bindings,
    } = build_channels(channels, request_timeout)?;
    if channels.is_empty() {
        return Err("gateway has no enabled channel".to_string());
    }
    let mut feishu_webhook_server =
        start_feishu_webhook_server(&bind_addr, feishu_webhook_bindings).await?;

    let shutdown_signal = crate::wait_for_shutdown_signal();
    tokio::pin!(shutdown_signal);

    loop {
        let now = Instant::now();
        let sleep_for = channels
            .iter()
            .map(|channel| channel.time_until_next_poll(now))
            .min()
            .unwrap_or(Duration::from_millis(250));
        tokio::select! {
            result = &mut shutdown_signal => {
                result?;
                break;
            }
            _ = tokio::time::sleep(sleep_for) => {}
        }

        for channel in &mut channels {
            let channel_name = channel.name().to_string();
            if let Err(error) = channel.poll_if_due(&mut router).await {
                eprintln!("warning: channel '{channel_name}' poll failed: {error}");
            }
        }
    }

    if let Some(handle) = feishu_webhook_server.take() {
        handle.abort();
        let _ = handle.await;
    }

    Ok(())
}

fn default_session_root() -> PathBuf {
    crate::config::current_conf_dir()
        .join("agent")
        .join("sessions")
}

struct BuiltChannels {
    channels: Vec<Box<dyn Channel>>,
    feishu_webhook_bindings: Vec<FeishuWebhookBinding>,
}

fn build_channels(
    channels: Vec<GatewayChannelConfig>,
    request_timeout: Duration,
) -> Result<BuiltChannels, String> {
    let mut built_channels: Vec<Box<dyn Channel>> = Vec::new();
    let mut feishu_webhook_bindings = Vec::new();
    for channel in channels {
        match channel {
            GatewayChannelConfig::Telegram(telegram) => {
                built_channels.push(Box::new(TelegramChannel::new(telegram, request_timeout)?));
            }
            GatewayChannelConfig::Feishu(feishu) => {
                let (channel, binding) = FeishuChannel::new(feishu, request_timeout)?;
                built_channels.push(Box::new(channel));
                feishu_webhook_bindings.push(binding);
            }
        }
    }
    Ok(BuiltChannels {
        channels: built_channels,
        feishu_webhook_bindings,
    })
}

async fn start_feishu_webhook_server(
    bind_addr: &str,
    bindings: Vec<FeishuWebhookBinding>,
) -> Result<Option<JoinHandle<()>>, String> {
    if bindings.is_empty() {
        return Ok(None);
    }
    let listener = tokio::net::TcpListener::bind(bind_addr)
        .await
        .map_err(|error| format!("bind feishu webhook listener on {bind_addr} failed: {error}"))?;
    println!("[gateway] feishu webhook: http://{bind_addr}/webhook/feishu/{{channel_name}}");
    let app = build_feishu_webhook_router(bindings);
    let handle = tokio::spawn(async move {
        if let Err(error) = axum::serve(listener, app).await {
            eprintln!("warning: feishu webhook server stopped: {error}");
        }
    });
    Ok(Some(handle))
}

fn startup_log_lines(
    cwd: &Path,
    session_root: &Path,
    bind_addr: &str,
    request_timeout: Duration,
    model: &Model,
    channels: &[GatewayChannelConfig],
) -> Vec<String> {
    let mut lines = vec![
        "[gateway] starting runtime".to_string(),
        format!("[gateway] cwd: {}", cwd.display()),
        format!("[gateway] session_root: {}", session_root.display()),
        format!("[gateway] bind_addr: {bind_addr}"),
        format!(
            "[gateway] model: provider={} api={} id={}",
            model.provider, model.api, model.id
        ),
        format!(
            "[gateway] request_timeout_ms: {}",
            request_timeout.as_millis()
        ),
        format!("[gateway] configured_channels: {}", channels.len()),
    ];

    for channel in channels {
        match channel {
            GatewayChannelConfig::Telegram(config) => {
                lines.push(format!(
                    "[gateway] channel telegram name={} poll_interval_ms={} update_limit={} allowed_users={} proxy_configured={}",
                    config.name,
                    config.poll_interval.as_millis(),
                    config.update_limit,
                    config.allowed_user_ids.len(),
                    config.proxy_url.is_some()
                ));
            }
            GatewayChannelConfig::Feishu(config) => {
                lines.push(format!(
                    "[gateway] channel feishu name={} mode=webhook allowed_users={} poll_interval_ms={} proxy_configured={}",
                    config.name,
                    config.allowed_user_ids.len(),
                    config.poll_interval.as_millis(),
                    config.proxy_url.is_some()
                ));
            }
        }
    }

    lines
}

pub fn session_key(channel_name: &str, user_id: &str) -> String {
    format!("{channel_name}:{user_id}")
}

pub fn session_month_dir(session_root: &Path, now: chrono::DateTime<Local>) -> PathBuf {
    session_root
        .join(format!("{:04}", now.year()))
        .join(format!("{:02}", now.month()))
}

pub fn extract_assistant_reply(messages: &[Message]) -> String {
    let mut last_text: Option<String> = None;
    let mut last_error: Option<String> = None;

    for message in messages {
        if let Message::Assistant {
            content,
            stop_reason,
            error_message,
            ..
        } = message
        {
            let text = content
                .iter()
                .filter_map(|block| match block {
                    AssistantContentBlock::Text { text, .. } => Some(text.as_str()),
                    AssistantContentBlock::Thinking { .. }
                    | AssistantContentBlock::ToolCall { .. } => None,
                })
                .collect::<String>();
            if !text.trim().is_empty() {
                last_text = Some(text);
            }
            if *stop_reason == StopReason::Error {
                last_error = error_message.clone();
            }
        }
    }

    last_text
        .or(last_error)
        .unwrap_or_else(|| "Done.".to_string())
}

fn create_gateway_session(
    cwd: &Path,
    session_root: &Path,
    channel_name: &str,
    user_id: &str,
    model: &Model,
    api_key: Option<String>,
    reuse_existing: bool,
) -> Result<AgentSession, String> {
    let manager = create_session_manager(cwd, session_root, channel_name, user_id, reuse_existing)?;
    Ok(build_session_from_manager(cwd, model, api_key, manager))
}

fn create_session_manager(
    cwd: &Path,
    session_root: &Path,
    channel_name: &str,
    user_id: &str,
    reuse_existing: bool,
) -> Result<SessionManager, String> {
    let now = Local::now();
    let month_dir = session_month_dir(session_root, now);
    fs::create_dir_all(&month_dir).map_err(|error| {
        format!(
            "create gateway session month dir {} failed: {error}",
            month_dir.display()
        )
    })?;

    let prefix = route_file_prefix(channel_name, user_id);
    let mut existing_files = fs::read_dir(&month_dir)
        .map_err(|error| {
            format!(
                "read gateway session dir {} failed: {error}",
                month_dir.display()
            )
        })?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(&prefix) && name.ends_with(".jsonl"))
        })
        .collect::<Vec<_>>();
    existing_files.sort();
    if reuse_existing {
        if let Some(existing) = existing_files.pop() {
            return SessionManager::load(existing);
        }
    }

    let cwd_text = cwd
        .to_str()
        .ok_or_else(|| format!("gateway cwd is not valid UTF-8: {}", cwd.display()))?;
    let manager = SessionManager::create(cwd_text, &month_dir)?;
    let created_path = manager
        .session_file()
        .cloned()
        .ok_or_else(|| "session manager did not return session_file path".to_string())?;
    drop(manager);

    let unix_millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let route_path = month_dir.join(format!("{prefix}{unix_millis}.jsonl"));
    fs::rename(&created_path, &route_path).map_err(|error| {
        format!(
            "rename gateway session {} -> {} failed: {error}",
            created_path.display(),
            route_path.display()
        )
    })?;
    SessionManager::load(route_path)
}

fn is_new_session_command(input: &str) -> bool {
    let trimmed = input.trim();
    if trimmed.eq_ignore_ascii_case("/new") {
        return true;
    }
    if let Some(mention) = trimmed.strip_prefix("/new@") {
        return !mention.trim().is_empty() && !mention.chars().any(char::is_whitespace);
    }
    false
}

fn build_gateway_system_prompt(cwd: &Path) -> String {
    format!(
        "{DEFAULT_PROMPT_INTRO}\n\nCurrent working directory: {}",
        cwd.display()
    )
}

fn sanitize_session_segment(segment: &str) -> String {
    let cleaned = segment
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '-'
            }
        })
        .collect::<String>();
    let trimmed = cleaned.trim_matches('-');
    if trimmed.is_empty() {
        "unknown".to_string()
    } else {
        trimmed.to_string()
    }
}

fn route_file_prefix(channel_name: &str, user_id: &str) -> String {
    format!(
        "gateway-{}-{}-",
        sanitize_session_segment(channel_name),
        sanitize_session_segment(user_id)
    )
}

fn build_session_from_manager(
    cwd: &Path,
    model: &Model,
    api_key: Option<String>,
    manager: SessionManager,
) -> AgentSession {
    let tools = create_coding_tools(cwd);
    let stream_api_key = api_key.clone();
    let stream_fn = Arc::new(
        move |model: Model, context: pixy_ai::Context, options: Option<SimpleStreamOptions>| {
            let mut resolved_options = options.unwrap_or_default();
            if resolved_options.stream.api_key.is_none() {
                resolved_options.stream.api_key = stream_api_key.clone();
            }
            pixy_ai::stream_simple(model, context, Some(resolved_options))
        },
    );
    let config = AgentSessionConfig {
        model: model.clone(),
        system_prompt: build_gateway_system_prompt(cwd),
        stream_fn,
        tools,
    };
    AgentSession::new(manager, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn usage_stub() -> pixy_ai::Usage {
        pixy_ai::Usage {
            input: 1,
            output: 1,
            cache_read: 0,
            cache_write: 0,
            total_tokens: 2,
            cost: pixy_ai::Cost {
                input: 0.0,
                output: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
                total: 0.0,
            },
        }
    }

    #[test]
    fn session_key_uses_channel_and_user_id() {
        assert_eq!(session_key("tg-main", "10001"), "tg-main:10001");
    }

    #[test]
    fn session_month_dir_uses_year_and_month_levels() {
        let now = chrono::DateTime::parse_from_rfc3339("2026-02-25T10:11:12+08:00")
            .expect("fixed timestamp should parse")
            .with_timezone(&Local);
        let dir = session_month_dir(Path::new("/tmp/sessions"), now);
        assert_eq!(dir, PathBuf::from("/tmp/sessions/2026/02"));
    }

    #[test]
    fn route_file_prefix_sanitizes_invalid_characters() {
        let prefix = route_file_prefix("tg/main", "user:#1");
        assert_eq!(prefix, "gateway-tg-main-user--1-");
    }

    #[test]
    fn new_session_command_matches_exact_new_token() {
        assert!(is_new_session_command("/new"));
        assert!(is_new_session_command(" /new "));
        assert!(is_new_session_command("/new@pixy_bot"));
        assert!(!is_new_session_command("/new please"));
        assert!(!is_new_session_command("hello /new"));
    }

    #[test]
    fn create_session_manager_force_new_ignores_existing_route_file() {
        let temp = tempdir().expect("tempdir");
        let session_root = temp.path().join("sessions");
        let cwd = temp.path();

        let first = create_session_manager(cwd, &session_root, "tg-main", "10001", false)
            .expect("first session should be created");
        let first_file = first.session_file().expect("first session file").clone();
        drop(first);

        let reused = create_session_manager(cwd, &session_root, "tg-main", "10001", true)
            .expect("reuse session should load existing route session");
        let reused_file = reused.session_file().expect("reused session file").clone();
        assert_eq!(reused_file, first_file);

        std::thread::sleep(Duration::from_millis(2));
        let forced = create_session_manager(cwd, &session_root, "tg-main", "10001", false)
            .expect("forced new session should not reuse existing route file");
        let forced_file = forced.session_file().expect("forced session file").clone();
        assert_ne!(forced_file, first_file);
    }

    #[test]
    fn extract_assistant_reply_prefers_last_assistant_text() {
        let messages = vec![
            Message::Assistant {
                content: vec![AssistantContentBlock::Text {
                    text: "first".to_string(),
                    text_signature: None,
                }],
                api: "openai-responses".to_string(),
                provider: "openai".to_string(),
                model: "gpt-5.3-codex".to_string(),
                usage: usage_stub(),
                stop_reason: StopReason::Stop,
                error_message: None,
                timestamp: 0,
            },
            Message::Assistant {
                content: vec![AssistantContentBlock::Text {
                    text: "second".to_string(),
                    text_signature: None,
                }],
                api: "openai-responses".to_string(),
                provider: "openai".to_string(),
                model: "gpt-5.3-codex".to_string(),
                usage: usage_stub(),
                stop_reason: StopReason::Stop,
                error_message: None,
                timestamp: 0,
            },
        ];

        assert_eq!(extract_assistant_reply(&messages), "second");
    }

    #[test]
    fn startup_log_lines_include_runtime_overview_and_channels() {
        let model = Model {
            id: "gpt-5.3-codex".to_string(),
            name: "gpt-5.3-codex".to_string(),
            api: "openai-responses".to_string(),
            provider: "openai".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            reasoning: true,
            reasoning_effort: Some(pixy_ai::ThinkingLevel::Medium),
            input: vec!["text".to_string()],
            cost: pixy_ai::Cost {
                input: 0.0,
                output: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
                total: 0.0,
            },
            context_window: 200_000,
            max_tokens: 8_192,
        };
        let lines = startup_log_lines(
            Path::new("/workspace"),
            Path::new("/sessions"),
            "0.0.0.0:8080",
            Duration::from_millis(15_000),
            &model,
            &[
                GatewayChannelConfig::Telegram(crate::config::TelegramChannelConfig {
                    name: "tg-main".to_string(),
                    bot_token: "secret-token".to_string(),
                    proxy_url: Some("socks5://127.0.0.1:7891".to_string()),
                    poll_interval: Duration::from_millis(1500),
                    update_limit: 50,
                    allowed_user_ids: vec!["10001".to_string(), "10002".to_string()],
                }),
                GatewayChannelConfig::Feishu(crate::config::FeishuChannelConfig {
                    name: "feishu-main".to_string(),
                    app_id: "cli_xxx".to_string(),
                    app_secret: "secret".to_string(),
                    verification_token: "token".to_string(),
                    proxy_url: None,
                    poll_interval: Duration::from_millis(100),
                    allowed_user_ids: vec!["ou_1".to_string()],
                }),
            ],
        );

        let joined = lines.join("\n");
        assert!(
            joined.contains("[gateway] starting runtime"),
            "startup logs should include runtime boot line"
        );
        assert!(
            joined.contains("channel telegram name=tg-main"),
            "startup logs should include telegram channel details"
        );
        assert!(
            joined.contains("proxy_configured=true"),
            "startup logs should show proxy configured state for telegram channels"
        );
        assert!(
            joined.contains("channel feishu name=feishu-main"),
            "startup logs should include feishu channel details"
        );
        assert!(
            !joined.contains("secret-token"),
            "startup logs should not include channel secrets"
        );
    }

    #[test]
    fn build_channels_includes_feishu_and_telegram_entries() {
        let channels = vec![
            GatewayChannelConfig::Telegram(crate::config::TelegramChannelConfig {
                name: "tg-main".to_string(),
                bot_token: "token".to_string(),
                proxy_url: None,
                poll_interval: Duration::from_millis(1500),
                update_limit: 50,
                allowed_user_ids: vec!["10001".to_string()],
            }),
            GatewayChannelConfig::Feishu(crate::config::FeishuChannelConfig {
                name: "feishu-main".to_string(),
                app_id: "cli_xxx".to_string(),
                app_secret: "secret".to_string(),
                verification_token: "token".to_string(),
                proxy_url: None,
                poll_interval: Duration::from_millis(100),
                allowed_user_ids: vec!["ou_abc".to_string()],
            }),
        ];
        let built = build_channels(channels, Duration::from_secs(5))
            .expect("build channels should succeed for telegram + feishu");
        assert_eq!(
            built.channels.len(),
            2,
            "both channels should be instantiated"
        );
        assert_eq!(
            built.feishu_webhook_bindings.len(),
            1,
            "feishu channel should register one webhook binding"
        );
    }
}
