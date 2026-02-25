use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;

use pixy_ai::Model;
use pixy_coding_agent::RuntimeLoadOptions;
use serde::Deserialize;

#[derive(Debug, Clone)]
pub struct GatewayConfig {
    pub enabled: bool,
    pub bind_addr: String,
    pub request_timeout: Duration,
    pub transport_retry_count: Option<usize>,
    pub model: Model,
    pub api_key: Option<String>,
    pub channels: Vec<GatewayChannelConfig>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GatewayChannelConfig {
    Telegram(TelegramChannelConfig),
    Feishu(FeishuChannelConfig),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TelegramChannelConfig {
    pub name: String,
    pub bot_token: String,
    pub proxy_url: Option<String>,
    pub poll_interval: Duration,
    pub update_limit: u8,
    pub allowed_user_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeishuChannelConfig {
    pub name: String,
    pub app_id: String,
    pub app_secret: String,
    pub verification_token: String,
    pub proxy_url: Option<String>,
    pub poll_interval: Duration,
    pub allowed_user_ids: Vec<String>,
}

#[derive(Debug, Deserialize, Default)]
struct PixyTomlFile {
    #[serde(default)]
    gateway: PixyTomlGateway,
    #[serde(default)]
    transport_retry_count: Option<usize>,
    #[serde(default)]
    env: HashMap<String, String>,
}

#[derive(Debug, Deserialize, Default)]
struct PixyTomlGateway {
    #[serde(default)]
    enabled: Option<bool>,
    #[serde(default)]
    bind: Option<String>,
    #[serde(default)]
    request_timeout_ms: Option<u64>,
    #[serde(default)]
    channels: Vec<PixyTomlGatewayChannel>,
}

#[derive(Debug, Deserialize, Default)]
struct PixyTomlGatewayChannel {
    name: String,
    kind: String,
    #[serde(default)]
    enabled: Option<bool>,
    #[serde(default)]
    bot_token: Option<String>,
    #[serde(default)]
    app_id: Option<String>,
    #[serde(default)]
    app_secret: Option<String>,
    #[serde(default)]
    verification_token: Option<String>,
    #[serde(default)]
    proxy_url: Option<String>,
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    poll_interval_ms: Option<u64>,
    #[serde(default)]
    update_limit: Option<u8>,
    #[serde(default)]
    allowed_user_ids: Vec<String>,
}

const DEFAULT_CONF_DIR_NAME: &str = ".pixy";
static CONF_DIR: OnceLock<PathBuf> = OnceLock::new();

pub fn init_conf_dir(conf_dir: Option<PathBuf>) {
    let resolved = conf_dir
        .as_deref()
        .map(resolve_conf_dir_arg)
        .unwrap_or_else(default_conf_dir);
    let _ = CONF_DIR.set(resolved);
}

pub(crate) fn current_conf_dir() -> PathBuf {
    CONF_DIR.get().cloned().unwrap_or_else(default_conf_dir)
}

fn default_conf_dir() -> PathBuf {
    home_dir().join(DEFAULT_CONF_DIR_NAME)
}

fn resolve_conf_dir_arg(path: &Path) -> PathBuf {
    let expanded = expand_path_with_home(path);
    if expanded.is_absolute() {
        expanded
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(expanded)
    }
}

fn expand_path_with_home(path: &Path) -> PathBuf {
    let raw = path.to_string_lossy();
    if raw == "~" {
        return home_dir();
    }
    if let Some(suffix) = raw.strip_prefix("~/") {
        return home_dir().join(suffix);
    }
    path.to_path_buf()
}

fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

pub fn default_pixy_config_path() -> PathBuf {
    current_conf_dir().join("pixy.toml")
}

pub fn load_gateway_config(path: &Path) -> Result<GatewayConfig, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|error| format!("read {} failed: {error}", path.display()))?;
    parse_gateway_config_with_seed(&content, RuntimeLoadOptions::runtime_router_seed())
}

pub(crate) fn gateway_runtime_load_options() -> RuntimeLoadOptions {
    RuntimeLoadOptions {
        conf_dir: Some(current_conf_dir()),
        load_skills: true,
        include_default_skills: true,
        ..RuntimeLoadOptions::default()
    }
}

pub(crate) fn parse_gateway_config_with_seed(
    content: &str,
    router_seed: u64,
) -> Result<GatewayConfig, String> {
    let parsed: PixyTomlFile =
        toml::from_str(content).map_err(|error| format!("parse pixy.toml failed: {error}"))?;
    let runtime_options = gateway_runtime_load_options();
    let runtime = runtime_options.resolve_runtime_from_toml_with_seed(
        Path::new("."),
        content,
        router_seed,
    )?;
    let channels = resolve_gateway_channels(&parsed.gateway.channels, &parsed.env)?;
    let request_timeout =
        Duration::from_millis(parsed.gateway.request_timeout_ms.unwrap_or(20_000));
    let bind_addr = parsed
        .gateway
        .bind
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("0.0.0.0:8080")
        .to_string();

    Ok(GatewayConfig {
        enabled: parsed.gateway.enabled.unwrap_or(false),
        bind_addr,
        request_timeout,
        transport_retry_count: parsed.transport_retry_count,
        model: runtime.model,
        api_key: runtime.api_key,
        channels,
    })
}

fn resolve_gateway_channels(
    channels: &[PixyTomlGatewayChannel],
    env_map: &HashMap<String, String>,
) -> Result<Vec<GatewayChannelConfig>, String> {
    let mut resolved = Vec::new();
    for channel in channels {
        if channel.enabled == Some(false) {
            continue;
        }
        let channel_name = channel.name.trim();
        if channel_name.is_empty() {
            continue;
        }
        let kind = channel.kind.trim().to_ascii_lowercase();
        match kind.as_str() {
            "telegram" => {
                let mode = channel
                    .mode
                    .as_deref()
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .unwrap_or("polling");
                if !mode.eq_ignore_ascii_case("polling") {
                    return Err(format!(
                        "telegram channel '{}' only supports polling mode",
                        channel_name
                    ));
                }
                let bot_token = channel
                    .bot_token
                    .as_deref()
                    .and_then(|value| resolve_config_value(value, env_map))
                    .ok_or_else(|| {
                        format!("telegram channel '{}' is missing bot_token", channel_name)
                    })?;
                let proxy_url = channel
                    .proxy_url
                    .as_deref()
                    .and_then(|value| resolve_config_value(value, env_map));
                let allowed_user_ids = normalize_allowed_user_ids(&channel.allowed_user_ids);
                if allowed_user_ids.is_empty() {
                    return Err(format!(
                        "telegram channel '{}' requires non-empty allowed_user_ids",
                        channel_name
                    ));
                }
                let update_limit = channel.update_limit.unwrap_or(50);
                if update_limit == 0 {
                    return Err(format!(
                        "telegram channel '{}' update_limit must be greater than 0",
                        channel_name
                    ));
                }
                resolved.push(GatewayChannelConfig::Telegram(TelegramChannelConfig {
                    name: channel_name.to_string(),
                    bot_token,
                    proxy_url,
                    poll_interval: Duration::from_millis(channel.poll_interval_ms.unwrap_or(1_500)),
                    update_limit,
                    allowed_user_ids,
                }));
            }
            "feishu" => {
                let mode = channel
                    .mode
                    .as_deref()
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .unwrap_or("webhook");
                if !mode.eq_ignore_ascii_case("webhook") {
                    return Err(format!(
                        "feishu channel '{}' only supports webhook mode",
                        channel_name
                    ));
                }
                let app_id = channel
                    .app_id
                    .as_deref()
                    .and_then(|value| resolve_config_value(value, env_map))
                    .ok_or_else(|| {
                        format!("feishu channel '{}' is missing app_id", channel_name)
                    })?;
                let app_secret = channel
                    .app_secret
                    .as_deref()
                    .and_then(|value| resolve_config_value(value, env_map))
                    .ok_or_else(|| {
                        format!("feishu channel '{}' is missing app_secret", channel_name)
                    })?;
                let verification_token = channel
                    .verification_token
                    .as_deref()
                    .and_then(|value| resolve_config_value(value, env_map))
                    .ok_or_else(|| {
                        format!(
                            "feishu channel '{}' is missing verification_token",
                            channel_name
                        )
                    })?;
                let proxy_url = channel
                    .proxy_url
                    .as_deref()
                    .and_then(|value| resolve_config_value(value, env_map));
                let allowed_user_ids = normalize_allowed_user_ids(&channel.allowed_user_ids);
                if allowed_user_ids.is_empty() {
                    return Err(format!(
                        "feishu channel '{}' requires non-empty allowed_user_ids",
                        channel_name
                    ));
                }
                resolved.push(GatewayChannelConfig::Feishu(FeishuChannelConfig {
                    name: channel_name.to_string(),
                    app_id,
                    app_secret,
                    verification_token,
                    proxy_url,
                    poll_interval: Duration::from_millis(channel.poll_interval_ms.unwrap_or(100)),
                    allowed_user_ids,
                }));
            }
            other => {
                return Err(format!(
                    "gateway channel '{}' has unsupported kind '{}'",
                    channel_name, other
                ));
            }
        }
    }
    Ok(resolved)
}

fn normalize_allowed_user_ids(values: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for value in values {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            continue;
        }
        if out.iter().any(|existing: &String| existing == trimmed) {
            continue;
        }
        out.push(trimmed.to_string());
    }
    out
}

fn resolve_config_value(value: &str, env_map: &HashMap<String, String>) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Some(env_key) = trimmed.strip_prefix('$') {
        return env_map
            .get(env_key)
            .cloned()
            .or_else(|| std::env::var(env_key).ok())
            .filter(|resolved| !resolved.trim().is_empty());
    }
    Some(trimmed.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gateway_runtime_load_options_enable_skills_by_default() {
        let options = gateway_runtime_load_options();
        assert!(options.load_skills);
        assert!(options.include_default_skills);
        assert_eq!(options.conf_dir, Some(current_conf_dir()));
    }

    #[test]
    fn parse_gateway_config_resolves_polling_telegram_channel() {
        let content = r#"
[env]
TELEGRAM_BOT_TOKEN = "token-from-env"
TG_PROXY_URL = "socks5://127.0.0.1:7891"

[llm]
default_provider = "openai"

[[llm.providers]]
name = "openai"
kind = "chat"
provider = "openai"
api = "openai-responses"
base_url = "https://api.openai.com/v1"
api_key = "literal"
model = "gpt-5.3-codex"
weight = 1

[gateway]
enabled = true
request_timeout_ms = 15000

[[gateway.channels]]
name = "tg-main"
kind = "telegram"
enabled = true
bot_token = "$TELEGRAM_BOT_TOKEN"
proxy_url = "$TG_PROXY_URL"
mode = "polling"
poll_interval_ms = 1234
update_limit = 55
allowed_user_ids = ["10001", "10002"]
"#;

        let config =
            parse_gateway_config_with_seed(content, 0).expect("config should parse successfully");
        assert!(config.enabled, "gateway should be enabled");
        assert_eq!(config.request_timeout, Duration::from_millis(15_000));
        assert_eq!(config.transport_retry_count, None);
        assert_eq!(config.model.provider, "openai");
        assert_eq!(config.model.id, "gpt-5.3-codex");
        assert_eq!(config.api_key.as_deref(), Some("literal"));

        let telegram = config
            .channels
            .iter()
            .find_map(|channel| match channel {
                GatewayChannelConfig::Telegram(config) => Some(config),
                GatewayChannelConfig::Feishu(_) => None,
            })
            .expect("telegram channel should be present");
        assert_eq!(telegram.name, "tg-main");
        assert_eq!(telegram.bot_token, "token-from-env");
        assert_eq!(
            telegram.proxy_url.as_deref(),
            Some("socks5://127.0.0.1:7891")
        );
        assert_eq!(telegram.poll_interval, Duration::from_millis(1234));
        assert_eq!(telegram.update_limit, 55);
        assert_eq!(telegram.allowed_user_ids, vec!["10001", "10002"]);
    }

    #[test]
    fn parse_gateway_config_rejects_non_polling_telegram_mode() {
        let content = r#"
[llm]
default_provider = "openai"

[[llm.providers]]
name = "openai"
kind = "chat"
provider = "openai"
api = "openai-responses"
base_url = "https://api.openai.com/v1"
api_key = "literal"
model = "gpt-5.3-codex"
weight = 1

[gateway]
enabled = true

[[gateway.channels]]
name = "tg-main"
kind = "telegram"
enabled = true
bot_token = "literal"
mode = "webhook"
allowed_user_ids = ["10001"]
"#;

        let error = parse_gateway_config_with_seed(content, 0)
            .expect_err("webhook mode should be rejected for telegram");
        assert!(
            error.contains("polling"),
            "error should mention polling-only requirement"
        );
    }

    #[test]
    fn parse_gateway_config_rejects_empty_allowed_user_ids_for_telegram() {
        let content = r#"
[llm]
default_provider = "openai"

[[llm.providers]]
name = "openai"
kind = "chat"
provider = "openai"
api = "openai-responses"
base_url = "https://api.openai.com/v1"
api_key = "literal"
model = "gpt-5.3-codex"
weight = 1

[gateway]
enabled = true

[[gateway.channels]]
name = "tg-main"
kind = "telegram"
enabled = true
bot_token = "literal"
mode = "polling"
allowed_user_ids = []
"#;

        let error = parse_gateway_config_with_seed(content, 0)
            .expect_err("telegram channel should require allowed_user_ids");
        assert!(
            error.contains("allowed_user_ids"),
            "error should mention missing allowlist"
        );
    }

    #[test]
    fn wildcard_default_provider_uses_weights_for_chat_providers() {
        let content = r#"
[llm]
default_provider = "*"

[[llm.providers]]
name = "openai"
kind = "chat"
provider = "openai"
api = "openai-responses"
base_url = "https://api.openai.com/v1"
api_key = "openai"
model = "gpt-5.3-codex"
weight = 90

[[llm.providers]]
name = "anthropic"
kind = "chat"
provider = "anthropic"
api = "anthropic-messages"
base_url = "https://api.anthropic.com/v1"
api_key = "anthropic"
model = "claude-3-5-sonnet-latest"
weight = 10

[gateway]
enabled = true

[[gateway.channels]]
name = "tg-main"
kind = "telegram"
enabled = true
bot_token = "literal"
mode = "polling"
allowed_user_ids = ["10001"]
"#;

        let anthropic = parse_gateway_config_with_seed(content, 5)
            .expect("config should parse")
            .model;
        let openai = parse_gateway_config_with_seed(content, 95)
            .expect("config should parse")
            .model;
        assert_eq!(anthropic.provider, "anthropic");
        assert_eq!(openai.provider, "openai");
    }

    #[test]
    fn parse_gateway_config_resolves_feishu_channel_from_env() {
        let content = r#"
[env]
FEISHU_APP_ID = "cli_test_app_id"
FEISHU_APP_SECRET = "test-secret"
FEISHU_VERIFICATION_TOKEN = "verify-token"

[llm]
default_provider = "openai"

[[llm.providers]]
name = "openai"
kind = "chat"
provider = "openai"
api = "openai-responses"
base_url = "https://api.openai.com/v1"
api_key = "literal"
model = "gpt-5.3-codex"
weight = 1

[gateway]
enabled = true
bind = "0.0.0.0:18080"

[[gateway.channels]]
name = "feishu-main"
kind = "feishu"
enabled = true
app_id = "$FEISHU_APP_ID"
app_secret = "$FEISHU_APP_SECRET"
verification_token = "$FEISHU_VERIFICATION_TOKEN"
allowed_user_ids = ["ou_abc", "ou_def"]
"#;

        let config =
            parse_gateway_config_with_seed(content, 0).expect("config should parse successfully");
        assert_eq!(config.bind_addr, "0.0.0.0:18080");
        let feishu = config
            .channels
            .iter()
            .find_map(|channel| match channel {
                GatewayChannelConfig::Telegram(_) => None,
                GatewayChannelConfig::Feishu(config) => Some(config),
            })
            .expect("feishu channel should be present");
        assert_eq!(feishu.name, "feishu-main");
        assert_eq!(feishu.app_id, "cli_test_app_id");
        assert_eq!(feishu.app_secret, "test-secret");
        assert_eq!(feishu.verification_token, "verify-token");
        assert_eq!(feishu.allowed_user_ids, vec!["ou_abc", "ou_def"]);
    }

    #[test]
    fn parse_gateway_config_rejects_empty_allowed_user_ids_for_feishu() {
        let content = r#"
[llm]
default_provider = "openai"

[[llm.providers]]
name = "openai"
kind = "chat"
provider = "openai"
api = "openai-responses"
base_url = "https://api.openai.com/v1"
api_key = "literal"
model = "gpt-5.3-codex"
weight = 1

[gateway]
enabled = true

[[gateway.channels]]
name = "feishu-main"
kind = "feishu"
enabled = true
app_id = "cli_xxx"
app_secret = "secret"
verification_token = "token"
allowed_user_ids = []
"#;

        let error = parse_gateway_config_with_seed(content, 0)
            .expect_err("feishu channel should require allowed_user_ids");
        assert!(
            error.contains("allowed_user_ids"),
            "error should mention missing allowlist"
        );
    }

    #[test]
    fn parse_gateway_config_rejects_non_webhook_feishu_mode() {
        let content = r#"
[llm]
default_provider = "openai"

[[llm.providers]]
name = "openai"
kind = "chat"
provider = "openai"
api = "openai-responses"
base_url = "https://api.openai.com/v1"
api_key = "literal"
model = "gpt-5.3-codex"
weight = 1

[gateway]
enabled = true

[[gateway.channels]]
name = "feishu-main"
kind = "feishu"
enabled = true
mode = "polling"
app_id = "cli_xxx"
app_secret = "secret"
verification_token = "token"
allowed_user_ids = ["ou_abc"]
"#;

        let error = parse_gateway_config_with_seed(content, 0)
            .expect_err("non-webhook feishu mode should be rejected");
        assert!(
            error.contains("webhook"),
            "error should mention webhook-only requirement"
        );
    }
}
