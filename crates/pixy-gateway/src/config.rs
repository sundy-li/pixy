use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;

use pixy_ai::Model;
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
    llm: PixyTomlLlm,
    #[serde(default)]
    gateway: PixyTomlGateway,
    #[serde(default)]
    transport_retry_count: Option<usize>,
    #[serde(default)]
    env: HashMap<String, String>,
}

#[derive(Debug, Deserialize, Default)]
struct PixyTomlLlm {
    #[serde(default)]
    default_provider: Option<String>,
    #[serde(default)]
    providers: Vec<PixyTomlProvider>,
}

#[derive(Debug, Deserialize, Default)]
struct PixyTomlProvider {
    name: String,
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    provider: Option<String>,
    #[serde(default)]
    api: Option<String>,
    #[serde(default)]
    base_url: Option<String>,
    #[serde(default)]
    api_key: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    weight: Option<u8>,
    #[serde(default)]
    context_window: Option<u32>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    reasoning: Option<bool>,
    #[serde(default)]
    reasoning_effort: Option<pixy_ai::ThinkingLevel>,
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
    parse_gateway_config_with_seed(&content, runtime_router_seed())
}

pub(crate) fn parse_gateway_config_with_seed(
    content: &str,
    router_seed: u64,
) -> Result<GatewayConfig, String> {
    let parsed: PixyTomlFile =
        toml::from_str(content).map_err(|error| format!("parse pixy.toml failed: {error}"))?;
    let model_selection = resolve_model_selection(&parsed.llm, &parsed.env, router_seed)?;
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
        model: model_selection.model,
        api_key: model_selection.api_key,
        channels,
    })
}

#[derive(Debug, Clone)]
struct ResolvedModelSelection {
    model: Model,
    api_key: Option<String>,
}

fn runtime_router_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0)
}

fn resolve_model_selection(
    llm: &PixyTomlLlm,
    env_map: &HashMap<String, String>,
    router_seed: u64,
) -> Result<ResolvedModelSelection, String> {
    let mut chat_providers = llm
        .providers
        .iter()
        .filter(|provider| is_chat_provider(provider.kind.as_deref()))
        .collect::<Vec<_>>();
    chat_providers.sort_by(|left, right| left.name.cmp(&right.name));

    if chat_providers.is_empty() {
        return Err("pixy.toml llm.providers has no chat provider for gateway".to_string());
    }

    for provider in &chat_providers {
        let weight = provider.weight.unwrap_or(1);
        if weight >= 100 {
            return Err(format!(
                "Provider '{}' has invalid weight {weight}, expected value < 100",
                provider.name
            ));
        }
    }

    let selected_provider_name =
        resolve_provider_name_for_gateway(llm, &chat_providers, router_seed)?;
    let selected_provider = chat_providers
        .into_iter()
        .find(|provider| provider.name == selected_provider_name)
        .ok_or_else(|| {
            format!(
                "gateway provider '{}' not found in llm.providers",
                selected_provider_name
            )
        })?;

    build_model_from_provider(selected_provider, env_map)
}

fn resolve_provider_name_for_gateway(
    llm: &PixyTomlLlm,
    chat_providers: &[&PixyTomlProvider],
    router_seed: u64,
) -> Result<String, String> {
    if let Some(default_provider) = llm
        .default_provider
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        if default_provider == "*" {
            let total_weight: u64 = chat_providers
                .iter()
                .map(|provider| provider.weight.unwrap_or(1) as u64)
                .sum();
            if total_weight == 0 {
                return Err(
                    "default_provider='*' requires at least one chat provider with non-zero weight"
                        .to_string(),
                );
            }
            let mut cursor = router_seed % total_weight;
            for provider in chat_providers {
                let weight = provider.weight.unwrap_or(1) as u64;
                if weight == 0 {
                    continue;
                }
                if cursor < weight {
                    return Ok(provider.name.clone());
                }
                cursor -= weight;
            }
        } else {
            return Ok(default_provider.to_string());
        }
    }

    if chat_providers.len() == 1 {
        return Ok(chat_providers[0].name.clone());
    }

    Ok(chat_providers[0].name.clone())
}

fn build_model_from_provider(
    provider: &PixyTomlProvider,
    env_map: &HashMap<String, String>,
) -> Result<ResolvedModelSelection, String> {
    let provider_name = provider
        .provider
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| provider.name.clone());
    let model_id = provider
        .model
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .or_else(|| default_model_for_provider(&provider_name))
        .ok_or_else(|| format!("Unable to resolve model for provider '{}'", provider.name))?;
    let api = provider
        .api
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .or_else(|| infer_api_for_provider(&provider_name))
        .ok_or_else(|| format!("Unable to resolve API for provider '{}'", provider.name))?;

    let base_url = provider
        .base_url
        .as_deref()
        .and_then(|value| resolve_config_value(value, env_map))
        .or_else(|| default_base_url_for_api(&api))
        .ok_or_else(|| {
            format!(
                "Unable to resolve base URL for provider '{}'",
                provider.name
            )
        })?;

    let api_key = provider
        .api_key
        .as_deref()
        .and_then(|value| resolve_config_value(value, env_map))
        .or_else(|| infer_api_key_from_settings(&provider_name, env_map))
        .or_else(|| std::env::var(primary_env_key_for_provider(&provider_name)).ok());

    let reasoning = provider
        .reasoning
        .unwrap_or_else(|| default_reasoning_enabled_for_api(&api));
    let reasoning_effort = provider.reasoning_effort.clone().or_else(|| {
        if reasoning {
            Some(pixy_ai::ThinkingLevel::Medium)
        } else {
            None
        }
    });

    Ok(ResolvedModelSelection {
        model: Model {
            id: model_id.clone(),
            name: model_id,
            api,
            provider: provider_name,
            base_url,
            reasoning,
            reasoning_effort,
            input: vec!["text".to_string()],
            cost: pixy_ai::Cost {
                input: 0.0,
                output: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
                total: 0.0,
            },
            context_window: provider.context_window.unwrap_or(200_000),
            max_tokens: provider.max_tokens.unwrap_or(8_192),
        },
        api_key,
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

fn is_chat_provider(kind: Option<&str>) -> bool {
    kind.map(str::trim)
        .map_or(true, |kind| kind.eq_ignore_ascii_case("chat"))
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

fn infer_api_key_from_settings(
    provider: &str,
    env_map: &HashMap<String, String>,
) -> Option<String> {
    let provider_upper = provider.to_uppercase().replace('-', "_");
    let provider_api_key = format!("{provider_upper}_API_KEY");
    let provider_auth_token = format!("{provider_upper}_AUTH_TOKEN");

    [
        env_map.get(&provider_api_key).cloned(),
        env_map.get(&provider_auth_token).cloned(),
        env_map.get("OPENAI_API_KEY").cloned(),
        env_map.get("ANTHROPIC_API_KEY").cloned(),
        env_map.get("ANTHROPIC_AUTH_TOKEN").cloned(),
        env_map.get("GOOGLE_API_KEY").cloned(),
    ]
    .into_iter()
    .flatten()
    .find(|value| !value.trim().is_empty())
}

fn primary_env_key_for_provider(provider: &str) -> &'static str {
    match provider {
        "anthropic" | "anthropic-messages" => "ANTHROPIC_API_KEY",
        "openai"
        | "openai-completions"
        | "openai-responses"
        | "openai-codex-responses"
        | "azure-openai"
        | "azure-openai-responses"
        | "codex" => "OPENAI_API_KEY",
        "google" | "google-generative-ai" | "google-gemini-cli" | "google-vertex" => {
            "GOOGLE_API_KEY"
        }
        "bedrock" | "amazon-bedrock" | "bedrock-converse-stream" => "AWS_ACCESS_KEY_ID",
        _ => "OPENAI_API_KEY",
    }
}

fn infer_api_for_provider(provider: &str) -> Option<String> {
    match provider {
        "openai" => Some("openai-responses".to_string()),
        "openai-completions" => Some("openai-completions".to_string()),
        "openai-responses" => Some("openai-responses".to_string()),
        "openai-codex-responses" | "codex" => Some("openai-codex-responses".to_string()),
        "azure-openai" | "azure-openai-responses" => Some("azure-openai-responses".to_string()),
        "anthropic" | "anthropic-messages" => Some("anthropic-messages".to_string()),
        "google" | "google-generative-ai" => Some("google-generative-ai".to_string()),
        "google-gemini-cli" => Some("google-gemini-cli".to_string()),
        "google-vertex" => Some("google-vertex".to_string()),
        "bedrock" | "amazon-bedrock" | "bedrock-converse-stream" => {
            Some("bedrock-converse-stream".to_string())
        }
        _ => None,
    }
}

fn default_model_for_provider(provider: &str) -> Option<String> {
    match provider {
        "openai" | "openai-completions" => Some("gpt-5.3-codex".to_string()),
        "openai-responses" | "azure-openai" | "azure-openai-responses" => {
            Some("gpt-5.3-codex".to_string())
        }
        "openai-codex-responses" | "codex" => Some("codex-mini-latest".to_string()),
        "anthropic" | "anthropic-messages" => Some("claude-3-5-sonnet-latest".to_string()),
        "google" | "google-generative-ai" | "google-gemini-cli" | "google-vertex" => {
            Some("gemini-2.5-flash".to_string())
        }
        "bedrock" | "amazon-bedrock" | "bedrock-converse-stream" => {
            Some("anthropic.claude-3-5-sonnet-20241022-v2:0".to_string())
        }
        _ => None,
    }
}

fn default_base_url_for_api(api: &str) -> Option<String> {
    match api {
        "openai-completions" | "openai-responses" | "openai-codex-responses" => {
            Some("https://api.openai.com/v1".to_string())
        }
        "anthropic-messages" => Some("https://api.anthropic.com/v1".to_string()),
        "google-generative-ai" => {
            Some("https://generativelanguage.googleapis.com/v1beta".to_string())
        }
        _ => None,
    }
}

fn default_reasoning_enabled_for_api(api: &str) -> bool {
    matches!(
        api,
        "openai-responses"
            | "openai-completions"
            | "openai-codex-responses"
            | "azure-openai-responses"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

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
