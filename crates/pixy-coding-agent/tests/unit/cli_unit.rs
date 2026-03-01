use std::collections::HashMap;
use std::collections::HashSet;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use super::*;
use crate::SkillSource;
use crate::{load_skills, LoadSkillsOptions, RuntimeLoadOptions};
use pixy_ai::{Cost, Model, DEFAULT_TRANSPORT_RETRY_COUNT};
use serde::Deserialize;
use tempfile::tempdir;

fn test_chat_args() -> ChatArgs {
    ChatArgs {
        api: None,
        provider: None,
        model: None,
        base_url: None,
        context_window: None,
        max_tokens: None,
        agent_dir: None,
        cwd: None,
        session_dir: None,
        session_file: None,
        system_prompt: Some("test".to_string()),
        prompt: None,
        continue_first: false,
        no_tools: false,
        skills: vec![],
        no_skills: false,
        hide_tool_results: false,
        no_tui: false,
        theme: None,
    }
}

#[derive(Debug, Clone, Default)]
struct AgentSettingsFile {
    default_provider: Option<String>,
    theme: Option<String>,
    transport_retry_count: Option<usize>,
    skills: Vec<String>,
    env: HashMap<String, String>,
}

#[derive(Debug, Clone, Default)]
struct ModelsFile {
    providers: HashMap<String, ProviderConfig>,
}

#[derive(Debug, Clone, Default)]
struct ProviderConfig {
    provider: Option<String>,
    kind: Option<String>,
    api: Option<String>,
    base_url: Option<String>,
    api_key: Option<String>,
    default_model: Option<String>,
    weight: u8,
    models: Vec<ProviderModelConfig>,
}

fn default_provider_weight() -> u8 {
    1
}

#[derive(Debug, Clone, Default, Deserialize)]
struct ProviderModelConfig {
    id: String,
    api: Option<String>,
    #[serde(rename = "baseUrl")]
    base_url: Option<String>,
    #[serde(rename = "reasoning", default)]
    reasoning: Option<bool>,
    #[serde(rename = "reasoningEffort", default)]
    reasoning_effort: Option<pixy_ai::ThinkingLevel>,
    #[serde(rename = "contextWindow")]
    context_window: Option<u32>,
    #[serde(rename = "maxTokens")]
    max_tokens: Option<u32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct PixyTomlFile {
    #[serde(default)]
    llm: PixyTomlLlm,
    theme: Option<String>,
    #[serde(default)]
    transport_retry_count: Option<usize>,
    #[serde(default)]
    skills: Vec<String>,
    #[serde(default)]
    env: HashMap<String, String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct PixyTomlLlm {
    #[serde(default)]
    default_provider: Option<String>,
    #[serde(default)]
    providers: Vec<PixyTomlProvider>,
}

#[derive(Debug, Clone, Default, Deserialize)]
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
    #[serde(default = "default_provider_weight")]
    weight: u8,
    #[serde(default)]
    reasoning: Option<bool>,
    #[serde(default)]
    reasoning_effort: Option<pixy_ai::ThinkingLevel>,
    #[serde(default)]
    context_window: Option<u32>,
    #[serde(default)]
    max_tokens: Option<u32>,
}

#[derive(Debug, Clone, Default)]
struct AgentLocalConfig {
    settings: AgentSettingsFile,
    models: ModelsFile,
}

#[derive(Debug, Clone)]
struct ResolvedRuntimeConfig {
    model: Model,
    model_catalog: Vec<Model>,
    api_key: Option<String>,
}

#[derive(Debug, Clone)]
pub struct LLMRouter {
    slots: Vec<Slot>,
}

#[derive(Debug, Clone)]
struct Slot {
    provider: String,
    weight: u8,
}

impl LLMRouter {
    fn from_provider_configs(providers: &HashMap<String, ProviderConfig>) -> Result<Self, String> {
        let mut slots = Vec::with_capacity(providers.len());
        for (provider, config) in providers {
            if !is_chat_provider(config) {
                continue;
            }
            if config.weight >= 100 {
                return Err(format!(
                    "Provider '{provider}' has invalid weight {}, expected value < 100",
                    config.weight
                ));
            }
            slots.push(Slot {
                provider: provider.clone(),
                weight: config.weight,
            });
        }
        slots.sort_by(|left, right| left.provider.cmp(&right.provider));
        Ok(Self { slots })
    }

    fn select_provider(&self, seed: u64) -> Option<String> {
        let total_weight: u64 = self.slots.iter().map(|slot| slot.weight as u64).sum();
        if total_weight == 0 {
            return None;
        }

        let mut cursor = seed % total_weight;
        for slot in &self.slots {
            let weight = slot.weight as u64;
            if weight == 0 {
                continue;
            }
            if cursor < weight {
                return Some(slot.provider.clone());
            }
            cursor -= weight;
        }
        None
    }
}

fn load_agent_local_config(_agent_dir: &Path) -> Result<AgentLocalConfig, String> {
    let config_path = default_pixy_config_path();
    let config = read_toml_if_exists::<PixyTomlFile>(&config_path)?;
    Ok(config
        .map(convert_pixy_toml_to_local_config)
        .unwrap_or_default())
}

fn convert_pixy_toml_to_local_config(config: PixyTomlFile) -> AgentLocalConfig {
    let mut providers = HashMap::new();
    for provider in config.llm.providers {
        if provider.name.trim().is_empty() {
            continue;
        }

        let provider_key = provider.name.trim().to_string();
        let model_id = provider.model.clone().map(|value| value.trim().to_string());
        let provider_model =
            model_id
                .clone()
                .filter(|id| !id.is_empty())
                .map(|id| ProviderModelConfig {
                    id,
                    api: provider.api.clone(),
                    base_url: provider.base_url.clone(),
                    reasoning: provider.reasoning,
                    reasoning_effort: provider.reasoning_effort.clone(),
                    context_window: provider.context_window,
                    max_tokens: provider.max_tokens,
                });

        let provider_config = ProviderConfig {
            provider: provider
                .provider
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string),
            kind: provider.kind,
            api: provider.api,
            base_url: provider.base_url,
            api_key: provider.api_key,
            default_model: model_id.filter(|id| !id.is_empty()),
            weight: provider.weight,
            models: provider_model.into_iter().collect(),
        };
        providers.insert(provider_key, provider_config);
    }

    AgentLocalConfig {
        settings: AgentSettingsFile {
            default_provider: config.llm.default_provider,
            theme: config.theme,
            transport_retry_count: config.transport_retry_count,
            skills: config.skills,
            env: config.env,
        },
        models: ModelsFile { providers },
    }
}

fn resolve_transport_retry_count(settings: &AgentSettingsFile) -> usize {
    settings
        .transport_retry_count
        .unwrap_or(DEFAULT_TRANSPORT_RETRY_COUNT)
}

fn resolve_runtime_config_for_tests(
    args: &ChatArgs,
    local: &AgentLocalConfig,
) -> Result<ResolvedRuntimeConfig, String> {
    resolve_runtime_config_for_tests_with_seed(args, local, runtime_router_seed())
}

fn runtime_router_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0)
}

fn resolve_runtime_config_for_tests_with_seed(
    args: &ChatArgs,
    local: &AgentLocalConfig,
    router_seed: u64,
) -> Result<ResolvedRuntimeConfig, String> {
    CliRuntimeConfigResolver::new(args, local, router_seed).resolve_runtime_config_with_seed()
}

struct CliRuntimeConfigResolver<'a> {
    args: &'a ChatArgs,
    local: &'a AgentLocalConfig,
    router_seed: u64,
}

impl<'a> CliRuntimeConfigResolver<'a> {
    fn new(args: &'a ChatArgs, local: &'a AgentLocalConfig, router_seed: u64) -> Self {
        Self {
            args,
            local,
            router_seed,
        }
    }

    fn resolve_runtime_config_with_seed(&self) -> Result<ResolvedRuntimeConfig, String> {
        self.resolve_runtime_config_with_seed_impl()
    }

    fn resolve_runtime_config_with_seed_impl(&self) -> Result<ResolvedRuntimeConfig, String> {
        let cli_model_parts = split_provider_model(self.args.model.as_deref());
        let explicit_provider =
            first_non_empty([self.args.provider.clone(), cli_model_parts.0.clone()]);
        let routed_provider = if explicit_provider.is_none() {
            resolve_weighted_provider_selection(self.local, self.router_seed)?
        } else {
            None
        };
        let settings_provider = self
            .local
            .settings
            .default_provider
            .clone()
            .filter(|value| value.trim() != "*");

        let provider = first_non_empty([
            explicit_provider,
            routed_provider,
            settings_provider,
            infer_single_provider(self.local),
            Some("openai".to_string()),
        ])
        .ok_or_else(|| "Unable to resolve provider".to_string())?;

        let provider_config = self.local.models.providers.get(&provider);
        if let Some(config) = provider_config {
            if config.weight >= 100 {
                return Err(format!(
                    "Provider '{provider}' has invalid weight {}, expected value < 100",
                    config.weight
                ));
            }
            if !is_chat_provider(config) {
                let kind = config.kind.as_deref().unwrap_or("unknown");
                return Err(format!(
                    "Provider '{provider}' has kind '{kind}', expected 'chat' for coding sessions"
                ));
            }
        }
        let provider_name = resolve_provider_name(&provider, provider_config);
        let model_id = first_non_empty([
            cli_model_parts.1.clone(),
            provider_config_default_model(provider_config),
            default_model_for_provider(&provider_name),
        ])
        .ok_or_else(|| format!("Unable to resolve model for provider '{provider}'"))?;

        let selected_model_cfg = provider_config.and_then(|provider_config| {
            provider_config
                .models
                .iter()
                .find(|model| model.id == model_id)
        });

        let api = first_non_empty([
            self.args.api.clone(),
            selected_model_cfg.and_then(|model| model.api.clone()),
            provider_config.and_then(|provider| provider.api.clone()),
            infer_api_for_provider(&provider_name),
        ])
        .ok_or_else(|| format!("Unable to resolve API for provider '{provider}'"))?;

        let cli_base_url = self
            .args
            .base_url
            .as_ref()
            .and_then(|value| resolve_config_value(value, &self.local.settings.env));
        let model_base_url = selected_model_cfg
            .and_then(|model| model.base_url.as_ref())
            .and_then(|value| resolve_config_value(value, &self.local.settings.env));
        let provider_base_url = provider_config
            .and_then(|provider| provider.base_url.as_ref())
            .and_then(|value| resolve_config_value(value, &self.local.settings.env));
        let base_url = first_non_empty([
            cli_base_url,
            model_base_url,
            provider_base_url,
            default_base_url_for_api(&api),
        ])
        .ok_or_else(|| format!("Unable to resolve base URL for api '{api}'"))?;

        let context_window = self
            .args
            .context_window
            .or_else(|| selected_model_cfg.and_then(|model| model.context_window))
            .unwrap_or(200_000);
        let max_tokens = self
            .args
            .max_tokens
            .or_else(|| selected_model_cfg.and_then(|model| model.max_tokens))
            .unwrap_or(8_192);

        let api_key = provider_config
            .and_then(|provider_cfg| provider_cfg.api_key.as_ref())
            .and_then(|value| resolve_config_value(value, &self.local.settings.env))
            .or_else(|| infer_api_key_from_settings(&provider_name, &self.local.settings.env))
            .or_else(|| std::env::var(primary_env_key_for_provider(&provider_name)).ok());

        let reasoning = selected_model_cfg
            .and_then(|cfg| cfg.reasoning)
            .unwrap_or_else(|| default_reasoning_enabled_for_api(&api));
        let reasoning_effort = selected_model_cfg
            .and_then(|cfg| cfg.reasoning_effort.clone())
            .or_else(|| {
                if reasoning {
                    Some(pixy_ai::ThinkingLevel::Medium)
                } else {
                    None
                }
            });

        let model = Model {
            id: model_id.clone(),
            name: model_id,
            api,
            provider: provider_name.clone(),
            base_url,
            reasoning,
            reasoning_effort,
            input: vec!["text".to_string()],
            cost: Cost {
                input: 0.0,
                output: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
                total: 0.0,
            },
            context_window,
            max_tokens,
        };

        let mut model_catalog = provider_config
            .map(|provider_cfg| {
                provider_cfg
                    .models
                    .iter()
                    .map(|entry| {
                        model_from_config(
                            entry,
                            &provider_name,
                            &model.api,
                            &model.base_url,
                            model.context_window,
                            model.max_tokens,
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        dedupe_models(&mut model_catalog);

        if let Some(position) = model_catalog
            .iter()
            .position(|entry| entry.provider == model.provider && entry.id == model.id)
        {
            model_catalog.remove(position);
        }
        model_catalog.insert(0, model.clone());

        Ok(ResolvedRuntimeConfig {
            model,
            model_catalog,
            api_key,
        })
    }
}

fn resolve_weighted_provider_selection(
    local: &AgentLocalConfig,
    router_seed: u64,
) -> Result<Option<String>, String> {
    let use_weighted_provider = local
        .settings
        .default_provider
        .as_deref()
        .map(str::trim)
        .is_some_and(|provider| provider == "*");
    if !use_weighted_provider {
        return Ok(None);
    }

    let router = LLMRouter::from_provider_configs(&local.models.providers)?;
    let provider = router.select_provider(router_seed).ok_or_else(|| {
        "default_provider='*' requires at least one chat provider with non-zero weight in <conf_dir>/pixy.toml".to_string()
    })?;
    Ok(Some(provider))
}

fn provider_config_default_model(provider_config: Option<&ProviderConfig>) -> Option<String> {
    provider_config.and_then(|config| {
        first_non_empty([
            config.default_model.clone(),
            config.models.first().map(|entry| entry.id.clone()),
        ])
    })
}

fn resolve_provider_name(provider_key: &str, provider_config: Option<&ProviderConfig>) -> String {
    provider_config
        .and_then(|config| config.provider.clone())
        .and_then(|value| {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        })
        .unwrap_or_else(|| provider_key.to_string())
}

fn infer_single_provider(local: &AgentLocalConfig) -> Option<String> {
    let chat_providers = local
        .models
        .providers
        .iter()
        .filter(|(_, config)| is_chat_provider(config))
        .map(|(provider, _)| provider.clone())
        .collect::<Vec<_>>();
    if chat_providers.len() == 1 {
        chat_providers.into_iter().next()
    } else {
        None
    }
}

fn is_chat_provider(config: &ProviderConfig) -> bool {
    config
        .kind
        .as_deref()
        .map(str::trim)
        .map_or(true, |kind| kind.eq_ignore_ascii_case("chat"))
}

fn model_from_config(
    config: &ProviderModelConfig,
    provider: &str,
    default_api: &str,
    default_base_url: &str,
    default_context_window: u32,
    default_max_tokens: u32,
) -> Model {
    let api = config
        .api
        .clone()
        .unwrap_or_else(|| default_api.to_string());
    let base_url = config
        .base_url
        .clone()
        .or_else(|| default_base_url_for_api(&api))
        .unwrap_or_else(|| default_base_url.to_string());
    let reasoning = config
        .reasoning
        .unwrap_or_else(|| default_reasoning_enabled_for_api(&api));
    let reasoning_effort = config.reasoning_effort.clone().or_else(|| {
        if reasoning {
            Some(pixy_ai::ThinkingLevel::Medium)
        } else {
            None
        }
    });

    Model {
        id: config.id.clone(),
        name: config.id.clone(),
        api,
        provider: provider.to_string(),
        base_url,
        reasoning,
        reasoning_effort,
        input: vec!["text".to_string()],
        cost: Cost {
            input: 0.0,
            output: 0.0,
            cache_read: 0.0,
            cache_write: 0.0,
            total: 0.0,
        },
        context_window: config.context_window.unwrap_or(default_context_window),
        max_tokens: config.max_tokens.unwrap_or(default_max_tokens),
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

fn dedupe_models(models: &mut Vec<Model>) {
    let mut deduped = Vec::new();
    for model in std::mem::take(models) {
        if deduped
            .iter()
            .any(|existing: &Model| existing.provider == model.provider && existing.id == model.id)
        {
            continue;
        }
        deduped.push(model);
    }
    *models = deduped;
}

fn split_provider_model(input: Option<&str>) -> (Option<String>, Option<String>) {
    let Some(raw) = input.map(str::trim).filter(|value| !value.is_empty()) else {
        return (None, None);
    };

    if let Some((provider, model)) = raw.split_once('/') {
        if !provider.is_empty() && !model.is_empty() {
            return (Some(provider.to_string()), Some(model.to_string()));
        }
    }

    (None, Some(raw.to_string()))
}

fn first_non_empty<const N: usize>(candidates: [Option<String>; N]) -> Option<String> {
    for candidate in candidates {
        if let Some(value) = candidate {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
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

fn load_runtime_skills(
    args: &ChatArgs,
    local: &AgentLocalConfig,
    cwd: &Path,
    agent_dir: &Path,
) -> Vec<Skill> {
    let mut options = LoadSkillsOptions::new(cwd.to_path_buf(), agent_dir.to_path_buf());
    options.include_defaults = !args.no_skills;
    options.skill_paths = local.settings.skills.clone();
    options.skill_paths.extend(args.skills.clone());

    let result = load_skills(options);
    result.skills
}

#[test]
fn cli_accepts_explicit_cli_subcommand() {
    let parsed = Cli::try_parse_from(["pixy", "cli", "--no-tui"]);
    assert!(
        parsed.is_ok(),
        "pixy cli should be accepted as explicit CLI entrypoint"
    );
}

#[test]
fn cli_accepts_gateway_start_daemon_subcommand() {
    let parsed = Cli::try_parse_from(["pixy", "gateway", "start", "--daemon"]);
    assert!(
        parsed.is_ok(),
        "pixy gateway start --daemon should be accepted"
    );
}

#[test]
fn cli_accepts_conf_dir_global_flag() {
    let parsed = Cli::try_parse_from(["pixy", "--conf-dir", "/tmp/pixy-conf", "gateway", "start"]);
    assert!(
        parsed.is_ok(),
        "pixy should accept --conf-dir as global flag"
    );
}

#[test]
fn gateway_command_tokens_include_daemon_flag_when_requested() {
    let tokens = gateway_command_tokens(
        &GatewaySubcommand::Start(GatewayStartArgs { daemon: true }),
        None,
    );
    assert_eq!(tokens, vec!["start".to_string(), "--daemon".to_string()]);
}

#[test]
fn gateway_command_tokens_encode_restart_subcommand() {
    let tokens = gateway_command_tokens(&GatewaySubcommand::Restart, None);
    assert_eq!(tokens, vec!["restart".to_string()]);
}

#[test]
fn gateway_command_tokens_include_conf_dir_when_provided() {
    let conf_dir = Path::new("/tmp/pixy-conf");
    let tokens = gateway_command_tokens(&GatewaySubcommand::Stop, Some(conf_dir));
    assert_eq!(
        tokens,
        vec![
            "--conf-dir".to_string(),
            "/tmp/pixy-conf".to_string(),
            "stop".to_string(),
        ]
    );
}

#[test]
fn new_repl_command_matches_exact_new_token() {
    use crate::cli_app::{ReplCommand, ReplCommandParser};

    assert_eq!(
        ReplCommandParser::parse("/new"),
        Some(ReplCommand::NewSession)
    );
    assert_eq!(
        ReplCommandParser::parse(" /new "),
        Some(ReplCommand::NewSession)
    );
    assert_eq!(
        ReplCommandParser::parse("/new task"),
        Some(ReplCommand::Prompt {
            text: "/new task".to_string()
        })
    );
    assert_eq!(
        ReplCommandParser::parse("new"),
        Some(ReplCommand::Prompt {
            text: "new".to_string()
        })
    );
}

#[test]
fn repl_command_parser_recognizes_control_commands() {
    use crate::cli_app::{ReplCommand, ReplCommandParser};

    assert_eq!(
        ReplCommandParser::parse("/new"),
        Some(ReplCommand::NewSession)
    );
    assert_eq!(
        ReplCommandParser::parse("/resume"),
        Some(ReplCommand::Resume { target: None })
    );
    assert_eq!(
        ReplCommandParser::parse("/resume  abc.jsonl "),
        Some(ReplCommand::Resume {
            target: Some("abc.jsonl".to_string())
        })
    );
    assert_eq!(
        ReplCommandParser::parse("/continue"),
        Some(ReplCommand::Continue)
    );
    assert_eq!(
        ReplCommandParser::parse("/session"),
        Some(ReplCommand::Session)
    );
    assert_eq!(ReplCommandParser::parse("/help"), Some(ReplCommand::Help));
    assert_eq!(ReplCommandParser::parse("/quit"), Some(ReplCommand::Exit));
    assert_eq!(ReplCommandParser::parse("/exit"), Some(ReplCommand::Exit));
}

#[test]
fn repl_command_parser_treats_other_text_as_prompt() {
    use crate::cli_app::{ReplCommand, ReplCommandParser};

    assert_eq!(
        ReplCommandParser::parse(" summarize this file "),
        Some(ReplCommand::Prompt {
            text: "summarize this file".to_string()
        })
    );
    assert_eq!(ReplCommandParser::parse(""), None);
    assert_eq!(ReplCommandParser::parse("   "), None);
}

#[test]
fn resolves_default_provider_model_from_settings_and_models() {
    let args = ChatArgs {
        api: None,
        provider: None,
        model: None,
        base_url: None,
        context_window: None,
        max_tokens: None,
        agent_dir: None,
        cwd: None,
        session_dir: None,
        session_file: None,
        system_prompt: Some("test".to_string()),
        prompt: None,
        continue_first: false,
        no_tools: false,
        skills: vec![],
        no_skills: false,
        hide_tool_results: false,
        no_tui: false,
        theme: None,
    };

    let local = AgentLocalConfig {
        settings: AgentSettingsFile {
            default_provider: Some("anthropic".to_string()),
            theme: None,
            transport_retry_count: None,
            skills: vec![],
            env: HashMap::from([(
                "ANTHROPIC_AUTH_TOKEN".to_string(),
                "token-from-settings".to_string(),
            )]),
        },
        models: ModelsFile {
            providers: HashMap::from([(
                "anthropic".to_string(),
                ProviderConfig {
                    provider: None,
                    kind: None,
                    api: None,
                    base_url: Some("https://custom.anthropic.local".to_string()),
                    api_key: Some("model-json-token".to_string()),
                    default_model: Some("claude-opus-4-6".to_string()),
                    weight: 1,
                    models: vec![],
                },
            )]),
        },
    };

    let resolved =
        resolve_runtime_config_for_tests(&args, &local).expect("runtime config should resolve");
    assert_eq!(resolved.model.provider, "anthropic");
    assert_eq!(resolved.model.api, "anthropic-messages");
    assert_eq!(resolved.model.id, "claude-opus-4-6");
    assert_eq!(resolved.model.base_url, "https://custom.anthropic.local");
    assert_eq!(resolved.api_key.as_deref(), Some("model-json-token"));
}

#[test]
fn wildcard_default_provider_selects_provider_by_weight() {
    let args = test_chat_args();
    let local = AgentLocalConfig {
        settings: AgentSettingsFile {
            default_provider: Some("*".to_string()),
            ..AgentSettingsFile::default()
        },
        models: ModelsFile {
            providers: HashMap::from([
                (
                    "openai".to_string(),
                    ProviderConfig {
                        provider: None,
                        kind: None,
                        api: Some("openai-responses".to_string()),
                        base_url: Some("https://api.openai.com/v1".to_string()),
                        api_key: None,
                        default_model: Some("gpt-5.3-codex".to_string()),
                        weight: 80,
                        models: vec![],
                    },
                ),
                (
                    "anthropic".to_string(),
                    ProviderConfig {
                        provider: None,
                        kind: None,
                        api: Some("anthropic-messages".to_string()),
                        base_url: Some("https://api.anthropic.com/v1".to_string()),
                        api_key: None,
                        default_model: Some("claude-3-5-sonnet-latest".to_string()),
                        weight: 20,
                        models: vec![],
                    },
                ),
            ]),
        },
    };

    let weighted_anthropic = resolve_runtime_config_for_tests_with_seed(&args, &local, 10)
        .expect("runtime config should resolve");
    assert_eq!(weighted_anthropic.model.provider, "anthropic");

    let weighted_openai = resolve_runtime_config_for_tests_with_seed(&args, &local, 90)
        .expect("runtime config should resolve");
    assert_eq!(weighted_openai.model.provider, "openai");
}

#[test]
fn resolve_runtime_config_matches_runtime_module_with_same_seed() {
    let dir = tempdir().expect("tempdir");
    let pixy_dir = dir.path().join(".pixy");
    std::fs::create_dir_all(&pixy_dir).expect("create .pixy");
    std::fs::write(
        pixy_dir.join("pixy.toml"),
        r#"
[env]
PIXY_TEST_OPENAI_KEY = "sk-from-settings"

[llm]
default_provider = "*"

[[llm.providers]]
name = "openai"
kind = "chat"
provider = "openai"
api = "openai-responses"
base_url = "https://api.openai.com/v1"
api_key = "$PIXY_TEST_OPENAI_KEY"
model = "gpt-5.3-codex"
weight = 90

[[llm.providers]]
name = "anthropic"
kind = "chat"
provider = "anthropic"
api = "anthropic-messages"
base_url = "https://api.anthropic.com/v1"
api_key = "sk-anthropic"
model = "claude-3-5-sonnet-latest"
weight = 10
"#,
    )
    .expect("write pixy.toml");

    unsafe {
        std::env::set_var("HOME", dir.path());
    }

    let args = test_chat_args();
    let local = load_agent_local_config(Path::new(".")).expect("load local config");
    let cli_resolved = resolve_runtime_config_for_tests_with_seed(&args, &local, 42)
        .expect("cli runtime config should resolve");

    let runtime_options = RuntimeLoadOptions {
        conf_dir: Some(pixy_dir.clone()),
        agent_dir: Some(dir.path().join("agents")),
        load_skills: false,
        include_default_skills: false,
        skill_paths: vec![],
        overrides: RuntimeOverrides {
            api: args.api.clone(),
            provider: args.provider.clone(),
            model: args.model.clone(),
            base_url: args.base_url.clone(),
            context_window: args.context_window,
            max_tokens: args.max_tokens,
            ..RuntimeOverrides::default()
        },
    };
    let runtime_resolved = runtime_options
        .resolve_runtime_with_seed(Path::new("."), 42)
        .expect("runtime module should resolve");

    assert_eq!(cli_resolved.model.provider, runtime_resolved.model.provider);
    assert_eq!(cli_resolved.model.id, runtime_resolved.model.id);
    assert_eq!(cli_resolved.model.api, runtime_resolved.model.api);
    assert_eq!(cli_resolved.model.base_url, runtime_resolved.model.base_url);
    assert_eq!(
        cli_resolved.model.reasoning,
        runtime_resolved.model.reasoning
    );
    assert_eq!(
        cli_resolved.model.reasoning_effort,
        runtime_resolved.model.reasoning_effort
    );
    assert_eq!(cli_resolved.api_key, runtime_resolved.api_key);
}

#[test]
fn wildcard_default_provider_rejects_invalid_provider_weight() {
    let args = test_chat_args();
    let local = AgentLocalConfig {
        settings: AgentSettingsFile {
            default_provider: Some("*".to_string()),
            ..AgentSettingsFile::default()
        },
        models: ModelsFile {
            providers: HashMap::from([(
                "openai".to_string(),
                ProviderConfig {
                    provider: None,
                    kind: None,
                    api: Some("openai-responses".to_string()),
                    base_url: Some("https://api.openai.com/v1".to_string()),
                    api_key: None,
                    default_model: Some("gpt-5.3-codex".to_string()),
                    weight: 100,
                    models: vec![],
                },
            )]),
        },
    };

    let error = resolve_runtime_config_for_tests_with_seed(&args, &local, 0)
        .expect_err("invalid weight should be rejected");
    assert!(error.contains("expected value < 100"));
}

#[test]
fn fixed_provider_rejects_invalid_provider_weight() {
    let mut args = test_chat_args();
    args.provider = Some("openai".to_string());

    let local = AgentLocalConfig {
        settings: AgentSettingsFile::default(),
        models: ModelsFile {
            providers: HashMap::from([(
                "openai".to_string(),
                ProviderConfig {
                    provider: None,
                    kind: None,
                    api: Some("openai-responses".to_string()),
                    base_url: Some("https://api.openai.com/v1".to_string()),
                    api_key: None,
                    default_model: Some("gpt-5.3-codex".to_string()),
                    weight: 100,
                    models: vec![],
                },
            )]),
        },
    };

    let error = resolve_runtime_config_for_tests_with_seed(&args, &local, 0)
        .expect_err("invalid weight should be rejected");
    assert!(error.contains("expected value < 100"));
}

#[test]
fn explicit_provider_bypasses_weighted_router_validation() {
    let mut args = test_chat_args();
    args.provider = Some("openai".to_string());
    args.model = Some("gpt-5.3-codex".to_string());

    let local = AgentLocalConfig {
        settings: AgentSettingsFile {
            default_provider: Some("*".to_string()),
            ..AgentSettingsFile::default()
        },
        models: ModelsFile {
            providers: HashMap::from([
                (
                    "openai".to_string(),
                    ProviderConfig {
                        provider: None,
                        kind: None,
                        api: Some("openai-responses".to_string()),
                        base_url: Some("https://api.openai.com/v1".to_string()),
                        api_key: None,
                        default_model: Some("gpt-5.3-codex".to_string()),
                        weight: 10,
                        models: vec![],
                    },
                ),
                (
                    "anthropic".to_string(),
                    ProviderConfig {
                        provider: None,
                        kind: None,
                        api: Some("anthropic-messages".to_string()),
                        base_url: Some("https://api.anthropic.com/v1".to_string()),
                        api_key: None,
                        default_model: Some("claude-3-5-sonnet-latest".to_string()),
                        weight: 100,
                        models: vec![],
                    },
                ),
            ]),
        },
    };

    let resolved = resolve_runtime_config_for_tests_with_seed(&args, &local, 0)
        .expect("runtime config should resolve");
    assert_eq!(resolved.model.provider, "openai");
    assert_eq!(resolved.model.id, "gpt-5.3-codex");
}

#[test]
fn fixed_default_provider_bypasses_weighted_router() {
    let args = test_chat_args();
    let local = AgentLocalConfig {
        settings: AgentSettingsFile {
            default_provider: Some("openai".to_string()),
            ..AgentSettingsFile::default()
        },
        models: ModelsFile {
            providers: HashMap::from([
                (
                    "openai".to_string(),
                    ProviderConfig {
                        provider: None,
                        kind: None,
                        api: Some("openai-responses".to_string()),
                        base_url: Some("https://api.openai.com/v1".to_string()),
                        api_key: None,
                        default_model: Some("gpt-5.3-codex".to_string()),
                        weight: 1,
                        models: vec![],
                    },
                ),
                (
                    "anthropic".to_string(),
                    ProviderConfig {
                        provider: None,
                        kind: None,
                        api: Some("anthropic-messages".to_string()),
                        base_url: Some("https://api.anthropic.com/v1".to_string()),
                        api_key: None,
                        default_model: Some("claude-3-5-sonnet-latest".to_string()),
                        weight: 99,
                        models: vec![],
                    },
                ),
            ]),
        },
    };

    let resolved = resolve_runtime_config_for_tests_with_seed(&args, &local, 0)
        .expect("runtime config should resolve");
    assert_eq!(resolved.model.provider, "openai");
}

#[test]
fn resolve_runtime_config_uses_first_provider_model_when_default_model_missing() {
    let args = test_chat_args();
    let local = AgentLocalConfig {
        settings: AgentSettingsFile {
            default_provider: Some("openai".to_string()),
            ..AgentSettingsFile::default()
        },
        models: ModelsFile {
            providers: HashMap::from([(
                "openai".to_string(),
                ProviderConfig {
                    provider: None,
                    kind: None,
                    api: Some("openai-responses".to_string()),
                    base_url: Some("https://api.openai.com/v1".to_string()),
                    api_key: None,
                    default_model: None,
                    weight: 1,
                    models: vec![
                        ProviderModelConfig {
                            id: "gpt-4.1".to_string(),
                            api: None,
                            base_url: None,
                            reasoning: None,
                            reasoning_effort: None,
                            context_window: None,
                            max_tokens: None,
                        },
                        ProviderModelConfig {
                            id: "gpt-5.3-codex".to_string(),
                            api: None,
                            base_url: None,
                            reasoning: None,
                            reasoning_effort: None,
                            context_window: None,
                            max_tokens: None,
                        },
                    ],
                },
            )]),
        },
    };

    let resolved =
        resolve_runtime_config_for_tests(&args, &local).expect("runtime config should resolve");
    assert_eq!(resolved.model.id, "gpt-4.1");
}

#[test]
fn cli_model_with_provider_prefix_overrides_settings_provider() {
    let args = ChatArgs {
        api: None,
        provider: None,
        model: Some("openai/gpt-4o-mini".to_string()),
        base_url: None,
        context_window: None,
        max_tokens: None,
        agent_dir: None,
        cwd: None,
        session_dir: None,
        session_file: None,
        system_prompt: Some("test".to_string()),
        prompt: None,
        continue_first: false,
        no_tools: false,
        skills: vec![],
        no_skills: false,
        hide_tool_results: false,
        no_tui: false,
        theme: None,
    };

    let local = AgentLocalConfig {
        settings: AgentSettingsFile {
            default_provider: Some("anthropic".to_string()),
            theme: None,
            transport_retry_count: None,
            skills: vec![],
            env: HashMap::new(),
        },
        models: ModelsFile::default(),
    };

    let resolved =
        resolve_runtime_config_for_tests(&args, &local).expect("runtime config should resolve");
    assert_eq!(resolved.model.provider, "openai");
    assert_eq!(resolved.model.api, "openai-responses");
    assert_eq!(resolved.model.id, "gpt-4o-mini");
}

#[test]
fn provider_api_key_can_be_resolved_from_settings_env() {
    let args = ChatArgs {
        api: None,
        provider: Some("anthropic".to_string()),
        model: Some("claude-opus-4-6".to_string()),
        base_url: Some("https://api.anthropic.com/v1".to_string()),
        context_window: None,
        max_tokens: None,
        agent_dir: None,
        cwd: None,
        session_dir: None,
        session_file: None,
        system_prompt: Some("test".to_string()),
        prompt: None,
        continue_first: false,
        no_tools: false,
        skills: vec![],
        no_skills: false,
        hide_tool_results: false,
        no_tui: false,
        theme: None,
    };

    let local = AgentLocalConfig {
        settings: AgentSettingsFile {
            default_provider: None,
            theme: None,
            transport_retry_count: None,
            skills: vec![],
            env: HashMap::from([(
                "ANTHROPIC_AUTH_TOKEN".to_string(),
                "anthropic-token".to_string(),
            )]),
        },
        models: ModelsFile::default(),
    };

    let resolved =
        resolve_runtime_config_for_tests(&args, &local).expect("runtime config should resolve");
    assert_eq!(resolved.api_key.as_deref(), Some("anthropic-token"));
}

#[test]
fn provider_base_url_can_be_resolved_from_settings_env() {
    let mut args = test_chat_args();
    args.provider = Some("openai".to_string());
    args.model = Some("gpt-5.3-codex".to_string());

    let local = AgentLocalConfig {
        settings: AgentSettingsFile {
            env: HashMap::from([(
                "LLM_BASE_URL".to_string(),
                "https://codex.databend.cloud".to_string(),
            )]),
            ..AgentSettingsFile::default()
        },
        models: ModelsFile {
            providers: HashMap::from([(
                "openai".to_string(),
                ProviderConfig {
                    provider: Some("openai".to_string()),
                    kind: Some("chat".to_string()),
                    api: Some("openai-responses".to_string()),
                    base_url: Some("$LLM_BASE_URL".to_string()),
                    api_key: None,
                    default_model: Some("gpt-5.3-codex".to_string()),
                    weight: 1,
                    models: vec![],
                },
            )]),
        },
    };

    let resolved =
        resolve_runtime_config_for_tests(&args, &local).expect("runtime config should resolve");
    assert_eq!(resolved.model.base_url, "https://codex.databend.cloud");
}

#[test]
fn resolve_runtime_config_builds_model_catalog_with_selected_model_first() {
    let args = ChatArgs {
        api: None,
        provider: Some("openai".to_string()),
        model: Some("gpt-5.3-codex".to_string()),
        base_url: None,
        context_window: None,
        max_tokens: None,
        agent_dir: None,
        cwd: None,
        session_dir: None,
        session_file: None,
        system_prompt: Some("test".to_string()),
        prompt: None,
        continue_first: false,
        no_tools: false,
        skills: vec![],
        no_skills: false,
        hide_tool_results: false,
        no_tui: false,
        theme: None,
    };

    let local = AgentLocalConfig {
        settings: AgentSettingsFile::default(),
        models: ModelsFile {
            providers: HashMap::from([(
                "openai".to_string(),
                ProviderConfig {
                    provider: None,
                    kind: None,
                    api: Some("openai-completions".to_string()),
                    base_url: Some("https://api.openai.com/v1".to_string()),
                    api_key: None,
                    default_model: None,
                    weight: 1,
                    models: vec![
                        ProviderModelConfig {
                            id: "gpt-4.1".to_string(),
                            api: None,
                            base_url: None,
                            reasoning: None,
                            reasoning_effort: None,
                            context_window: None,
                            max_tokens: None,
                        },
                        ProviderModelConfig {
                            id: "gpt-5.3-codex".to_string(),
                            api: None,
                            base_url: None,
                            reasoning: None,
                            reasoning_effort: None,
                            context_window: None,
                            max_tokens: None,
                        },
                    ],
                },
            )]),
        },
    };

    let resolved =
        resolve_runtime_config_for_tests(&args, &local).expect("runtime config should resolve");
    assert_eq!(resolved.model_catalog.len(), 2);
    assert_eq!(resolved.model_catalog[0].id, "gpt-5.3-codex");
    assert_eq!(resolved.model_catalog[1].id, "gpt-4.1");
}

#[test]
fn load_tui_keybindings_reads_supported_actions() {
    let dir = tempdir().expect("tempdir");
    let config_path = dir.path().join("keybindings.json");
    std::fs::write(
        &config_path,
        r#"{
  "clear": "ctrl+l",
  "exit": "ctrl+q",
  "interrupt": ["escape", "ctrl+c"],
  "cycleThinkingLevel": "shift+tab",
  "cycleModelForward": "ctrl+p",
  "cycleModelBackward": "shift+ctrl+p",
  "cyclePermissionMode": "ctrl+l",
  "selectModel": "ctrl+k",
  "expandTools": ["invalid", "ctrl+e"],
  "toggleThinking": "ctrl+y",
  "followUp": "alt+enter",
  "dequeue": "alt+up"
}"#,
    )
    .expect("write keybindings");

    let bindings = load_tui_keybindings(dir.path()).expect("bindings should parse");
    assert_eq!(
        bindings.clear,
        vec![parse_key_id("ctrl+l").expect("parse ctrl+l")]
    );
    assert_eq!(
        bindings.quit,
        vec![parse_key_id("ctrl+q").expect("parse ctrl+q")]
    );
    assert_eq!(
        bindings.interrupt,
        vec![
            parse_key_id("escape").expect("parse escape"),
            parse_key_id("ctrl+c").expect("parse ctrl+c")
        ]
    );
    assert_eq!(
        bindings.cycle_thinking_level,
        vec![parse_key_id("shift+tab").expect("parse shift+tab")]
    );
    assert_eq!(
        bindings.cycle_model_forward,
        vec![parse_key_id("ctrl+p").expect("parse ctrl+p")]
    );
    assert_eq!(
        bindings.cycle_model_backward,
        vec![parse_key_id("shift+ctrl+p").expect("parse shift+ctrl+p")]
    );
    assert_eq!(
        bindings.cycle_permission_mode,
        vec![parse_key_id("ctrl+l").expect("parse ctrl+l")]
    );
    assert_eq!(
        bindings.select_model,
        vec![parse_key_id("ctrl+k").expect("parse ctrl+k")]
    );
    assert_eq!(
        bindings.expand_tools,
        vec![parse_key_id("ctrl+e").expect("parse ctrl+e")]
    );
    assert_eq!(
        bindings.toggle_thinking,
        vec![parse_key_id("ctrl+y").expect("parse ctrl+y")]
    );
    assert_eq!(
        bindings.continue_run,
        vec![parse_key_id("alt+enter").expect("parse alt+enter")]
    );
    assert_eq!(
        bindings.dequeue,
        vec![parse_key_id("alt+up").expect("parse alt+up")]
    );
}

#[test]
fn load_tui_keybindings_ignores_invalid_json() {
    let dir = tempdir().expect("tempdir");
    let config_path = dir.path().join("keybindings.json");
    std::fs::write(&config_path, "{").expect("write invalid keybindings");

    assert!(load_tui_keybindings(dir.path()).is_none());
}

#[test]
fn resolve_tui_theme_name_prefers_cli_then_settings_then_default() {
    assert_eq!(
        resolve_tui_theme_name(Some("light"), Some("dark")).expect("cli theme should win"),
        "light"
    );
    assert_eq!(
        resolve_tui_theme_name(None, Some("light")).expect("settings theme should be used"),
        "light"
    );
    assert_eq!(
        resolve_tui_theme_name(None, None).expect("default theme should be dark"),
        "dark"
    );
}

#[test]
fn resolve_tui_theme_name_rejects_unknown_values() {
    let error = resolve_tui_theme_name(Some("solarized"), None).expect_err("invalid theme");
    assert!(error.contains("unsupported theme"));
    assert!(error.contains("dark"));
    assert!(error.contains("light"));
}

#[test]
fn cli_stream_renderer_formats_deltas_and_tool_lines() {
    let mut renderer = CliStreamRenderer::new(Vec::<u8>::new(), true);

    renderer
        .on_update(AgentSessionStreamUpdate::AssistantTextDelta(
            "hello".to_string(),
        ))
        .expect("delta write succeeds");
    renderer
        .on_update(AgentSessionStreamUpdate::AssistantTextDelta(
            " world".to_string(),
        ))
        .expect("delta write succeeds");
    renderer
        .on_update(AgentSessionStreamUpdate::AssistantLine(String::new()))
        .expect("assistant line succeeds");
    renderer
        .on_update(AgentSessionStreamUpdate::ToolLine(
            "[tool:read:ok]".to_string(),
        ))
        .expect("tool line succeeds");
    renderer
        .on_update(AgentSessionStreamUpdate::ToolLine(
            "tool output".to_string(),
        ))
        .expect("tool line succeeds");
    renderer.finish().expect("finish succeeds");

    let output = String::from_utf8(renderer.into_inner()).expect("utf-8 output");
    assert_eq!(output, "hello world\n[tool:read:ok]\ntool output\n");
}

#[test]
fn cli_stream_renderer_hides_tool_lines_when_disabled() {
    let mut renderer = CliStreamRenderer::new(Vec::<u8>::new(), false);

    renderer
        .on_update(AgentSessionStreamUpdate::AssistantTextDelta(
            "hi".to_string(),
        ))
        .expect("delta write succeeds");
    renderer
        .on_update(AgentSessionStreamUpdate::AssistantLine(String::new()))
        .expect("assistant line succeeds");
    renderer
        .on_update(AgentSessionStreamUpdate::ToolLine(
            "[tool:read:ok]".to_string(),
        ))
        .expect("tool line ignored");
    renderer.finish().expect("finish succeeds");

    let output = String::from_utf8(renderer.into_inner()).expect("utf-8 output");
    assert_eq!(output, "hi\n");
}

#[test]
fn render_messages_omits_read_tool_text_blocks() {
    let messages = vec![Message::ToolResult {
        tool_call_id: "call-read-1".to_string(),
        tool_name: "read".to_string(),
        content: vec![ToolResultContentBlock::Text {
            text: "secret file body".to_string(),
            text_signature: None,
        }],
        details: None,
        is_error: false,
        timestamp: 1_700_000_000_010,
    }];

    let mut output = Vec::<u8>::new();
    render_messages_to_writer(&messages, true, &mut output).expect("render succeeds");
    assert_eq!(
        String::from_utf8(output).expect("utf-8"),
        "[tool:read:ok]\n",
        "read tool output body should be hidden in CLI message rendering"
    );
}

#[test]
fn render_messages_keeps_non_read_tool_text_blocks() {
    let messages = vec![Message::ToolResult {
        tool_call_id: "call-write-1".to_string(),
        tool_name: "write".to_string(),
        content: vec![ToolResultContentBlock::Text {
            text: "write diff stat".to_string(),
            text_signature: None,
        }],
        details: None,
        is_error: false,
        timestamp: 1_700_000_000_020,
    }];

    let mut output = Vec::<u8>::new();
    render_messages_to_writer(&messages, true, &mut output).expect("render succeeds");
    assert_eq!(
        String::from_utf8(output).expect("utf-8"),
        "[tool:write:ok]\nwrite diff stat\n"
    );
}

#[test]
fn cli_stream_renderer_updates_thinking_line_in_place() {
    let mut renderer = CliStreamRenderer::new(Vec::<u8>::new(), true);

    renderer
        .on_update(AgentSessionStreamUpdate::AssistantLine(
            "[thinking] a".to_string(),
        ))
        .expect("thinking line succeeds");
    renderer
        .on_update(AgentSessionStreamUpdate::AssistantLine(
            "[thinking] ab".to_string(),
        ))
        .expect("thinking update succeeds");
    renderer
        .on_update(AgentSessionStreamUpdate::AssistantLine(String::new()))
        .expect("flush line succeeds");
    renderer.finish().expect("finish succeeds");

    let output = String::from_utf8(renderer.into_inner()).expect("utf-8 output");
    assert_eq!(output, "[thinking] a\r\u{1b}[2K[thinking] ab\n");
}

#[test]
fn cli_stream_renderer_replaces_multiline_thinking_block() {
    let mut renderer = CliStreamRenderer::new(Vec::<u8>::new(), true);

    renderer
        .on_update(AgentSessionStreamUpdate::AssistantLine(
            "[thinking] line1\nline2".to_string(),
        ))
        .expect("initial multiline thinking succeeds");
    renderer
        .on_update(AgentSessionStreamUpdate::AssistantLine(
            "[thinking] line1\nline2 updated".to_string(),
        ))
        .expect("multiline thinking update succeeds");
    renderer
        .on_update(AgentSessionStreamUpdate::AssistantLine(String::new()))
        .expect("flush line succeeds");
    renderer.finish().expect("finish succeeds");

    let output = String::from_utf8(renderer.into_inner()).expect("utf-8 output");
    assert_eq!(
        output,
        "[thinking] line1\nline2\r\u{1b}[2K\u{1b}[1A\r\u{1b}[2K[thinking] line1\nline2 updated\n"
    );
}

#[test]
fn cli_stream_renderer_renders_fenced_code_blocks_with_markdown_style() {
    let mut renderer = CliStreamRenderer::new(Vec::<u8>::new(), true);

    renderer
        .on_update(AgentSessionStreamUpdate::AssistantTextDelta(
            "Before\n```rust\nfn main() {\n".to_string(),
        ))
        .expect("delta write succeeds");
    renderer
        .on_update(AgentSessionStreamUpdate::AssistantTextDelta(
            "    println!(\"hi\");\n}\n```\nAfter".to_string(),
        ))
        .expect("delta write succeeds");
    renderer.finish().expect("finish succeeds");

    let output = String::from_utf8(renderer.into_inner()).expect("utf-8 output");
    assert!(output.contains("Before\n"));
    assert!(output.contains("\u{1b}[48;5;236m\u{1b}[38;5;111mfn"));
    assert!(
        output.contains("\u{1b}[48;5;236m\u{1b}[38;5;150m\"hi\""),
        "string token should use markdown code string style"
    );
    assert!(
        output.contains("println"),
        "code block content should still be present"
    );
    assert!(
        !output.contains("```"),
        "fence markers should not be printed"
    );
    assert!(output.ends_with("After\n"));
}

#[test]
fn resolve_runtime_config_supports_google_provider_inference() {
    let args = ChatArgs {
        api: None,
        provider: Some("google".to_string()),
        model: Some("gemini-2.5-flash".to_string()),
        base_url: None,
        context_window: None,
        max_tokens: None,
        agent_dir: None,
        cwd: None,
        session_dir: None,
        session_file: None,
        system_prompt: Some("test".to_string()),
        prompt: None,
        continue_first: false,
        no_tools: false,
        skills: vec![],
        no_skills: false,
        hide_tool_results: false,
        no_tui: false,
        theme: None,
    };
    let local = AgentLocalConfig::default();

    let resolved =
        resolve_runtime_config_for_tests(&args, &local).expect("runtime config should resolve");
    assert_eq!(resolved.model.api, "google-generative-ai");
    assert_eq!(
        resolved.model.base_url,
        "https://generativelanguage.googleapis.com/v1beta"
    );
}

#[test]
fn resolve_runtime_config_defaults_reasoning_to_medium_when_unconfigured() {
    let args = ChatArgs {
        api: None,
        provider: Some("openai".to_string()),
        model: Some("gpt-5.3-codex".to_string()),
        base_url: None,
        context_window: None,
        max_tokens: None,
        agent_dir: None,
        cwd: None,
        session_dir: None,
        session_file: None,
        system_prompt: Some("test".to_string()),
        prompt: None,
        continue_first: false,
        no_tools: false,
        skills: vec![],
        no_skills: false,
        hide_tool_results: false,
        no_tui: false,
        theme: None,
    };
    let local = AgentLocalConfig::default();

    let resolved =
        resolve_runtime_config_for_tests(&args, &local).expect("runtime config should resolve");
    assert!(resolved.model.reasoning);
    assert_eq!(
        resolved.model.reasoning_effort,
        Some(pixy_ai::ThinkingLevel::Medium)
    );
}

#[test]
fn default_agent_dir_uses_pixy_home_directory() {
    unsafe {
        std::env::set_var("HOME", "/tmp/pixy-home");
    }
    assert_eq!(
        default_agent_dir(),
        PathBuf::from("/tmp/pixy-home/.pixy/agents")
    );
}

#[test]
fn load_agent_local_config_reads_pixy_toml() {
    let dir = tempdir().expect("tempdir");
    let pixy_dir = dir.path().join(".pixy");
    std::fs::create_dir_all(&pixy_dir).expect("create .pixy");
    std::fs::write(
        pixy_dir.join("pixy.toml"),
        r#"
theme = "light"
transport_retry_count = 7
skills = ["./skills"]

[env]
OPENAI_KEY = "sk-test"

[llm]
default_provider = "*"

[[llm.providers]]
name = "openai"
kind = "chat"
provider = "openai"
api = "openai-responses"
base_url = "https://api.openai.com/v1"
api_key = "$OPENAI_KEY"
model = "gpt-5.3-codex"
weight = 30
"#,
    )
    .expect("write pixy.toml");

    unsafe {
        std::env::set_var("HOME", dir.path());
    }

    let local = load_agent_local_config(Path::new(".")).expect("load config");
    assert_eq!(local.settings.theme.as_deref(), Some("light"));
    assert_eq!(local.settings.transport_retry_count, Some(7));
    assert_eq!(local.settings.default_provider.as_deref(), Some("*"));
    assert_eq!(
        local.settings.env.get("OPENAI_KEY").map(String::as_str),
        Some("sk-test")
    );

    let provider = local
        .models
        .providers
        .get("openai")
        .expect("openai provider exists");
    assert_eq!(provider.provider.as_deref(), Some("openai"));
    assert_eq!(provider.kind.as_deref(), Some("chat"));
    assert_eq!(provider.default_model.as_deref(), Some("gpt-5.3-codex"));
    assert_eq!(provider.weight, 30);
    assert_eq!(provider.models.len(), 1);
    assert_eq!(provider.models[0].id, "gpt-5.3-codex");
}

#[test]
fn parse_runtime_log_config_from_toml_reads_log_section() {
    let config = parse_runtime_log_config_from_toml(
        r#"
[env]
LOG_LEVEL = "debug"

[log]
path = "/tmp/pixy-logs"
level = "$LOG_LEVEL"
rotate_size_mb = 12
stdout = true
"#,
        "pixy.log",
    )
    .expect("log config should parse");

    assert_eq!(config.file_path, PathBuf::from("/tmp/pixy-logs/pixy.log"));
    assert_eq!(config.level, "debug");
    assert_eq!(config.rotate_size_bytes, 12 * 1024 * 1024);
    assert!(config.stdout);
}

#[test]
fn parse_runtime_log_config_from_toml_uses_defaults_when_missing() {
    let config = parse_runtime_log_config_from_toml(
        r#"
[llm]
default_provider = "openai"
"#,
        "pixy.log",
    )
    .expect("config without log section should still parse");

    assert!(config.file_path.ends_with(Path::new(".pixy/logs/pixy.log")));
    assert_eq!(config.level, "info");
    assert_eq!(config.rotate_size_bytes, 100 * 1024 * 1024);
    assert!(!config.stdout);
}

#[test]
fn wildcard_router_ignores_embedding_provider_kind() {
    let args = test_chat_args();
    let local = AgentLocalConfig {
        settings: AgentSettingsFile {
            default_provider: Some("*".to_string()),
            ..AgentSettingsFile::default()
        },
        models: ModelsFile {
            providers: HashMap::from([
                (
                    "embed".to_string(),
                    ProviderConfig {
                        provider: Some("openai".to_string()),
                        kind: Some("embedding".to_string()),
                        api: Some("openai-responses".to_string()),
                        base_url: Some("https://api.openai.com/v1".to_string()),
                        api_key: None,
                        default_model: Some("text-embedding-3-small".to_string()),
                        weight: 99,
                        models: vec![],
                    },
                ),
                (
                    "chat".to_string(),
                    ProviderConfig {
                        provider: Some("openai".to_string()),
                        kind: Some("chat".to_string()),
                        api: Some("openai-responses".to_string()),
                        base_url: Some("https://api.openai.com/v1".to_string()),
                        api_key: None,
                        default_model: Some("gpt-5.3-codex".to_string()),
                        weight: 1,
                        models: vec![],
                    },
                ),
            ]),
        },
    };

    let resolved = resolve_runtime_config_for_tests_with_seed(&args, &local, 0)
        .expect("runtime config should resolve");
    assert_eq!(resolved.model.provider, "openai");
    assert_eq!(resolved.model.id, "gpt-5.3-codex");
}

#[test]
fn resolve_transport_retry_count_defaults_to_five() {
    assert_eq!(
        resolve_transport_retry_count(&AgentSettingsFile::default()),
        DEFAULT_TRANSPORT_RETRY_COUNT
    );
}

#[test]
fn resolve_transport_retry_count_reads_settings_value() {
    let settings = AgentSettingsFile {
        transport_retry_count: Some(2),
        ..AgentSettingsFile::default()
    };
    assert_eq!(resolve_transport_retry_count(&settings), 2);
}

#[test]
fn load_runtime_skills_merges_settings_and_cli_paths() {
    let dir = tempdir().expect("temp dir");
    let settings_skill = dir.path().join("settings-skill");
    let cli_skill = dir.path().join("cli-skill");
    std::fs::create_dir_all(&settings_skill).expect("create settings skill dir");
    std::fs::create_dir_all(&cli_skill).expect("create cli skill dir");
    std::fs::write(
        settings_skill.join("SKILL.md"),
        r#"---
name: settings-skill
description: from settings
---
"#,
    )
    .expect("write settings skill");
    std::fs::write(
        cli_skill.join("SKILL.md"),
        r#"---
name: cli-skill
description: from cli
---
"#,
    )
    .expect("write cli skill");

    let mut args = test_chat_args();
    args.no_skills = true;
    args.skills = vec!["cli-skill".to_string()];

    let local = AgentLocalConfig {
        settings: AgentSettingsFile {
            skills: vec!["settings-skill".to_string()],
            ..AgentSettingsFile::default()
        },
        models: ModelsFile::default(),
    };
    let skills = load_runtime_skills(&args, &local, dir.path(), &dir.path().join(".pixy/agents"));
    let names = skills
        .iter()
        .map(|skill| skill.name.clone())
        .collect::<HashSet<_>>();
    assert!(names.contains("settings-skill"));
    assert!(names.contains("cli-skill"));
}

#[test]
fn load_runtime_skills_honors_no_skills_for_default_discovery() {
    let dir = tempdir().expect("temp dir");
    let agent_dir = dir.path().join(".pixy/agents");
    let default_skill = agent_dir.join("skills").join("default-skill");
    std::fs::create_dir_all(&default_skill).expect("create default skill dir");
    std::fs::write(
        default_skill.join("SKILL.md"),
        r#"---
name: default-skill
description: default discovered skill
---
"#,
    )
    .expect("write default skill");

    let mut args = test_chat_args();
    args.no_skills = true;

    let local = AgentLocalConfig::default();
    let skills = load_runtime_skills(&args, &local, dir.path(), &agent_dir);
    assert!(skills.is_empty());
}

#[test]
fn resolve_runtime_config_reads_reasoning_and_effort_from_model_config() {
    let mut args = test_chat_args();
    args.provider = Some("openai".to_string());
    args.model = Some("gpt-4.1".to_string());

    let local = AgentLocalConfig {
        settings: AgentSettingsFile::default(),
        models: ModelsFile {
            providers: HashMap::from([(
                "openai".to_string(),
                ProviderConfig {
                    provider: None,
                    kind: None,
                    api: Some("openai-completions".to_string()),
                    base_url: Some("https://api.openai.com/v1".to_string()),
                    api_key: None,
                    default_model: None,
                    weight: 1,
                    models: vec![ProviderModelConfig {
                        id: "gpt-4.1".to_string(),
                        api: None,
                        base_url: None,
                        reasoning: Some(true),
                        reasoning_effort: Some(pixy_ai::ThinkingLevel::High),
                        context_window: None,
                        max_tokens: None,
                    }],
                },
            )]),
        },
    };

    let resolved =
        resolve_runtime_config_for_tests(&args, &local).expect("runtime config should resolve");
    assert!(resolved.model.reasoning);
    assert_eq!(
        resolved.model.reasoning_effort,
        Some(pixy_ai::ThinkingLevel::High)
    );
}

#[test]
fn build_startup_resource_lines_is_empty_for_d_style_banner() {
    let dir = tempdir().expect("temp dir");
    let agent_dir = dir.path().join(".pixy/agents");
    let cwd = dir.path().join("workspace");
    std::fs::create_dir_all(&agent_dir).expect("create agent dir");
    std::fs::create_dir_all(&cwd).expect("create cwd");

    let skills = vec![Skill {
        name: "project-skill".to_string(),
        description: "project".to_string(),
        file_path: cwd.join(".agents/skills/project/SKILL.md"),
        base_dir: cwd.join(".agents/skills/project"),
        source: SkillSource::Project,
        disable_model_invocation: false,
    }];

    let lines = build_startup_resource_lines(&cwd, &agent_dir, &skills);
    assert!(
        lines.is_empty(),
        "startup banner should omit extra resource rows"
    );
}

#[test]
fn format_status_model_label_prefers_custom_name() {
    assert_eq!(
        format_status_model_label("Databend GPT-5.3 Codex", "gpt-5.3-codex", true),
        "Databend GPT-5.3 Codex:gpt-5.3-codex"
    );
}

#[test]
fn format_status_model_label_uses_title_case_model_when_name_missing() {
    assert_eq!(
        format_status_model_label("", "gpt-5.3-codex", false),
        "gpt-5.3-codex"
    );
}

#[test]
fn format_status_model_label_uses_title_case_model_when_name_matches_model() {
    assert_eq!(
        format_status_model_label("gpt-5.3-codex", "gpt-5.3-codex", true),
        "gpt-5.3-codex:gpt-5.3-codex"
    );
}

#[test]
fn resolve_resume_picker_selection_defaults_to_latest() {
    let candidates = vec![PathBuf::from("/tmp/session-a.jsonl")];
    let selected = resolve_resume_picker_selection(&candidates, "")
        .expect("selection should parse")
        .expect("selection should not cancel");
    assert_eq!(selected, candidates[0]);
}

#[test]
fn resolve_resume_picker_selection_supports_numeric_choice() {
    let candidates = vec![
        PathBuf::from("/tmp/session-a.jsonl"),
        PathBuf::from("/tmp/session-b.jsonl"),
    ];
    let selected = resolve_resume_picker_selection(&candidates, "2")
        .expect("selection should parse")
        .expect("selection should not cancel");
    assert_eq!(selected, candidates[1]);
}

#[test]
fn resolve_resume_picker_selection_supports_cancel() {
    let candidates = vec![PathBuf::from("/tmp/session-a.jsonl")];
    let selected =
        resolve_resume_picker_selection(&candidates, "q").expect("selection should parse");
    assert!(selected.is_none());
}

#[test]
fn resolve_resume_picker_selection_rejects_out_of_range_choice() {
    let candidates = vec![PathBuf::from("/tmp/session-a.jsonl")];
    let error =
        resolve_resume_picker_selection(&candidates, "2").expect_err("selection should fail");
    assert!(error.contains("between 1 and 1"));
}
