use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use pixy_ai::{Cost, DEFAULT_TRANSPORT_RETRY_COUNT, Model};
use serde::Deserialize;

use crate::{LoadSkillsOptions, Skill, SkillDiagnostic, load_skills};

const DEFAULT_PIXY_HOME_DIR_NAME: &str = ".pixy";

#[derive(Debug, Clone, Default)]
pub struct RuntimeOverrides {
    pub fixed_model: Option<Model>,
    pub fixed_model_catalog: Option<Vec<Model>>,
    pub fixed_api_key: Option<String>,
    pub api: Option<String>,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub base_url: Option<String>,
    pub context_window: Option<u32>,
    pub max_tokens: Option<u32>,
}

impl RuntimeOverrides {
    pub fn from_fixed_model(model: Model, api_key: Option<String>) -> Self {
        Self {
            fixed_model: Some(model),
            fixed_model_catalog: None,
            fixed_api_key: api_key,
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone)]
pub struct RuntimeLoadOptions {
    pub conf_dir: Option<PathBuf>,
    pub agent_dir: Option<PathBuf>,
    pub load_skills: bool,
    pub include_default_skills: bool,
    pub skill_paths: Vec<String>,
    pub overrides: RuntimeOverrides,
}

impl Default for RuntimeLoadOptions {
    fn default() -> Self {
        Self {
            conf_dir: None,
            agent_dir: None,
            load_skills: true,
            include_default_skills: true,
            skill_paths: vec![],
            overrides: RuntimeOverrides::default(),
        }
    }
}

impl RuntimeLoadOptions {
    pub fn from_fixed_model(model: Model, api_key: Option<String>) -> Self {
        Self {
            load_skills: false,
            include_default_skills: false,
            overrides: RuntimeOverrides::from_fixed_model(model, api_key),
            ..Self::default()
        }
    }

    pub fn runtime_router_seed() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos() as u64)
            .unwrap_or(0)
    }

    pub fn resolve_runtime(&self, cwd: &Path) -> Result<ResolvedRuntime, String> {
        self.resolve_runtime_with_seed(cwd, Self::runtime_router_seed())
    }

    pub fn resolve_runtime_with_seed(
        &self,
        cwd: &Path,
        router_seed: u64,
    ) -> Result<ResolvedRuntime, String> {
        let conf_dir = pixy_home_dir(self.conf_dir.as_deref());
        let config_path = conf_dir.join("pixy.toml");
        let content = if config_path.exists() {
            std::fs::read_to_string(&config_path)
                .map_err(|error| format!("read {} failed: {error}", config_path.display()))?
        } else {
            String::new()
        };
        let mut local = load_agent_local_config_from_toml(&content)?;
        let runtime = RuntimeConfigResolver::new(&self.overrides, &local, router_seed)
            .resolve_runtime_config_with_seed()?;

        let (skills, skill_diagnostics) = if self.load_skills {
            let agent_dir = self
                .agent_dir
                .clone()
                .unwrap_or_else(|| conf_dir.join("agent"));
            let mut load_options = LoadSkillsOptions::new(cwd.to_path_buf(), agent_dir);
            load_options.include_defaults = self.include_default_skills;
            load_options.skill_paths = local.settings.skills.clone();
            load_options.skill_paths.extend(self.skill_paths.clone());
            let loaded = load_skills(load_options);
            (loaded.skills, loaded.diagnostics)
        } else {
            (vec![], vec![])
        };

        Ok(ResolvedRuntime {
            model: runtime.model,
            model_catalog: runtime.model_catalog,
            api_key: runtime.api_key,
            skills,
            skill_diagnostics,
            theme: local.settings.theme.take(),
            transport_retry_count: local
                .settings
                .transport_retry_count
                .unwrap_or(DEFAULT_TRANSPORT_RETRY_COUNT),
        })
    }

    pub fn resolve_runtime_from_toml_with_seed(
        &self,
        cwd: &Path,
        content: &str,
        router_seed: u64,
    ) -> Result<ResolvedRuntime, String> {
        let mut local = load_agent_local_config_from_toml(content)?;
        let runtime = RuntimeConfigResolver::new(&self.overrides, &local, router_seed)
            .resolve_runtime_config_with_seed()?;

        let (skills, skill_diagnostics) = if self.load_skills {
            let conf_dir = pixy_home_dir(self.conf_dir.as_deref());
            let agent_dir = self
                .agent_dir
                .clone()
                .unwrap_or_else(|| conf_dir.join("agent"));
            let mut load_options = LoadSkillsOptions::new(cwd.to_path_buf(), agent_dir);
            load_options.include_defaults = self.include_default_skills;
            load_options.skill_paths = local.settings.skills.clone();
            load_options.skill_paths.extend(self.skill_paths.clone());
            let loaded = load_skills(load_options);
            (loaded.skills, loaded.diagnostics)
        } else {
            (vec![], vec![])
        };

        Ok(ResolvedRuntime {
            model: runtime.model,
            model_catalog: runtime.model_catalog,
            api_key: runtime.api_key,
            skills,
            skill_diagnostics,
            theme: local.settings.theme.take(),
            transport_retry_count: local
                .settings
                .transport_retry_count
                .unwrap_or(DEFAULT_TRANSPORT_RETRY_COUNT),
        })
    }
}

#[derive(Debug, Clone)]
pub struct ResolvedRuntime {
    pub model: Model,
    pub model_catalog: Vec<Model>,
    pub api_key: Option<String>,
    pub skills: Vec<Skill>,
    pub skill_diagnostics: Vec<SkillDiagnostic>,
    pub theme: Option<String>,
    pub transport_retry_count: usize,
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

fn default_provider_weight() -> u8 {
    1
}

fn load_agent_local_config_from_toml(content: &str) -> Result<AgentLocalConfig, String> {
    let config = if content.trim().is_empty() {
        PixyTomlFile::default()
    } else {
        toml::from_str::<PixyTomlFile>(content)
            .map_err(|error| format!("parse pixy.toml failed: {error}"))?
    };
    Ok(convert_pixy_toml_to_local_config(config))
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

struct RuntimeConfigResolver<'a> {
    overrides: &'a RuntimeOverrides,
    local: &'a AgentLocalConfig,
    router_seed: u64,
}

impl<'a> RuntimeConfigResolver<'a> {
    fn new(overrides: &'a RuntimeOverrides, local: &'a AgentLocalConfig, router_seed: u64) -> Self {
        Self {
            overrides,
            local,
            router_seed,
        }
    }

    fn resolve_runtime_config_with_seed(&self) -> Result<ResolvedRuntimeConfig, String> {
        self.resolve_runtime_config_with_seed_impl()
    }

    fn resolve_runtime_config_with_seed_impl(&self) -> Result<ResolvedRuntimeConfig, String> {
        if let Some(model) = self.overrides.fixed_model.clone() {
            let mut model_catalog = self
                .overrides
                .fixed_model_catalog
                .clone()
                .unwrap_or_else(|| vec![model.clone()]);
            dedupe_models(&mut model_catalog);
            if let Some(position) = model_catalog
                .iter()
                .position(|entry| entry.provider == model.provider && entry.id == model.id)
            {
                model_catalog.remove(position);
            }
            model_catalog.insert(0, model.clone());
            return Ok(ResolvedRuntimeConfig {
                model,
                model_catalog,
                api_key: self.overrides.fixed_api_key.clone(),
            });
        }

        let cli_model_parts = split_provider_model(self.overrides.model.as_deref());
        let explicit_provider =
            first_non_empty([self.overrides.provider.clone(), cli_model_parts.0.clone()]);
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
            self.overrides.api.clone(),
            selected_model_cfg.and_then(|model| model.api.clone()),
            provider_config.and_then(|provider| provider.api.clone()),
            infer_api_for_provider(&provider_name),
        ])
        .ok_or_else(|| format!("Unable to resolve API for provider '{provider}'"))?;

        let cli_base_url = self
            .overrides
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
            .overrides
            .context_window
            .or_else(|| selected_model_cfg.and_then(|model| model.context_window))
            .unwrap_or(200_000);
        let max_tokens = self
            .overrides
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

fn default_reasoning_enabled_for_api(api: &str) -> bool {
    matches!(
        api,
        "openai-responses"
            | "openai-completions"
            | "openai-codex-responses"
            | "azure-openai-responses"
    )
}

fn pixy_home_dir(conf_dir: Option<&Path>) -> PathBuf {
    conf_dir
        .map(resolve_pixy_home_arg)
        .unwrap_or_else(default_pixy_home_dir)
}

fn resolve_pixy_home_arg(path: &Path) -> PathBuf {
    let raw = path.to_string_lossy();
    let expanded = if raw == "~" {
        home_dir()
    } else if let Some(suffix) = raw.strip_prefix("~/") {
        home_dir().join(suffix)
    } else {
        path.to_path_buf()
    };

    if expanded.is_absolute() {
        expanded
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(expanded)
    }
}

fn default_pixy_home_dir() -> PathBuf {
    home_dir().join(DEFAULT_PIXY_HOME_DIR_NAME)
}

fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn resolve_runtime_from_toml_resolves_model_and_runtime_settings() {
        let content = r#"
theme = "light"
transport_retry_count = 7

[llm]
default_provider = "openai"

[[llm.providers]]
name = "openai"
kind = "chat"
provider = "openai"
api = "openai-responses"
base_url = "https://api.openai.com/v1"
api_key = "key"
model = "gpt-5.3-codex"
weight = 1
"#;

        let mut options = RuntimeLoadOptions::default();
        options.load_skills = false;
        let resolved = options
            .resolve_runtime_from_toml_with_seed(Path::new("."), content, 0)
            .expect("runtime should resolve");

        assert_eq!(resolved.model.provider, "openai");
        assert_eq!(resolved.model.id, "gpt-5.3-codex");
        assert_eq!(resolved.api_key.as_deref(), Some("key"));
        assert_eq!(resolved.transport_retry_count, 7);
        assert_eq!(resolved.theme.as_deref(), Some("light"));
        assert!(resolved.skills.is_empty());
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
"#;

        let mut options = RuntimeLoadOptions::default();
        options.load_skills = false;
        let anthropic = options
            .resolve_runtime_from_toml_with_seed(Path::new("."), content, 5)
            .expect("runtime should resolve")
            .model;
        let openai = options
            .resolve_runtime_from_toml_with_seed(Path::new("."), content, 95)
            .expect("runtime should resolve")
            .model;

        assert_eq!(anthropic.provider, "anthropic");
        assert_eq!(openai.provider, "openai");
    }

    #[test]
    fn resolve_runtime_from_toml_loads_config_and_explicit_skills() {
        let dir = tempdir().expect("tempdir");
        let cwd = dir.path().join("workspace");
        std::fs::create_dir_all(&cwd).expect("create cwd");

        let settings_skill_dir = dir.path().join("settings-skill");
        std::fs::create_dir_all(&settings_skill_dir).expect("create settings skill dir");
        std::fs::write(
            settings_skill_dir.join("SKILL.md"),
            r#"---
name: settings-skill
description: Loaded from settings.
---
"#,
        )
        .expect("write settings skill");

        let explicit_skill_dir = dir.path().join("explicit-skill");
        std::fs::create_dir_all(&explicit_skill_dir).expect("create explicit skill dir");
        std::fs::write(
            explicit_skill_dir.join("SKILL.md"),
            r#"---
name: explicit-skill
description: Loaded from explicit path.
---
"#,
        )
        .expect("write explicit skill");

        let content = format!(
            r#"
skills = ["{}"]

[llm]
default_provider = "openai"

[[llm.providers]]
name = "openai"
kind = "chat"
provider = "openai"
api = "openai-responses"
base_url = "https://api.openai.com/v1"
api_key = "key"
model = "gpt-5.3-codex"
weight = 1
"#,
            settings_skill_dir.display()
        );

        let options = RuntimeLoadOptions {
            conf_dir: Some(dir.path().join(".pixy")),
            agent_dir: Some(dir.path().join("agent")),
            load_skills: true,
            include_default_skills: false,
            skill_paths: vec![explicit_skill_dir.display().to_string()],
            overrides: RuntimeOverrides::default(),
        };
        let resolved = options
            .resolve_runtime_from_toml_with_seed(&cwd, &content, 0)
            .expect("runtime should resolve");

        let mut names = resolved
            .skills
            .iter()
            .map(|skill| skill.name.clone())
            .collect::<Vec<_>>();
        names.sort();
        assert_eq!(names, vec!["explicit-skill", "settings-skill"]);
    }

    #[test]
    fn resolve_runtime_from_fixed_model_uses_options_override() {
        let model = Model {
            id: "gpt-5.3-codex".to_string(),
            name: "gpt-5.3-codex".to_string(),
            api: "openai-responses".to_string(),
            provider: "openai".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            reasoning: true,
            reasoning_effort: Some(pixy_ai::ThinkingLevel::Medium),
            input: vec!["text".to_string()],
            cost: Cost {
                input: 0.0,
                output: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
                total: 0.0,
            },
            context_window: 200_000,
            max_tokens: 8_192,
        };
        let options = RuntimeLoadOptions::from_fixed_model(model.clone(), Some("key".to_string()));
        let resolved = options
            .resolve_runtime(Path::new("."))
            .expect("runtime should resolve from fixed model");

        assert_eq!(resolved.model, model);
        assert_eq!(resolved.model_catalog, vec![model]);
        assert_eq!(resolved.api_key.as_deref(), Some("key"));
        assert!(
            resolved.skills.is_empty(),
            "fixed model options disable skills by default"
        );
    }

    #[test]
    fn resolve_runtime_prefers_settings_env_over_process_env_for_placeholder_values() {
        let content = r#"
[env]
PIXY_TEST_PROVIDER_TOKEN = "from-settings"
PIXY_TEST_PROVIDER_BASE = "https://settings.example/v1"

[llm]
default_provider = "openai"

[[llm.providers]]
name = "openai"
kind = "chat"
provider = "openai"
api = "openai-responses"
base_url = "$PIXY_TEST_PROVIDER_BASE"
api_key = "$PIXY_TEST_PROVIDER_TOKEN"
model = "gpt-5.3-codex"
weight = 1
"#;

        unsafe {
            std::env::set_var("PIXY_TEST_PROVIDER_TOKEN", "from-process");
            std::env::set_var("PIXY_TEST_PROVIDER_BASE", "https://process.example/v1");
        }

        let mut options = RuntimeLoadOptions::default();
        options.load_skills = false;
        let resolved = options
            .resolve_runtime_from_toml_with_seed(Path::new("."), content, 0)
            .expect("runtime should resolve");

        assert_eq!(resolved.api_key.as_deref(), Some("from-settings"));
        assert_eq!(resolved.model.base_url, "https://settings.example/v1");
    }

    #[test]
    fn resolve_runtime_uses_process_env_when_placeholder_is_missing_from_settings_env() {
        let content = r#"
[llm]
default_provider = "openai"

[[llm.providers]]
name = "openai"
kind = "chat"
provider = "openai"
api = "openai-responses"
base_url = "https://api.openai.com/v1"
api_key = "$PIXY_TEST_PROCESS_ONLY_TOKEN"
model = "gpt-5.3-codex"
weight = 1
"#;

        unsafe {
            std::env::set_var("PIXY_TEST_PROCESS_ONLY_TOKEN", "from-process-only");
        }

        let mut options = RuntimeLoadOptions::default();
        options.load_skills = false;
        let resolved = options
            .resolve_runtime_from_toml_with_seed(Path::new("."), content, 0)
            .expect("runtime should resolve");

        assert_eq!(resolved.api_key.as_deref(), Some("from-process-only"));
    }

    #[test]
    fn wildcard_default_provider_ignores_non_chat_providers() {
        let content = r#"
[llm]
default_provider = "*"

[[llm.providers]]
name = "embedder"
kind = "embedding"
provider = "openai"
api = "openai-responses"
base_url = "https://api.openai.com/v1"
api_key = "embed-key"
model = "text-embedding-3-small"
weight = 99

[[llm.providers]]
name = "openai"
kind = "chat"
provider = "openai"
api = "openai-responses"
base_url = "https://api.openai.com/v1"
api_key = "chat-key"
model = "gpt-5.3-codex"
weight = 1
"#;

        let mut options = RuntimeLoadOptions::default();
        options.load_skills = false;
        let resolved = options
            .resolve_runtime_from_toml_with_seed(Path::new("."), content, 0)
            .expect("runtime should resolve");

        assert_eq!(resolved.model.provider, "openai");
        assert_eq!(resolved.model.id, "gpt-5.3-codex");
        assert_eq!(resolved.api_key.as_deref(), Some("chat-key"));
    }

    #[test]
    fn resolve_runtime_defaults_reasoning_to_medium_for_openai_compat_api() {
        let content = r#"
[llm]
default_provider = "openai"

[[llm.providers]]
name = "openai"
kind = "chat"
provider = "openai"
api = "openai-responses"
base_url = "https://api.openai.com/v1"
api_key = "key"
model = "gpt-5.3-codex"
weight = 1
"#;

        let mut options = RuntimeLoadOptions::default();
        options.load_skills = false;
        let resolved = options
            .resolve_runtime_from_toml_with_seed(Path::new("."), content, 0)
            .expect("runtime should resolve");

        assert!(resolved.model.reasoning);
        assert_eq!(
            resolved.model.reasoning_effort,
            Some(pixy_ai::ThinkingLevel::Medium)
        );
    }
}
