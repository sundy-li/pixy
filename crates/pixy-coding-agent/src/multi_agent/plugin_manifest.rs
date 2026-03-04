use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer, Serialize};

use crate::{DeclarativeHookSpec, SubAgentMode, SubAgentPromptMetadata, SubAgentSpec};

use super::{policy::DispatchPolicyConfig, resolve_subagent_model_target};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultiAgentPluginManifest {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default, deserialize_with = "deserialize_subagents")]
    pub subagents: Vec<SubAgentSpec>,
    #[serde(default)]
    pub hooks: Vec<DeclarativeHookSpec>,
    #[serde(default)]
    pub policy: DispatchPolicyConfig,
}

impl MultiAgentPluginManifest {
    pub fn validate(&self) -> Result<(), String> {
        let name = self.name.trim();
        if name.is_empty() {
            return Err("plugin manifest name cannot be empty".to_string());
        }

        let mut seen_subagents = std::collections::BTreeSet::new();
        for spec in &self.subagents {
            spec.validate()
                .map_err(|error| format!("plugin '{name}' has invalid subagent: {error}"))?;
            let subagent_name = spec.name.trim().to_string();
            if !seen_subagents.insert(subagent_name.clone()) {
                return Err(format!(
                    "plugin '{name}' defines duplicate subagent '{subagent_name}'"
                ));
            }
        }
        for (index, hook) in self.hooks.iter().enumerate() {
            hook.validate().map_err(|error| {
                format!("plugin '{name}' has invalid hook #{}: {error}", index + 1)
            })?;
        }
        self.policy
            .validate()
            .map_err(|error| format!("plugin '{name}' has invalid policy: {error}"))?;
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
struct RawSubAgentSpec {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    mode: Option<SubAgentMode>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    provider: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default, alias = "allowed_tools", alias = "allow_tools")]
    tools: Vec<String>,
    #[serde(default, alias = "deny_tools")]
    blocked_tools: Vec<String>,
    #[serde(default)]
    metadata: Option<SubAgentPromptMetadata>,
}

impl RawSubAgentSpec {
    fn into_spec(self, fallback_name: Option<&str>) -> Result<SubAgentSpec, String> {
        let explicit_name = self
            .name
            .as_deref()
            .map(str::trim)
            .filter(|v| !v.is_empty());
        let key_name = fallback_name.map(str::trim).filter(|v| !v.is_empty());
        if let (Some(explicit), Some(from_key)) = (explicit_name, key_name) {
            if explicit != from_key {
                return Err(format!(
                    "subagent name '{explicit}' does not match key '{from_key}'"
                ));
            }
        }

        let name = explicit_name.or(key_name).unwrap_or_default().to_string();
        let description = self
            .description
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("Configured subagent")
            .to_string();
        let model = resolve_subagent_model_target(self.provider.as_deref(), self.model.as_deref())?;

        Ok(SubAgentSpec {
            name,
            description,
            mode: self.mode.unwrap_or(SubAgentMode::SubAgent),
            prompt: self
                .prompt
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string),
            model,
            tools: self.tools,
            blocked_tools: self.blocked_tools,
            metadata: self.metadata,
        })
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
enum RawSubAgents {
    List(Vec<RawSubAgentSpec>),
    Map(BTreeMap<String, RawSubAgentSpec>),
}

fn deserialize_subagents<'de, D>(deserializer: D) -> Result<Vec<SubAgentSpec>, D::Error>
where
    D: Deserializer<'de>,
{
    let Some(raw) = Option::<RawSubAgents>::deserialize(deserializer)? else {
        return Ok(vec![]);
    };

    let mut specs = Vec::new();
    match raw {
        RawSubAgents::List(items) => {
            for raw_item in items {
                let spec = raw_item.into_spec(None).map_err(serde::de::Error::custom)?;
                specs.push(spec);
            }
        }
        RawSubAgents::Map(items) => {
            for (name, raw_item) in items {
                let spec = raw_item
                    .into_spec(Some(&name))
                    .map_err(serde::de::Error::custom)?;
                specs.push(spec);
            }
        }
    }

    Ok(specs)
}

#[cfg(test)]
mod tests {
    use super::MultiAgentPluginManifest;

    #[test]
    fn manifest_supports_map_style_subagents() {
        let manifest: MultiAgentPluginManifest = toml::from_str(
            r#"
name = "map-plugin"

[subagents.code]
description = "Implementation worker"
mode = "subagent"
prompt = "Implement the task."
tools = ["read", "edit", "bash"]
blocked_tools = ["task"]

[subagents.code.metadata]
category = "specialist"
prompt_alias = "Code Worker"
use_when = ["Implementation work"]
avoid_when = ["Final review"]
"#,
        )
        .expect("manifest should parse");

        assert_eq!(manifest.subagents.len(), 1);
        assert_eq!(manifest.subagents[0].name, "code");
        assert_eq!(manifest.subagents[0].tools, vec!["read", "edit", "bash"]);
        assert_eq!(manifest.subagents[0].blocked_tools, vec!["task"]);
    }

    #[test]
    fn manifest_rejects_map_name_mismatch() {
        let error = toml::from_str::<MultiAgentPluginManifest>(
            r#"
name = "map-plugin"

[subagents.code]
name = "other"
description = "Implementation worker"
mode = "subagent"
"#,
        )
        .expect_err("mismatched name should fail");

        assert!(error.to_string().contains("does not match key"));
    }

    #[test]
    fn manifest_supports_subagent_provider_field() {
        let manifest: MultiAgentPluginManifest = toml::from_str(
            r#"
name = "map-plugin"

[subagents.code]
description = "Implementation worker"
mode = "subagent"
provider = "openai"
model = "gpt-5.3-codex"
"#,
        )
        .expect("manifest should parse");

        assert_eq!(manifest.subagents.len(), 1);
        assert_eq!(
            manifest.subagents[0].model.as_deref(),
            Some("openai/gpt-5.3-codex")
        );
    }
}
