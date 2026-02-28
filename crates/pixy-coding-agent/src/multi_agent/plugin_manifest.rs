use serde::{Deserialize, Serialize};

use crate::{DeclarativeHookSpec, SubAgentSpec};

use super::policy::DispatchPolicyConfig;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultiAgentPluginManifest {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
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
