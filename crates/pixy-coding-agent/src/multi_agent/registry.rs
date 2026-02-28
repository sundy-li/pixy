use std::collections::BTreeMap;

use super::types::SubAgentSpec;

pub trait SubAgentResolver: Send + Sync {
    fn resolve(&self, subagent_type: &str) -> Option<SubAgentSpec>;
    fn list(&self) -> Vec<SubAgentSpec>;
}

#[derive(Clone, Debug, Default)]
pub struct DefaultSubAgentRegistry {
    by_name: BTreeMap<String, SubAgentSpec>,
}

impl DefaultSubAgentRegistry {
    pub fn builder() -> SubAgentRegistryBuilder {
        SubAgentRegistryBuilder::default()
    }

    pub fn from_specs(specs: impl IntoIterator<Item = SubAgentSpec>) -> Result<Self, String> {
        let mut builder = Self::builder();
        for spec in specs {
            builder = builder.register_builtin(spec)?;
        }
        Ok(builder.build())
    }
}

impl SubAgentResolver for DefaultSubAgentRegistry {
    fn resolve(&self, subagent_type: &str) -> Option<SubAgentSpec> {
        self.by_name.get(subagent_type).cloned()
    }

    fn list(&self) -> Vec<SubAgentSpec> {
        self.by_name.values().cloned().collect()
    }
}

#[derive(Clone, Debug, Default)]
pub struct SubAgentRegistryBuilder {
    by_name: BTreeMap<String, SubAgentSpec>,
    source_by_name: BTreeMap<String, String>,
}

impl SubAgentRegistryBuilder {
    pub fn register_builtin(self, spec: SubAgentSpec) -> Result<Self, String> {
        let mut builder = self;
        builder.register_builtin_mut(spec)?;
        Ok(builder)
    }

    pub fn register_builtin_mut(&mut self, spec: SubAgentSpec) -> Result<(), String> {
        self.register_with_source_mut(spec, "runtime".to_string())
    }

    pub fn register_plugin_subagent(
        self,
        plugin_name: &str,
        spec: SubAgentSpec,
    ) -> Result<Self, String> {
        let mut builder = self;
        builder.register_plugin_subagent_mut(plugin_name, spec)?;
        Ok(builder)
    }

    pub fn register_plugin_subagent_mut(
        &mut self,
        plugin_name: &str,
        spec: SubAgentSpec,
    ) -> Result<(), String> {
        let plugin_name = plugin_name.trim();
        if plugin_name.is_empty() {
            return Err("plugin_name cannot be empty".to_string());
        }
        self.register_with_source_mut(spec, format!("plugin:{plugin_name}"))
    }

    fn register_with_source_mut(
        &mut self,
        spec: SubAgentSpec,
        source: String,
    ) -> Result<(), String> {
        spec.validate()?;
        let key = spec.name.trim().to_string();
        if self.by_name.contains_key(&key) {
            let existing = self
                .source_by_name
                .get(&key)
                .cloned()
                .unwrap_or_else(|| "unknown".to_string());
            return Err(format!(
                "duplicate subagent name '{key}' (from {source}, already defined by {existing})"
            ));
        }
        self.by_name.insert(key.clone(), spec);
        self.source_by_name.insert(key, source);
        Ok(())
    }

    pub fn build(self) -> DefaultSubAgentRegistry {
        DefaultSubAgentRegistry {
            by_name: self.by_name,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DefaultSubAgentRegistry, SubAgentResolver};
    use crate::{SubAgentMode, SubAgentSpec};

    fn spec(name: &str) -> SubAgentSpec {
        SubAgentSpec {
            name: name.to_string(),
            description: format!("{name} helper"),
            mode: SubAgentMode::SubAgent,
        }
    }

    #[test]
    fn builder_registers_and_resolves_subagent() {
        let registry = DefaultSubAgentRegistry::builder()
            .register_builtin(spec("general"))
            .expect("register should pass")
            .build();

        let resolved = registry
            .resolve("general")
            .expect("general subagent should be resolvable");

        assert_eq!(resolved.name, "general");
    }

    #[test]
    fn builder_rejects_duplicate_name() {
        let error = DefaultSubAgentRegistry::builder()
            .register_builtin(spec("general"))
            .expect("first register should pass")
            .register_builtin(spec("general"))
            .expect_err("duplicate should fail");

        assert!(error.contains("duplicate"));
        assert!(error.contains("general"));
    }

    #[test]
    fn registry_lists_subagents_in_sorted_order() {
        let registry = DefaultSubAgentRegistry::builder()
            .register_builtin(spec("explore"))
            .expect("register explore")
            .register_builtin(spec("general"))
            .expect("register general")
            .build();

        let names = registry
            .list()
            .into_iter()
            .map(|spec| spec.name)
            .collect::<Vec<_>>();

        assert_eq!(names, vec!["explore", "general"]);
    }

    #[test]
    fn builder_rejects_duplicate_name_from_plugin_source() {
        let error = DefaultSubAgentRegistry::builder()
            .register_builtin(spec("general"))
            .expect("register runtime subagent")
            .register_plugin_subagent("my-plugin", spec("general"))
            .expect_err("duplicate from plugin should fail");

        assert!(error.contains("duplicate subagent"));
        assert!(error.contains("plugin:my-plugin"));
    }

    #[test]
    fn builder_mut_registration_supports_incremental_updates() {
        let mut builder = DefaultSubAgentRegistry::builder();
        builder
            .register_builtin_mut(spec("general"))
            .expect("register runtime subagent");
        builder
            .register_plugin_subagent_mut("my-plugin", spec("explore"))
            .expect("register plugin subagent");

        let registry = builder.build();
        assert!(registry.resolve("general").is_some());
        assert!(registry.resolve("explore").is_some());
    }
}
