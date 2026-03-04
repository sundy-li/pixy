use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::{DeclarativeHookSpec, SubAgentMode, SubAgentPromptMetadata, SubAgentSpec};

use super::{resolve_subagent_model_target, DispatchPolicyConfig, MultiAgentPluginManifest};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadedPluginManifest {
    pub path: PathBuf,
    pub manifest: MultiAgentPluginManifest,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MergedPluginConfig {
    pub subagents: Vec<PluginSubAgentSpec>,
    pub hooks: Vec<DeclarativeHookSpec>,
    pub policy: DispatchPolicyConfig,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PluginSubAgentSpec {
    pub plugin_name: String,
    pub spec: SubAgentSpec,
}

pub fn load_plugin_manifests(
    plugin_paths: &[PathBuf],
) -> Result<Vec<LoadedPluginManifest>, String> {
    let mut loaded = Vec::with_capacity(plugin_paths.len());
    for path in plugin_paths {
        let content = std::fs::read_to_string(path)
            .map_err(|error| format!("read plugin manifest {} failed: {error}", path.display()))?;
        let manifest = toml::from_str::<MultiAgentPluginManifest>(&content)
            .map_err(|error| format!("parse plugin manifest {} failed: {error}", path.display()))?;
        manifest.validate()?;
        loaded.push(LoadedPluginManifest {
            path: path.clone(),
            manifest,
        });
    }

    loaded.sort_by(|left, right| left.manifest.name.cmp(&right.manifest.name));

    for pair in loaded.windows(2) {
        if pair[0].manifest.name == pair[1].manifest.name {
            return Err(format!(
                "duplicate plugin name '{}' found in {} and {}",
                pair[0].manifest.name,
                pair[0].path.display(),
                pair[1].path.display()
            ));
        }
    }

    Ok(loaded)
}

pub fn load_and_merge_plugins(plugin_paths: &[PathBuf]) -> Result<MergedPluginConfig, String> {
    let loaded = load_plugin_manifests(plugin_paths)?;
    let mut merged = MergedPluginConfig::default();
    for plugin in loaded {
        let merged_subagents = merge_plugin_subagent_sources(&plugin)?;
        merged.hooks.extend(plugin.manifest.hooks);
        for spec in merged_subagents {
            merged.subagents.push(PluginSubAgentSpec {
                plugin_name: plugin.manifest.name.clone(),
                spec,
            });
        }
        merged.policy.merge_from(&plugin.manifest.policy);
    }
    merged.policy.validate()?;
    Ok(merged)
}

pub fn load_and_merge_plugins_from_paths(
    plugin_paths: &[impl AsRef<Path>],
) -> Result<MergedPluginConfig, String> {
    let resolved = plugin_paths
        .iter()
        .map(|path| path.as_ref().to_path_buf())
        .collect::<Vec<_>>();
    load_and_merge_plugins(&resolved)
}

fn merge_plugin_subagent_sources(
    plugin: &LoadedPluginManifest,
) -> Result<Vec<SubAgentSpec>, String> {
    let mut merged = plugin.manifest.subagents.clone();
    let sidecar = load_sidecar_subagents_from_agents_dir(plugin)?;
    let mut seen = merged
        .iter()
        .map(|spec| spec.name.trim().to_string())
        .collect::<BTreeSet<_>>();

    for spec in sidecar {
        let subagent_name = spec.name.trim().to_string();
        if !seen.insert(subagent_name.clone()) {
            return Err(format!(
                "plugin '{}' defines duplicate subagent '{}' across manifest and agents directory",
                plugin.manifest.name, subagent_name
            ));
        }
        merged.push(spec);
    }

    Ok(merged)
}

fn load_sidecar_subagents_from_agents_dir(
    plugin: &LoadedPluginManifest,
) -> Result<Vec<SubAgentSpec>, String> {
    let Some(parent_dir) = plugin.path.parent() else {
        return Ok(vec![]);
    };
    let agents_dir = parent_dir.join("agents");
    if !agents_dir.exists() {
        return Ok(vec![]);
    }
    if !agents_dir.is_dir() {
        return Err(format!(
            "plugin '{}' expects '{}' to be a directory",
            plugin.manifest.name,
            agents_dir.display()
        ));
    }

    let mut agent_files = std::fs::read_dir(&agents_dir)
        .map_err(|error| {
            format!(
                "read plugin '{}' agents directory {} failed: {error}",
                plugin.manifest.name,
                agents_dir.display()
            )
        })?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_file())
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("toml"))
        })
        .collect::<Vec<_>>();
    agent_files.sort();

    let mut specs = Vec::with_capacity(agent_files.len());
    for agent_file in agent_files {
        let content = std::fs::read_to_string(&agent_file).map_err(|error| {
            format!(
                "read plugin '{}' agent file {} failed: {error}",
                plugin.manifest.name,
                agent_file.display()
            )
        })?;
        let raw = toml::from_str::<PluginAgentFile>(&content).map_err(|error| {
            format!(
                "parse plugin '{}' agent file {} failed: {error}",
                plugin.manifest.name,
                agent_file.display()
            )
        })?;

        let fallback_name = agent_file
            .file_stem()
            .and_then(|stem| stem.to_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                format!(
                    "invalid plugin '{}' agent file name '{}'",
                    plugin.manifest.name,
                    agent_file.display()
                )
            })?;

        let spec = raw.into_spec(Some(fallback_name)).map_err(|error| {
            format!(
                "plugin '{}' agent file {} is invalid: {error}",
                plugin.manifest.name,
                agent_file.display()
            )
        })?;
        spec.validate().map_err(|error| {
            format!(
                "plugin '{}' agent file {} is invalid: {error}",
                plugin.manifest.name,
                agent_file.display()
            )
        })?;
        specs.push(spec);
    }

    Ok(specs)
}

#[derive(Clone, Debug, Default, Deserialize)]
struct PluginAgentFile {
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

impl PluginAgentFile {
    fn into_spec(self, fallback_name: Option<&str>) -> Result<SubAgentSpec, String> {
        let explicit_name = self
            .name
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let key_name = fallback_name
            .map(str::trim)
            .filter(|value| !value.is_empty());
        if let (Some(explicit), Some(from_key)) = (explicit_name, key_name) {
            if explicit != from_key {
                return Err(format!(
                    "subagent name '{explicit}' does not match file name '{from_key}'"
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

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use tempfile::tempdir;

    use super::load_and_merge_plugins;
    use super::load_plugin_manifests;

    fn write_plugin(path: &PathBuf, content: &str) {
        fs::write(path, content).expect("write plugin file");
    }

    fn write_agent(path: &PathBuf, content: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create agents dir");
        }
        fs::write(path, content).expect("write agent file");
    }

    #[test]
    fn plugin_loader_parses_manifest_from_configured_paths() {
        let dir = tempdir().expect("tempdir");
        let plugin = dir.path().join("alpha.toml");
        write_plugin(
            &plugin,
            r#"
name = "alpha"

[[subagents]]
name = "general"
description = "General helper"
mode = "subagent"
"#,
        );

        let loaded = load_plugin_manifests(&[plugin]).expect("plugin should parse");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].manifest.name, "alpha");
        assert_eq!(loaded[0].manifest.subagents.len(), 1);
    }

    #[test]
    fn plugin_loader_rejects_duplicate_plugin_names() {
        let dir = tempdir().expect("tempdir");
        let plugin_a = dir.path().join("a.toml");
        let plugin_b = dir.path().join("b.toml");
        let content = r#"
name = "duplicate"
"#;
        write_plugin(&plugin_a, content);
        write_plugin(&plugin_b, content);

        let error = load_plugin_manifests(&[plugin_a, plugin_b])
            .expect_err("duplicate plugin names should be rejected");
        assert!(error.contains("duplicate plugin"));
    }

    #[test]
    fn plugin_loader_merges_subagents_in_deterministic_order() {
        let dir = tempdir().expect("tempdir");
        let plugin_z = dir.path().join("zeta.toml");
        let plugin_a = dir.path().join("alpha.toml");
        write_plugin(
            &plugin_z,
            r#"
name = "zeta"

[[subagents]]
name = "z-agent"
description = "z helper"
mode = "subagent"
"#,
        );
        write_plugin(
            &plugin_a,
            r#"
name = "alpha"

[[subagents]]
name = "a-agent"
description = "a helper"
mode = "subagent"
"#,
        );

        let merged = load_and_merge_plugins(&[plugin_z, plugin_a]).expect("plugins should merge");
        let names = merged
            .subagents
            .iter()
            .map(|item| item.spec.name.clone())
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["a-agent", "z-agent"]);
    }

    #[test]
    fn plugin_loader_supports_documented_fixture_manifest() {
        let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../docs/fixtures/multi-agent/plugins/basic-plugin.toml");

        let loaded = load_plugin_manifests(&[fixture]).expect("fixture plugin should parse");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].manifest.name, "basic-plugin");
        assert!(!loaded[0].manifest.subagents.is_empty());
    }

    #[test]
    fn plugin_loader_supports_map_style_subagent_manifest() {
        let dir = tempdir().expect("tempdir");
        let plugin = dir.path().join("map-style.toml");
        write_plugin(
            &plugin,
            r#"
name = "map-style"

[subagents.code]
description = "Implementation worker"
mode = "subagent"
prompt = "Implement requested changes."
tools = ["read", "edit", "bash"]
blocked_tools = ["task"]
"#,
        );

        let loaded = load_plugin_manifests(&[plugin]).expect("plugin should parse");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].manifest.subagents.len(), 1);
        let code = &loaded[0].manifest.subagents[0];
        assert_eq!(code.name, "code");
        assert_eq!(code.tools, vec!["read", "edit", "bash"]);
        assert_eq!(code.blocked_tools, vec!["task"]);
    }

    #[test]
    fn plugin_loader_loads_sidecar_agents_from_agents_directory() {
        let dir = tempdir().expect("tempdir");
        let plugin = dir.path().join("mission-plugin.toml");
        write_plugin(
            &plugin,
            r#"
name = "mission-plugin"
"#,
        );
        write_agent(
            &dir.path().join("agents/code.toml"),
            r#"
description = "Implementation worker"
mode = "subagent"
tools = ["read", "edit", "bash"]
"#,
        );
        write_agent(
            &dir.path().join("agents/review.toml"),
            r#"
description = "Review worker"
mode = "subagent"
tools = ["read", "bash"]
blocked_tools = ["edit", "write"]
"#,
        );

        let merged = load_and_merge_plugins(&[plugin]).expect("plugins should merge");
        let names = merged
            .subagents
            .iter()
            .map(|item| item.spec.name.clone())
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["code", "review"]);
    }

    #[test]
    fn plugin_loader_rejects_duplicate_subagent_across_manifest_and_agents_directory() {
        let dir = tempdir().expect("tempdir");
        let plugin = dir.path().join("mission-plugin.toml");
        write_plugin(
            &plugin,
            r#"
name = "mission-plugin"

[[subagents]]
name = "code"
description = "Implementation worker"
mode = "subagent"
"#,
        );
        write_agent(
            &dir.path().join("agents/code.toml"),
            r#"
description = "Implementation worker sidecar"
mode = "subagent"
"#,
        );

        let error = load_and_merge_plugins(&[plugin]).expect_err("duplicate subagent should fail");
        assert!(error.contains("duplicate subagent"));
    }

    #[test]
    fn plugin_loader_sidecar_agent_supports_provider_field() {
        let dir = tempdir().expect("tempdir");
        let plugin = dir.path().join("mission-plugin.toml");
        write_plugin(
            &plugin,
            r#"
name = "mission-plugin"
"#,
        );
        write_agent(
            &dir.path().join("agents/code.toml"),
            r#"
description = "Implementation worker"
mode = "subagent"
provider = "openai"
model = "gpt-5.3-codex"
"#,
        );

        let merged = load_and_merge_plugins(&[plugin]).expect("plugins should merge");
        assert_eq!(merged.subagents.len(), 1);
        assert_eq!(
            merged.subagents[0].spec.model.as_deref(),
            Some("openai/gpt-5.3-codex")
        );
    }

    #[test]
    fn plugin_loader_merges_hooks_in_deterministic_order() {
        let dir = tempdir().expect("tempdir");
        let plugin_z = dir.path().join("zeta.toml");
        let plugin_a = dir.path().join("alpha.toml");
        write_plugin(
            &plugin_z,
            r#"
name = "zeta"

[[hooks]]
name = "z-hook"
stage = "before_task_dispatch"

[[hooks.actions]]
type = "route_to"
subagent = "z-agent"
"#,
        );
        write_plugin(
            &plugin_a,
            r#"
name = "alpha"

[[hooks]]
name = "a-hook"
stage = "before_task_dispatch"

[[hooks.actions]]
type = "route_to"
subagent = "a-agent"
"#,
        );

        let merged = load_and_merge_plugins(&[plugin_z, plugin_a]).expect("plugins should merge");
        let hook_names = merged
            .hooks
            .iter()
            .map(|hook| hook.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(hook_names, vec!["a-hook", "z-hook"]);
    }

    #[test]
    fn plugin_loader_rejects_invalid_hook_spec() {
        let dir = tempdir().expect("tempdir");
        let plugin = dir.path().join("invalid-hook.toml");
        write_plugin(
            &plugin,
            r#"
name = "invalid"

[[hooks]]
name = "missing-actions"
stage = "before_task_dispatch"
"#,
        );

        let error = load_plugin_manifests(&[plugin]).expect_err("invalid hooks should fail");
        assert!(error.contains("invalid hook"));
    }
}
