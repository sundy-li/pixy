use std::path::{Path, PathBuf};

use crate::{DeclarativeHookSpec, SubAgentSpec};

use super::{DispatchPolicyConfig, MultiAgentPluginManifest};

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
        merged.hooks.extend(plugin.manifest.hooks);
        for spec in plugin.manifest.subagents {
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
