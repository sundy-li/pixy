use std::collections::{HashMap, HashSet};
use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use serde::Deserialize;

const MAX_NAME_LENGTH: usize = 64;
const MAX_DESCRIPTION_LENGTH: usize = 1_024;
const PIXY_CONFIG_DIR_NAME: &str = ".pixy";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SkillSource {
    User,
    Project,
    Path,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub file_path: PathBuf,
    pub base_dir: PathBuf,
    pub source: SkillSource,
    pub disable_model_invocation: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SkillDiagnosticKind {
    Warning,
    Collision,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkillDiagnostic {
    pub kind: SkillDiagnosticKind,
    pub message: String,
    pub path: PathBuf,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LoadSkillsResult {
    pub skills: Vec<Skill>,
    pub diagnostics: Vec<SkillDiagnostic>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadSkillsOptions {
    pub cwd: PathBuf,
    pub agent_dir: PathBuf,
    pub skill_paths: Vec<String>,
    pub include_defaults: bool,
}

impl LoadSkillsOptions {
    pub fn new(cwd: PathBuf, agent_dir: PathBuf) -> Self {
        Self {
            cwd,
            agent_dir,
            skill_paths: vec![],
            include_defaults: true,
        }
    }
}

pub fn load_skills(options: LoadSkillsOptions) -> LoadSkillsResult {
    let mut aggregated = LoadSkillsResult::default();
    let mut loaded_real_paths = HashSet::<PathBuf>::new();
    let mut loaded_names = HashMap::<String, PathBuf>::new();

    if options.include_defaults {
        merge_load_result(
            load_skills_from_dir(&options.agent_dir.join("skills"), SkillSource::User),
            &mut aggregated,
            &mut loaded_real_paths,
            &mut loaded_names,
        );

        if let Some(home) = home_dir() {
            merge_load_result(
                load_skills_from_dir(&home.join(".agents").join("skills"), SkillSource::User),
                &mut aggregated,
                &mut loaded_real_paths,
                &mut loaded_names,
            );
        }

        merge_load_result(
            load_skills_from_dir(
                &options.cwd.join(PIXY_CONFIG_DIR_NAME).join("skills"),
                SkillSource::Project,
            ),
            &mut aggregated,
            &mut loaded_real_paths,
            &mut loaded_names,
        );

        for dir in project_ancestor_dirs(&options.cwd) {
            merge_load_result(
                load_skills_from_dir(&dir.join(".agents").join("skills"), SkillSource::Project),
                &mut aggregated,
                &mut loaded_real_paths,
                &mut loaded_names,
            );
        }
    }

    for raw_path in &options.skill_paths {
        let resolved_path = resolve_skill_path(raw_path, &options.cwd);
        if !resolved_path.exists() {
            aggregated.diagnostics.push(SkillDiagnostic {
                kind: SkillDiagnosticKind::Warning,
                message: "skill path does not exist".to_string(),
                path: resolved_path,
            });
            continue;
        }

        let source = infer_explicit_source(&resolved_path, &options);

        if resolved_path.is_dir() {
            merge_load_result(
                load_skills_from_dir(&resolved_path, source),
                &mut aggregated,
                &mut loaded_real_paths,
                &mut loaded_names,
            );
            continue;
        }

        if resolved_path.is_file() && is_markdown_file(&resolved_path) {
            let (skill, diagnostics) = load_skill_from_file(&resolved_path, source);
            merge_load_result(
                LoadSkillsResult {
                    skills: skill.into_iter().collect(),
                    diagnostics,
                },
                &mut aggregated,
                &mut loaded_real_paths,
                &mut loaded_names,
            );
            continue;
        }

        aggregated.diagnostics.push(SkillDiagnostic {
            kind: SkillDiagnosticKind::Warning,
            message: "skill path is not a markdown file".to_string(),
            path: resolved_path,
        });
    }

    aggregated
}

fn merge_load_result(
    result: LoadSkillsResult,
    aggregated: &mut LoadSkillsResult,
    loaded_real_paths: &mut HashSet<PathBuf>,
    loaded_names: &mut HashMap<String, PathBuf>,
) {
    aggregated.diagnostics.extend(result.diagnostics);
    for skill in result.skills {
        let real_path = std::fs::canonicalize(&skill.file_path).unwrap_or(skill.file_path.clone());
        if loaded_real_paths.contains(&real_path) {
            continue;
        }

        if let Some(existing_path) = loaded_names.get(&skill.name) {
            aggregated.diagnostics.push(SkillDiagnostic {
                kind: SkillDiagnosticKind::Collision,
                message: format!("name \"{}\" collision", skill.name),
                path: skill.file_path.clone(),
            });
            aggregated.diagnostics.push(SkillDiagnostic {
                kind: SkillDiagnosticKind::Warning,
                message: format!("keeping first skill at {}", existing_path.display()),
                path: skill.file_path.clone(),
            });
            continue;
        }

        loaded_real_paths.insert(real_path);
        loaded_names.insert(skill.name.clone(), skill.file_path.clone());
        aggregated.skills.push(skill);
    }
}

pub fn load_skills_from_dir(dir: &Path, source: SkillSource) -> LoadSkillsResult {
    let mut result = LoadSkillsResult::default();
    load_skills_from_dir_internal(dir, &source, true, &mut result);
    result
}

pub fn format_skills_for_prompt(skills: &[Skill]) -> String {
    let visible_skills = skills
        .iter()
        .filter(|skill| !skill.disable_model_invocation)
        .collect::<Vec<_>>();

    if visible_skills.is_empty() {
        return String::new();
    }

    let mut lines = vec![
        String::new(),
        String::new(),
        "The following skills provide specialized instructions for specific tasks.".to_string(),
        "Use the read tool to load a skill's file when the task matches its description."
            .to_string(),
        "After reading a matching skill, execute it with available tools. Do not stop at command suggestions unless the user explicitly asks for command-only output."
            .to_string(),
        "When a skill file references a relative path, resolve it against the skill directory (parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.".to_string(),
        String::new(),
        "<available_skills>".to_string(),
    ];

    for skill in visible_skills {
        lines.push("  <skill>".to_string());
        lines.push(format!("    <name>{}</name>", escape_xml(&skill.name)));
        lines.push(format!(
            "    <description>{}</description>",
            escape_xml(&skill.description)
        ));
        lines.push(format!(
            "    <location>{}</location>",
            escape_xml(skill.file_path.to_string_lossy().as_ref())
        ));
        lines.push("  </skill>".to_string());
    }

    lines.push("</available_skills>".to_string());
    lines.join("\n")
}

fn load_skills_from_dir_internal(
    dir: &Path,
    source: &SkillSource,
    include_root_markdown: bool,
    out: &mut LoadSkillsResult,
) {
    if !dir.exists() {
        return;
    }

    let mut entries = match std::fs::read_dir(dir) {
        Ok(read_dir) => read_dir.filter_map(Result::ok).collect::<Vec<_>>(),
        Err(_) => return,
    };
    entries.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

    for entry in entries {
        let name = entry.file_name();
        let name_text = name.to_string_lossy();
        if name_text.starts_with('.') || name_text == "node_modules" {
            continue;
        }

        let path = entry.path();
        let metadata = match std::fs::symlink_metadata(&path) {
            Ok(value) => value,
            Err(_) => continue,
        };
        let file_type = metadata.file_type();

        let (is_file, is_dir) = if file_type.is_symlink() {
            match std::fs::metadata(&path) {
                Ok(linked) => (linked.is_file(), linked.is_dir()),
                Err(_) => continue,
            }
        } else {
            (file_type.is_file(), file_type.is_dir())
        };

        if is_dir {
            load_skills_from_dir_internal(&path, source, false, out);
            continue;
        }

        if !is_file {
            continue;
        }

        let is_root_markdown_file = include_root_markdown && is_markdown_file(&path);
        let is_nested_skill_file = !include_root_markdown && name == OsStr::new("SKILL.md");
        if !is_root_markdown_file && !is_nested_skill_file {
            continue;
        }

        let (skill, diagnostics) = load_skill_from_file(&path, source.clone());
        out.diagnostics.extend(diagnostics);
        if let Some(skill) = skill {
            out.skills.push(skill);
        }
    }
}

fn load_skill_from_file(
    file_path: &Path,
    source: SkillSource,
) -> (Option<Skill>, Vec<SkillDiagnostic>) {
    let mut diagnostics = vec![];
    let raw_content = match std::fs::read_to_string(file_path) {
        Ok(content) => content,
        Err(error) => {
            diagnostics.push(SkillDiagnostic {
                kind: SkillDiagnosticKind::Warning,
                message: error.to_string(),
                path: file_path.to_path_buf(),
            });
            return (None, diagnostics);
        }
    };

    let frontmatter = match parse_frontmatter(&raw_content) {
        Ok(frontmatter) => frontmatter,
        Err(error) => {
            diagnostics.push(SkillDiagnostic {
                kind: SkillDiagnosticKind::Warning,
                message: error,
                path: file_path.to_path_buf(),
            });
            return (None, diagnostics);
        }
    };

    let parent_dir_name = file_path
        .parent()
        .and_then(Path::file_name)
        .and_then(OsStr::to_str)
        .unwrap_or_default();
    let name = frontmatter
        .name
        .clone()
        .unwrap_or_else(|| parent_dir_name.to_string());

    for message in validate_name(&name, parent_dir_name) {
        diagnostics.push(SkillDiagnostic {
            kind: SkillDiagnosticKind::Warning,
            message,
            path: file_path.to_path_buf(),
        });
    }

    for message in validate_description(frontmatter.description.as_deref()) {
        diagnostics.push(SkillDiagnostic {
            kind: SkillDiagnosticKind::Warning,
            message,
            path: file_path.to_path_buf(),
        });
    }

    let Some(description) = frontmatter
        .description
        .as_ref()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
    else {
        return (None, diagnostics);
    };

    (
        Some(Skill {
            name,
            description: description.to_string(),
            file_path: file_path.to_path_buf(),
            base_dir: file_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf(),
            source,
            disable_model_invocation: frontmatter.disable_model_invocation,
        }),
        diagnostics,
    )
}

#[derive(Debug, Deserialize, Default)]
struct SkillFrontmatter {
    name: Option<String>,
    description: Option<String>,
    #[serde(default)]
    #[serde(rename = "disable-model-invocation")]
    disable_model_invocation: bool,
}

fn parse_frontmatter(content: &str) -> Result<SkillFrontmatter, String> {
    let normalized = content.replace("\r\n", "\n").replace('\r', "\n");
    if !normalized.starts_with("---\n") {
        return Ok(SkillFrontmatter::default());
    }

    let start = 4usize;
    let Some(relative_end_index) = normalized[start..].find("\n---") else {
        return Ok(SkillFrontmatter::default());
    };
    let end_index = start + relative_end_index;
    let yaml = &normalized[start..end_index];
    serde_yaml::from_str::<SkillFrontmatter>(yaml).map_err(|error| error.to_string())
}

fn validate_name(name: &str, parent_dir_name: &str) -> Vec<String> {
    let mut errors = vec![];
    if name != parent_dir_name {
        errors.push(format!(
            "name \"{name}\" does not match parent directory \"{parent_dir_name}\""
        ));
    }
    if name.len() > MAX_NAME_LENGTH {
        errors.push(format!(
            "name exceeds {MAX_NAME_LENGTH} characters ({})",
            name.len()
        ));
    }
    if !name
        .chars()
        .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '-')
    {
        errors.push(
            "name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)"
                .to_string(),
        );
    }
    if name.starts_with('-') || name.ends_with('-') {
        errors.push("name must not start or end with a hyphen".to_string());
    }
    if name.contains("--") {
        errors.push("name must not contain consecutive hyphens".to_string());
    }
    errors
}

fn validate_description(description: Option<&str>) -> Vec<String> {
    let mut errors = vec![];
    match description {
        Some(value) if value.trim().is_empty() => {
            errors.push("description is required".to_string());
        }
        Some(value) if value.len() > MAX_DESCRIPTION_LENGTH => {
            errors.push(format!(
                "description exceeds {MAX_DESCRIPTION_LENGTH} characters ({})",
                value.len()
            ));
        }
        None => errors.push("description is required".to_string()),
        _ => {}
    }
    errors
}

fn escape_xml(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn is_markdown_file(path: &Path) -> bool {
    path.extension()
        .and_then(OsStr::to_str)
        .map(|ext| ext.eq_ignore_ascii_case("md"))
        .unwrap_or(false)
}

fn resolve_skill_path(input: &str, cwd: &Path) -> PathBuf {
    let normalized = normalize_tilde_path(input);
    if normalized.is_absolute() {
        normalized
    } else {
        cwd.join(normalized)
    }
}

fn normalize_tilde_path(input: &str) -> PathBuf {
    let trimmed = input.trim();
    if trimmed == "~" {
        return home_dir().unwrap_or_else(|| PathBuf::from("~"));
    }
    if let Some(stripped) = trimmed.strip_prefix("~/") {
        if let Some(home) = home_dir() {
            return home.join(stripped);
        }
    }
    PathBuf::from(trimmed)
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

fn infer_explicit_source(path: &Path, options: &LoadSkillsOptions) -> SkillSource {
    if !options.include_defaults {
        if is_under_path(path, &options.agent_dir.join("skills")) {
            return SkillSource::User;
        }
        if is_under_path(path, &options.cwd.join(PIXY_CONFIG_DIR_NAME).join("skills")) {
            return SkillSource::Project;
        }
    }
    SkillSource::Path
}

fn is_under_path(target: &Path, root: &Path) -> bool {
    let normalized_target = std::fs::canonicalize(target).unwrap_or_else(|_| target.to_path_buf());
    let normalized_root = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    normalized_target == normalized_root || normalized_target.starts_with(&normalized_root)
}

fn project_ancestor_dirs(cwd: &Path) -> Vec<PathBuf> {
    let git_root = find_git_root(cwd);
    let mut dirs = vec![];
    let mut current = Some(cwd);

    while let Some(dir) = current {
        dirs.push(dir.to_path_buf());

        if git_root.as_deref() == Some(dir) {
            break;
        }
        if git_root.is_none() && dir.parent().is_none() {
            break;
        }

        current = dir.parent();
    }

    dirs
}

fn find_git_root(start: &Path) -> Option<PathBuf> {
    let mut current = Some(start);
    while let Some(dir) = current {
        if dir.join(".git").exists() {
            return Some(dir.to_path_buf());
        }
        current = dir.parent();
    }
    None
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use tempfile::tempdir;

    use super::*;

    #[test]
    fn load_skills_from_dir_loads_valid_skill() {
        let dir = tempdir().expect("temp dir");
        let skill_dir = dir.path().join("valid-skill");
        std::fs::create_dir_all(&skill_dir).expect("create skill dir");
        std::fs::write(
            skill_dir.join("SKILL.md"),
            r#"---
name: valid-skill
description: A valid skill for testing.
---

# Valid Skill
"#,
        )
        .expect("write skill");

        let result = load_skills_from_dir(dir.path(), SkillSource::Path);
        assert_eq!(
            result.skills.len(),
            1,
            "diagnostics: {:?}",
            result.diagnostics
        );
        assert_eq!(result.skills[0].name, "valid-skill");
        assert!(result.diagnostics.is_empty());
    }

    #[test]
    fn format_skills_for_prompt_renders_visible_skills_only() {
        let skills = vec![
            Skill {
                name: "visible-skill".to_string(),
                description: "Visible skill".to_string(),
                file_path: PathBuf::from("/skills/visible/SKILL.md"),
                base_dir: PathBuf::from("/skills/visible"),
                source: SkillSource::Path,
                disable_model_invocation: false,
            },
            Skill {
                name: "hidden-skill".to_string(),
                description: "Hidden skill".to_string(),
                file_path: PathBuf::from("/skills/hidden/SKILL.md"),
                base_dir: PathBuf::from("/skills/hidden"),
                source: SkillSource::Path,
                disable_model_invocation: true,
            },
        ];

        let prompt = format_skills_for_prompt(&skills);
        assert!(prompt.contains("<available_skills>"));
        assert!(prompt.contains("<name>visible-skill</name>"));
        assert!(
            prompt.contains("After reading a matching skill, execute it with available tools.")
        );
        assert!(!prompt.contains("hidden-skill"));
    }

    #[test]
    fn load_skills_resolves_relative_skill_paths() {
        let dir = tempdir().expect("temp dir");
        let skill_dir = dir.path().join("extra").join("path-skill");
        std::fs::create_dir_all(&skill_dir).expect("create skill dir");
        std::fs::write(
            skill_dir.join("SKILL.md"),
            r#"---
name: path-skill
description: Loaded from explicit relative path.
---
"#,
        )
        .expect("write skill");

        let mut options =
            LoadSkillsOptions::new(dir.path().to_path_buf(), dir.path().join(".pixy"));
        options.include_defaults = false;
        options.skill_paths = vec!["extra/path-skill".to_string()];

        let result = load_skills(options);
        assert_eq!(
            result.skills.len(),
            1,
            "diagnostics: {:?}",
            result.diagnostics
        );
        assert_eq!(result.skills[0].name, "path-skill");
        assert_eq!(result.skills[0].source, SkillSource::Path);
    }

    #[test]
    fn load_skills_skips_missing_description() {
        let dir = tempdir().expect("temp dir");
        let skill_dir = dir.path().join("missing-description");
        std::fs::create_dir_all(&skill_dir).expect("create skill dir");
        std::fs::write(
            skill_dir.join("SKILL.md"),
            r#"---
name: missing-description
---
"#,
        )
        .expect("write skill");

        let result = load_skills_from_dir(dir.path(), SkillSource::Path);
        assert!(result.skills.is_empty());
        assert!(
            result
                .diagnostics
                .iter()
                .any(|item| item.message.contains("description is required"))
        );
    }

    #[test]
    fn load_skills_reports_name_collision_and_keeps_first() {
        let dir = tempdir().expect("temp dir");
        let first_dir = dir.path().join("alpha").join("same-skill");
        let second_dir = dir.path().join("beta").join("same-skill");
        std::fs::create_dir_all(&first_dir).expect("create first");
        std::fs::create_dir_all(&second_dir).expect("create second");
        std::fs::write(
            first_dir.join("SKILL.md"),
            r#"---
name: same-skill
description: first
---
"#,
        )
        .expect("write first");
        std::fs::write(
            second_dir.join("SKILL.md"),
            r#"---
name: same-skill
description: second
---
"#,
        )
        .expect("write second");

        let mut options =
            LoadSkillsOptions::new(dir.path().to_path_buf(), dir.path().join(".pixy/agent"));
        options.include_defaults = false;
        options.skill_paths = vec!["alpha".to_string(), "beta".to_string()];

        let result = load_skills(options);
        assert_eq!(result.skills.len(), 1);
        assert_eq!(result.skills[0].description, "first");
        assert!(
            result
                .diagnostics
                .iter()
                .any(|item| item.kind == SkillDiagnosticKind::Collision)
        );
    }

    #[test]
    fn load_skills_scans_agents_dir_up_to_repo_root() {
        let dir = tempdir().expect("temp dir");
        let repo_root = dir.path().join("repo");
        let nested = repo_root.join("a").join("b");
        std::fs::create_dir_all(&nested).expect("create nested");
        std::fs::create_dir_all(repo_root.join(".git")).expect("create git dir");

        let repo_skill = repo_root.join(".agents").join("skills").join("repo-skill");
        std::fs::create_dir_all(&repo_skill).expect("create repo skill dir");
        std::fs::write(
            repo_skill.join("SKILL.md"),
            r#"---
name: repo-skill
description: skill from repo root.
---
"#,
        )
        .expect("write repo skill");

        let outside_skill = dir.path().join(".agents").join("skills").join("outside");
        std::fs::create_dir_all(&outside_skill).expect("create outside dir");
        std::fs::write(
            outside_skill.join("SKILL.md"),
            r#"---
name: outside
description: should not be discovered.
---
"#,
        )
        .expect("write outside skill");

        let options = LoadSkillsOptions::new(nested, dir.path().join(".pixy/agent"));
        let result = load_skills(options);

        let names = result
            .skills
            .iter()
            .map(|skill| skill.name.clone())
            .collect::<HashSet<_>>();
        assert!(names.contains("repo-skill"));
        assert!(!names.contains("outside"));
    }
}
