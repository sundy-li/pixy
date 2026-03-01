use std::path::{Path, PathBuf};

use crate::{format_skills_for_prompt, Skill, SkillSource, SubAgentSpec};
use chrono::Local;
use pixy_agent_core::AgentTool;

const DEFAULT_PROMPT_INTRO: &str = "You are pixy, an expert coding assistant and coding agent harness. You help users by reading files, executing commands, editing code, and writing new files.";

pub fn build_system_prompt(
    custom_prompt: Option<&str>,
    cwd: &Path,
    tools: &[AgentTool],
    skills: &[Skill],
) -> String {
    let now_text = Local::now()
        .format("%A, %B %-d, %Y, %I:%M:%S %p %Z")
        .to_string();
    let tool_names: Vec<&str> = tools.iter().map(|tool| tool.name.as_str()).collect();
    build_system_prompt_with_now(custom_prompt, cwd, &tool_names, skills, &now_text)
}

pub fn append_multi_agent_prompt_section(
    prompt: &mut String,
    tools: &[AgentTool],
    subagents: &[SubAgentSpec],
) {
    if !tools.iter().any(|tool| tool.name == "task") {
        return;
    }

    let mut sorted = subagents
        .iter()
        .filter(|spec| !spec.name.trim().is_empty())
        .collect::<Vec<_>>();
    if sorted.is_empty() {
        return;
    }
    sorted.sort_by(|left, right| left.name.cmp(&right.name));

    let mut lines = vec![
        "<MULTI_AGENT>".to_string(),
        "Task delegation is enabled through the `task` tool.".to_string(),
        "Available subagents:".to_string(),
    ];
    for spec in sorted {
        lines.push(format!("- {}: {}", spec.name, spec.description));
    }
    lines.push("</MULTI_AGENT>".to_string());

    append_prompt_section(prompt, &lines.join("\n"));
}

fn build_system_prompt_with_now(
    custom_prompt: Option<&str>,
    cwd: &Path,
    selected_tools: &[&str],
    skills: &[Skill],
    now_text: &str,
) -> String {
    let mut prompt = build_default_prompt(custom_prompt, cwd, selected_tools);
    let has_read_tool = selected_tools.contains(&"read");
    if has_read_tool {
        let skills_prompt = format_skills_for_prompt(skills);
        if !skills_prompt.is_empty() {
            prompt.push_str(&skills_prompt);
        }
    }
    if let Some(workspace_agents) = load_workspace_agents_prompt(cwd) {
        append_prompt_section(&mut prompt, &workspace_agents);
    }
    let workspace_skills = format_workspace_skills_for_prompt(cwd, skills);
    if !workspace_skills.is_empty() {
        append_prompt_section(&mut prompt, &workspace_skills);
    }

    if !prompt.ends_with('\n') {
        prompt.push('\n');
    }
    prompt.push('\n');
    prompt.push_str(&format!(
        "<context>\nCurrent date and time: {now_text}\n</context>\n\n<workspace_context>\nCurrent working directory: {}\n</workspace_context>",
        cwd.display()
    ));

    prompt
}

fn append_prompt_section(prompt: &mut String, section: &str) {
    if section.trim().is_empty() {
        return;
    }
    if !prompt.ends_with('\n') {
        prompt.push('\n');
    }
    prompt.push('\n');
    prompt.push_str(section.trim_end());
}

fn load_workspace_agents_prompt(cwd: &Path) -> Option<String> {
    let path = find_workspace_context_file(cwd)?;
    let content = match std::fs::read_to_string(&path) {
        Ok(content) => content,
        Err(error) => {
            eprintln!(
                "warning: could not read workspace context file {}: {error}",
                path.display()
            );
            return None;
        }
    };
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return None;
    }

    Some(format!(
        "<WORKSPACE_AGENTS>\n{trimmed}\n</WORKSPACE_AGENTS>"
    ))
}

fn find_workspace_context_file(cwd: &Path) -> Option<PathBuf> {
    for name in ["AGENTS.md", "CLAUDE.md"] {
        let candidate = cwd.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn format_workspace_skills_for_prompt(cwd: &Path, skills: &[Skill]) -> String {
    let mut workspace_skills = skills
        .iter()
        .filter(|skill| !skill.disable_model_invocation)
        .filter(|skill| is_workspace_skill(skill, cwd))
        .collect::<Vec<_>>();
    workspace_skills.sort_by(|a, b| {
        a.name
            .cmp(&b.name)
            .then_with(|| a.file_path.cmp(&b.file_path))
    });
    workspace_skills.dedup_by(|a, b| a.name == b.name && a.file_path == b.file_path);

    if workspace_skills.is_empty() {
        return String::new();
    }

    let mut lines = vec!["<WORKSPACE_SKILLS>".to_string()];
    for skill in workspace_skills {
        lines.push("  <SKILL>".to_string());
        lines.push(format!("    <NAME>{}</NAME>", escape_xml(&skill.name)));
        lines.push(format!(
            "    <DESCRIPTION>{}</DESCRIPTION>",
            escape_xml(&skill.description)
        ));
        lines.push(format!(
            "    <LOCATION>{}</LOCATION>",
            escape_xml(skill.file_path.to_string_lossy().as_ref())
        ));
        lines.push("  </SKILL>".to_string());
    }
    lines.push("</WORKSPACE_SKILLS>".to_string());
    lines.join("\n")
}

fn is_workspace_skill(skill: &Skill, cwd: &Path) -> bool {
    if skill.source == SkillSource::Project {
        return true;
    }

    let workspace_root = normalize_path_for_compare(cwd, cwd);
    let skill_path = normalize_path_for_compare(&skill.file_path, cwd);
    skill_path.starts_with(&workspace_root)
}

fn normalize_path_for_compare(path: &Path, cwd: &Path) -> PathBuf {
    let candidate = if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    };
    std::fs::canonicalize(&candidate).unwrap_or(candidate)
}

fn escape_xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn build_default_prompt(
    custom_prompt: Option<&str>,
    cwd: &Path,
    selected_tools: &[&str],
) -> String {
    let prompt = resolve_prompt_body(custom_prompt, cwd);
    let tool_lines: Vec<String> = selected_tools
        .iter()
        .copied()
        .filter_map(|name| {
            tool_description(name).map(|description| format!("- {name}: {description}"))
        })
        .collect();
    let tools_text = if tool_lines.is_empty() {
        "(none)".to_string()
    } else {
        tool_lines.join("\n")
    };

    let guidelines = build_guidelines(selected_tools);
    let sections = [
        format!("<identity>\n{prompt}\n</identity>"),
        "<runtime_contract>\n\
1) Use available tools for concrete actions (file operations, shell commands, and log inspection).\n\
2) Do not ask the user to manually run commands or edit files when tools can do it directly.\n\
3) Prefer execution over command-only suggestions unless the user explicitly asks for commands only.\n\
4) Ask for confirmation only before clearly destructive or irreversible actions.\n\
</runtime_contract>"
            .to_string(),
        format!(
            "<tools_contract>\n\
You have access to the following built-in tools:\n\
{tools_text}\n\
\n\
Guidelines:\n\
{guidelines}\n\
</tools_contract>"
        ),
    ];

    sections.join("\n\n")
}

fn resolve_prompt_body(custom_prompt: Option<&str>, cwd: &Path) -> String {
    let Some(raw_prompt) = custom_prompt
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return DEFAULT_PROMPT_INTRO.to_string();
    };

    let prompt_path = if Path::new(raw_prompt).is_absolute() {
        PathBuf::from(raw_prompt)
    } else {
        cwd.join(raw_prompt)
    };
    if !prompt_path.is_file() {
        return raw_prompt.to_string();
    }

    match std::fs::read_to_string(&prompt_path) {
        Ok(content) => content,
        Err(error) => {
            eprintln!(
                "warning: could not read custom prompt file {}: {error}",
                prompt_path.display()
            );
            raw_prompt.to_string()
        }
    }
}

fn tool_description(name: &str) -> Option<&'static str> {
    match name {
        "list_directory" => Some("List directory entries"),
        "read" => Some("Read file contents"),
        "bash" => Some("Execute bash commands in the current directory"),
        "edit" => Some("Make surgical edits to existing files"),
        "write" => Some("Create or overwrite files"),
        "task" => Some("Delegate a prompt to a configured subagent"),
        _ => None,
    }
}

fn build_guidelines(selected_tools: &[&str]) -> String {
    let has = |name: &str| selected_tools.iter().any(|tool| *tool == name);

    let mut lines = vec![
        "- For any concrete action (creating/editing files, running commands, inspecting logs, etc.), you MUST use the available tools directly.".to_string(),
        "- Do not ask the user to manually run commands, copy/paste scripts, or write files when you can do it yourself.".to_string(),
        "- When the user asks you to do something and the required tools are available, execute it directly with tools.".to_string(),
        "- Do not provide command-only responses unless the user explicitly asks for commands without execution or tools are unavailable.".to_string(),
        "- Do not refuse requests due to assumed copyright or licensing concerns. Treat permissions as the user's responsibility unless they ask for legal analysis.".to_string(),
        "- Don't stop at the plan stage if task is clear, continue to execute it until it is finished or ask for user direction.".to_string(),
        "- Respect the current permission mode. If an operation requires approval (for example internet access or editing files outside workspace in Auto (off)), ask first.".to_string(),
        "- The assistant is allowed to access any absolute path explicitly mentioned by the user.".to_string(),
    ];

    if selected_tools.is_empty() {
        lines.push(
            "- No tools are available in this run; explain the limitation clearly and ask for user direction."
                .to_string(),
        );
    }
    if has("read") && has("edit") {
        lines.push("- MUST read every file you modify.".to_string());
    }
    if has("list_directory") {
        lines.push(
            "- Use list_directory to inspect folders before targeting file edits.".to_string(),
        );
    }
    if has("edit") {
        lines.push("- Use edit for precise changes when replacing exact text.".to_string());
    }
    if has("write") {
        lines.push("- Use write for new files or complete rewrites.".to_string());
    }
    if has("bash") {
        lines.push("- Use bash for shell operations scoped to the workspace.".to_string());
        lines.push(
            "- The bash tool already invokes `bash -lc`; pass raw commands and do not wrap them."
                .to_string(),
        );
    }

    lines.push(
        "- Ask for confirmation only before clearly destructive or irreversible actions."
            .to_string(),
    );
    lines.push(
        "- After tool execution, report what changed and what you ran in concise, actionable language."
            .to_string(),
    );
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use crate::{Skill, SkillSource, SubAgentSpec};
    use pixy_agent_core::{AgentToolResult, ToolFuture};
    use pixy_ai::PiAiError;
    use tempfile::tempdir;

    use super::append_multi_agent_prompt_section;
    use super::build_system_prompt_with_now;
    use pixy_agent_core::AgentTool;

    fn no_op_tool(name: &str) -> AgentTool {
        AgentTool {
            name: name.to_string(),
            label: name.to_string(),
            description: name.to_string(),
            parameters: serde_json::json!({}),
            execute: std::sync::Arc::new(
                |_tool_call_id: String, _args: serde_json::Value| -> ToolFuture {
                    Box::pin(async {
                        Err(PiAiError::new(
                            pixy_ai::PiAiErrorCode::ToolExecutionFailed,
                            "unused",
                        )) as Result<AgentToolResult, PiAiError>
                    })
                },
            ),
        }
    }

    #[test]
    fn default_prompt_uses_sectioned_structure() {
        let prompt = build_system_prompt_with_now(
            None,
            Path::new("/workspace"),
            &["list_directory", "read", "bash", "edit", "write"],
            &[],
            "Monday, February 23, 2026, 11:00:00 AM UTC",
        );

        assert!(prompt.contains("<identity>"));
        assert!(prompt.contains("<runtime_contract>"));
        assert!(prompt.contains("<tools_contract>"));
        assert!(prompt.contains("<context>"));
        assert!(prompt.contains("<workspace_context>"));
    }

    #[test]
    fn default_prompt_includes_tools_and_runtime_context() {
        let prompt = build_system_prompt_with_now(
            None,
            Path::new("/workspace"),
            &["list_directory", "read", "bash", "edit", "write"],
            &[],
            "Monday, February 23, 2026, 11:00:00 AM UTC",
        );

        assert!(prompt.contains("You are pixy"));
        assert!(prompt.contains("You have access to the following built-in tools:"));
        assert!(prompt.contains("- list_directory:"));
        assert!(prompt.contains("- read:"));
        assert!(prompt.contains("- bash:"));
        assert!(prompt.contains("- edit:"));
        assert!(prompt.contains("- write:"));
        assert!(
            prompt.contains("Current date and time: Monday, February 23, 2026, 11:00:00 AM UTC")
        );
        assert!(prompt.contains("Current working directory: /workspace"));
    }

    #[test]
    fn empty_tools_show_none() {
        let prompt = build_system_prompt_with_now(
            None,
            Path::new("/workspace"),
            &[],
            &[],
            "Monday, February 23, 2026, 11:00:00 AM UTC",
        );

        assert!(prompt.contains("You have access to the following built-in tools:\n(none)"));
    }

    #[test]
    fn custom_prompt_file_replaces_default_body() {
        let dir = tempdir().expect("temp dir");
        let prompt_path = dir.path().join("prompt.txt");
        std::fs::write(&prompt_path, "You are custom from file.").expect("write prompt file");

        let prompt = build_system_prompt_with_now(
            Some("prompt.txt"),
            dir.path(),
            &["list_directory", "read", "bash", "edit", "write"],
            &[],
            "Monday, February 23, 2026, 11:00:00 AM UTC",
        );

        assert!(prompt.contains("You are custom from file."));
        assert!(!prompt.contains("Available tools:"));
        assert!(prompt.contains("Current working directory:"));
    }

    #[test]
    fn prompt_includes_available_skills_when_read_tool_is_available() {
        let prompt = build_system_prompt_with_now(
            None,
            Path::new("/workspace"),
            &["list_directory", "read", "bash", "edit", "write"],
            &[
                Skill {
                    name: "visible-skill".to_string(),
                    description: "Visible skill".to_string(),
                    file_path: "/skills/visible/SKILL.md".into(),
                    base_dir: "/skills/visible".into(),
                    source: SkillSource::Path,
                    disable_model_invocation: false,
                },
                Skill {
                    name: "hidden-skill".to_string(),
                    description: "Hidden skill".to_string(),
                    file_path: "/skills/hidden/SKILL.md".into(),
                    base_dir: "/skills/hidden".into(),
                    source: SkillSource::Path,
                    disable_model_invocation: true,
                },
            ],
            "Monday, February 23, 2026, 11:00:00 AM UTC",
        );

        assert!(prompt.contains("<available_skills>"));
        assert!(prompt.contains("<name>visible-skill</name>"));
        assert!(!prompt.contains("hidden-skill"));
    }

    #[test]
    fn prompt_omits_skills_when_read_tool_is_not_available() {
        let prompt = build_system_prompt_with_now(
            None,
            Path::new("/workspace"),
            &["list_directory", "bash", "edit", "write"],
            &[Skill {
                name: "visible-skill".to_string(),
                description: "Visible skill".to_string(),
                file_path: "/skills/visible/SKILL.md".into(),
                base_dir: "/skills/visible".into(),
                source: SkillSource::Path,
                disable_model_invocation: false,
            }],
            "Monday, February 23, 2026, 11:00:00 AM UTC",
        );

        assert!(!prompt.contains("<available_skills>"));
    }

    #[test]
    fn default_prompt_includes_action_and_authorization_guidelines() {
        let prompt = build_system_prompt_with_now(
            None,
            Path::new("/workspace"),
            &["list_directory", "read", "bash", "edit", "write"],
            &[],
            "Monday, February 23, 2026, 11:00:00 AM UTC",
        );

        assert!(prompt.contains(
            "When the user asks you to do something and the required tools are available, execute it directly with tools."
        ));
        assert!(prompt
            .contains("Do not refuse requests due to assumed copyright or licensing concerns."));
        assert!(prompt.contains(
            "The bash tool already invokes `bash -lc`; pass raw commands and do not wrap them."
        ));
    }

    #[test]
    fn prompt_includes_workspace_agents_file_when_present() {
        let dir = tempdir().expect("temp dir");
        std::fs::write(dir.path().join("AGENTS.md"), "workspace agents prompt")
            .expect("write agents");

        let prompt = build_system_prompt_with_now(
            None,
            dir.path(),
            &["list_directory", "read", "bash", "edit", "write"],
            &[],
            "Monday, February 23, 2026, 11:00:00 AM UTC",
        );

        assert!(prompt.contains("<WORKSPACE_AGENTS>"));
        assert!(prompt.contains("workspace agents prompt"));
        assert!(prompt.contains("</WORKSPACE_AGENTS>"));
    }

    #[test]
    fn prompt_falls_back_to_workspace_claude_file() {
        let dir = tempdir().expect("temp dir");
        std::fs::write(dir.path().join("CLAUDE.md"), "workspace claude prompt")
            .expect("write claude");

        let prompt = build_system_prompt_with_now(
            None,
            dir.path(),
            &["list_directory", "read", "bash", "edit", "write"],
            &[],
            "Monday, February 23, 2026, 11:00:00 AM UTC",
        );

        assert!(prompt.contains("<WORKSPACE_AGENTS>"));
        assert!(prompt.contains("workspace claude prompt"));
    }

    #[test]
    fn prompt_includes_workspace_skills_only() {
        let dir = tempdir().expect("temp dir");
        let project_skill_path = dir.path().join(".agents/skills/demo/SKILL.md");
        let path_skill_path = dir.path().join(".skills/local/SKILL.md");

        let prompt = build_system_prompt_with_now(
            None,
            dir.path(),
            &["list_directory", "bash", "edit", "write"],
            &[
                Skill {
                    name: "project-skill".to_string(),
                    description: "Project skill".to_string(),
                    file_path: project_skill_path,
                    base_dir: PathBuf::from("/skills/project"),
                    source: SkillSource::Project,
                    disable_model_invocation: false,
                },
                Skill {
                    name: "path-skill".to_string(),
                    description: "Path skill in workspace".to_string(),
                    file_path: path_skill_path,
                    base_dir: PathBuf::from("/skills/path"),
                    source: SkillSource::Path,
                    disable_model_invocation: false,
                },
                Skill {
                    name: "user-skill".to_string(),
                    description: "User skill".to_string(),
                    file_path: PathBuf::from("/users/demo/.agents/skills/user/SKILL.md"),
                    base_dir: PathBuf::from("/users/demo/.agents/skills/user"),
                    source: SkillSource::User,
                    disable_model_invocation: false,
                },
            ],
            "Monday, February 23, 2026, 11:00:00 AM UTC",
        );

        assert!(prompt.contains("<WORKSPACE_SKILLS>"));
        assert!(prompt.contains("project-skill"));
        assert!(prompt.contains("path-skill"));
        assert!(!prompt.contains("user-skill"));
        assert!(prompt.contains("</WORKSPACE_SKILLS>"));
    }

    #[test]
    fn append_multi_agent_prompt_section_includes_subagent_names_when_task_tool_present() {
        let mut prompt = "base prompt".to_string();
        let tools = vec![no_op_tool("task")];
        let subagents = vec![
            SubAgentSpec {
                name: "general".to_string(),
                description: "General helper".to_string(),
                mode: crate::SubAgentMode::SubAgent,
            },
            SubAgentSpec {
                name: "explore".to_string(),
                description: "Exploration helper".to_string(),
                mode: crate::SubAgentMode::SubAgent,
            },
        ];

        append_multi_agent_prompt_section(&mut prompt, &tools, &subagents);

        assert!(prompt.contains("<MULTI_AGENT>"));
        assert!(prompt.contains("explore"));
        assert!(prompt.contains("general"));
    }

    #[test]
    fn append_multi_agent_prompt_section_skips_when_task_tool_absent() {
        let mut prompt = "base prompt".to_string();
        let tools = vec![no_op_tool("read")];
        let subagents = vec![SubAgentSpec {
            name: "general".to_string(),
            description: "General helper".to_string(),
            mode: crate::SubAgentMode::SubAgent,
        }];

        append_multi_agent_prompt_section(&mut prompt, &tools, &subagents);

        assert!(!prompt.contains("<MULTI_AGENT>"));
    }
}
