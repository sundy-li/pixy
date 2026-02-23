use std::path::Path;

use chrono::Local;
use pixy_agent_core::AgentTool;

const DEFAULT_PROMPT_INTRO: &str =
    "You are pixy, a pragmatic coding assistant working in the user's local workspace.";

pub fn build_system_prompt(custom_prompt: Option<&str>, cwd: &Path, tools: &[AgentTool]) -> String {
    let now_text = Local::now()
        .format("%A, %B %-d, %Y, %I:%M:%S %p %Z")
        .to_string();
    let tool_names: Vec<&str> = tools.iter().map(|tool| tool.name.as_str()).collect();
    build_system_prompt_with_now(custom_prompt, cwd, &tool_names, &now_text)
}

fn build_system_prompt_with_now(
    custom_prompt: Option<&str>,
    cwd: &Path,
    selected_tools: &[&str],
    now_text: &str,
) -> String {
    let mut prompt = if let Some(custom) = custom_prompt.and_then(normalize_custom_prompt) {
        load_custom_prompt(custom, cwd)
    } else {
        build_default_prompt(selected_tools)
    };

    if !prompt.ends_with('\n') {
        prompt.push('\n');
    }
    prompt.push_str(&format!("Current date and time: {now_text}\n"));
    prompt.push_str(&format!("Current working directory: {}", cwd.display()));
    prompt
}

fn normalize_custom_prompt(value: &str) -> Option<&str> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

fn load_custom_prompt(custom_prompt: &str, cwd: &Path) -> String {
    let path = Path::new(custom_prompt);
    let candidate = if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    };

    if candidate.is_file() {
        match std::fs::read_to_string(&candidate) {
            Ok(content) => return content,
            Err(error) => {
                eprintln!(
                    "warning: could not read system prompt file {}: {error}",
                    candidate.display()
                );
            }
        }
    }

    custom_prompt.to_string()
}

fn build_default_prompt(selected_tools: &[&str]) -> String {
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

    format!("{DEFAULT_PROMPT_INTRO}\n\nAvailable tools:\n{tools_text}\n\nGuidelines:\n{guidelines}")
}

fn tool_description(name: &str) -> Option<&'static str> {
    match name {
        "read" => Some("Read file contents"),
        "bash" => Some("Execute bash commands in the workspace"),
        "edit" => Some("Make surgical edits to existing files"),
        "write" => Some("Create or overwrite files"),
        _ => None,
    }
}

fn build_guidelines(selected_tools: &[&str]) -> String {
    let has = |name: &str| selected_tools.iter().any(|tool| *tool == name);

    let mut lines = vec![
        "- For any concrete action (creating/editing files, running commands, inspecting logs, etc.), you MUST use the available tools directly.".to_string(),
        "- Do not ask the user to manually run commands, copy/paste scripts, or write files when you can do it yourself.".to_string(),
    ];

    if selected_tools.is_empty() {
        lines.push(
            "- No tools are available in this run; explain the limitation clearly and ask for user direction."
                .to_string(),
        );
    }
    if has("read") && has("edit") {
        lines.push("- Use read before edit to confirm exact file content.".to_string());
    }
    if has("edit") {
        lines.push("- Use edit for precise changes when replacing exact text.".to_string());
    }
    if has("write") {
        lines.push("- Use write for new files or complete rewrites.".to_string());
    }
    if has("bash") {
        lines.push("- Use bash for shell operations scoped to the workspace.".to_string());
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
    use std::path::Path;

    use tempfile::tempdir;

    use super::build_system_prompt_with_now;

    #[test]
    fn default_prompt_includes_tools_and_runtime_context() {
        let prompt = build_system_prompt_with_now(
            None,
            Path::new("/workspace"),
            &["read", "bash", "edit", "write"],
            "Monday, February 23, 2026, 11:00:00 AM UTC",
        );

        assert!(prompt.contains("You are pixy"));
        assert!(prompt.contains("Available tools:"));
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
            "Monday, February 23, 2026, 11:00:00 AM UTC",
        );

        assert!(prompt.contains("Available tools:\n(none)"));
    }

    #[test]
    fn custom_prompt_file_replaces_default_body() {
        let dir = tempdir().expect("temp dir");
        let prompt_path = dir.path().join("prompt.txt");
        std::fs::write(&prompt_path, "You are custom from file.").expect("write prompt file");

        let prompt = build_system_prompt_with_now(
            Some("prompt.txt"),
            dir.path(),
            &["read", "bash", "edit", "write"],
            "Monday, February 23, 2026, 11:00:00 AM UTC",
        );

        assert!(prompt.contains("You are custom from file."));
        assert!(!prompt.contains("Available tools:"));
        assert!(prompt.contains("Current working directory:"));
    }
}
