use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use pixy_agent_core::{AgentTool, AgentToolExecutor, AgentToolResult};
use pixy_ai::PiAiError;
use serde_json::{json, Value};

use super::common::{
    first_changed_line, format_diff_stat_line, get_required_string, invalid_tool_args,
    line_change_counts, resolve_to_cwd, text_result, tool_execution_failed,
};

pub fn create_edit_tool(cwd: impl AsRef<Path>) -> AgentTool {
    let cwd = cwd.as_ref().to_path_buf();
    AgentTool {
        name: "edit".to_string(),
        label: "edit".to_string(),
        description: "Replace exactly one unique text fragment in a UTF-8 file.".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Path to edit, absolute or relative to workspace cwd." },
                "oldText": { "type": "string", "description": "Exact original text to replace. Must be unique in file." },
                "newText": { "type": "string", "description": "Replacement text." }
            },
            "required": ["path", "oldText", "newText"],
            "additionalProperties": false
        }),
        execute: Arc::new(EditToolExecutor { cwd }),
    }
}

struct EditToolExecutor {
    cwd: PathBuf,
}

#[async_trait]
impl AgentToolExecutor for EditToolExecutor {
    async fn execute(
        &self,
        _tool_call_id: String,
        args: Value,
    ) -> Result<AgentToolResult, PiAiError> {
        let cwd = self.cwd.clone();
        execute_edit_tool(&cwd, args)
    }
}

fn execute_edit_tool(cwd: &Path, args: Value) -> Result<AgentToolResult, PiAiError> {
    let path = get_required_string(&args, "path")?;
    let old_text = get_required_string(&args, "oldText")?;
    let new_text = get_required_string(&args, "newText")?;
    if old_text.is_empty() {
        return Err(invalid_tool_args("`oldText` must not be empty"));
    }

    let absolute_path = resolve_to_cwd(cwd, &path);
    let content = fs::read_to_string(&absolute_path)
        .map_err(|error| tool_execution_failed(format!("Failed to read {path}: {error}")))?;
    let occurrences = content.matches(&old_text).count();
    if occurrences == 0 {
        return Err(tool_execution_failed(format!(
            "Could not find the exact text in {path}. The old text must match exactly."
        )));
    }
    if occurrences > 1 {
        return Err(tool_execution_failed(format!(
            "Found {occurrences} occurrences of the text in {path}. The text must be unique."
        )));
    }

    let updated = content.replacen(&old_text, &new_text, 1);
    if updated == content {
        return Err(tool_execution_failed(format!(
            "No changes made to {path}. The replacement produced identical content."
        )));
    }

    fs::write(&absolute_path, updated.as_bytes())
        .map_err(|error| tool_execution_failed(format!("Failed to write {path}: {error}")))?;
    let (insertions, deletions) = line_change_counts(&content, &updated);
    Ok(text_result(
        format_diff_stat_line(&path, &content, &updated),
        json!({
            "path": path,
            "firstChangedLine": first_changed_line(&content, &updated),
            "occurrences": 1,
            "insertions": insertions,
            "deletions": deletions,
            "changedLines": insertions + deletions,
        }),
    ))
}
