use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use pixy_agent_core::{AgentTool, AgentToolExecutor, AgentToolResult};
use pixy_ai::PiAiError;
use serde_json::{Value, json};

use super::common::{
    format_diff_stat_line, get_required_string, line_change_counts, resolve_to_cwd, text_result,
    tool_execution_failed,
};

pub fn create_write_tool(cwd: impl AsRef<Path>) -> AgentTool {
    let cwd = cwd.as_ref().to_path_buf();
    AgentTool {
        name: "write".to_string(),
        label: "write".to_string(),
        description: "Write UTF-8 text content to a file, creating parent directories if needed."
            .to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Path to write, absolute or relative to workspace cwd." },
                "content": { "type": "string", "description": "Full file content to write." }
            },
            "required": ["path", "content"],
            "additionalProperties": false
        }),
        execute: Arc::new(WriteToolExecutor { cwd }),
    }
}

struct WriteToolExecutor {
    cwd: PathBuf,
}

#[async_trait]
impl AgentToolExecutor for WriteToolExecutor {
    async fn execute(
        &self,
        _tool_call_id: String,
        args: Value,
    ) -> Result<AgentToolResult, PiAiError> {
        let cwd = self.cwd.clone();
        execute_write_tool(&cwd, args)
    }
}

fn execute_write_tool(cwd: &Path, args: Value) -> Result<AgentToolResult, PiAiError> {
    let path = get_required_string(&args, "path")?;
    let content = get_required_string(&args, "content")?;
    let absolute_path = resolve_to_cwd(cwd, &path);
    let previous_content = match fs::read(&absolute_path) {
        Ok(bytes) => String::from_utf8_lossy(&bytes).to_string(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(_) => String::new(),
    };
    if let Some(parent) = absolute_path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            tool_execution_failed(format!("Failed to create parent directories: {error}"))
        })?;
    }

    fs::write(&absolute_path, &content)
        .map_err(|error| tool_execution_failed(format!("Failed to write {path}: {error}")))?;
    let (insertions, deletions) = line_change_counts(&previous_content, &content);
    Ok(text_result(
        format_diff_stat_line(&path, &previous_content, &content),
        json!({
            "path": path,
            "bytes": content.len(),
            "insertions": insertions,
            "deletions": deletions,
            "changedLines": insertions + deletions,
        }),
    ))
}
