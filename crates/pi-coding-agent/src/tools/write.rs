use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use pi_agent_core::{AgentTool, AgentToolExecutor, AgentToolResult, ToolFuture};
use serde_json::{Value, json};

use super::common::{get_required_string, resolve_to_cwd, text_result};

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

impl AgentToolExecutor for WriteToolExecutor {
    fn execute(&self, _tool_call_id: String, args: Value) -> ToolFuture {
        let cwd = self.cwd.clone();
        Box::pin(async move { execute_write_tool(&cwd, args) })
    }
}

fn execute_write_tool(cwd: &Path, args: Value) -> Result<AgentToolResult, String> {
    let path = get_required_string(&args, "path")?;
    let content = get_required_string(&args, "content")?;
    let absolute_path = resolve_to_cwd(cwd, &path);
    if let Some(parent) = absolute_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("Failed to create parent directories: {error}"))?;
    }

    fs::write(&absolute_path, &content)
        .map_err(|error| format!("Failed to write {path}: {error}"))?;
    Ok(text_result(
        format!("Successfully wrote {} bytes to {}", content.len(), path),
        json!({
            "path": path,
            "bytes": content.len(),
        }),
    ))
}
