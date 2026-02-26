use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use pixy_agent_core::{AgentTool, AgentToolExecutor, AgentToolResult};
use pixy_ai::PiAiError;
use serde_json::{Value, json};

use super::common::{invalid_tool_args, text_result};

pub fn create_list_directory_tool(cwd: impl AsRef<Path>) -> AgentTool {
    let cwd = cwd.as_ref().to_path_buf();
    AgentTool {
        name: "list_directory".to_string(),
        label: "list_directory".to_string(),
        description: "List directory entries.".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list. Empty value lists workspace root."
                }
            },
            "additionalProperties": false
        }),
        execute: Arc::new(ListDirectoryToolExecutor { cwd }),
    }
}

struct ListDirectoryToolExecutor {
    cwd: PathBuf,
}

#[async_trait]
impl AgentToolExecutor for ListDirectoryToolExecutor {
    async fn execute(
        &self,
        _tool_call_id: String,
        args: Value,
    ) -> Result<AgentToolResult, PiAiError> {
        let cwd = self.cwd.clone();
        execute_list_directory_tool(&cwd, args)
    }
}

fn execute_list_directory_tool(cwd: &Path, args: Value) -> Result<AgentToolResult, PiAiError> {
    let path = match args.get("path") {
        Some(value) if value.is_null() => String::new(),
        Some(value) => value
            .as_str()
            .map(|value| value.to_string())
            .ok_or_else(|| invalid_tool_args("Missing or invalid `path`"))?,
        None => String::new(),
    };
    let requested_path = path.trim();

    let target = if requested_path.is_empty() {
        cwd.to_path_buf()
    } else {
        let candidate = PathBuf::from(requested_path);
        if candidate.is_absolute() {
            candidate
        } else {
            cwd.join(candidate)
        }
    };

    if !target.exists() {
        return Ok(text_result(
            format!("Directory not found: {requested_path}"),
            json!({
                "path": requested_path,
                "error": "not_found",
            }),
        ));
    }
    if !target.is_dir() {
        return Ok(text_result(
            format!("Not a directory: {requested_path}"),
            json!({
                "path": requested_path,
                "error": "not_directory",
            }),
        ));
    }

    match std::fs::read_dir(&target) {
        Ok(entries) => {
            let mut items = entries
                .filter_map(|entry| entry.ok())
                .map(|entry| {
                    let name = entry.file_name().to_string_lossy().to_string();
                    let entry_type = entry.file_type().ok();
                    if entry_type.is_some_and(|kind| kind.is_dir()) {
                        format!("  {name}/")
                    } else {
                        let size = entry.metadata().ok().map(|meta| meta.len()).unwrap_or(0);
                        format!("  {name}  ({size} bytes)")
                    }
                })
                .collect::<Vec<_>>();
            items.sort();
            let text = if items.is_empty() {
                "(empty directory)".to_string()
            } else {
                items.join("\n")
            };
            Ok(text_result(
                text,
                json!({
                    "path": requested_path,
                    "resolvedPath": target.display().to_string(),
                    "entryCount": items.len(),
                }),
            ))
        }
        Err(error) => Ok(text_result(
            format!("Error listing directory: {error}"),
            json!({
                "path": requested_path,
                "error": "read_dir_failed",
            }),
        )),
    }
}
