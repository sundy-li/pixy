use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use pixy_agent_core::{AgentTool, AgentToolExecutor, AgentToolResult};
use pixy_ai::PiAiError;
use serde_json::{Value, json};

use super::common::{
    DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, get_optional_usize, get_required_string,
    invalid_tool_args, resolve_to_cwd, text_result, tool_execution_failed, truncate_head,
    truncated_by_str,
};

pub fn create_read_tool(cwd: impl AsRef<Path>) -> AgentTool {
    let cwd = cwd.as_ref().to_path_buf();
    AgentTool {
        name: "read".to_string(),
        label: "read".to_string(),
        description: "Read UTF-8 text file content from disk. Supports offset/limit pagination."
            .to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Path to the file, absolute or relative to workspace cwd, no need to ask for permission." },
                "offset": { "type": "integer", "minimum": 1, "description": "1-based start line offset." },
                "limit": { "type": "integer", "minimum": 1, "description": "Maximum number of lines to return." }
            },
            "required": ["path"],
            "additionalProperties": false
        }),
        execute: Arc::new(ReadToolExecutor { cwd }),
    }
}

struct ReadToolExecutor {
    cwd: PathBuf,
}

#[async_trait]
impl AgentToolExecutor for ReadToolExecutor {
    async fn execute(
        &self,
        _tool_call_id: String,
        args: Value,
    ) -> Result<AgentToolResult, PiAiError> {
        let path = get_required_string(&args, "path")?;
        let offset = get_optional_usize(&args, "offset")?.unwrap_or(1);
        if offset == 0 {
            return Err(invalid_tool_args("`offset` must be >= 1"));
        }
        let limit = get_optional_usize(&args, "limit")?;
        if let Some(limit_value) = limit {
            if limit_value == 0 {
                return Err(invalid_tool_args("`limit` must be >= 1"));
            }
        }

        let absolute_path = resolve_to_cwd(&self.cwd, &path);
        let bytes = fs::read(&absolute_path)
            .map_err(|error| tool_execution_failed(format!("Failed to read {path}: {error}")))?;
        let full_content = String::from_utf8(bytes).map_err(|_| {
            tool_execution_failed(format!("Failed to read {path}: file is not valid UTF-8"))
        })?;
        let all_lines: Vec<&str> = full_content.split('\n').collect();

        if offset > all_lines.len() {
            return Err(invalid_tool_args(format!(
                "Offset {offset} is beyond end of file ({} lines total)",
                all_lines.len()
            )));
        }

        let start_index = offset - 1;
        let end_index = match limit {
            Some(limit_value) => start_index.saturating_add(limit_value).min(all_lines.len()),
            None => all_lines.len(),
        };
        let selected = all_lines[start_index..end_index].join("\n");
        let truncation = truncate_head(&selected, DEFAULT_MAX_LINES, DEFAULT_MAX_BYTES);

        let mut output = truncation.content.clone();
        if output.is_empty() && full_content.is_empty() {
            output = "(empty file)".to_string();
        }

        if truncation.truncated {
            let shown_start = offset;
            let shown_end = shown_start + truncation.output_lines.saturating_sub(1);
            let next_offset = shown_end + 1;
            output.push_str(&format!(
                "\n\n[Showing lines {shown_start}-{shown_end} of {}. Use offset={next_offset} to continue.]",
                all_lines.len()
            ));
        } else if end_index < all_lines.len() {
            let next_offset = end_index + 1;
            let remaining = all_lines.len() - end_index;
            output.push_str(&format!(
                "\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]"
            ));
        }

        Ok(text_result(
            output,
            json!({
                "path": path,
                "offset": offset,
                "limit": limit,
                "totalLines": truncation.total_lines,
                "outputLines": truncation.output_lines,
                "truncated": truncation.truncated,
                "truncatedBy": truncation.truncated_by.map(truncated_by_str),
                "outputBytes": truncation.output_bytes,
                "totalBytes": truncation.total_bytes,
            }),
        ))
    }
}
