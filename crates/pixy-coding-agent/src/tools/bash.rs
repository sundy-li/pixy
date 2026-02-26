use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use pixy_agent_core::{AgentTool, AgentToolExecutor, AgentToolResult};
use pixy_ai::PiAiError;
use serde_json::{Value, json};
use tokio::process::Command;
use tokio::time::timeout;

use crate::bash_command::normalize_nested_bash_lc;

use super::common::{
    DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, format_timeout, get_optional_f64, get_required_string,
    invalid_tool_args, text_result, tool_execution_failed, truncate_tail, truncated_by_str,
};

pub fn create_bash_tool(cwd: impl AsRef<Path>) -> AgentTool {
    let cwd = cwd.as_ref().to_path_buf();
    AgentTool {
        name: "bash".to_string(),
        label: "bash".to_string(),
        description:
            "Execute a shell command in the cwd and return combined stdout/stderr. This tool already runs via `bash -lc`."
                .to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "command": { "type": "string", "description": "Shell command to execute (do not prefix with `bash -lc`; the tool already does that)." },
                "timeout": { "type": "number", "exclusiveMinimum": 0, "description": "Optional timeout in seconds." }
            },
            "required": ["command"],
            "additionalProperties": false
        }),
        execute: Arc::new(BashToolExecutor { cwd }),
    }
}

struct BashToolExecutor {
    cwd: PathBuf,
}

#[async_trait]
impl AgentToolExecutor for BashToolExecutor {
    async fn execute(
        &self,
        _tool_call_id: String,
        args: Value,
    ) -> Result<AgentToolResult, PiAiError> {
        let cwd = self.cwd.clone();
        execute_bash_tool(&cwd, args).await
    }
}

async fn execute_bash_tool(cwd: &Path, args: Value) -> Result<AgentToolResult, PiAiError> {
    if !cwd.exists() {
        return Err(tool_execution_failed(format!(
            "Working directory does not exist: {}",
            cwd.display()
        )));
    }

    let command = get_required_string(&args, "command")?;
    let normalized_command = normalize_nested_bash_lc(&command);
    let timeout_seconds = get_optional_f64(&args, "timeout")?;
    if let Some(seconds) = timeout_seconds {
        if seconds <= 0.0 {
            return Err(invalid_tool_args("`timeout` must be > 0"));
        }
    }

    let mut process = Command::new("bash");
    process
        .arg("-lc")
        .arg(normalized_command.as_ref())
        .current_dir(cwd);

    let output = match timeout_seconds {
        Some(seconds) => timeout(Duration::from_secs_f64(seconds), process.output())
            .await
            .map_err(|_| {
                tool_execution_failed(format!(
                    "Command timed out after {} seconds",
                    format_timeout(seconds)
                ))
            })?
            .map_err(|error| {
                tool_execution_failed(format!("Failed to execute command: {error}"))
            })?,
        None => process.output().await.map_err(|error| {
            tool_execution_failed(format!("Failed to execute command: {error}"))
        })?,
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let mut combined = String::new();
    if !stdout.is_empty() {
        combined.push_str(&stdout);
    }
    if !stderr.is_empty() {
        if !combined.is_empty() && !combined.ends_with('\n') {
            combined.push('\n');
        }
        combined.push_str(&stderr);
    }
    if combined.is_empty() {
        combined = "(no output)".to_string();
    }

    let truncation = truncate_tail(&combined, DEFAULT_MAX_LINES, DEFAULT_MAX_BYTES);
    let mut output_text = truncation.content.clone();
    if truncation.truncated {
        output_text.push_str(&format!(
            "\n\n[Output truncated: showing {} of {} lines ({} bytes).]",
            truncation.output_lines, truncation.total_lines, truncation.output_bytes
        ));
    }

    if !output.status.success() {
        if let Some(code) = output.status.code() {
            output_text.push_str(&format!("\n\nCommand exited with code {code}"));
        } else {
            output_text.push_str("\n\nCommand exited with unknown status");
        }
        return Err(tool_execution_failed(output_text));
    }

    Ok(text_result(
        output_text,
        json!({
            "exitCode": output.status.code(),
            "truncated": truncation.truncated,
            "truncatedBy": truncation.truncated_by.map(truncated_by_str),
            "outputLines": truncation.output_lines,
            "totalLines": truncation.total_lines,
            "outputBytes": truncation.output_bytes,
            "totalBytes": truncation.total_bytes,
        }),
    ))
}
