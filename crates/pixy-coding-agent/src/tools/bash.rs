use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use pixy_agent_core::{AgentTool, AgentToolExecutor, AgentToolResult, ToolFuture};
use serde_json::{Value, json};
use tokio::process::Command;
use tokio::time::timeout;

use super::common::{
    DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, format_timeout, get_optional_f64, get_required_string,
    text_result, truncate_tail, truncated_by_str,
};

pub fn create_bash_tool(cwd: impl AsRef<Path>) -> AgentTool {
    let cwd = cwd.as_ref().to_path_buf();
    AgentTool {
        name: "bash".to_string(),
        label: "bash".to_string(),
        description:
            "Execute a bash command in the workspace cwd and return combined stdout/stderr."
                .to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "command": { "type": "string", "description": "Shell command to execute." },
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

impl AgentToolExecutor for BashToolExecutor {
    fn execute(&self, _tool_call_id: String, args: Value) -> ToolFuture {
        let cwd = self.cwd.clone();
        Box::pin(async move { execute_bash_tool(&cwd, args).await })
    }
}

async fn execute_bash_tool(cwd: &Path, args: Value) -> Result<AgentToolResult, String> {
    if !cwd.exists() {
        return Err(format!(
            "Working directory does not exist: {}",
            cwd.display()
        ));
    }

    let command = get_required_string(&args, "command")?;
    let timeout_seconds = get_optional_f64(&args, "timeout")?;
    if let Some(seconds) = timeout_seconds {
        if seconds <= 0.0 {
            return Err("`timeout` must be > 0".to_string());
        }
    }

    let mut process = Command::new("bash");
    process.arg("-lc").arg(command).current_dir(cwd);

    let output = match timeout_seconds {
        Some(seconds) => timeout(Duration::from_secs_f64(seconds), process.output())
            .await
            .map_err(|_| {
                format!(
                    "Command timed out after {} seconds",
                    format_timeout(seconds)
                )
            })?
            .map_err(|error| format!("Failed to execute command: {error}"))?,
        None => process
            .output()
            .await
            .map_err(|error| format!("Failed to execute command: {error}"))?,
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
        return Err(output_text);
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
