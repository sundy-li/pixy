use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use pi_agent_core::{AgentTool, AgentToolResult};
use pi_ai::ToolResultContentBlock;
use serde_json::{Value, json};
use tokio::process::Command;
use tokio::time::timeout;

const DEFAULT_MAX_LINES: usize = 1024;
const DEFAULT_MAX_BYTES: usize = 64 * 1024;

#[derive(Clone, Copy)]
enum TruncatedBy {
    Lines,
    Bytes,
}

struct TruncateResult {
    content: String,
    output_lines: usize,
    output_bytes: usize,
    total_lines: usize,
    total_bytes: usize,
    truncated: bool,
    truncated_by: Option<TruncatedBy>,
}

pub fn create_coding_tools(cwd: impl AsRef<Path>) -> Vec<AgentTool> {
    let cwd = cwd.as_ref().to_path_buf();
    vec![
        create_read_tool(&cwd),
        create_bash_tool(&cwd),
        create_edit_tool(&cwd),
        create_write_tool(&cwd),
    ]
}

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
                "path": { "type": "string", "description": "Path to the file, absolute or relative to workspace cwd." },
                "offset": { "type": "integer", "minimum": 1, "description": "1-based start line offset." },
                "limit": { "type": "integer", "minimum": 1, "description": "Maximum number of lines to return." }
            },
            "required": ["path"],
            "additionalProperties": false
        }),
        execute: Arc::new(move |_tool_call_id: String, args: Value| {
            let cwd = cwd.clone();
            Box::pin(async move { execute_read_tool(&cwd, args) })
        }),
    }
}

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
        execute: Arc::new(move |_tool_call_id: String, args: Value| {
            let cwd = cwd.clone();
            Box::pin(async move { execute_write_tool(&cwd, args) })
        }),
    }
}

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
        execute: Arc::new(move |_tool_call_id: String, args: Value| {
            let cwd = cwd.clone();
            Box::pin(async move { execute_edit_tool(&cwd, args) })
        }),
    }
}

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
        execute: Arc::new(move |_tool_call_id: String, args: Value| {
            let cwd = cwd.clone();
            Box::pin(async move { execute_bash_tool(&cwd, args).await })
        }),
    }
}

fn execute_read_tool(cwd: &Path, args: Value) -> Result<AgentToolResult, String> {
    let path = get_required_string(&args, "path")?;
    let offset = get_optional_usize(&args, "offset")?.unwrap_or(1);
    if offset == 0 {
        return Err("`offset` must be >= 1".to_string());
    }
    let limit = get_optional_usize(&args, "limit")?;
    if let Some(limit_value) = limit {
        if limit_value == 0 {
            return Err("`limit` must be >= 1".to_string());
        }
    }

    let absolute_path = resolve_to_cwd(cwd, &path);
    let bytes =
        fs::read(&absolute_path).map_err(|error| format!("Failed to read {path}: {error}"))?;
    let full_content = String::from_utf8(bytes)
        .map_err(|_| format!("Failed to read {path}: file is not valid UTF-8"))?;
    let all_lines: Vec<&str> = full_content.split('\n').collect();

    if offset > all_lines.len() {
        return Err(format!(
            "Offset {offset} is beyond end of file ({} lines total)",
            all_lines.len()
        ));
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
        format!(
            "Successfully wrote {} bytes to {}",
            content.as_bytes().len(),
            path
        ),
        json!({
            "path": path,
            "bytes": content.as_bytes().len(),
        }),
    ))
}

fn execute_edit_tool(cwd: &Path, args: Value) -> Result<AgentToolResult, String> {
    let path = get_required_string(&args, "path")?;
    let old_text = get_required_string(&args, "oldText")?;
    let new_text = get_required_string(&args, "newText")?;
    if old_text.is_empty() {
        return Err("`oldText` must not be empty".to_string());
    }

    let absolute_path = resolve_to_cwd(cwd, &path);
    let content = fs::read_to_string(&absolute_path)
        .map_err(|error| format!("Failed to read {path}: {error}"))?;
    let occurrences = content.matches(&old_text).count();
    if occurrences == 0 {
        return Err(format!(
            "Could not find the exact text in {path}. The old text must match exactly."
        ));
    }
    if occurrences > 1 {
        return Err(format!(
            "Found {occurrences} occurrences of the text in {path}. The text must be unique."
        ));
    }

    let updated = content.replacen(&old_text, &new_text, 1);
    if updated == content {
        return Err(format!(
            "No changes made to {path}. The replacement produced identical content."
        ));
    }

    fs::write(&absolute_path, updated.as_bytes())
        .map_err(|error| format!("Failed to write {path}: {error}"))?;
    Ok(text_result(
        format!("Successfully replaced text in {path}."),
        json!({
            "path": path,
            "firstChangedLine": first_changed_line(&content, &updated),
            "occurrences": 1,
        }),
    ))
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

fn truncate_head(content: &str, max_lines: usize, max_bytes: usize) -> TruncateResult {
    let total_lines = content.lines().count().max(1);
    let total_bytes = content.as_bytes().len();
    let mut lines: Vec<&str> = content.split('\n').collect();
    let mut truncated = false;
    let mut truncated_by = None;

    if lines.len() > max_lines {
        lines.truncate(max_lines);
        truncated = true;
        truncated_by = Some(TruncatedBy::Lines);
    }

    let mut output = lines.join("\n");
    if output.as_bytes().len() > max_bytes {
        output = truncate_prefix_bytes(&output, max_bytes);
        truncated = true;
        truncated_by = Some(TruncatedBy::Bytes);
    }

    let output_lines = if output.is_empty() {
        0
    } else {
        output.lines().count()
    };
    let output_bytes = output.as_bytes().len();

    TruncateResult {
        content: output,
        output_lines,
        output_bytes,
        total_lines,
        total_bytes,
        truncated,
        truncated_by,
    }
}

fn truncate_tail(content: &str, max_lines: usize, max_bytes: usize) -> TruncateResult {
    let total_lines = content.lines().count().max(1);
    let total_bytes = content.as_bytes().len();
    let mut lines: Vec<&str> = content.split('\n').collect();
    let mut truncated = false;
    let mut truncated_by = None;

    if lines.len() > max_lines {
        lines = lines.split_off(lines.len() - max_lines);
        truncated = true;
        truncated_by = Some(TruncatedBy::Lines);
    }

    let mut output = lines.join("\n");
    if output.as_bytes().len() > max_bytes {
        output = truncate_suffix_bytes(&output, max_bytes);
        truncated = true;
        truncated_by = Some(TruncatedBy::Bytes);
    }

    let output_lines = if output.is_empty() {
        0
    } else {
        output.lines().count()
    };
    let output_bytes = output.as_bytes().len();

    TruncateResult {
        content: output,
        output_lines,
        output_bytes,
        total_lines,
        total_bytes,
        truncated,
        truncated_by,
    }
}

fn truncate_prefix_bytes(content: &str, max_bytes: usize) -> String {
    if content.as_bytes().len() <= max_bytes {
        return content.to_string();
    }

    let mut end = max_bytes;
    while end > 0 && !content.is_char_boundary(end) {
        end -= 1;
    }
    content[..end].to_string()
}

fn truncate_suffix_bytes(content: &str, max_bytes: usize) -> String {
    if content.as_bytes().len() <= max_bytes {
        return content.to_string();
    }

    let mut start = content.as_bytes().len() - max_bytes;
    while start < content.as_bytes().len() && !content.is_char_boundary(start) {
        start += 1;
    }
    content[start..].to_string()
}

fn resolve_to_cwd(cwd: &Path, file_path: &str) -> PathBuf {
    let normalized = if let Some(stripped) = file_path.strip_prefix('@') {
        stripped
    } else {
        file_path
    };

    let expanded = expand_home(normalized);
    let path = PathBuf::from(expanded);
    if path.is_absolute() {
        path
    } else {
        cwd.join(path)
    }
}

fn expand_home(path: &str) -> String {
    if path == "~" {
        return std::env::var("HOME").unwrap_or_else(|_| path.to_string());
    }
    if let Some(rest) = path.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return format!("{home}/{rest}");
        }
    }
    path.to_string()
}

fn get_required_string(args: &Value, key: &str) -> Result<String, String> {
    args.get(key)
        .and_then(Value::as_str)
        .map(|value| value.to_string())
        .ok_or_else(|| format!("Missing or invalid `{key}`"))
}

fn get_optional_usize(args: &Value, key: &str) -> Result<Option<usize>, String> {
    match args.get(key) {
        None => Ok(None),
        Some(value) if value.is_null() => Ok(None),
        Some(value) => {
            if let Some(raw) = value.as_u64() {
                return usize::try_from(raw)
                    .map(Some)
                    .map_err(|_| format!("`{key}` is too large"));
            }
            if let Some(raw) = value.as_i64() {
                if raw < 0 {
                    return Err(format!("`{key}` must be >= 0"));
                }
                return usize::try_from(raw as u64)
                    .map(Some)
                    .map_err(|_| format!("`{key}` is too large"));
            }
            Err(format!("Missing or invalid `{key}`"))
        }
    }
}

fn get_optional_f64(args: &Value, key: &str) -> Result<Option<f64>, String> {
    match args.get(key) {
        None => Ok(None),
        Some(value) if value.is_null() => Ok(None),
        Some(value) => value
            .as_f64()
            .map(Some)
            .ok_or_else(|| format!("Missing or invalid `{key}`")),
    }
}

fn text_result(text: String, details: Value) -> AgentToolResult {
    AgentToolResult {
        content: vec![ToolResultContentBlock::Text {
            text,
            text_signature: None,
        }],
        details,
    }
}

fn first_changed_line(before: &str, after: &str) -> Option<usize> {
    let before_lines: Vec<&str> = before.split('\n').collect();
    let after_lines: Vec<&str> = after.split('\n').collect();
    let common = before_lines.len().min(after_lines.len());

    for index in 0..common {
        if before_lines[index] != after_lines[index] {
            return Some(index + 1);
        }
    }
    if before_lines.len() != after_lines.len() {
        return Some(common + 1);
    }
    None
}

fn truncated_by_str(value: TruncatedBy) -> &'static str {
    match value {
        TruncatedBy::Lines => "lines",
        TruncatedBy::Bytes => "bytes",
    }
}

fn format_timeout(seconds: f64) -> String {
    let rounded = seconds.round();
    if (seconds - rounded).abs() < f64::EPSILON {
        (rounded as i64).to_string()
    } else {
        format!("{seconds}")
    }
}
