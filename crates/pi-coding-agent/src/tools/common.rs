use std::path::{Path, PathBuf};

use pi_agent_core::AgentToolResult;
use pi_ai::ToolResultContentBlock;
use serde_json::Value;

pub(super) const DEFAULT_MAX_LINES: usize = 1024;
pub(super) const DEFAULT_MAX_BYTES: usize = 64 * 1024;

#[derive(Clone, Copy)]
pub(super) enum TruncatedBy {
    Lines,
    Bytes,
}

pub(super) struct TruncateResult {
    pub content: String,
    pub output_lines: usize,
    pub output_bytes: usize,
    pub total_lines: usize,
    pub total_bytes: usize,
    pub truncated: bool,
    pub truncated_by: Option<TruncatedBy>,
}

pub(super) fn truncate_head(content: &str, max_lines: usize, max_bytes: usize) -> TruncateResult {
    let total_lines = content.lines().count().max(1);
    let total_bytes = content.len();
    let mut lines: Vec<&str> = content.split('\n').collect();
    let mut truncated = false;
    let mut truncated_by = None;

    if lines.len() > max_lines {
        lines.truncate(max_lines);
        truncated = true;
        truncated_by = Some(TruncatedBy::Lines);
    }

    let mut output = lines.join("\n");
    if output.len() > max_bytes {
        output = truncate_prefix_bytes(&output, max_bytes);
        truncated = true;
        truncated_by = Some(TruncatedBy::Bytes);
    }

    let output_lines = if output.is_empty() {
        0
    } else {
        output.lines().count()
    };
    let output_bytes = output.len();

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

pub(super) fn truncate_tail(content: &str, max_lines: usize, max_bytes: usize) -> TruncateResult {
    let total_lines = content.lines().count().max(1);
    let total_bytes = content.len();
    let mut lines: Vec<&str> = content.split('\n').collect();
    let mut truncated = false;
    let mut truncated_by = None;

    if lines.len() > max_lines {
        lines = lines.split_off(lines.len() - max_lines);
        truncated = true;
        truncated_by = Some(TruncatedBy::Lines);
    }

    let mut output = lines.join("\n");
    if output.len() > max_bytes {
        output = truncate_suffix_bytes(&output, max_bytes);
        truncated = true;
        truncated_by = Some(TruncatedBy::Bytes);
    }

    let output_lines = if output.is_empty() {
        0
    } else {
        output.lines().count()
    };
    let output_bytes = output.len();

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

pub(super) fn truncate_prefix_bytes(content: &str, max_bytes: usize) -> String {
    if content.len() <= max_bytes {
        return content.to_string();
    }

    let mut end = max_bytes;
    while end > 0 && !content.is_char_boundary(end) {
        end -= 1;
    }
    content[..end].to_string()
}

pub(super) fn truncate_suffix_bytes(content: &str, max_bytes: usize) -> String {
    if content.len() <= max_bytes {
        return content.to_string();
    }

    let mut start = content.len() - max_bytes;
    while start < content.len() && !content.is_char_boundary(start) {
        start += 1;
    }
    content[start..].to_string()
}

pub(super) fn resolve_to_cwd(cwd: &Path, file_path: &str) -> PathBuf {
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

pub(super) fn expand_home(path: &str) -> String {
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

pub(super) fn get_required_string(args: &Value, key: &str) -> Result<String, String> {
    args.get(key)
        .and_then(Value::as_str)
        .map(|value| value.to_string())
        .ok_or_else(|| format!("Missing or invalid `{key}`"))
}

pub(super) fn get_optional_usize(args: &Value, key: &str) -> Result<Option<usize>, String> {
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

pub(super) fn get_optional_f64(args: &Value, key: &str) -> Result<Option<f64>, String> {
    match args.get(key) {
        None => Ok(None),
        Some(value) if value.is_null() => Ok(None),
        Some(value) => value
            .as_f64()
            .map(Some)
            .ok_or_else(|| format!("Missing or invalid `{key}`")),
    }
}

pub(super) fn text_result(text: String, details: Value) -> AgentToolResult {
    AgentToolResult {
        content: vec![ToolResultContentBlock::Text {
            text,
            text_signature: None,
        }],
        details,
    }
}

pub(super) fn first_changed_line(before: &str, after: &str) -> Option<usize> {
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

pub(super) fn truncated_by_str(value: TruncatedBy) -> &'static str {
    match value {
        TruncatedBy::Lines => "lines",
        TruncatedBy::Bytes => "bytes",
    }
}

pub(super) fn format_timeout(seconds: f64) -> String {
    let rounded = seconds.round();
    if (seconds - rounded).abs() < f64::EPSILON {
        (rounded as i64).to_string()
    } else {
        format!("{seconds}")
    }
}
