use std::collections::HashMap;
use std::path::{Path, PathBuf};

use pixy_agent_core::AgentToolResult;
use pixy_ai::{PiAiError, PiAiErrorCode, ToolResultContentBlock};
use serde_json::Value;

pub(super) const DEFAULT_MAX_LINES: usize = 4096;
pub(super) const DEFAULT_MAX_BYTES: usize = 256 * 1024;

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

pub(super) fn get_required_string(args: &Value, key: &str) -> Result<String, PiAiError> {
    args.get(key)
        .and_then(Value::as_str)
        .map(|value| value.to_string())
        .ok_or_else(|| invalid_tool_args(format!("Missing or invalid `{key}`")))
}

pub(super) fn get_optional_usize(args: &Value, key: &str) -> Result<Option<usize>, PiAiError> {
    match args.get(key) {
        None => Ok(None),
        Some(value) if value.is_null() => Ok(None),
        Some(value) => {
            if let Some(raw) = value.as_u64() {
                return usize::try_from(raw)
                    .map(Some)
                    .map_err(|_| invalid_tool_args(format!("`{key}` is too large")));
            }
            if let Some(raw) = value.as_i64() {
                if raw < 0 {
                    return Err(invalid_tool_args(format!("`{key}` must be >= 0")));
                }
                return usize::try_from(raw as u64)
                    .map(Some)
                    .map_err(|_| invalid_tool_args(format!("`{key}` is too large")));
            }
            Err(invalid_tool_args(format!("Missing or invalid `{key}`")))
        }
    }
}

pub(super) fn get_optional_f64(args: &Value, key: &str) -> Result<Option<f64>, PiAiError> {
    match args.get(key) {
        None => Ok(None),
        Some(value) if value.is_null() => Ok(None),
        Some(value) => value
            .as_f64()
            .map(Some)
            .ok_or_else(|| invalid_tool_args(format!("Missing or invalid `{key}`"))),
    }
}

pub(super) fn invalid_tool_args(message: impl Into<String>) -> PiAiError {
    PiAiError::new(PiAiErrorCode::ToolArgumentsInvalid, message.into())
}

pub(super) fn tool_execution_failed(message: impl Into<String>) -> PiAiError {
    PiAiError::new(PiAiErrorCode::ToolExecutionFailed, message.into())
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

pub(super) fn line_change_counts(before: &str, after: &str) -> (usize, usize) {
    let mut before_map = line_multiset(before);
    let after_map = line_multiset(after);
    let mut added = 0usize;
    let mut removed = 0usize;

    for (line, after_count) in after_map {
        match before_map.remove(&line) {
            Some(before_count) if after_count >= before_count => {
                added = added.saturating_add(after_count - before_count);
            }
            Some(before_count) => {
                removed = removed.saturating_add(before_count - after_count);
            }
            None => {
                added = added.saturating_add(after_count);
            }
        }
    }

    for (_, before_count) in before_map {
        removed = removed.saturating_add(before_count);
    }

    (added, removed)
}

pub(super) fn format_diff_stat_line(path: &str, before: &str, after: &str) -> String {
    const PATH_WIDTH: usize = 48;
    const BAR_WIDTH: usize = 20;

    let (added, removed) = line_change_counts(before, after);
    let changed = added.saturating_add(removed);
    let bar = diff_stat_bar(added, removed, BAR_WIDTH);
    let display_path = truncate_path_for_stat(path, PATH_WIDTH);

    if bar.is_empty() {
        format!("{display_path:<width$} | {changed:>4}", width = PATH_WIDTH)
    } else {
        format!(
            "{display_path:<width$} | {changed:>4} {bar}",
            width = PATH_WIDTH
        )
    }
}

fn line_multiset(content: &str) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for line in content.lines() {
        *counts.entry(line.to_string()).or_insert(0) += 1;
    }
    counts
}

fn diff_stat_bar(added: usize, removed: usize, width: usize) -> String {
    let total = added.saturating_add(removed);
    if total == 0 || width == 0 {
        return String::new();
    }

    if removed == 0 {
        return "+".repeat(width);
    }
    if added == 0 {
        return "-".repeat(width);
    }

    let mut plus_width = (added.saturating_mul(width) + total / 2) / total;
    if plus_width == 0 {
        plus_width = 1;
    } else if plus_width >= width {
        plus_width = width.saturating_sub(1);
    }
    let minus_width = width.saturating_sub(plus_width);
    format!("{}{}", "+".repeat(plus_width), "-".repeat(minus_width))
}

fn truncate_path_for_stat(path: &str, max_chars: usize) -> String {
    if path.chars().count() <= max_chars {
        return path.to_string();
    }

    if max_chars <= 3 {
        return ".".repeat(max_chars);
    }

    let suffix_chars = max_chars - 3;
    let suffix = path
        .chars()
        .rev()
        .take(suffix_chars)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<String>();
    format!("...{suffix}")
}
