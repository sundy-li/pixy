use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use pixy_agent_core::{AgentTool, AgentToolExecutor, AgentToolResult};
use pixy_ai::{PiAiError, PiAiErrorCode, ToolResultContentBlock};
use serde_json::{json, Value};

use crate::memory::{MemoryFlushContext, MemoryManager};

/// Build a `memory` tool backed by a shared `MemoryManager`.
pub fn create_memory_tool(
    memory: Arc<Mutex<MemoryManager>>,
    default_max_results: usize,
    default_min_score: f32,
) -> AgentTool {
    AgentTool {
        name: "memory".to_string(),
        label: "memory".to_string(),
        description: "Record/search/session-flush persistent memory. Actions: record, search, get, flush, cleanup."
            .to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["record", "search", "get", "flush", "cleanup"],
                    "description": "Memory action to execute."
                },
                "content": { "type": "string", "description": "Content for record action." },
                "query": { "type": "string", "description": "Query for search action." },
                "date": { "type": "string", "description": "Date for get action, format YYYY-MM-DD. Empty means today." },
                "max_results": { "type": "integer", "minimum": 1, "description": "Optional search result cap." },
                "min_score": { "type": "number", "minimum": 0.0, "maximum": 1.0, "description": "Optional search min score threshold." },
                "session_id": { "type": "string" },
                "agent_id": { "type": "string" },
                "token_count": { "type": "integer", "minimum": 0 },
                "compaction_count": { "type": "integer", "minimum": 0 },
                "summary": { "type": "string" },
                "notes": { "type": "array", "items": { "type": "string" } },
                "decisions": { "type": "array", "items": { "type": "string" } },
                "todos": { "type": "array", "items": { "type": "string" } },
                "metadata": { "type": "object" }
            },
            "required": ["action"],
            "additionalProperties": false
        }),
        execute: Arc::new(MemoryToolExecutor {
            memory,
            default_max_results: default_max_results.max(1),
            default_min_score: default_min_score.clamp(0.0, 1.0),
        }),
    }
}

struct MemoryToolExecutor {
    memory: Arc<Mutex<MemoryManager>>,
    default_max_results: usize,
    default_min_score: f32,
}

#[async_trait]
impl AgentToolExecutor for MemoryToolExecutor {
    async fn execute(
        &self,
        _tool_call_id: String,
        args: Value,
    ) -> Result<AgentToolResult, PiAiError> {
        let action = required_string(&args, "action")?;
        match action.as_str() {
            "record" => self.execute_record(&args),
            "search" => self.execute_search(&args),
            "get" => self.execute_get(&args),
            "flush" => self.execute_flush(&args),
            "cleanup" => self.execute_cleanup(),
            _ => Err(invalid_tool_args(format!(
                "unsupported memory action '{action}'"
            ))),
        }
    }
}

impl MemoryToolExecutor {
    fn execute_record(&self, args: &Value) -> Result<AgentToolResult, PiAiError> {
        let content = required_string(args, "content")?;
        let manager = lock_memory(&self.memory)?;
        manager
            .record(&content)
            .map_err(|error| tool_execution_failed(error.to_string()))?;
        Ok(text_result(
            "Memory recorded.".to_string(),
            json!({
                "action": "record",
                "bytes": content.len(),
            }),
        ))
    }

    fn execute_search(&self, args: &Value) -> Result<AgentToolResult, PiAiError> {
        let query = required_string(args, "query")?;
        let max_results = optional_usize(args, "max_results")?.unwrap_or(self.default_max_results);
        let min_score = optional_f32(args, "min_score")?.unwrap_or(self.default_min_score);

        let manager = lock_memory(&self.memory)?;
        let results = manager
            .search_scored(&query, max_results, min_score)
            .map_err(|error| tool_execution_failed(error.to_string()))?;

        if results.is_empty() {
            return Ok(text_result(
                "No memory matched query.".to_string(),
                json!({
                    "action": "search",
                    "query": query,
                    "count": 0,
                }),
            ));
        }

        let text = results
            .iter()
            .enumerate()
            .map(|(index, result)| {
                format!(
                    "{}. [{}] score={:.3} {} :: {}",
                    index + 1,
                    result.date,
                    result.score,
                    result.path.display(),
                    result.snippet.replace('\n', " ")
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        let details = results
            .iter()
            .map(|result| {
                json!({
                    "path": result.path.display().to_string(),
                    "date": result.date.to_string(),
                    "score": result.score,
                    "snippet": result.snippet,
                    "lineNumbers": result.line_numbers,
                })
            })
            .collect::<Vec<_>>();

        Ok(text_result(
            text,
            json!({
                "action": "search",
                "query": query,
                "count": details.len(),
                "results": details,
            }),
        ))
    }

    fn execute_get(&self, args: &Value) -> Result<AgentToolResult, PiAiError> {
        let date = optional_string(args, "date")?;
        let manager = lock_memory(&self.memory)?;
        let content = if let Some(date_value) = date.as_deref() {
            manager
                .read_date_string(date_value)
                .map_err(|error| tool_execution_failed(error.to_string()))?
        } else {
            manager
                .read_today()
                .map_err(|error| tool_execution_failed(error.to_string()))?
        };
        let text = if content.trim().is_empty() {
            "(empty memory)".to_string()
        } else {
            content
        };
        Ok(text_result(
            text,
            json!({
                "action": "get",
                "date": date,
            }),
        ))
    }

    fn execute_flush(&self, args: &Value) -> Result<AgentToolResult, PiAiError> {
        let token_count = optional_usize(args, "token_count")?.unwrap_or(0);
        let compaction_count = optional_usize(args, "compaction_count")?.unwrap_or(0);
        let context = MemoryFlushContext {
            session_id: optional_string(args, "session_id")?,
            agent_id: optional_string(args, "agent_id")?,
            token_count,
            compaction_count,
            summary: optional_string(args, "summary")?,
            notes: optional_string_array(args, "notes")?,
            decisions: optional_string_array(args, "decisions")?,
            todos: optional_string_array(args, "todos")?,
            metadata: optional_json_object(args, "metadata")?,
        };

        let manager = lock_memory(&self.memory)?;
        manager
            .flush(&context)
            .map_err(|error| tool_execution_failed(error.to_string()))?;
        Ok(text_result(
            "Memory flush recorded.".to_string(),
            json!({
                "action": "flush",
                "session_id": context.session_id,
                "token_count": token_count,
                "compaction_count": compaction_count,
            }),
        ))
    }

    fn execute_cleanup(&self) -> Result<AgentToolResult, PiAiError> {
        let manager = lock_memory(&self.memory)?;
        let deleted = manager
            .cleanup()
            .map_err(|error| tool_execution_failed(error.to_string()))?;
        Ok(text_result(
            format!("Cleanup finished, removed {deleted} old memory files."),
            json!({
                "action": "cleanup",
                "deleted": deleted,
            }),
        ))
    }
}

fn lock_memory(
    memory: &Arc<Mutex<MemoryManager>>,
) -> Result<std::sync::MutexGuard<'_, MemoryManager>, PiAiError> {
    memory.lock().map_err(|_| {
        tool_execution_failed("memory manager lock poisoned; try again in a new session")
    })
}

fn required_string(args: &Value, key: &str) -> Result<String, PiAiError> {
    args.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .ok_or_else(|| invalid_tool_args(format!("missing or invalid `{key}`")))
}

fn optional_string(args: &Value, key: &str) -> Result<Option<String>, PiAiError> {
    match args.get(key) {
        None => Ok(None),
        Some(value) if value.is_null() => Ok(None),
        Some(value) => value
            .as_str()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(|value| Some(value.to_string()))
            .ok_or_else(|| invalid_tool_args(format!("missing or invalid `{key}`"))),
    }
}

fn optional_usize(args: &Value, key: &str) -> Result<Option<usize>, PiAiError> {
    match args.get(key) {
        None => Ok(None),
        Some(value) if value.is_null() => Ok(None),
        Some(value) => {
            let raw = value
                .as_u64()
                .ok_or_else(|| invalid_tool_args(format!("missing or invalid `{key}`")))?;
            usize::try_from(raw)
                .map(Some)
                .map_err(|_| invalid_tool_args(format!("`{key}` is too large")))
        }
    }
}

fn optional_f32(args: &Value, key: &str) -> Result<Option<f32>, PiAiError> {
    match args.get(key) {
        None => Ok(None),
        Some(value) if value.is_null() => Ok(None),
        Some(value) => value
            .as_f64()
            .map(|value| value as f32)
            .map(Some)
            .ok_or_else(|| invalid_tool_args(format!("missing or invalid `{key}`"))),
    }
}

fn optional_string_array(args: &Value, key: &str) -> Result<Vec<String>, PiAiError> {
    let Some(value) = args.get(key) else {
        return Ok(Vec::new());
    };
    if value.is_null() {
        return Ok(Vec::new());
    }
    let list = value
        .as_array()
        .ok_or_else(|| invalid_tool_args(format!("missing or invalid `{key}`")))?;
    let mut result = Vec::with_capacity(list.len());
    for item in list {
        let text = item
            .as_str()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| invalid_tool_args(format!("`{key}` must contain non-empty strings")))?;
        result.push(text.to_string());
    }
    Ok(result)
}

fn optional_json_object(args: &Value, key: &str) -> Result<Option<Value>, PiAiError> {
    match args.get(key) {
        None => Ok(None),
        Some(value) if value.is_null() => Ok(None),
        Some(value @ Value::Object(_)) => Ok(Some(value.clone())),
        Some(_) => Err(invalid_tool_args(format!("`{key}` must be a JSON object"))),
    }
}

fn invalid_tool_args(message: impl Into<String>) -> PiAiError {
    PiAiError::new(PiAiErrorCode::ToolArgumentsInvalid, message.into())
}

fn tool_execution_failed(message: impl Into<String>) -> PiAiError {
    PiAiError::new(PiAiErrorCode::ToolExecutionFailed, message.into())
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
