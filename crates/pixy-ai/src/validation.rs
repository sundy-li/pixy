use jsonschema::JSONSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::error::{PiAiError, PiAiErrorCode};
use crate::types::Tool;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

pub fn validate_tool_call(tools: &[Tool], tool_call: &ToolCall) -> Result<Value, PiAiError> {
    let Some(tool) = tools.iter().find(|tool| tool.name == tool_call.name) else {
        let available = tools
            .iter()
            .map(|tool| tool.name.clone())
            .collect::<Vec<_>>();
        return Err(PiAiError::new(
            PiAiErrorCode::ToolNotFound,
            format!("Tool '{}' not found", tool_call.name),
        )
        .with_details(json!({
            "toolName": tool_call.name,
            "availableTools": available,
        })));
    };

    validate_tool_arguments(tool, tool_call)
}

pub fn validate_tool_arguments(tool: &Tool, tool_call: &ToolCall) -> Result<Value, PiAiError> {
    let compiled = JSONSchema::compile(&tool.parameters).map_err(|error| {
        PiAiError::new(
            PiAiErrorCode::SchemaInvalid,
            format!("Invalid JSON schema for tool '{}': {error}", tool.name),
        )
        .with_details(json!({
            "toolName": tool.name,
        }))
    })?;

    if let Err(errors) = compiled.validate(&tool_call.arguments) {
        let validation_errors = errors
            .map(|error| {
                json!({
                    "path": error.instance_path.to_string(),
                    "message": error.to_string(),
                })
            })
            .collect::<Vec<_>>();

        return Err(PiAiError::new(
            PiAiErrorCode::ToolArgumentsInvalid,
            format!("Validation failed for tool '{}'", tool.name),
        )
        .with_details(json!({
            "toolName": tool.name,
            "toolCallId": tool_call.id,
            "arguments": tool_call.arguments,
            "validationErrors": validation_errors,
        })));
    }

    Ok(tool_call.arguments.clone())
}
