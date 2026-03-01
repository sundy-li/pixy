use pixy_ai::{validate_tool_call, PiAiErrorCode, Tool, ToolCall};
use serde_json::json;

fn sample_tool() -> Tool {
    Tool {
        name: "read".to_string(),
        description: "Read a file".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "offset": { "type": "integer", "minimum": 0 }
            },
            "required": ["path"],
            "additionalProperties": false
        }),
    }
}

#[test]
fn validate_tool_call_accepts_valid_arguments() {
    let tools = vec![sample_tool()];
    let call = ToolCall {
        id: "tool-1".to_string(),
        name: "read".to_string(),
        arguments: json!({
            "path": "README.md",
            "offset": 10
        }),
    };

    let validated = validate_tool_call(&tools, &call).expect("validation should pass");
    assert_eq!(validated["path"], json!("README.md"));
    assert_eq!(validated["offset"], json!(10));
}

#[test]
fn validate_tool_call_rejects_missing_tool() {
    let tools = vec![sample_tool()];
    let call = ToolCall {
        id: "tool-2".to_string(),
        name: "write".to_string(),
        arguments: json!({ "path": "x" }),
    };

    let error = validate_tool_call(&tools, &call).expect_err("missing tool should fail");
    assert_eq!(error.code, PiAiErrorCode::ToolNotFound);
    assert!(error.message.contains("Tool 'write' not found"));
}

#[test]
fn validate_tool_call_rejects_invalid_arguments_with_structured_details() {
    let tools = vec![sample_tool()];
    let call = ToolCall {
        id: "tool-3".to_string(),
        name: "read".to_string(),
        arguments: json!({
            "path": 10
        }),
    };

    let error = validate_tool_call(&tools, &call).expect_err("invalid arguments should fail");
    assert_eq!(error.code, PiAiErrorCode::ToolArgumentsInvalid);

    let details = error
        .details
        .expect("validation error should contain details");
    let validation_errors = details["validationErrors"]
        .as_array()
        .expect("validationErrors should be array");
    assert!(!validation_errors.is_empty());
}
