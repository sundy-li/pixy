use std::fmt::{Display, Formatter};

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PiAiErrorCode {
    ToolNotFound,
    ToolArgumentsInvalid,
    ToolExecutionFailed,
    SchemaInvalid,
    ProviderAuthMissing,
    ProviderHttp,
    ProviderTransport,
    ProviderProtocol,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PiAiError {
    pub code: PiAiErrorCode,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
}

impl PiAiError {
    pub fn new(code: PiAiErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            details: None,
        }
    }

    pub fn with_details(mut self, details: Value) -> Self {
        self.details = Some(details);
        self
    }

    pub fn as_compact_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| {
            format!(
                "{{\"code\":\"provider_protocol\",\"message\":\"{}\"}}",
                self.message.replace('\"', "\\\"")
            )
        })
    }
}

impl Display for PiAiError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.code, self.message)
    }
}

impl std::error::Error for PiAiError {}
