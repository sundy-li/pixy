use pixy_ai::{Message, UserContent, UserContentBlock};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const COMPACTION_SUMMARY_PREFIX: &str = "The conversation history before this point was compacted into the following summary:\n\n<summary>\n";
pub const COMPACTION_SUMMARY_SUFFIX: &str = "\n</summary>";

pub const BRANCH_SUMMARY_PREFIX: &str =
    "The following is a summary of a branch that this conversation came back from:\n\n<summary>\n";
pub const BRANCH_SUMMARY_SUFFIX: &str = "</summary>";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BashExecutionMessage {
    pub role: String,
    pub command: String,
    pub output: String,
    #[serde(rename = "exitCode")]
    pub exit_code: Option<i32>,
    pub cancelled: bool,
    pub truncated: bool,
    #[serde(rename = "fullOutputPath", skip_serializing_if = "Option::is_none")]
    pub full_output_path: Option<String>,
    pub timestamp: i64,
    #[serde(rename = "excludeFromContext", skip_serializing_if = "Option::is_none")]
    pub exclude_from_context: Option<bool>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CustomMessage {
    pub role: String,
    #[serde(rename = "customType")]
    pub custom_type: String,
    pub content: UserContent,
    pub display: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
    pub timestamp: i64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BranchSummaryMessage {
    pub role: String,
    pub summary: String,
    #[serde(rename = "fromId")]
    pub from_id: String,
    pub timestamp: i64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompactionSummaryMessage {
    pub role: String,
    pub summary: String,
    #[serde(rename = "tokensBefore")]
    pub tokens_before: u64,
    pub timestamp: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum CodingMessage {
    Agent(Message),
    BashExecution(BashExecutionMessage),
    Custom(CustomMessage),
    BranchSummary(BranchSummaryMessage),
    CompactionSummary(CompactionSummaryMessage),
}

pub fn bash_execution_to_text(msg: &BashExecutionMessage) -> String {
    let mut text = format!("Ran `{}`\n", msg.command);
    if msg.output.is_empty() {
        text.push_str("(no output)");
    } else {
        text.push_str("```\n");
        text.push_str(&msg.output);
        text.push_str("\n```");
    }

    if msg.cancelled {
        text.push_str("\n\n(command cancelled)");
    } else if let Some(code) = msg.exit_code {
        if code != 0 {
            text.push_str(&format!("\n\nCommand exited with code {code}"));
        }
    }

    if msg.truncated {
        if let Some(path) = &msg.full_output_path {
            text.push_str(&format!("\n\n[Output truncated. Full output: {path}]"));
        }
    }

    text
}

pub fn convert_to_llm(messages: &[CodingMessage]) -> Vec<Message> {
    messages
        .iter()
        .filter_map(|message| match message {
            CodingMessage::Agent(message) => Some(message.clone()),
            CodingMessage::BashExecution(message) => {
                if message.exclude_from_context.unwrap_or(false) {
                    return None;
                }
                Some(Message::User {
                    content: UserContent::Blocks(vec![UserContentBlock::Text {
                        text: bash_execution_to_text(message),
                        text_signature: None,
                    }]),
                    timestamp: message.timestamp,
                })
            }
            CodingMessage::Custom(message) => Some(Message::User {
                content: message.content.clone(),
                timestamp: message.timestamp,
            }),
            CodingMessage::BranchSummary(message) => Some(Message::User {
                content: UserContent::Blocks(vec![UserContentBlock::Text {
                    text: format!(
                        "{BRANCH_SUMMARY_PREFIX}{}{BRANCH_SUMMARY_SUFFIX}",
                        message.summary
                    ),
                    text_signature: None,
                }]),
                timestamp: message.timestamp,
            }),
            CodingMessage::CompactionSummary(message) => Some(Message::User {
                content: UserContent::Blocks(vec![UserContentBlock::Text {
                    text: format!(
                        "{COMPACTION_SUMMARY_PREFIX}{}{COMPACTION_SUMMARY_SUFFIX}",
                        message.summary
                    ),
                    text_signature: None,
                }]),
                timestamp: message.timestamp,
            }),
        })
        .collect()
}
