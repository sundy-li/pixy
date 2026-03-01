use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubAgentMode {
    #[serde(rename = "primary")]
    Primary,
    #[serde(rename = "subagent", alias = "sub_agent")]
    SubAgent,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubAgentSpec {
    pub name: String,
    pub description: String,
    pub mode: SubAgentMode,
}

impl SubAgentSpec {
    pub fn validate(&self) -> Result<(), String> {
        if self.name.trim().is_empty() {
            return Err("subagent name cannot be empty".to_string());
        }
        if self.description.trim().is_empty() {
            return Err("subagent description cannot be empty".to_string());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskToolInput {
    pub subagent_type: String,
    pub prompt: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
}

impl TaskToolInput {
    pub fn validate(&self) -> Result<(), String> {
        if self.subagent_type.trim().is_empty() {
            return Err("task subagent_type cannot be empty".to_string());
        }
        if self.prompt.trim().is_empty() {
            return Err("task prompt cannot be empty".to_string());
        }
        if let Some(task_id) = &self.task_id {
            if task_id.trim().is_empty() {
                return Err("task task_id cannot be empty when provided".to_string());
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskToolOutput {
    pub task_id: String,
    pub summary: String,
    pub child_session_file: String,
}

impl TaskToolOutput {
    pub fn validate(&self) -> Result<(), String> {
        if self.task_id.trim().is_empty() {
            return Err("task output task_id cannot be empty".to_string());
        }
        if self.summary.trim().is_empty() {
            return Err("task output summary cannot be empty".to_string());
        }
        if self.child_session_file.trim().is_empty() {
            return Err("task output child_session_file cannot be empty".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_tool_input_rejects_empty_subagent_type() {
        let input = TaskToolInput {
            subagent_type: "".to_string(),
            prompt: "scan project".to_string(),
            task_id: None,
        };

        let error = input
            .validate()
            .expect_err("empty subagent type should be rejected");
        assert!(error.contains("subagent_type"));
    }

    #[test]
    fn task_tool_input_rejects_empty_prompt() {
        let input = TaskToolInput {
            subagent_type: "general".to_string(),
            prompt: "".to_string(),
            task_id: None,
        };

        let error = input
            .validate()
            .expect_err("empty prompt should be rejected");
        assert!(error.contains("prompt"));
    }

    #[test]
    fn task_tool_output_rejects_empty_summary() {
        let output = TaskToolOutput {
            task_id: "task-1".to_string(),
            summary: "".to_string(),
            child_session_file: "/tmp/session.jsonl".to_string(),
        };

        let error = output
            .validate()
            .expect_err("empty summary should be rejected");
        assert!(error.contains("summary"));
    }

    #[test]
    fn subagent_spec_requires_non_empty_name() {
        let spec = SubAgentSpec {
            name: "".to_string(),
            description: "general helper".to_string(),
            mode: SubAgentMode::SubAgent,
        };

        let error = spec
            .validate()
            .expect_err("empty subagent name should be rejected");
        assert!(error.contains("name"));
    }
}
