use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubAgentMode {
    #[serde(rename = "primary")]
    Primary,
    #[serde(rename = "subagent", alias = "sub_agent")]
    SubAgent,
}

impl Default for SubAgentMode {
    fn default() -> Self {
        Self::SubAgent
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubAgentPromptTrigger {
    pub domain: String,
    pub trigger: String,
}

impl SubAgentPromptTrigger {
    fn validate(&self) -> Result<(), String> {
        if self.domain.trim().is_empty() {
            return Err("subagent metadata trigger domain cannot be empty".to_string());
        }
        if self.trigger.trim().is_empty() {
            return Err("subagent metadata trigger text cannot be empty".to_string());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubAgentPromptMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "promptAlias",
        alias = "prompt_alias"
    )]
    pub prompt_alias: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub use_when: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub avoid_when: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub triggers: Vec<SubAgentPromptTrigger>,
}

impl SubAgentPromptMetadata {
    fn validate(&self) -> Result<(), String> {
        if let Some(value) = self.category.as_deref() {
            if value.trim().is_empty() {
                return Err("subagent metadata category cannot be empty".to_string());
            }
        }
        if let Some(value) = self.cost.as_deref() {
            if value.trim().is_empty() {
                return Err("subagent metadata cost cannot be empty".to_string());
            }
        }
        if let Some(value) = self.prompt_alias.as_deref() {
            if value.trim().is_empty() {
                return Err("subagent metadata prompt_alias cannot be empty".to_string());
            }
        }
        for use_when in &self.use_when {
            if use_when.trim().is_empty() {
                return Err("subagent metadata use_when entries cannot be empty".to_string());
            }
        }
        for avoid_when in &self.avoid_when {
            if avoid_when.trim().is_empty() {
                return Err("subagent metadata avoid_when entries cannot be empty".to_string());
            }
        }
        for trigger in &self.triggers {
            trigger.validate()?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubAgentSpec {
    pub name: String,
    pub description: String,
    pub mode: SubAgentMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        alias = "allowed_tools",
        alias = "allow_tools"
    )]
    pub tools: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty", alias = "deny_tools")]
    pub blocked_tools: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<SubAgentPromptMetadata>,
}

impl SubAgentSpec {
    pub fn validate(&self) -> Result<(), String> {
        if self.name.trim().is_empty() {
            return Err("subagent name cannot be empty".to_string());
        }
        if self.description.trim().is_empty() {
            return Err("subagent description cannot be empty".to_string());
        }
        if let Some(prompt) = &self.prompt {
            if prompt.trim().is_empty() {
                return Err("subagent prompt cannot be empty when provided".to_string());
            }
        }
        if let Some(model) = &self.model {
            if model.trim().is_empty() {
                return Err("subagent model cannot be empty when provided".to_string());
            }
        }
        let allow = normalized_tool_names(&self.tools, "tools")?;
        let deny = normalized_tool_names(&self.blocked_tools, "blocked_tools")?;
        let deny_lookup = deny.iter().collect::<BTreeSet<_>>();
        for tool in allow {
            if deny_lookup.contains(&tool) {
                return Err(format!(
                    "subagent tool '{tool}' cannot exist in both tools and blocked_tools"
                ));
            }
        }
        if let Some(metadata) = &self.metadata {
            metadata.validate()?;
        }
        Ok(())
    }

    pub fn normalized_allowed_tools(&self) -> Vec<String> {
        normalize_tool_names_best_effort(&self.tools)
    }

    pub fn normalized_blocked_tools(&self) -> Vec<String> {
        normalize_tool_names_best_effort(&self.blocked_tools)
    }
}

pub fn resolve_subagent_model_target(
    provider: Option<&str>,
    model: Option<&str>,
) -> Result<Option<String>, String> {
    let provider = provider.map(str::trim).filter(|value| !value.is_empty());
    let model = model.map(str::trim).filter(|value| !value.is_empty());

    match (provider, model) {
        (None, None) => Ok(None),
        (Some(provider), None) => Ok(Some(provider.to_string())),
        (None, Some(model)) => Ok(Some(model.to_string())),
        (Some(provider), Some(model)) => {
            if let Some((model_provider, model_id)) = split_provider_model_target(model) {
                if model_provider != provider {
                    return Err(format!(
                        "subagent provider '{provider}' conflicts with model target '{model}'"
                    ));
                }
                return Ok(Some(format!("{model_provider}/{model_id}")));
            }

            if model.contains('/') {
                return Err(format!(
                    "subagent model target '{model}' is invalid; expected '<provider>/<model>'"
                ));
            }

            Ok(Some(format!("{provider}/{model}")))
        }
    }
}

fn split_provider_model_target(raw: &str) -> Option<(&str, &str)> {
    let (provider, model) = raw.split_once('/')?;
    let provider = provider.trim();
    let model = model.trim();
    if provider.is_empty() || model.is_empty() {
        return None;
    }
    Some((provider, model))
}

fn normalize_tool_names_best_effort(raw: &[String]) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut normalized = Vec::with_capacity(raw.len());
    for value in raw {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            continue;
        }
        let text = trimmed.to_string();
        if seen.insert(text.clone()) {
            normalized.push(text);
        }
    }
    normalized
}

fn normalized_tool_names(raw: &[String], field: &str) -> Result<Vec<String>, String> {
    let mut seen = BTreeSet::new();
    let mut normalized = Vec::with_capacity(raw.len());
    for value in raw {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return Err(format!("subagent {field} entries cannot be empty"));
        }
        let text = trimmed.to_string();
        if !seen.insert(text.clone()) {
            return Err(format!("subagent {field} contains duplicate tool '{text}'"));
        }
        normalized.push(text);
    }
    Ok(normalized)
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
            prompt: None,
            model: None,
            tools: vec![],
            blocked_tools: vec![],
            metadata: None,
        };

        let error = spec
            .validate()
            .expect_err("empty subagent name should be rejected");
        assert!(error.contains("name"));
    }

    #[test]
    fn subagent_spec_rejects_conflicting_tool_rules() {
        let spec = SubAgentSpec {
            name: "general".to_string(),
            description: "general helper".to_string(),
            mode: SubAgentMode::SubAgent,
            prompt: None,
            model: None,
            tools: vec!["bash".to_string()],
            blocked_tools: vec!["bash".to_string()],
            metadata: None,
        };

        let error = spec
            .validate()
            .expect_err("conflicting tool rules should be rejected");
        assert!(error.contains("both tools and blocked_tools"));
    }

    #[test]
    fn subagent_spec_accepts_oh_my_opencode_style_metadata() {
        let spec = SubAgentSpec {
            name: "frontend-ui-ux-engineer".to_string(),
            description: "visual specialist".to_string(),
            mode: SubAgentMode::SubAgent,
            prompt: Some("Make UI stunning.".to_string()),
            model: Some("google/gemini-3-pro-preview".to_string()),
            tools: vec!["read".to_string(), "bash".to_string()],
            blocked_tools: vec!["write".to_string()],
            metadata: Some(SubAgentPromptMetadata {
                category: Some("specialist".to_string()),
                cost: Some("CHEAP".to_string()),
                prompt_alias: Some("Frontend UI/UX Engineer".to_string()),
                use_when: vec!["Visual changes".to_string()],
                avoid_when: vec!["Pure logic".to_string()],
                triggers: vec![SubAgentPromptTrigger {
                    domain: "Frontend UI/UX".to_string(),
                    trigger: "Visual changes only".to_string(),
                }],
            }),
        };

        spec.validate().expect("spec should validate");
    }

    #[test]
    fn resolve_subagent_model_target_combines_provider_and_model() {
        let resolved = resolve_subagent_model_target(Some("openai"), Some("gpt-5.3-codex"))
            .expect("provider + model should resolve");
        assert_eq!(resolved.as_deref(), Some("openai/gpt-5.3-codex"));
    }

    #[test]
    fn resolve_subagent_model_target_rejects_provider_and_prefixed_model_mismatch() {
        let error = resolve_subagent_model_target(
            Some("openai"),
            Some("anthropic/claude-4-6-sonnet-latest"),
        )
        .expect_err("mismatched provider/model should fail");
        assert!(error.contains("conflicts"));
    }
}
