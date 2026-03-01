use serde::{Deserialize, Serialize};

use super::SubAgentResolver;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PolicyRuleEffect {
    Allow,
    Deny,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DispatchPolicyRule {
    pub subagent: String,
    #[serde(default = "default_tool_name")]
    pub tool: String,
    pub effect: PolicyRuleEffect,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DispatchPolicyConfig {
    #[serde(default)]
    pub fallback_subagent: Option<String>,
    #[serde(default)]
    pub rules: Vec<DispatchPolicyRule>,
}

impl DispatchPolicyConfig {
    pub fn validate(&self) -> Result<(), String> {
        if let Some(fallback_subagent) = &self.fallback_subagent {
            if fallback_subagent.trim().is_empty() {
                return Err("policy fallback_subagent cannot be empty when provided".to_string());
            }
        }

        for rule in &self.rules {
            if rule.subagent.trim().is_empty() {
                return Err("policy rule subagent cannot be empty".to_string());
            }
            if rule.tool.trim().is_empty() {
                return Err("policy rule tool cannot be empty".to_string());
            }
        }

        Ok(())
    }

    pub fn merge_from(&mut self, other: &DispatchPolicyConfig) {
        if let Some(fallback_subagent) = other
            .fallback_subagent
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            self.fallback_subagent = Some(fallback_subagent.to_string());
        }
        self.rules.extend(other.rules.clone());
    }

    /// Evaluate policy for a dispatch target.
    ///
    /// Rules are matched in declaration order. The first matching rule decides
    /// allow/deny behavior ("first-match-wins").
    pub fn evaluate(
        &self,
        tool_name: &str,
        requested_subagent: &str,
        resolver: &dyn SubAgentResolver,
    ) -> DispatchPolicyDecision {
        let requested_subagent = requested_subagent.trim().to_string();
        let mut resolved_subagent = requested_subagent.clone();
        let mut routing_hint_applied = false;

        if resolver.resolve(&requested_subagent).is_none() {
            if let Some(fallback_subagent) = self
                .fallback_subagent
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
            {
                if resolver.resolve(fallback_subagent).is_some() {
                    resolved_subagent = fallback_subagent.to_string();
                    routing_hint_applied = true;
                }
            }
        }

        let mut blocked = false;
        let mut reason = None;
        for rule in &self.rules {
            if !policy_tool_matches(rule, tool_name) {
                continue;
            }
            if !policy_subagent_matches(rule, &resolved_subagent) {
                continue;
            }
            if matches!(rule.effect, PolicyRuleEffect::Deny) {
                blocked = true;
                reason = Some(format!(
                    "task dispatch denied by policy rule (tool='{}', subagent='{}')",
                    rule.tool, rule.subagent
                ));
            }
            break;
        }

        DispatchPolicyDecision {
            requested_subagent,
            resolved_subagent,
            routing_hint_applied,
            blocked,
            reason,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DispatchPolicyDecision {
    pub requested_subagent: String,
    pub resolved_subagent: String,
    pub routing_hint_applied: bool,
    pub blocked: bool,
    pub reason: Option<String>,
}

fn default_tool_name() -> String {
    "task".to_string()
}

fn policy_tool_matches(rule: &DispatchPolicyRule, tool_name: &str) -> bool {
    let tool = rule.tool.trim();
    let requested = tool_name.trim();
    tool == "*" || tool == requested
}

fn policy_subagent_matches(rule: &DispatchPolicyRule, resolved_subagent: &str) -> bool {
    let subagent = rule.subagent.trim();
    subagent == "*" || subagent == resolved_subagent
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DefaultSubAgentRegistry, SubAgentMode, SubAgentSpec};

    fn registry() -> DefaultSubAgentRegistry {
        DefaultSubAgentRegistry::builder()
            .register_builtin(SubAgentSpec {
                name: "general".to_string(),
                description: "General helper".to_string(),
                mode: SubAgentMode::SubAgent,
            })
            .expect("register general")
            .register_builtin(SubAgentSpec {
                name: "explore".to_string(),
                description: "Explore helper".to_string(),
                mode: SubAgentMode::SubAgent,
            })
            .expect("register explore")
            .build()
    }

    #[test]
    fn policy_denies_subagent_by_tool_scope() {
        let policy = DispatchPolicyConfig {
            fallback_subagent: None,
            rules: vec![
                DispatchPolicyRule {
                    subagent: "explore".to_string(),
                    tool: "task".to_string(),
                    effect: PolicyRuleEffect::Deny,
                },
                DispatchPolicyRule {
                    subagent: "*".to_string(),
                    tool: "task".to_string(),
                    effect: PolicyRuleEffect::Allow,
                },
            ],
        };

        let decision = policy.evaluate("task", "explore", &registry());
        assert!(decision.blocked);
        assert_eq!(decision.resolved_subagent, "explore");
    }

    #[test]
    fn policy_applies_fallback_subagent_when_requested_one_missing() {
        let policy = DispatchPolicyConfig {
            fallback_subagent: Some("general".to_string()),
            rules: vec![],
        };

        let decision = policy.evaluate("task", "missing", &registry());
        assert!(!decision.blocked);
        assert_eq!(decision.resolved_subagent, "general");
        assert!(decision.routing_hint_applied);
    }
}
