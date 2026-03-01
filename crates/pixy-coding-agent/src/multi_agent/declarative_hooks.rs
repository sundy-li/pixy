use std::process::Command;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::{
    create_multi_agent_plugin_runtime, AfterTaskResultHookContext, BeforeTaskDispatchHookContext,
    BeforeToolDefinitionHookContext, BeforeUserMessageHookContext, MultiAgentHook,
    MultiAgentPluginRuntime,
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeclarativeHookStage {
    BeforeUserMessage,
    BeforeTaskDispatch,
    AfterTaskResult,
    BeforeToolDefinition,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DeclarativeHookAction {
    SetField {
        field: String,
        value: String,
    },
    AppendField {
        field: String,
        value: String,
    },
    RouteTo {
        subagent: String,
    },
    Bash {
        command: String,
        field: String,
        #[serde(default)]
        append: bool,
    },
}

impl DeclarativeHookAction {
    fn validate(&self, hook_name: &str) -> Result<(), String> {
        match self {
            Self::SetField { field, .. } | Self::AppendField { field, .. } => {
                if field.trim().is_empty() {
                    return Err(format!("hook '{hook_name}' action field cannot be empty"));
                }
            }
            Self::RouteTo { subagent } => {
                if subagent.trim().is_empty() {
                    return Err(format!(
                        "hook '{hook_name}' route_to subagent cannot be empty"
                    ));
                }
            }
            Self::Bash { command, field, .. } => {
                if command.trim().is_empty() {
                    return Err(format!("hook '{hook_name}' bash command cannot be empty"));
                }
                if field.trim().is_empty() {
                    return Err(format!("hook '{hook_name}' bash field cannot be empty"));
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeclarativeHookSpec {
    #[serde(default)]
    pub name: String,
    pub stage: DeclarativeHookStage,
    #[serde(default)]
    pub tool_name: Option<String>,
    #[serde(default)]
    pub subagent: Option<String>,
    #[serde(default)]
    pub prompt_contains: Option<String>,
    #[serde(default)]
    pub actions: Vec<DeclarativeHookAction>,
}

impl DeclarativeHookSpec {
    pub fn validate(&self) -> Result<(), String> {
        let hook_name = if self.name.trim().is_empty() {
            "unnamed-hook"
        } else {
            self.name.trim()
        };

        if self.actions.is_empty() {
            return Err(format!("hook '{hook_name}' has no actions"));
        }

        for action in &self.actions {
            action.validate(hook_name)?;
        }

        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
struct DeclarativeHookEngine {
    hooks: Vec<DeclarativeHookSpec>,
}

impl DeclarativeHookEngine {
    fn from_specs(hooks: Vec<DeclarativeHookSpec>) -> Result<Self, String> {
        for hook in &hooks {
            hook.validate()?;
        }
        Ok(Self { hooks })
    }

    fn matching_hooks(
        &self,
        stage: DeclarativeHookStage,
        input: HookMatchInput<'_>,
    ) -> Vec<&DeclarativeHookSpec> {
        self.hooks
            .iter()
            .filter(|hook| hook.stage == stage && hook_matches(hook, input))
            .collect::<Vec<_>>()
    }
}

impl MultiAgentHook for DeclarativeHookEngine {
    fn before_user_message(&self, ctx: &mut BeforeUserMessageHookContext) {
        let input = HookMatchInput {
            tool_name: None,
            subagent: None,
            text: Some(&ctx.message),
        };
        for hook in self.matching_hooks(DeclarativeHookStage::BeforeUserMessage, input) {
            for action in &hook.actions {
                apply_before_user_message_action(ctx, action);
            }
        }
    }

    fn before_task_dispatch(&self, ctx: &mut BeforeTaskDispatchHookContext) {
        let input = HookMatchInput {
            tool_name: Some("task"),
            subagent: Some(&ctx.input.subagent_type),
            text: Some(&ctx.input.prompt),
        };
        for hook in self.matching_hooks(DeclarativeHookStage::BeforeTaskDispatch, input) {
            for action in &hook.actions {
                apply_before_task_dispatch_action(ctx, action);
            }
        }
    }

    fn after_task_result(&self, ctx: &mut AfterTaskResultHookContext) {
        let input = HookMatchInput {
            tool_name: Some("task"),
            subagent: Some(&ctx.resolved_subagent),
            text: Some(&ctx.output.summary),
        };
        for hook in self.matching_hooks(DeclarativeHookStage::AfterTaskResult, input) {
            for action in &hook.actions {
                apply_after_task_result_action(ctx, action);
            }
        }
    }

    fn before_tool_definition(&self, ctx: &mut BeforeToolDefinitionHookContext) {
        let input = HookMatchInput {
            tool_name: Some(&ctx.tool_name),
            subagent: None,
            text: Some(&ctx.description),
        };
        for hook in self.matching_hooks(DeclarativeHookStage::BeforeToolDefinition, input) {
            for action in &hook.actions {
                apply_before_tool_definition_action(ctx, action);
            }
        }
    }
}

pub fn create_multi_agent_plugin_runtime_from_specs(
    hooks: &[DeclarativeHookSpec],
) -> Result<MultiAgentPluginRuntime, String> {
    if hooks.is_empty() {
        return Ok(MultiAgentPluginRuntime::default());
    }
    let engine = DeclarativeHookEngine::from_specs(hooks.to_vec())?;
    Ok(create_multi_agent_plugin_runtime(vec![Arc::new(engine)]))
}

#[derive(Clone, Copy)]
struct HookMatchInput<'a> {
    tool_name: Option<&'a str>,
    subagent: Option<&'a str>,
    text: Option<&'a str>,
}

fn hook_matches(hook: &DeclarativeHookSpec, input: HookMatchInput<'_>) -> bool {
    if let Some(expected_tool) = hook
        .tool_name
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        if input.tool_name.map(str::trim) != Some(expected_tool) {
            return false;
        }
    }

    if let Some(expected_subagent) = hook
        .subagent
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        if input.subagent.map(str::trim) != Some(expected_subagent) {
            return false;
        }
    }

    if let Some(contains_text) = hook
        .prompt_contains
        .as_deref()
        .filter(|value| !value.is_empty())
    {
        let Some(source_text) = input.text else {
            return false;
        };
        if !source_text.contains(contains_text) {
            return false;
        }
    }

    true
}

fn apply_before_user_message_action(
    ctx: &mut BeforeUserMessageHookContext,
    action: &DeclarativeHookAction,
) {
    match action {
        DeclarativeHookAction::SetField { field, value } if field == "message" => {
            ctx.message = value.clone();
        }
        DeclarativeHookAction::AppendField { field, value } if field == "message" => {
            ctx.message.push_str(value);
        }
        DeclarativeHookAction::Bash {
            command,
            field,
            append,
        } if field == "message" => {
            let output = run_bash(command);
            if *append {
                ctx.message.push_str(&output);
            } else {
                ctx.message = output;
            }
        }
        _ => {}
    }
}

fn apply_before_task_dispatch_action(
    ctx: &mut BeforeTaskDispatchHookContext,
    action: &DeclarativeHookAction,
) {
    match action {
        DeclarativeHookAction::RouteTo { subagent } => {
            ctx.input.subagent_type = subagent.clone();
        }
        DeclarativeHookAction::SetField { field, value } => {
            set_before_task_dispatch_field(ctx, field, value.clone(), false);
        }
        DeclarativeHookAction::AppendField { field, value } => {
            set_before_task_dispatch_field(ctx, field, value.clone(), true);
        }
        DeclarativeHookAction::Bash {
            command,
            field,
            append,
        } => {
            let output = run_bash(command);
            set_before_task_dispatch_field(ctx, field, output, *append);
        }
    }
}

fn set_before_task_dispatch_field(
    ctx: &mut BeforeTaskDispatchHookContext,
    field: &str,
    incoming: String,
    append: bool,
) {
    match field {
        "input.prompt" => {
            if append {
                ctx.input.prompt.push_str(&incoming);
            } else {
                ctx.input.prompt = incoming;
            }
        }
        "input.subagent_type" => {
            if append {
                ctx.input.subagent_type.push_str(&incoming);
            } else {
                ctx.input.subagent_type = incoming;
            }
        }
        "input.task_id" => {
            if append {
                let mut combined = ctx.input.task_id.take().unwrap_or_default();
                combined.push_str(&incoming);
                ctx.input.task_id = if combined.trim().is_empty() {
                    None
                } else {
                    Some(combined)
                };
            } else {
                ctx.input.task_id = if incoming.trim().is_empty() {
                    None
                } else {
                    Some(incoming)
                };
            }
        }
        _ => {}
    }
}

fn apply_after_task_result_action(
    ctx: &mut AfterTaskResultHookContext,
    action: &DeclarativeHookAction,
) {
    match action {
        DeclarativeHookAction::SetField { field, value } => {
            set_after_task_result_field(ctx, field, value.clone(), false);
        }
        DeclarativeHookAction::AppendField { field, value } => {
            set_after_task_result_field(ctx, field, value.clone(), true);
        }
        DeclarativeHookAction::Bash {
            command,
            field,
            append,
        } => {
            let output = run_bash(command);
            set_after_task_result_field(ctx, field, output, *append);
        }
        DeclarativeHookAction::RouteTo { .. } => {}
    }
}

fn set_after_task_result_field(
    ctx: &mut AfterTaskResultHookContext,
    field: &str,
    incoming: String,
    append: bool,
) {
    match field {
        "output.summary" => {
            if append {
                ctx.output.summary.push_str(&incoming);
            } else {
                ctx.output.summary = incoming;
            }
        }
        "output.task_id" => {
            if append {
                ctx.output.task_id.push_str(&incoming);
            } else {
                ctx.output.task_id = incoming;
            }
        }
        "output.child_session_file" => {
            if append {
                ctx.output.child_session_file.push_str(&incoming);
            } else {
                ctx.output.child_session_file = incoming;
            }
        }
        "resolved_subagent" => {
            if append {
                ctx.resolved_subagent.push_str(&incoming);
            } else {
                ctx.resolved_subagent = incoming;
            }
        }
        _ => {}
    }
}

fn apply_before_tool_definition_action(
    ctx: &mut BeforeToolDefinitionHookContext,
    action: &DeclarativeHookAction,
) {
    match action {
        DeclarativeHookAction::SetField { field, value } if field == "description" => {
            ctx.description = value.clone();
        }
        DeclarativeHookAction::AppendField { field, value } if field == "description" => {
            ctx.description.push_str(value);
        }
        DeclarativeHookAction::Bash {
            command,
            field,
            append,
        } if field == "description" => {
            let output = run_bash(command);
            if *append {
                ctx.description.push_str(&output);
            } else {
                ctx.description = output;
            }
        }
        _ => {}
    }
}

fn run_bash(command: &str) -> String {
    match Command::new("bash").arg("-lc").arg(command).output() {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout)
                .trim_end()
                .to_string();
            let stderr = String::from_utf8_lossy(&output.stderr)
                .trim_end()
                .to_string();
            if output.status.success() {
                stdout
            } else if !stderr.is_empty() {
                format!(
                    "[hook bash exit {}] {}",
                    output.status.code().unwrap_or(-1),
                    stderr
                )
            } else {
                format!(
                    "[hook bash exit {}] {}",
                    output.status.code().unwrap_or(-1),
                    stdout
                )
            }
        }
        Err(error) => format!("[hook bash failed] {error}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TaskToolInput, TaskToolOutput};

    #[test]
    fn declarative_hook_route_to_updates_subagent() {
        let runtime = create_multi_agent_plugin_runtime_from_specs(&[DeclarativeHookSpec {
            name: "route-review".to_string(),
            stage: DeclarativeHookStage::BeforeTaskDispatch,
            tool_name: None,
            subagent: None,
            prompt_contains: Some("review".to_string()),
            actions: vec![DeclarativeHookAction::RouteTo {
                subagent: "review".to_string(),
            }],
        }])
        .expect("runtime should build");

        let mut ctx = BeforeTaskDispatchHookContext {
            input: TaskToolInput {
                subagent_type: "general".to_string(),
                prompt: "please review this patch".to_string(),
                task_id: None,
            },
        };
        runtime.before_task_dispatch(&mut ctx);

        assert_eq!(ctx.input.subagent_type, "review");
    }

    #[test]
    fn declarative_hook_bash_can_append_summary_in_after_task_result() {
        let runtime = create_multi_agent_plugin_runtime_from_specs(&[DeclarativeHookSpec {
            name: "append-git-status".to_string(),
            stage: DeclarativeHookStage::AfterTaskResult,
            tool_name: None,
            subagent: Some("review".to_string()),
            prompt_contains: None,
            actions: vec![DeclarativeHookAction::Bash {
                command: "printf HOOK".to_string(),
                field: "output.summary".to_string(),
                append: true,
            }],
        }])
        .expect("runtime should build");

        let mut ctx = AfterTaskResultHookContext {
            output: TaskToolOutput {
                task_id: "task-1".to_string(),
                summary: "done".to_string(),
                child_session_file: "/tmp/child.jsonl".to_string(),
            },
            resolved_subagent: "review".to_string(),
            routing_hint_applied: false,
        };
        runtime.after_task_result(&mut ctx);

        assert_eq!(ctx.output.summary, "doneHOOK");
    }

    #[test]
    fn declarative_hook_rejects_empty_actions() {
        let result = create_multi_agent_plugin_runtime_from_specs(&[DeclarativeHookSpec {
            name: "invalid".to_string(),
            stage: DeclarativeHookStage::BeforeUserMessage,
            tool_name: None,
            subagent: None,
            prompt_contains: None,
            actions: vec![],
        }]);
        let error = match result {
            Ok(_) => panic!("invalid hook should be rejected"),
            Err(error) => error,
        };

        assert!(error.contains("no actions"));
    }
}
