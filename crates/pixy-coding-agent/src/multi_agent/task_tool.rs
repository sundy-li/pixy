use std::sync::Arc;

use async_trait::async_trait;
use pixy_agent_core::{AgentTool, AgentToolExecutor, AgentToolResult};
use pixy_ai::{PiAiError, PiAiErrorCode, ToolResultContentBlock};
use serde_json::{Value, json};

use super::dispatcher::TaskDispatcher;
use crate::TaskToolInput;

pub fn create_task_tool(dispatcher: Arc<TaskDispatcher>) -> AgentTool {
    AgentTool {
        name: "task".to_string(),
        label: "task".to_string(),
        description:
            "Delegate work to a registered subagent, optionally reusing prior task context with task_id."
                .to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "subagent_type": { "type": "string", "description": "Registered subagent type name." },
                "prompt": { "type": "string", "description": "Task prompt passed to the subagent." },
                "task_id": { "type": "string", "description": "Optional child-session reuse identifier." }
            },
            "required": ["subagent_type", "prompt"],
            "additionalProperties": false
        }),
        execute: Arc::new(TaskToolExecutor { dispatcher }),
    }
}

struct TaskToolExecutor {
    dispatcher: Arc<TaskDispatcher>,
}

#[async_trait]
impl AgentToolExecutor for TaskToolExecutor {
    async fn execute(
        &self,
        _tool_call_id: String,
        args: Value,
    ) -> Result<AgentToolResult, PiAiError> {
        let input: TaskToolInput = serde_json::from_value(args).map_err(|error| {
            PiAiError::new(
                PiAiErrorCode::ToolArgumentsInvalid,
                format!("invalid task tool arguments: {error}"),
            )
        })?;
        input
            .validate()
            .map_err(|error| PiAiError::new(PiAiErrorCode::ToolArgumentsInvalid, error))?;

        let dispatched = self.dispatcher.dispatch(input).await?;
        let mut details = serde_json::to_value(&dispatched.output).map_err(|error| {
            PiAiError::new(
                PiAiErrorCode::ToolExecutionFailed,
                format!("serialize task tool output failed: {error}"),
            )
        })?;
        if let Some(object) = details.as_object_mut() {
            object.insert(
                "resolved_subagent".to_string(),
                json!(dispatched.resolved_subagent),
            );
            object.insert(
                "routing_hint_applied".to_string(),
                json!(dispatched.routing_hint_applied),
            );
        }

        Ok(AgentToolResult {
            content: vec![ToolResultContentBlock::Text {
                text: format!("<task_result>\n{}\n</task_result>", dispatched.summary),
                text_signature: None,
            }],
            details,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use pixy_ai::{
        AssistantContentBlock, AssistantMessage, AssistantMessageEvent,
        AssistantMessageEventStream, Cost, DoneReason, Model, StopReason, Usage,
    };
    use serde_json::json;
    use tempfile::tempdir;
    use tokio::sync::Mutex;

    use super::create_task_tool;
    use crate::multi_agent::{TaskDispatcher, TaskDispatcherConfig};
    use crate::{
        ChildSessionStore, DefaultSubAgentRegistry, DispatchPolicyConfig, MultiAgentPluginRuntime,
        SubAgentMode, SubAgentResolver, SubAgentSpec,
    };

    fn sample_model() -> Model {
        Model {
            id: "test-model".to_string(),
            name: "Test Model".to_string(),
            api: "openai-responses".to_string(),
            provider: "openai".to_string(),
            base_url: "http://localhost".to_string(),
            reasoning: false,
            reasoning_effort: None,
            input: vec!["text".to_string()],
            cost: Cost {
                input: 0.0,
                output: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
                total: 0.0,
            },
            context_window: 128_000,
            max_tokens: 8_192,
        }
    }

    fn sample_usage() -> Usage {
        Usage {
            input: 1,
            output: 1,
            cache_read: 0,
            cache_write: 0,
            total_tokens: 2,
            cost: Cost {
                input: 0.0,
                output: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
                total: 0.0,
            },
        }
    }

    fn done_stream(text: String) -> AssistantMessageEventStream {
        let message = AssistantMessage {
            role: "assistant".to_string(),
            content: vec![AssistantContentBlock::Text {
                text,
                text_signature: None,
            }],
            api: "openai-responses".to_string(),
            provider: "openai".to_string(),
            model: "test-model".to_string(),
            usage: sample_usage(),
            stop_reason: StopReason::Stop,
            error_message: None,
            timestamp: 1,
        };
        let stream = AssistantMessageEventStream::new();
        stream.push(AssistantMessageEvent::Start {
            partial: message.clone(),
        });
        stream.push(AssistantMessageEvent::Done {
            reason: DoneReason::Stop,
            message,
        });
        stream
    }

    fn registry() -> Arc<dyn SubAgentResolver> {
        let built = DefaultSubAgentRegistry::builder()
            .register_builtin(SubAgentSpec {
                name: "general".to_string(),
                description: "General helper".to_string(),
                mode: SubAgentMode::SubAgent,
            })
            .expect("register general")
            .build();
        Arc::new(built)
    }

    #[tokio::test]
    async fn task_tool_executes_dispatcher_and_returns_structured_details() {
        let dir = tempdir().expect("tempdir");

        let dispatcher = Arc::new(TaskDispatcher::new(TaskDispatcherConfig {
            cwd: dir.path().to_path_buf(),
            parent_session_id: "parent-session".to_string(),
            parent_session_dir: dir.path().to_path_buf(),
            model: sample_model(),
            system_prompt: "You are parent".to_string(),
            stream_fn: Arc::new(move |_model, _context, _options| {
                Ok(done_stream("child completed".to_string()))
            }),
            child_tools: vec![],
            subagent_registry: registry(),
            session_store: Arc::new(Mutex::new(ChildSessionStore::new("parent-session"))),
            dispatch_policy: DispatchPolicyConfig::default(),
            plugin_runtime: Arc::new(MultiAgentPluginRuntime::default()),
            lifecycle_event_sink: None,
        }));

        let tool = create_task_tool(dispatcher);
        let result = tool
            .execute
            .execute(
                "tc-1".to_string(),
                json!({
                    "subagent_type": "general",
                    "prompt": "run child"
                }),
            )
            .await
            .expect("task tool should succeed");

        let text = match &result.content[0] {
            pixy_ai::ToolResultContentBlock::Text { text, .. } => text.clone(),
            _ => panic!("expected text tool result"),
        };
        assert!(text.contains("<task_result>"));
        assert!(text.contains("child completed"));
        assert!(result.details["task_id"].as_str().is_some());
        assert_eq!(result.details["summary"], "child completed");
        assert_eq!(result.details["resolved_subagent"], "general");
        assert_eq!(result.details["routing_hint_applied"], false);
    }

    #[tokio::test]
    async fn task_tool_rejects_invalid_arguments() {
        let dir = tempdir().expect("tempdir");

        let dispatcher = Arc::new(TaskDispatcher::new(TaskDispatcherConfig {
            cwd: dir.path().to_path_buf(),
            parent_session_id: "parent-session".to_string(),
            parent_session_dir: dir.path().to_path_buf(),
            model: sample_model(),
            system_prompt: "You are parent".to_string(),
            stream_fn: Arc::new(move |_model, _context, _options| {
                Ok(done_stream("child completed".to_string()))
            }),
            child_tools: vec![],
            subagent_registry: registry(),
            session_store: Arc::new(Mutex::new(ChildSessionStore::new("parent-session"))),
            dispatch_policy: DispatchPolicyConfig::default(),
            plugin_runtime: Arc::new(MultiAgentPluginRuntime::default()),
            lifecycle_event_sink: None,
        }));

        let tool = create_task_tool(dispatcher);
        let error = tool
            .execute
            .execute("tc-1".to_string(), json!({"subagent_type": "general"}))
            .await
            .expect_err("missing prompt should fail");

        assert!(error.message.contains("prompt"));
    }
}
