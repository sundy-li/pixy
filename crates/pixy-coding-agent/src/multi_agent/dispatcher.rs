use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use pixy_agent_core::{AgentTool, ParentChildRunEvent, ParentChildRunEventSink, StreamFn};
use pixy_ai::{AssistantContentBlock, Message, Model, PiAiError, PiAiErrorCode, StopReason};
use serde_json::json;
use tokio::sync::Mutex;

use crate::{
    AfterTaskResultHookContext, AgentSession, AgentSessionConfig, BeforeTaskDispatchHookContext,
    ChildSessionStore, DispatchPolicyConfig, MultiAgentPluginRuntime, SessionManager,
    SubAgentResolver, TaskToolInput, TaskToolOutput,
};

#[derive(Clone)]
pub struct TaskDispatcherConfig {
    pub cwd: PathBuf,
    pub parent_session_id: String,
    pub parent_session_dir: PathBuf,
    pub model: Model,
    pub model_catalog: Vec<Model>,
    pub system_prompt: String,
    pub stream_fn: StreamFn,
    /// Tool set exposed to child sessions.
    /// In V1 this intentionally excludes `task` to avoid recursive fan-out.
    pub child_tools: Vec<AgentTool>,
    pub subagent_registry: Arc<dyn SubAgentResolver>,
    pub session_store: Arc<Mutex<ChildSessionStore>>,
    pub dispatch_policy: DispatchPolicyConfig,
    pub plugin_runtime: Arc<MultiAgentPluginRuntime>,
    pub lifecycle_event_sink: Option<ParentChildRunEventSink>,
}

const UNRESOLVED_CHILD_SESSION_FILE: &str = "<child-session-unresolved>";
static TASK_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TaskDispatchResult {
    pub output: TaskToolOutput,
    pub summary: String,
    pub resolved_subagent: String,
    pub routing_hint_applied: bool,
    pub duration_ms: u64,
    pub trace_lines: Vec<String>,
}

#[derive(Clone)]
pub struct TaskDispatcher {
    config: TaskDispatcherConfig,
}

impl TaskDispatcher {
    pub fn new(config: TaskDispatcherConfig) -> Self {
        Self { config }
    }

    pub async fn dispatch(&self, input: TaskToolInput) -> Result<TaskDispatchResult, PiAiError> {
        let mut dispatch_ctx = BeforeTaskDispatchHookContext { input };
        self.config
            .plugin_runtime
            .before_task_dispatch(&mut dispatch_ctx);
        let input = dispatch_ctx.input;
        input
            .validate()
            .map_err(|error| PiAiError::new(PiAiErrorCode::ToolArgumentsInvalid, error))?;

        let policy_decision = self.config.dispatch_policy.evaluate(
            "task",
            &input.subagent_type,
            self.config.subagent_registry.as_ref(),
        );
        if policy_decision.blocked {
            return Err(PiAiError::new(
                PiAiErrorCode::ToolExecutionFailed,
                policy_decision
                    .reason
                    .as_deref()
                    .unwrap_or("task dispatch blocked"),
            )
            .with_details(json!({
                "kind": "task_dispatch_blocked",
                "tool": "task",
                "requested_subagent": policy_decision.requested_subagent,
                "resolved_subagent": policy_decision.resolved_subagent,
                "routing_hint_applied": policy_decision.routing_hint_applied,
            })));
        }

        let subagent = self
            .config
            .subagent_registry
            .resolve(&policy_decision.resolved_subagent)
            .ok_or_else(|| {
                PiAiError::new(
                    PiAiErrorCode::ToolArgumentsInvalid,
                    format!(
                        "unknown subagent_type '{}' (resolved '{}')",
                        input.subagent_type, policy_decision.resolved_subagent
                    ),
                )
            })?;
        let subagent_name = subagent.name.clone();
        let parent_session_id = self.config.parent_session_id.clone();
        let child_model =
            resolve_child_model(&self.config.model, &self.config.model_catalog, &subagent)
                .map_err(|error| {
                    PiAiError::new(
                        PiAiErrorCode::ToolArgumentsInvalid,
                        format!(
                            "subagent '{}' model configuration is invalid: {error}",
                            subagent_name
                        ),
                    )
                })?;
        let child_tools = resolve_child_tools(&self.config.child_tools, &subagent);

        let task_id = input
            .task_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .unwrap_or_else(generate_task_id);

        let child_session_file = match self.resolve_or_create_child_session_file(&task_id).await {
            Ok(path) => path,
            Err(error) => {
                self.emit_lifecycle_event(ParentChildRunEvent::ChildRunError {
                    parent_session_id: parent_session_id.clone(),
                    child_session_file: UNRESOLVED_CHILD_SESSION_FILE.to_string(),
                    task_id: task_id.clone(),
                    subagent: subagent_name.clone(),
                    error: error.message.clone(),
                });
                return Err(error);
            }
        };
        let child_session_file_text = child_session_file.to_string_lossy().to_string();

        self.emit_lifecycle_event(ParentChildRunEvent::ChildRunStart {
            parent_session_id: parent_session_id.clone(),
            child_session_file: child_session_file_text.clone(),
            task_id: task_id.clone(),
            subagent: subagent_name.clone(),
        });
        let run_started_at = Instant::now();

        let child_manager = SessionManager::load(&child_session_file).map_err(|error| {
            let error_message = format!(
                "failed to load child session {}: {error}",
                child_session_file.display()
            );
            self.emit_lifecycle_event(ParentChildRunEvent::ChildRunError {
                parent_session_id: parent_session_id.clone(),
                child_session_file: child_session_file_text.clone(),
                task_id: task_id.clone(),
                subagent: subagent_name.clone(),
                error: error_message.clone(),
            });
            PiAiError::new(PiAiErrorCode::ToolExecutionFailed, error_message)
        })?;

        let mut child_session = AgentSession::new(
            child_manager,
            AgentSessionConfig {
                model: child_model,
                system_prompt: build_child_system_prompt(&self.config.system_prompt, &subagent),
                stream_fn: self.config.stream_fn.clone(),
                // Child sessions in V1 intentionally do not get task tool to avoid recursive fan-out.
                tools: child_tools,
            },
        );
        child_session.set_multi_agent_plugin_runtime(self.config.plugin_runtime.clone());

        let produced = child_session.prompt(&input.prompt).await.map_err(|error| {
            let error_message = format!("subagent '{}' failed: {error}", subagent_name);
            self.emit_lifecycle_event(ParentChildRunEvent::ChildRunError {
                parent_session_id: parent_session_id.clone(),
                child_session_file: child_session_file_text.clone(),
                task_id: task_id.clone(),
                subagent: subagent_name.clone(),
                error: error_message.clone(),
            });
            PiAiError::new(PiAiErrorCode::ToolExecutionFailed, error_message)
        })?;
        let trace_lines = collect_subagent_trace_lines(&produced);
        if let Some((stop_reason, error_message)) = last_assistant_stop_reason(&produced) {
            if matches!(stop_reason, StopReason::Error | StopReason::Aborted) {
                let failure = error_message.unwrap_or_else(|| {
                    format!(
                        "subagent '{}' ended with stop_reason={stop_reason:?}",
                        subagent_name
                    )
                });
                self.emit_lifecycle_event(ParentChildRunEvent::ChildRunError {
                    parent_session_id: parent_session_id.clone(),
                    child_session_file: child_session_file_text.clone(),
                    task_id: task_id.clone(),
                    subagent: subagent_name.clone(),
                    error: failure.clone(),
                });
                return Err(PiAiError::new(PiAiErrorCode::ToolExecutionFailed, failure));
            }
        }
        let summary = last_assistant_text(&produced)
            .unwrap_or_else(|| "Subagent completed without assistant text output.".to_string());

        let output = TaskToolOutput {
            task_id: task_id.clone(),
            summary: summary.clone(),
            child_session_file: child_session_file.to_string_lossy().to_string(),
        };
        let mut after_ctx = AfterTaskResultHookContext {
            output,
            resolved_subagent: subagent.name,
            routing_hint_applied: policy_decision.routing_hint_applied,
        };
        self.config.plugin_runtime.after_task_result(&mut after_ctx);
        after_ctx.output.validate().map_err(|error| {
            self.emit_lifecycle_event(ParentChildRunEvent::ChildRunError {
                parent_session_id: parent_session_id.clone(),
                child_session_file: child_session_file_text.clone(),
                task_id: task_id.clone(),
                subagent: subagent_name.clone(),
                error: error.clone(),
            });
            PiAiError::new(PiAiErrorCode::ToolExecutionFailed, error)
        })?;

        let duration_ms = u64::try_from(run_started_at.elapsed().as_millis()).unwrap_or(u64::MAX);

        self.emit_lifecycle_event(ParentChildRunEvent::ChildRunEnd {
            parent_session_id,
            child_session_file: child_session_file_text,
            task_id: task_id.clone(),
            subagent: subagent_name,
            duration_ms,
            summary: after_ctx.output.summary.clone(),
        });

        Ok(TaskDispatchResult {
            summary: after_ctx.output.summary.clone(),
            output: after_ctx.output,
            resolved_subagent: after_ctx.resolved_subagent,
            routing_hint_applied: after_ctx.routing_hint_applied,
            duration_ms,
            trace_lines,
        })
    }

    fn emit_lifecycle_event(&self, event: ParentChildRunEvent) {
        if let Some(sink) = &self.config.lifecycle_event_sink {
            sink(event);
        }
    }

    async fn resolve_or_create_child_session_file(
        &self,
        task_id: &str,
    ) -> Result<PathBuf, PiAiError> {
        {
            let store = self.config.session_store.lock().await;
            if let Some(path) = store.resolve(task_id) {
                if path.exists() {
                    return Ok(path);
                }
            }
        }

        let manager = SessionManager::create_with_parent(
            self.config.cwd.to_string_lossy().as_ref(),
            &self.config.parent_session_dir,
            Some(&self.config.parent_session_id),
        )
        .map_err(|error| {
            PiAiError::new(
                PiAiErrorCode::ToolExecutionFailed,
                format!("failed to create child session for task '{task_id}': {error}"),
            )
        })?;

        let session_file = manager.session_file().cloned().ok_or_else(|| {
            PiAiError::new(
                PiAiErrorCode::ToolExecutionFailed,
                "child session manager returned no session file path",
            )
        })?;

        let mut store = self.config.session_store.lock().await;
        store.insert(task_id, &session_file).map_err(|error| {
            PiAiError::new(
                PiAiErrorCode::ToolExecutionFailed,
                format!("failed to register child session for task '{task_id}': {error}"),
            )
        })?;

        Ok(session_file)
    }
}

fn resolve_child_model(
    default_model: &Model,
    model_catalog: &[Model],
    subagent: &crate::SubAgentSpec,
) -> Result<Model, String> {
    let Some(raw_target) = subagent
        .model
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(default_model.clone());
    };

    if let Some((provider, model_id)) = split_provider_model(raw_target) {
        if default_model.provider == provider && default_model.id == model_id {
            return Ok(default_model.clone());
        }
        if let Some(found) = model_catalog
            .iter()
            .find(|model| model.provider == provider && model.id == model_id)
        {
            return Ok(found.clone());
        }
        return Err(format!(
            "model '{}' not found in runtime model catalog",
            raw_target
        ));
    }

    if default_model.provider == raw_target {
        return Ok(default_model.clone());
    }
    if let Some(found) = model_catalog
        .iter()
        .find(|model| model.provider == raw_target)
    {
        return Ok(found.clone());
    }

    if default_model.id == raw_target {
        return Ok(default_model.clone());
    }
    if let Some(found) = model_catalog.iter().find(|model| model.id == raw_target) {
        return Ok(found.clone());
    }

    Err(format!(
        "model '{}' not found in runtime model catalog",
        raw_target
    ))
}

fn split_provider_model(raw: &str) -> Option<(String, String)> {
    let (provider, model) = raw.split_once('/')?;
    let provider = provider.trim();
    let model = model.trim();
    if provider.is_empty() || model.is_empty() {
        return None;
    }
    Some((provider.to_string(), model.to_string()))
}

fn resolve_child_tools(base_tools: &[AgentTool], subagent: &crate::SubAgentSpec) -> Vec<AgentTool> {
    let allow = subagent
        .normalized_allowed_tools()
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    let deny = subagent
        .normalized_blocked_tools()
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    let use_allow = !allow.is_empty();

    base_tools
        .iter()
        .filter(|tool| !use_allow || allow.contains(&tool.name))
        .filter(|tool| !deny.contains(&tool.name))
        .cloned()
        .collect()
}

fn build_child_system_prompt(parent_system_prompt: &str, subagent: &crate::SubAgentSpec) -> String {
    let mut prompt = format!(
        "{parent_system_prompt}\n\n<subagent_context>\nYou are running as subagent '{}'. Focus on the delegated task and report concise actionable results.\nSubagent description: {}\n</subagent_context>",
        subagent.name,
        subagent.description.trim()
    );
    if let Some(extra_prompt) = subagent
        .prompt
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        prompt.push_str("\n\n<subagent_profile>\n");
        prompt.push_str(extra_prompt);
        prompt.push_str("\n</subagent_profile>");
    }
    prompt
}

fn last_assistant_text(messages: &[Message]) -> Option<String> {
    messages.iter().rev().find_map(|message| {
        let Message::Assistant { content, .. } = message else {
            return None;
        };

        let text = content
            .iter()
            .filter_map(|block| match block {
                AssistantContentBlock::Text { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
            .trim()
            .to_string();

        if text.is_empty() {
            None
        } else {
            Some(text)
        }
    })
}

fn last_assistant_stop_reason(messages: &[Message]) -> Option<(StopReason, Option<String>)> {
    messages.iter().rev().find_map(|message| {
        let Message::Assistant {
            stop_reason,
            error_message,
            ..
        } = message
        else {
            return None;
        };
        Some((stop_reason.clone(), error_message.clone()))
    })
}

fn collect_subagent_trace_lines(messages: &[Message]) -> Vec<String> {
    let mut lines = Vec::new();
    for message in messages {
        let Message::ToolResult {
            tool_name, content, ..
        } = message
        else {
            continue;
        };
        lines.push(format!("• Ran {tool_name}"));
        if *tool_name == "read" {
            continue;
        }
        for block in content {
            match block {
                pixy_ai::ToolResultContentBlock::Text { text, .. } => {
                    lines.extend(
                        text.lines()
                            .map(str::trim_end)
                            .filter(|line| !line.is_empty())
                            .map(str::to_string),
                    );
                }
                pixy_ai::ToolResultContentBlock::Image { .. } => {
                    lines.push("(image tool result omitted)".to_string());
                }
            }
        }
    }
    lines
}

fn generate_task_id() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0);
    let counter = TASK_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("task-{millis}-{counter}")
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::sync::Mutex as StdMutex;

    use pixy_agent_core::{AgentTool, AgentToolResult, ParentChildRunEvent, ToolFuture};
    use pixy_ai::{
        AssistantContentBlock, AssistantMessage, AssistantMessageEvent,
        AssistantMessageEventStream, Cost, DoneReason, Model, PiAiError, PiAiErrorCode, StopReason,
        ToolResultContentBlock, Usage,
    };
    use tempfile::tempdir;
    use tokio::sync::Mutex;

    use super::*;
    use crate::{
        ChildSessionStore, DefaultSubAgentRegistry, MultiAgentPluginRuntime, SubAgentMode,
        SubAgentResolver, SubAgentSpec, TaskToolInput,
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
                prompt: None,
                model: None,
                tools: vec![],
                blocked_tools: vec![],
                metadata: None,
            })
            .expect("register general")
            .build();
        Arc::new(built)
    }

    fn no_op_tool(name: &str) -> AgentTool {
        AgentTool {
            name: name.to_string(),
            label: name.to_string(),
            description: format!("{name} tool"),
            parameters: serde_json::json!({}),
            execute: Arc::new(
                |_tool_call_id: String, _args: serde_json::Value| -> ToolFuture {
                    Box::pin(async {
                        Err(PiAiError::new(PiAiErrorCode::ToolExecutionFailed, "unused"))
                            as Result<AgentToolResult, PiAiError>
                    })
                },
            ),
        }
    }

    #[tokio::test]
    async fn dispatch_returns_summary_and_task_output() {
        let dir = tempdir().expect("tempdir");
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_clone = calls.clone();

        let dispatcher = TaskDispatcher::new(TaskDispatcherConfig {
            cwd: dir.path().to_path_buf(),
            parent_session_id: "parent-session".to_string(),
            parent_session_dir: dir.path().to_path_buf(),
            model: sample_model(),
            model_catalog: vec![sample_model()],
            system_prompt: "You are parent".to_string(),
            stream_fn: Arc::new(move |_model, _context, _options| {
                calls_clone.fetch_add(1, Ordering::SeqCst);
                Ok(done_stream("child done".to_string()))
            }),
            child_tools: vec![],
            subagent_registry: registry(),
            session_store: Arc::new(Mutex::new(ChildSessionStore::new("parent-session"))),
            dispatch_policy: DispatchPolicyConfig::default(),
            plugin_runtime: Arc::new(MultiAgentPluginRuntime::default()),
            lifecycle_event_sink: None,
        });

        let result = dispatcher
            .dispatch(TaskToolInput {
                subagent_type: "general".to_string(),
                prompt: "investigate".to_string(),
                task_id: None,
            })
            .await
            .expect("dispatch should succeed");

        assert_eq!(result.summary, "child done");
        assert_eq!(result.resolved_subagent, "general");
        assert!(!result.routing_hint_applied);
        assert!(!result.output.task_id.is_empty());
        assert!(std::path::Path::new(&result.output.child_session_file).exists());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn dispatch_rejects_unknown_subagent_type() {
        let dir = tempdir().expect("tempdir");

        let dispatcher = TaskDispatcher::new(TaskDispatcherConfig {
            cwd: dir.path().to_path_buf(),
            parent_session_id: "parent-session".to_string(),
            parent_session_dir: dir.path().to_path_buf(),
            model: sample_model(),
            model_catalog: vec![sample_model()],
            system_prompt: "You are parent".to_string(),
            stream_fn: Arc::new(move |_model, _context, _options| {
                Ok(done_stream("child done".to_string()))
            }),
            child_tools: vec![],
            subagent_registry: registry(),
            session_store: Arc::new(Mutex::new(ChildSessionStore::new("parent-session"))),
            dispatch_policy: DispatchPolicyConfig::default(),
            plugin_runtime: Arc::new(MultiAgentPluginRuntime::default()),
            lifecycle_event_sink: None,
        });

        let error = dispatcher
            .dispatch(TaskToolInput {
                subagent_type: "missing".to_string(),
                prompt: "investigate".to_string(),
                task_id: None,
            })
            .await
            .expect_err("dispatch should reject unknown subagent");

        assert!(error.message.contains("subagent"));
    }

    #[tokio::test]
    async fn dispatch_reuses_child_session_when_task_id_repeats() {
        let dir = tempdir().expect("tempdir");
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_clone = calls.clone();

        let dispatcher = TaskDispatcher::new(TaskDispatcherConfig {
            cwd: dir.path().to_path_buf(),
            parent_session_id: "parent-session".to_string(),
            parent_session_dir: dir.path().to_path_buf(),
            model: sample_model(),
            model_catalog: vec![sample_model()],
            system_prompt: "You are parent".to_string(),
            stream_fn: Arc::new(move |_model, _context, _options| {
                let turn = calls_clone.fetch_add(1, Ordering::SeqCst) + 1;
                Ok(done_stream(format!("turn {turn}")))
            }),
            child_tools: vec![],
            subagent_registry: registry(),
            session_store: Arc::new(Mutex::new(ChildSessionStore::new("parent-session"))),
            dispatch_policy: DispatchPolicyConfig::default(),
            plugin_runtime: Arc::new(MultiAgentPluginRuntime::default()),
            lifecycle_event_sink: None,
        });

        let first = dispatcher
            .dispatch(TaskToolInput {
                subagent_type: "general".to_string(),
                prompt: "first".to_string(),
                task_id: Some("task-123".to_string()),
            })
            .await
            .expect("first dispatch");
        let second = dispatcher
            .dispatch(TaskToolInput {
                subagent_type: "general".to_string(),
                prompt: "second".to_string(),
                task_id: Some("task-123".to_string()),
            })
            .await
            .expect("second dispatch");

        assert_eq!(first.output.task_id, "task-123");
        assert_eq!(second.output.task_id, "task-123");
        assert_eq!(
            first.output.child_session_file,
            second.output.child_session_file
        );

        let loaded = crate::SessionManager::load(&first.output.child_session_file)
            .expect("child session should load");
        let context = loaded.build_session_context();
        assert!(
            context.messages.len() >= 4,
            "two prompts should persist at least two user+assistant turns"
        );
    }

    #[test]
    fn collect_subagent_trace_lines_keeps_tool_flow_and_hides_read_body() {
        let messages = vec![
            Message::ToolResult {
                tool_call_id: "call-read".to_string(),
                tool_name: "read".to_string(),
                content: vec![ToolResultContentBlock::Text {
                    text: "secret body".to_string(),
                    text_signature: None,
                }],
                details: None,
                is_error: false,
                timestamp: 1,
            },
            Message::ToolResult {
                tool_call_id: "call-write".to_string(),
                tool_name: "write".to_string(),
                content: vec![ToolResultContentBlock::Text {
                    text: "updated file".to_string(),
                    text_signature: None,
                }],
                details: None,
                is_error: false,
                timestamp: 2,
            },
        ];

        let lines = super::collect_subagent_trace_lines(&messages);
        assert_eq!(lines[0], "• Ran read");
        assert!(!lines.iter().any(|line| line.contains("secret body")));
        assert!(lines.iter().any(|line| line == "• Ran write"));
        assert!(lines.iter().any(|line| line == "updated file"));
    }

    #[test]
    fn generate_task_id_produces_unique_ids_in_process() {
        let mut ids = HashSet::new();
        for _ in 0..1024 {
            let id = super::generate_task_id();
            assert!(ids.insert(id), "task id should be unique");
        }
    }

    #[tokio::test]
    async fn dispatch_emits_lifecycle_error_when_child_session_creation_fails() {
        let dir = tempdir().expect("tempdir");
        let blocked_session_root = dir.path().join("not-a-directory");
        std::fs::write(&blocked_session_root, "occupied").expect("write blocker");
        let events = Arc::new(StdMutex::new(Vec::<ParentChildRunEvent>::new()));
        let events_for_sink = events.clone();

        let dispatcher = TaskDispatcher::new(TaskDispatcherConfig {
            cwd: dir.path().to_path_buf(),
            parent_session_id: "parent-session".to_string(),
            parent_session_dir: blocked_session_root,
            model: sample_model(),
            model_catalog: vec![sample_model()],
            system_prompt: "You are parent".to_string(),
            stream_fn: Arc::new(move |_model, _context, _options| {
                Ok(done_stream("child done".to_string()))
            }),
            child_tools: vec![],
            subagent_registry: registry(),
            session_store: Arc::new(Mutex::new(ChildSessionStore::new("parent-session"))),
            dispatch_policy: DispatchPolicyConfig::default(),
            plugin_runtime: Arc::new(MultiAgentPluginRuntime::default()),
            lifecycle_event_sink: Some(Arc::new(move |event| {
                events_for_sink.lock().expect("lock events").push(event);
            })),
        });

        let error = dispatcher
            .dispatch(TaskToolInput {
                subagent_type: "general".to_string(),
                prompt: "fails before child session starts".to_string(),
                task_id: Some("task-fail-create".to_string()),
            })
            .await
            .expect_err("dispatch should fail");
        assert!(error.message.contains("failed to create child session"));

        let events = events.lock().expect("lock events");
        assert!(events.iter().any(|event| {
            matches!(
                event,
                ParentChildRunEvent::ChildRunError {
                    task_id,
                    child_session_file,
                    ..
                } if task_id == "task-fail-create"
                    && child_session_file == UNRESOLVED_CHILD_SESSION_FILE
            )
        }));
    }

    #[test]
    fn resolve_child_model_uses_subagent_model_override() {
        let default_model = sample_model();
        let alternate_model = Model {
            id: "gemini-3-pro-preview".to_string(),
            name: "Gemini".to_string(),
            api: "google-generative-ai".to_string(),
            provider: "google".to_string(),
            base_url: "https://generativelanguage.googleapis.com".to_string(),
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
            context_window: 1_048_576,
            max_tokens: 8_192,
        };
        let subagent = SubAgentSpec {
            name: "code".to_string(),
            description: "Code worker".to_string(),
            mode: SubAgentMode::SubAgent,
            prompt: None,
            model: Some("google/gemini-3-pro-preview".to_string()),
            tools: vec![],
            blocked_tools: vec![],
            metadata: None,
        };

        let resolved = resolve_child_model(
            &default_model,
            &[default_model.clone(), alternate_model.clone()],
            &subagent,
        )
        .expect("model override should resolve");
        assert_eq!(resolved.provider, "google");
        assert_eq!(resolved.id, "gemini-3-pro-preview");
    }

    #[test]
    fn resolve_child_model_rejects_unknown_override() {
        let subagent = SubAgentSpec {
            name: "code".to_string(),
            description: "Code worker".to_string(),
            mode: SubAgentMode::SubAgent,
            prompt: None,
            model: Some("missing-provider/missing-model".to_string()),
            tools: vec![],
            blocked_tools: vec![],
            metadata: None,
        };

        let error = resolve_child_model(&sample_model(), &[sample_model()], &subagent)
            .expect_err("unknown model should fail");
        assert!(error.contains("not found"));
    }

    #[test]
    fn resolve_child_model_uses_provider_override() {
        let default_model = sample_model();
        let alternate_model = Model {
            id: "claude-4-6-sonnet-latest".to_string(),
            name: "Claude".to_string(),
            api: "anthropic-messages".to_string(),
            provider: "anthropic".to_string(),
            base_url: "https://api.anthropic.com".to_string(),
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
            context_window: 200_000,
            max_tokens: 8_192,
        };
        let subagent = SubAgentSpec {
            name: "review".to_string(),
            description: "Review worker".to_string(),
            mode: SubAgentMode::SubAgent,
            prompt: None,
            model: Some("anthropic".to_string()),
            tools: vec![],
            blocked_tools: vec![],
            metadata: None,
        };

        let resolved = resolve_child_model(
            &default_model,
            &[default_model.clone(), alternate_model.clone()],
            &subagent,
        )
        .expect("provider override should resolve");
        assert_eq!(resolved.provider, "anthropic");
        assert_eq!(resolved.id, "claude-4-6-sonnet-latest");
    }

    #[test]
    fn resolve_child_tools_applies_allow_and_deny_rules() {
        let tools = vec![no_op_tool("read"), no_op_tool("edit"), no_op_tool("bash")];
        let subagent = SubAgentSpec {
            name: "code".to_string(),
            description: "Code worker".to_string(),
            mode: SubAgentMode::SubAgent,
            prompt: None,
            model: None,
            tools: vec!["read".to_string(), "bash".to_string()],
            blocked_tools: vec!["bash".to_string()],
            metadata: None,
        };

        let filtered = resolve_child_tools(&tools, &subagent);
        let names = filtered
            .iter()
            .map(|tool| tool.name.clone())
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["read"]);
    }

    #[test]
    fn build_child_system_prompt_includes_subagent_profile_prompt() {
        let subagent = SubAgentSpec {
            name: "review".to_string(),
            description: "Review worker".to_string(),
            mode: SubAgentMode::SubAgent,
            prompt: Some("You MUST start with REVIEW_STATUS: PASS or FAIL.".to_string()),
            model: None,
            tools: vec![],
            blocked_tools: vec![],
            metadata: None,
        };

        let prompt = build_child_system_prompt("You are parent.", &subagent);
        assert!(prompt.contains("<subagent_context>"));
        assert!(prompt.contains("Review worker"));
        assert!(prompt.contains("<subagent_profile>"));
        assert!(prompt.contains("REVIEW_STATUS"));
    }
}
