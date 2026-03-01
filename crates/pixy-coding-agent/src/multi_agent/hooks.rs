use crate::{TaskToolInput, TaskToolOutput};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BeforeUserMessageHookContext {
    pub message: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BeforeTaskDispatchHookContext {
    pub input: TaskToolInput,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AfterTaskResultHookContext {
    pub output: TaskToolOutput,
    pub resolved_subagent: String,
    pub routing_hint_applied: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BeforeToolDefinitionHookContext {
    pub tool_name: String,
    pub description: String,
}

pub trait MultiAgentHook: Send + Sync {
    fn before_user_message(&self, _ctx: &mut BeforeUserMessageHookContext) {}
    fn before_task_dispatch(&self, _ctx: &mut BeforeTaskDispatchHookContext) {}
    fn after_task_result(&self, _ctx: &mut AfterTaskResultHookContext) {}
    fn before_tool_definition(&self, _ctx: &mut BeforeToolDefinitionHookContext) {}
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::{
        create_multi_agent_plugin_runtime, MultiAgentPluginRuntime, TaskToolInput, TaskToolOutput,
    };

    #[derive(Clone)]
    struct RecordingHook {
        name: &'static str,
        log: Arc<Mutex<Vec<String>>>,
    }

    impl RecordingHook {
        fn record(&self, hook_name: &str) {
            self.log
                .lock()
                .expect("lock log")
                .push(format!("{}:{hook_name}", self.name));
        }
    }

    impl MultiAgentHook for RecordingHook {
        fn before_user_message(&self, ctx: &mut BeforeUserMessageHookContext) {
            self.record("before_user_message");
            ctx.message.push_str(&format!(" [{}]", self.name));
        }

        fn before_task_dispatch(&self, ctx: &mut BeforeTaskDispatchHookContext) {
            self.record("before_task_dispatch");
            ctx.input.prompt.push_str(&format!(" [{}]", self.name));
        }

        fn after_task_result(&self, ctx: &mut AfterTaskResultHookContext) {
            self.record("after_task_result");
            ctx.output.summary.push_str(&format!(" [{}]", self.name));
        }

        fn before_tool_definition(&self, ctx: &mut BeforeToolDefinitionHookContext) {
            self.record("before_tool_definition");
            ctx.description.push_str(&format!(" [{}]", self.name));
        }
    }

    fn runtime_with_two_hooks(log: Arc<Mutex<Vec<String>>>) -> MultiAgentPluginRuntime {
        create_multi_agent_plugin_runtime(vec![
            Arc::new(RecordingHook {
                name: "h1",
                log: log.clone(),
            }),
            Arc::new(RecordingHook { name: "h2", log }),
        ])
    }

    #[test]
    fn hook_runtime_applies_hooks_in_registration_order_for_each_stage() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let runtime = runtime_with_two_hooks(log.clone());

        let mut user_ctx = BeforeUserMessageHookContext {
            message: "hello".to_string(),
        };
        runtime.before_user_message(&mut user_ctx);
        assert_eq!(user_ctx.message, "hello [h1] [h2]");

        let mut dispatch_ctx = BeforeTaskDispatchHookContext {
            input: TaskToolInput {
                subagent_type: "general".to_string(),
                prompt: "investigate".to_string(),
                task_id: Some("task-1".to_string()),
            },
        };
        runtime.before_task_dispatch(&mut dispatch_ctx);
        assert_eq!(dispatch_ctx.input.prompt, "investigate [h1] [h2]");

        let mut result_ctx = AfterTaskResultHookContext {
            output: TaskToolOutput {
                task_id: "task-1".to_string(),
                summary: "done".to_string(),
                child_session_file: "/tmp/child.jsonl".to_string(),
            },
            resolved_subagent: "general".to_string(),
            routing_hint_applied: false,
        };
        runtime.after_task_result(&mut result_ctx);
        assert_eq!(result_ctx.output.summary, "done [h1] [h2]");

        let mut tool_ctx = BeforeToolDefinitionHookContext {
            tool_name: "task".to_string(),
            description: "delegate".to_string(),
        };
        runtime.before_tool_definition(&mut tool_ctx);
        assert_eq!(tool_ctx.description, "delegate [h1] [h2]");

        let recorded = log.lock().expect("lock log").clone();
        assert_eq!(
            recorded,
            vec![
                "h1:before_user_message",
                "h2:before_user_message",
                "h1:before_task_dispatch",
                "h2:before_task_dispatch",
                "h1:after_task_result",
                "h2:after_task_result",
                "h1:before_tool_definition",
                "h2:before_tool_definition",
            ]
        );
    }
}
