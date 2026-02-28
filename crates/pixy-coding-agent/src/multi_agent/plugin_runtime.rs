use std::sync::Arc;

use super::{
    AfterTaskResultHookContext, BeforeTaskDispatchHookContext, BeforeToolDefinitionHookContext,
    BeforeUserMessageHookContext, MultiAgentHook,
};

#[derive(Clone, Default)]
pub struct MultiAgentPluginRuntime {
    hooks: Arc<Vec<Arc<dyn MultiAgentHook>>>,
}

impl MultiAgentPluginRuntime {
    pub fn from_hooks(hooks: Vec<Arc<dyn MultiAgentHook>>) -> Self {
        Self {
            hooks: Arc::new(hooks),
        }
    }

    pub fn before_user_message(&self, ctx: &mut BeforeUserMessageHookContext) {
        for hook in self.hooks.iter() {
            hook.before_user_message(ctx);
        }
    }

    pub fn before_task_dispatch(&self, ctx: &mut BeforeTaskDispatchHookContext) {
        for hook in self.hooks.iter() {
            hook.before_task_dispatch(ctx);
        }
    }

    pub fn after_task_result(&self, ctx: &mut AfterTaskResultHookContext) {
        for hook in self.hooks.iter() {
            hook.after_task_result(ctx);
        }
    }

    pub fn before_tool_definition(&self, ctx: &mut BeforeToolDefinitionHookContext) {
        for hook in self.hooks.iter() {
            hook.before_tool_definition(ctx);
        }
    }
}

pub fn create_multi_agent_plugin_runtime(
    hooks: Vec<Arc<dyn MultiAgentHook>>,
) -> MultiAgentPluginRuntime {
    MultiAgentPluginRuntime::from_hooks(hooks)
}
