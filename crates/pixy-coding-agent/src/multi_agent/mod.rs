mod declarative_hooks;
mod dispatcher;
mod hooks;
mod plugin_loader;
mod plugin_manifest;
mod plugin_runtime;
mod policy;
mod registry;
mod session_store;
mod task_tool;
mod types;

pub use declarative_hooks::{
    create_multi_agent_plugin_runtime_from_specs, DeclarativeHookAction, DeclarativeHookSpec,
    DeclarativeHookStage,
};
pub use dispatcher::{TaskDispatchResult, TaskDispatcher, TaskDispatcherConfig};
pub use hooks::{
    AfterTaskResultHookContext, BeforeTaskDispatchHookContext, BeforeToolDefinitionHookContext,
    BeforeUserMessageHookContext, MultiAgentHook,
};
pub use plugin_loader::{
    load_and_merge_plugins, load_and_merge_plugins_from_paths, load_plugin_manifests,
    LoadedPluginManifest, MergedPluginConfig, PluginSubAgentSpec,
};
pub use plugin_manifest::MultiAgentPluginManifest;
pub use plugin_runtime::{create_multi_agent_plugin_runtime, MultiAgentPluginRuntime};
pub use policy::{
    DispatchPolicyConfig, DispatchPolicyDecision, DispatchPolicyRule, PolicyRuleEffect,
};
pub use registry::{DefaultSubAgentRegistry, SubAgentRegistryBuilder, SubAgentResolver};
pub use session_store::ChildSessionStore;
pub use task_tool::create_task_tool;
pub use types::{SubAgentMode, SubAgentSpec, TaskToolInput, TaskToolOutput};
