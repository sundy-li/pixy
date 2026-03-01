//! Session-oriented coding agent orchestration.

mod agent_session;
mod agent_session_services;
mod bash_command;
pub mod cli;
mod cli_app;
pub mod memory;
mod memory_tool;
mod messages;
mod multi_agent;
mod runtime_config;
mod session_manager;
mod skills;
pub mod system_prompt;
mod tools;
mod tui_backend;

pub use agent_session::{
    create_session, create_session_from_runtime, AgentMode, AgentSession, AgentSessionConfig,
    AgentSessionStreamUpdate, AutoCompactionConfig, CreatedSession, SessionCreateOptions,
};
pub use memory_tool::create_memory_tool;
pub use messages::{
    bash_execution_to_text, convert_to_llm, BashExecutionMessage, BranchSummaryMessage,
    CodingMessage, CompactionSummaryMessage, CustomMessage, BRANCH_SUMMARY_PREFIX,
    BRANCH_SUMMARY_SUFFIX, COMPACTION_SUMMARY_PREFIX, COMPACTION_SUMMARY_SUFFIX,
};
pub use multi_agent::{
    create_multi_agent_plugin_runtime, create_multi_agent_plugin_runtime_from_specs,
    create_task_tool, load_and_merge_plugins, load_and_merge_plugins_from_paths,
    load_plugin_manifests, AfterTaskResultHookContext, BeforeTaskDispatchHookContext,
    BeforeToolDefinitionHookContext, BeforeUserMessageHookContext, ChildSessionStore,
    DeclarativeHookAction, DeclarativeHookSpec, DeclarativeHookStage, DefaultSubAgentRegistry,
    DispatchPolicyConfig, DispatchPolicyDecision, DispatchPolicyRule, LoadedPluginManifest,
    MergedPluginConfig, MultiAgentHook, MultiAgentPluginManifest, MultiAgentPluginRuntime,
    PluginSubAgentSpec, PolicyRuleEffect, SubAgentMode, SubAgentRegistryBuilder, SubAgentResolver,
    SubAgentSpec, TaskDispatchResult, TaskDispatcher, TaskDispatcherConfig, TaskToolInput,
    TaskToolOutput,
};
pub use runtime_config::{
    LLMRouter, ResolvedMemoryConfig, ResolvedMemorySearchConfig, ResolvedMultiAgentConfig,
    ResolvedRuntime, RuntimeLoadOptions, RuntimeOverrides,
};
pub use session_manager::{SessionContext, SessionManager, CURRENT_SESSION_VERSION};
pub use skills::{
    format_skills_for_prompt, load_skills, load_skills_from_dir, LoadSkillsOptions,
    LoadSkillsResult, Skill, SkillDiagnostic, SkillDiagnosticKind, SkillSource,
};
pub use system_prompt::build_system_prompt;
pub use tools::{
    create_bash_tool, create_coding_tools, create_coding_tools_with_extra, create_edit_tool,
    create_list_directory_tool, create_read_tool, create_write_tool,
};
