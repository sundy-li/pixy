//! Session-oriented coding agent orchestration.

mod agent_session;
mod agent_session_services;
mod bash_command;
pub mod cli;
mod cli_app;
mod messages;
mod runtime_config;
mod session_manager;
mod skills;
pub mod system_prompt;
mod tools;
mod tui_backend;

pub use agent_session::{
    AgentSession, AgentSessionConfig, AgentSessionStreamUpdate, AutoCompactionConfig,
    CreatedSession, SessionCreateOptions, create_session, create_session_from_runtime,
};
pub use messages::{
    BRANCH_SUMMARY_PREFIX, BRANCH_SUMMARY_SUFFIX, BashExecutionMessage, BranchSummaryMessage,
    COMPACTION_SUMMARY_PREFIX, COMPACTION_SUMMARY_SUFFIX, CodingMessage, CompactionSummaryMessage,
    CustomMessage, bash_execution_to_text, convert_to_llm,
};
pub use runtime_config::{LLMRouter, ResolvedRuntime, RuntimeLoadOptions, RuntimeOverrides};
pub use session_manager::{CURRENT_SESSION_VERSION, SessionContext, SessionManager};
pub use skills::{
    LoadSkillsOptions, LoadSkillsResult, Skill, SkillDiagnostic, SkillDiagnosticKind, SkillSource,
    format_skills_for_prompt, load_skills, load_skills_from_dir,
};
pub use system_prompt::build_system_prompt;
pub use tools::{
    create_bash_tool, create_coding_tools, create_edit_tool, create_list_directory_tool,
    create_read_tool, create_write_tool,
};
