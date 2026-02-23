//! Session-oriented coding agent orchestration.

mod agent_session;
mod messages;
mod session_manager;
mod skills;
mod tools;
mod tui_backend;

pub use agent_session::{
    AgentSession, AgentSessionConfig, AgentSessionStreamUpdate, AutoCompactionConfig,
};
pub use messages::{
    BRANCH_SUMMARY_PREFIX, BRANCH_SUMMARY_SUFFIX, BashExecutionMessage, BranchSummaryMessage,
    COMPACTION_SUMMARY_PREFIX, COMPACTION_SUMMARY_SUFFIX, CodingMessage, CompactionSummaryMessage,
    CustomMessage, bash_execution_to_text, convert_to_llm,
};
pub use session_manager::{CURRENT_SESSION_VERSION, SessionContext, SessionManager};
pub use skills::{
    LoadSkillsOptions, LoadSkillsResult, Skill, SkillDiagnostic, SkillDiagnosticKind, SkillSource,
    format_skills_for_prompt, load_skills, load_skills_from_dir,
};
pub use tools::{
    create_bash_tool, create_coding_tools, create_edit_tool, create_read_tool, create_write_tool,
};
