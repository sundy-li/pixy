use std::path::PathBuf;

use pixy_agent_core::AgentMessage;
use pixy_ai::Message;
use serde_json::Value;

use crate::agent_session::{AgentSessionStreamUpdate, SessionResumeCandidate};

pub(crate) struct SessionResumeService;

impl SessionResumeService {
    pub(crate) fn new() -> Self {
        Self
    }

    pub(crate) fn resolve_resume_session_target(
        &self,
        target: Option<&str>,
        current_session_file: Option<PathBuf>,
    ) -> Result<PathBuf, String> {
        super::agent_session::resolve_resume_session_target(target, current_session_file)
    }

    pub(crate) fn build_session_resume_candidate(
        &self,
        path: PathBuf,
    ) -> Result<SessionResumeCandidate, String> {
        super::agent_session::build_session_resume_candidate(path)
    }
}

pub(crate) struct AutoCompactionService;

impl AutoCompactionService {
    pub(crate) fn new() -> Self {
        Self
    }

    pub(crate) fn latest_assistant_message<'a>(
        &self,
        messages: &'a [AgentMessage],
    ) -> Option<&'a Message> {
        super::agent_session::latest_assistant_message(messages)
    }

    pub(crate) fn is_context_overflow_message(
        &self,
        message: &Message,
        context_window: u64,
    ) -> bool {
        super::agent_session::is_context_overflow_message(message, context_window)
    }

    pub(crate) fn latest_context_tokens_from_messages(
        &self,
        messages: &[AgentMessage],
    ) -> Option<u64> {
        super::agent_session::latest_context_tokens_from_messages(messages)
    }

    pub(crate) fn build_auto_compaction_summary(
        &self,
        messages: &[Message],
        context_tokens: u64,
        context_window: u64,
        max_summary_chars: usize,
    ) -> String {
        super::agent_session::build_auto_compaction_summary(
            messages,
            context_tokens,
            context_window,
            max_summary_chars,
        )
    }
}

pub(crate) struct StreamingToolLineRenderer;

impl StreamingToolLineRenderer {
    pub(crate) fn new() -> Self {
        Self
    }

    pub(crate) fn should_render_tool_result_content(&self, tool_name: &str) -> bool {
        super::agent_session::should_render_tool_result_content(tool_name)
    }

    pub(crate) fn format_tool_start_line(&self, tool_name: &str, args: &Value) -> String {
        super::agent_session::format_tool_start_line(tool_name, args)
    }

    pub(crate) fn render_messages_for_streaming(
        &self,
        messages: &[AgentMessage],
    ) -> Vec<AgentSessionStreamUpdate> {
        super::agent_session::render_messages_for_streaming(messages)
    }
}
