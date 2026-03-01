use std::path::PathBuf;

use pixy_agent_core::AgentAbortSignal;
use pixy_tui::{BackendFuture, ResumeCandidate, StreamUpdate, TuiBackend};

use crate::{cli_app::CliSession, AgentSession, AgentSessionStreamUpdate};

impl TuiBackend for AgentSession {
    fn prompt<'a>(&'a mut self, input: &'a str) -> BackendFuture<'a> {
        Box::pin(async move { AgentSession::prompt(self, input).await })
    }

    fn continue_run<'a>(&'a mut self) -> BackendFuture<'a> {
        Box::pin(async move { AgentSession::continue_run(self).await })
    }

    fn prompt_stream<'a>(
        &'a mut self,
        input: &'a str,
        abort_signal: Option<AgentAbortSignal>,
        on_update: &'a mut dyn FnMut(StreamUpdate),
    ) -> BackendFuture<'a> {
        Box::pin(async move {
            let mut mapper = ThinkingStreamMapper::default();
            AgentSession::prompt_streaming_with_abort(self, input, abort_signal, |update| {
                if let Some(mapped) = mapper.map(update) {
                    on_update(mapped);
                }
            })
            .await
        })
    }

    fn prompt_stream_with_blocks<'a>(
        &'a mut self,
        input: &'a str,
        blocks: Option<Vec<pixy_ai::UserContentBlock>>,
        abort_signal: Option<AgentAbortSignal>,
        on_update: &'a mut dyn FnMut(StreamUpdate),
    ) -> BackendFuture<'a> {
        Box::pin(async move {
            let mut mapper = ThinkingStreamMapper::default();
            AgentSession::prompt_streaming_blocks_with_abort(
                self,
                input,
                blocks,
                abort_signal,
                |update| {
                    if let Some(mapped) = mapper.map(update) {
                        on_update(mapped);
                    }
                },
            )
            .await
        })
    }

    fn continue_run_stream<'a>(
        &'a mut self,
        abort_signal: Option<AgentAbortSignal>,
        on_update: &'a mut dyn FnMut(StreamUpdate),
    ) -> BackendFuture<'a> {
        Box::pin(async move {
            let mut mapper = ThinkingStreamMapper::default();
            AgentSession::continue_run_streaming_with_abort(self, abort_signal, |update| {
                if let Some(mapped) = mapper.map(update) {
                    on_update(mapped);
                }
            })
            .await
        })
    }

    fn cycle_model_forward(&mut self) -> Result<Option<String>, String> {
        AgentSession::cycle_model_forward(self).map(|maybe_model| {
            maybe_model.map(|model| format!("model: {}/{}", model.provider, model.id))
        })
    }

    fn cycle_model_backward(&mut self) -> Result<Option<String>, String> {
        AgentSession::cycle_model_backward(self).map(|maybe_model| {
            maybe_model.map(|model| format!("model: {}/{}", model.provider, model.id))
        })
    }

    fn select_model(&mut self) -> Result<Option<String>, String> {
        AgentSession::select_model(self).map(|maybe_model| {
            maybe_model.map(|model| format!("model: {}/{}", model.provider, model.id))
        })
    }

    fn cycle_mode(&mut self) -> Result<Option<String>, String> {
        let mode = AgentSession::cycle_mode(self);
        Ok(Some(format!("mode: {}", mode.label())))
    }

    fn recent_resumable_sessions(
        &mut self,
        limit: usize,
    ) -> Result<Option<Vec<ResumeCandidate>>, String> {
        AgentSession::recent_resumable_session_candidates(self, limit).map(|candidates| {
            Some(
                candidates
                    .into_iter()
                    .map(|candidate| ResumeCandidate {
                        session_ref: candidate.path.display().to_string(),
                        title: candidate.title,
                        updated_at: candidate.updated_at,
                    })
                    .collect(),
            )
        })
    }

    fn resume_session(&mut self, session_ref: Option<&str>) -> Result<Option<String>, String> {
        AgentSession::resume(self, session_ref)
            .map(|path| Some(format!("session: {}", path.display())))
    }

    fn new_session(&mut self) -> Result<Option<String>, String> {
        AgentSession::start_new_session(self)
            .map(|path| Some(format!("session: {}", path.display())))
    }

    fn session_messages(&self) -> Option<Vec<pixy_ai::Message>> {
        Some(AgentSession::build_session_context(self).messages)
    }

    fn session_file(&self) -> Option<PathBuf> {
        AgentSession::session_file(self).cloned()
    }
}

impl TuiBackend for CliSession {
    fn prompt<'a>(&'a mut self, input: &'a str) -> BackendFuture<'a> {
        Box::pin(async move {
            let session = self.ensure_session()?;
            AgentSession::prompt(session, input).await
        })
    }

    fn continue_run<'a>(&'a mut self) -> BackendFuture<'a> {
        Box::pin(async move {
            let session = self.ensure_session()?;
            AgentSession::continue_run(session).await
        })
    }

    fn prompt_stream<'a>(
        &'a mut self,
        input: &'a str,
        abort_signal: Option<AgentAbortSignal>,
        on_update: &'a mut dyn FnMut(StreamUpdate),
    ) -> BackendFuture<'a> {
        Box::pin(async move {
            let session = self.ensure_session()?;
            let mut mapper = ThinkingStreamMapper::default();
            AgentSession::prompt_streaming_with_abort(session, input, abort_signal, |update| {
                if let Some(mapped) = mapper.map(update) {
                    on_update(mapped);
                }
            })
            .await
        })
    }

    fn prompt_stream_with_blocks<'a>(
        &'a mut self,
        input: &'a str,
        blocks: Option<Vec<pixy_ai::UserContentBlock>>,
        abort_signal: Option<AgentAbortSignal>,
        on_update: &'a mut dyn FnMut(StreamUpdate),
    ) -> BackendFuture<'a> {
        Box::pin(async move {
            let session = self.ensure_session()?;
            let mut mapper = ThinkingStreamMapper::default();
            AgentSession::prompt_streaming_blocks_with_abort(
                session,
                input,
                blocks,
                abort_signal,
                |update| {
                    if let Some(mapped) = mapper.map(update) {
                        on_update(mapped);
                    }
                },
            )
            .await
        })
    }

    fn continue_run_stream<'a>(
        &'a mut self,
        abort_signal: Option<AgentAbortSignal>,
        on_update: &'a mut dyn FnMut(StreamUpdate),
    ) -> BackendFuture<'a> {
        Box::pin(async move {
            let session = self.ensure_session()?;
            let mut mapper = ThinkingStreamMapper::default();
            AgentSession::continue_run_streaming_with_abort(session, abort_signal, |update| {
                if let Some(mapped) = mapper.map(update) {
                    on_update(mapped);
                }
            })
            .await
        })
    }

    fn cycle_model_forward(&mut self) -> Result<Option<String>, String> {
        let session = self.ensure_session()?;
        AgentSession::cycle_model_forward(session).map(|maybe_model| {
            maybe_model.map(|model| format!("model: {}/{}", model.provider, model.id))
        })
    }

    fn cycle_model_backward(&mut self) -> Result<Option<String>, String> {
        let session = self.ensure_session()?;
        AgentSession::cycle_model_backward(session).map(|maybe_model| {
            maybe_model.map(|model| format!("model: {}/{}", model.provider, model.id))
        })
    }

    fn select_model(&mut self) -> Result<Option<String>, String> {
        let session = self.ensure_session()?;
        AgentSession::select_model(session).map(|maybe_model| {
            maybe_model.map(|model| format!("model: {}/{}", model.provider, model.id))
        })
    }

    fn cycle_mode(&mut self) -> Result<Option<String>, String> {
        let session = self.ensure_session()?;
        let mode = AgentSession::cycle_mode(session);
        Ok(Some(format!("mode: {}", mode.label())))
    }

    fn recent_resumable_sessions(
        &mut self,
        limit: usize,
    ) -> Result<Option<Vec<ResumeCandidate>>, String> {
        self.recent_resumable_session_candidates(limit)
            .map(|candidates| {
                Some(
                    candidates
                        .into_iter()
                        .map(|candidate| ResumeCandidate {
                            session_ref: candidate.path.display().to_string(),
                            title: candidate.title,
                            updated_at: candidate.updated_at,
                        })
                        .collect(),
                )
            })
    }

    fn resume_session(&mut self, session_ref: Option<&str>) -> Result<Option<String>, String> {
        self.resume(session_ref)
            .map(|path| Some(format!("session: {}", path.display())))
    }

    fn new_session(&mut self) -> Result<Option<String>, String> {
        self.start_new_session()
            .map(|path| Some(format!("session: {}", path.display())))
    }

    fn session_messages(&self) -> Option<Vec<pixy_ai::Message>> {
        self.session_messages()
    }

    fn session_file(&self) -> Option<PathBuf> {
        self.session_file()
    }
}

#[derive(Default)]
struct ThinkingStreamMapper {
    thinking_buffer: String,
}

impl ThinkingStreamMapper {
    fn map(&mut self, update: AgentSessionStreamUpdate) -> Option<StreamUpdate> {
        match update {
            AgentSessionStreamUpdate::AssistantTextDelta(delta) => {
                self.thinking_buffer.clear();
                Some(StreamUpdate::AssistantTextDelta(delta))
            }
            AgentSessionStreamUpdate::AssistantLine(line) => self.map_assistant_line(line),
            AgentSessionStreamUpdate::ToolLine(line) => {
                self.thinking_buffer.clear();
                Some(StreamUpdate::ToolLine(line))
            }
        }
    }

    fn map_assistant_line(&mut self, line: String) -> Option<StreamUpdate> {
        let Some(next_thinking) = parse_thinking_line_content(line.as_str()) else {
            self.thinking_buffer.clear();
            return Some(StreamUpdate::AssistantLine(line));
        };

        if let Some(delta) = next_thinking.strip_prefix(self.thinking_buffer.as_str()) {
            self.thinking_buffer = next_thinking.to_string();
            if delta.is_empty() {
                None
            } else {
                Some(StreamUpdate::AssistantThinkingDelta(delta.to_string()))
            }
        } else {
            self.thinking_buffer = next_thinking.to_string();
            Some(StreamUpdate::AssistantLine(line))
        }
    }
}

fn parse_thinking_line_content(line: &str) -> Option<&str> {
    line.strip_prefix("[thinking]")
        .map(|rest| rest.strip_prefix(' ').unwrap_or(rest))
}

#[cfg(test)]
mod tests {
    use super::{AgentSessionStreamUpdate, StreamUpdate, ThinkingStreamMapper};

    #[test]
    fn mapper_turns_prefix_growing_thinking_snapshots_into_deltas() {
        let mut mapper = ThinkingStreamMapper::default();

        let first = mapper.map(AgentSessionStreamUpdate::AssistantLine(
            "[thinking] Ana".to_string(),
        ));
        let second = mapper.map(AgentSessionStreamUpdate::AssistantLine(
            "[thinking] Analyzing".to_string(),
        ));

        assert_eq!(
            first,
            Some(StreamUpdate::AssistantThinkingDelta("Ana".to_string()))
        );
        assert_eq!(
            second,
            Some(StreamUpdate::AssistantThinkingDelta("lyzing".to_string()))
        );
    }

    #[test]
    fn mapper_suppresses_duplicate_thinking_snapshot() {
        let mut mapper = ThinkingStreamMapper::default();

        let first = mapper.map(AgentSessionStreamUpdate::AssistantLine(
            "[thinking] same".to_string(),
        ));
        let duplicate = mapper.map(AgentSessionStreamUpdate::AssistantLine(
            "[thinking] same".to_string(),
        ));

        assert_eq!(
            first,
            Some(StreamUpdate::AssistantThinkingDelta("same".to_string()))
        );
        assert_eq!(duplicate, None);
    }

    #[test]
    fn mapper_falls_back_to_full_line_when_snapshot_is_not_prefix_growth() {
        let mut mapper = ThinkingStreamMapper::default();
        let _ = mapper.map(AgentSessionStreamUpdate::AssistantLine(
            "[thinking] abc".to_string(),
        ));

        let replacement = mapper.map(AgentSessionStreamUpdate::AssistantLine(
            "[thinking] ax".to_string(),
        ));

        assert_eq!(
            replacement,
            Some(StreamUpdate::AssistantLine("[thinking] ax".to_string()))
        );
    }
}
