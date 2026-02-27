use std::path::PathBuf;

use pixy_agent_core::AgentAbortSignal;
use pixy_tui::{BackendFuture, ResumeCandidate, StreamUpdate, TuiBackend};

use crate::{AgentSession, AgentSessionStreamUpdate, cli_app::CliSession};

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
            AgentSession::prompt_streaming_with_abort(self, input, abort_signal, |update| {
                on_update(map_stream_update(update))
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
            AgentSession::prompt_streaming_blocks_with_abort(
                self,
                input,
                blocks,
                abort_signal,
                |update| on_update(map_stream_update(update)),
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
            AgentSession::continue_run_streaming_with_abort(self, abort_signal, |update| {
                on_update(map_stream_update(update))
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
            AgentSession::prompt_streaming_with_abort(session, input, abort_signal, |update| {
                on_update(map_stream_update(update))
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
            AgentSession::prompt_streaming_blocks_with_abort(
                session,
                input,
                blocks,
                abort_signal,
                |update| on_update(map_stream_update(update)),
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
            AgentSession::continue_run_streaming_with_abort(session, abort_signal, |update| {
                on_update(map_stream_update(update))
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

fn map_stream_update(update: AgentSessionStreamUpdate) -> StreamUpdate {
    match update {
        AgentSessionStreamUpdate::AssistantTextDelta(delta) => {
            StreamUpdate::AssistantTextDelta(delta)
        }
        AgentSessionStreamUpdate::AssistantLine(line) => StreamUpdate::AssistantLine(line),
        AgentSessionStreamUpdate::ToolLine(line) => StreamUpdate::ToolLine(line),
    }
}
