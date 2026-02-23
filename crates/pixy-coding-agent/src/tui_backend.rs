use std::path::PathBuf;

use pixy_agent_core::AgentAbortSignal;
use pixy_tui::{BackendFuture, StreamUpdate, TuiBackend};

use crate::{AgentSession, AgentSessionStreamUpdate};

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

    fn resume_session(&mut self, session_ref: Option<&str>) -> Result<Option<String>, String> {
        AgentSession::resume(self, session_ref)
            .map(|path| Some(format!("session: {}", path.display())))
    }

    fn session_file(&self) -> Option<PathBuf> {
        AgentSession::session_file(self).cloned()
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
