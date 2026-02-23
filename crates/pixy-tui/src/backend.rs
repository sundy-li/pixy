use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;

use pixy_agent_core::AgentAbortSignal;
use pixy_ai::Message;

pub type BackendFuture<'a> = Pin<Box<dyn Future<Output = Result<Vec<Message>, String>> + 'a>>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StreamUpdate {
    AssistantTextDelta(String),
    AssistantLine(String),
    ToolLine(String),
}

pub trait TuiBackend {
    fn prompt<'a>(&'a mut self, input: &'a str) -> BackendFuture<'a>;
    fn continue_run<'a>(&'a mut self) -> BackendFuture<'a>;
    fn prompt_stream<'a>(
        &'a mut self,
        input: &'a str,
        abort_signal: Option<AgentAbortSignal>,
        on_update: &'a mut dyn FnMut(StreamUpdate),
    ) -> BackendFuture<'a>;
    fn continue_run_stream<'a>(
        &'a mut self,
        abort_signal: Option<AgentAbortSignal>,
        on_update: &'a mut dyn FnMut(StreamUpdate),
    ) -> BackendFuture<'a>;
    fn cycle_model_forward(&mut self) -> Result<Option<String>, String> {
        Ok(None)
    }
    fn cycle_model_backward(&mut self) -> Result<Option<String>, String> {
        Ok(None)
    }
    fn select_model(&mut self) -> Result<Option<String>, String> {
        Ok(None)
    }
    fn resume_session(&mut self, _session_ref: Option<&str>) -> Result<Option<String>, String> {
        Ok(None)
    }
    fn session_file(&self) -> Option<PathBuf>;
}
