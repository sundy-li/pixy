use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use pixy_agent_core::{
    AgentAbortSignal, AgentContext, AgentEvent, AgentLoopConfig, AgentMessage, AgentRetryConfig,
    AgentTool, StreamFn, agent_loop, agent_loop_continue,
};
use pixy_ai::{
    AssistantContentBlock, AssistantMessageEvent, Context as LlmContext, Message, Model,
    StopReason, ToolResultContentBlock, Usage, UserContent, UserContentBlock,
};
use serde_json::Value;

use crate::{SessionContext, SessionManager};

const AUTO_COMPACTION_SUMMARIZATION_SYSTEM_PROMPT: &str = "You are a context summarization assistant. Summarize conversation history for another coding assistant.";
const AUTO_COMPACTION_SUMMARIZATION_PROMPT: &str = "Summarize the conversation above so another LLM can continue the task. Include: user goal, completed work, current status, and concrete next steps. Preserve exact file paths, commands, and error messages where relevant. Keep it concise.";

pub struct AgentSessionConfig {
    pub model: Model,
    pub system_prompt: String,
    pub stream_fn: StreamFn,
    pub tools: Vec<AgentTool>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AgentSessionStreamUpdate {
    AssistantTextDelta(String),
    AssistantLine(String),
    ToolLine(String),
}

#[derive(Clone, Debug, PartialEq)]
pub struct AutoCompactionConfig {
    pub enabled: bool,
    pub reserve_tokens: u64,
    pub keep_recent_messages: usize,
    pub max_summary_chars: usize,
}

impl Default for AutoCompactionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            reserve_tokens: 16_384,
            keep_recent_messages: 8,
            max_summary_chars: 2_000,
        }
    }
}

pub struct AgentSession {
    session_manager: SessionManager,
    config: AgentSessionConfig,
    auto_compaction: AutoCompactionConfig,
    model_catalog: Vec<Model>,
    current_model_index: usize,
    retry_config: AgentRetryConfig,
}

impl AgentSession {
    pub fn new(session_manager: SessionManager, config: AgentSessionConfig) -> Self {
        let current_model = config.model.clone();
        Self {
            session_manager,
            config,
            auto_compaction: AutoCompactionConfig::default(),
            model_catalog: vec![current_model],
            current_model_index: 0,
            retry_config: AgentRetryConfig::default(),
        }
    }

    pub fn session_file(&self) -> Option<&PathBuf> {
        self.session_manager.session_file()
    }

    pub fn build_session_context(&self) -> SessionContext {
        self.session_manager.build_session_context()
    }

    pub fn resume(&mut self, target: Option<&str>) -> Result<PathBuf, String> {
        let target_path =
            resolve_resume_session_target(target, self.session_manager.session_file().cloned())?;
        let loaded = SessionManager::load(&target_path)?;
        self.session_manager = loaded;
        self.sync_model_from_session_state();
        Ok(target_path)
    }

    pub fn auto_compaction_config(&self) -> &AutoCompactionConfig {
        &self.auto_compaction
    }

    pub fn current_model(&self) -> &Model {
        &self.config.model
    }

    pub fn model_catalog(&self) -> &[Model] {
        &self.model_catalog
    }

    pub fn set_auto_compaction_config(&mut self, config: AutoCompactionConfig) {
        self.auto_compaction = config;
    }

    pub fn retry_config(&self) -> &AgentRetryConfig {
        &self.retry_config
    }

    pub fn set_retry_config(&mut self, retry_config: AgentRetryConfig) {
        self.retry_config = retry_config;
    }

    pub fn set_model_catalog(&mut self, models: Vec<Model>) {
        let current_provider = self.config.model.provider.clone();
        let current_model_id = self.config.model.id.clone();

        let mut catalog = Vec::new();
        for model in models {
            if catalog.iter().any(|existing: &Model| {
                existing.provider == model.provider && existing.id == model.id
            }) {
                continue;
            }
            catalog.push(model);
        }

        if catalog.is_empty() {
            catalog.push(self.config.model.clone());
            self.model_catalog = catalog;
            self.current_model_index = 0;
            return;
        }

        if let Some(position) = catalog
            .iter()
            .position(|model| model.provider == current_provider && model.id == current_model_id)
        {
            self.current_model_index = position;
            self.model_catalog = catalog;
            return;
        }

        catalog.insert(0, self.config.model.clone());
        self.model_catalog = catalog;
        self.current_model_index = 0;
    }

    pub fn cycle_model_forward(&mut self) -> Result<Option<Model>, String> {
        self.cycle_model(true)
    }

    pub fn cycle_model_backward(&mut self) -> Result<Option<Model>, String> {
        self.cycle_model(false)
    }

    pub fn select_model(&mut self) -> Result<Option<Model>, String> {
        // Current TUI has no selector popup yet, so map select action to next model.
        self.cycle_model_forward()
    }

    pub async fn prompt(&mut self, input: &str) -> Result<Vec<AgentMessage>, String> {
        self.prompt_internal(input, true).await
    }

    pub async fn prompt_streaming<F>(
        &mut self,
        input: &str,
        on_update: F,
    ) -> Result<Vec<AgentMessage>, String>
    where
        F: FnMut(AgentSessionStreamUpdate),
    {
        self.prompt_streaming_with_abort(input, None, on_update)
            .await
    }

    pub async fn prompt_streaming_with_abort<F>(
        &mut self,
        input: &str,
        abort_signal: Option<AgentAbortSignal>,
        mut on_update: F,
    ) -> Result<Vec<AgentMessage>, String>
    where
        F: FnMut(AgentSessionStreamUpdate),
    {
        let mut produced = self
            .run_prompt_once_streaming(input, abort_signal, Some(&mut on_update))
            .await?;
        if let Some(mut retry_messages) = self.maybe_handle_overflow_and_retry(&produced).await? {
            for update in render_messages_for_streaming(&retry_messages) {
                on_update(update);
            }
            produced.append(&mut retry_messages);
        }
        Ok(produced)
    }

    async fn prompt_internal(
        &mut self,
        input: &str,
        allow_overflow_retry: bool,
    ) -> Result<Vec<AgentMessage>, String> {
        let mut produced = self.run_prompt_once(input).await?;
        if allow_overflow_retry {
            if let Some(mut retry_messages) =
                self.maybe_handle_overflow_and_retry(&produced).await?
            {
                produced.append(&mut retry_messages);
            }
        }
        Ok(produced)
    }

    async fn run_prompt_once(&mut self, input: &str) -> Result<Vec<AgentMessage>, String> {
        let prompt = Message::User {
            content: UserContent::Text(input.to_string()),
            timestamp: now_millis(),
        };

        let context = self.agent_context_from_session();
        let stream = agent_loop(vec![prompt], context, self.loop_config(), None);
        let produced = stream
            .result()
            .await
            .ok_or_else(|| "Agent loop ended without a final result".to_string())?;

        self.persist_messages_and_maybe_compact(&produced).await?;
        Ok(produced)
    }

    async fn run_prompt_once_streaming(
        &mut self,
        input: &str,
        abort_signal: Option<AgentAbortSignal>,
        on_update: Option<&mut dyn FnMut(AgentSessionStreamUpdate)>,
    ) -> Result<Vec<AgentMessage>, String> {
        let prompt = Message::User {
            content: UserContent::Text(input.to_string()),
            timestamp: now_millis(),
        };

        let context = self.agent_context_from_session();
        let stream = agent_loop(vec![prompt], context, self.loop_config(), abort_signal);
        let produced = collect_agent_loop_result(stream, on_update).await?;

        self.persist_messages_and_maybe_compact(&produced).await?;
        Ok(produced)
    }

    pub async fn continue_run(&mut self) -> Result<Vec<AgentMessage>, String> {
        self.continue_internal(true).await
    }

    pub async fn continue_run_streaming<F>(
        &mut self,
        on_update: F,
    ) -> Result<Vec<AgentMessage>, String>
    where
        F: FnMut(AgentSessionStreamUpdate),
    {
        self.continue_run_streaming_with_abort(None, on_update)
            .await
    }

    pub async fn continue_run_streaming_with_abort<F>(
        &mut self,
        abort_signal: Option<AgentAbortSignal>,
        mut on_update: F,
    ) -> Result<Vec<AgentMessage>, String>
    where
        F: FnMut(AgentSessionStreamUpdate),
    {
        let mut produced = self
            .run_continue_once_streaming(abort_signal, Some(&mut on_update))
            .await?;
        if let Some(mut retry_messages) = self.maybe_handle_overflow_and_retry(&produced).await? {
            for update in render_messages_for_streaming(&retry_messages) {
                on_update(update);
            }
            produced.append(&mut retry_messages);
        }
        Ok(produced)
    }

    async fn continue_internal(
        &mut self,
        allow_overflow_retry: bool,
    ) -> Result<Vec<AgentMessage>, String> {
        let mut produced = self.run_continue_once().await?;
        if allow_overflow_retry {
            if let Some(mut retry_messages) =
                self.maybe_handle_overflow_and_retry(&produced).await?
            {
                produced.append(&mut retry_messages);
            }
        }
        Ok(produced)
    }

    async fn run_continue_once(&mut self) -> Result<Vec<AgentMessage>, String> {
        let context = self.agent_context_from_session();
        if context.messages.is_empty() {
            return Err("No messages to continue from".to_string());
        }

        let stream = if matches!(context.messages.last(), Some(Message::Assistant { .. })) {
            agent_loop(vec![], context, self.loop_config(), None)
        } else {
            agent_loop_continue(context, self.loop_config(), None)
        };
        let produced = stream
            .result()
            .await
            .ok_or_else(|| "Agent loop ended without a final result".to_string())?;

        self.persist_messages_and_maybe_compact(&produced).await?;
        Ok(produced)
    }

    async fn run_continue_once_streaming(
        &mut self,
        abort_signal: Option<AgentAbortSignal>,
        on_update: Option<&mut dyn FnMut(AgentSessionStreamUpdate)>,
    ) -> Result<Vec<AgentMessage>, String> {
        let context = self.agent_context_from_session();
        if context.messages.is_empty() {
            return Err("No messages to continue from".to_string());
        }

        let stream = if matches!(context.messages.last(), Some(Message::Assistant { .. })) {
            agent_loop(vec![], context, self.loop_config(), abort_signal)
        } else {
            agent_loop_continue(context, self.loop_config(), abort_signal)
        };
        let produced = collect_agent_loop_result(stream, on_update).await?;

        self.persist_messages_and_maybe_compact(&produced).await?;
        Ok(produced)
    }

    fn cycle_model(&mut self, forward: bool) -> Result<Option<Model>, String> {
        if self.model_catalog.len() <= 1 {
            return Ok(None);
        }

        let len = self.model_catalog.len();
        let next_index = if forward {
            (self.current_model_index + 1) % len
        } else if self.current_model_index == 0 {
            len - 1
        } else {
            self.current_model_index - 1
        };
        let next_model = self.model_catalog[next_index].clone();

        self.switch_model(next_index, next_model).map(Some)
    }

    fn sync_model_from_session_state(&mut self) {
        let Some((provider, model_id)) = self.session_manager.latest_model_change() else {
            return;
        };

        if let Some(index) = self
            .model_catalog
            .iter()
            .position(|model| model.provider == provider && model.id == model_id)
        {
            self.current_model_index = index;
            self.config.model = self.model_catalog[index].clone();
        }
    }

    fn switch_model(&mut self, index: usize, model: Model) -> Result<Model, String> {
        let previous_index = self.current_model_index;
        let previous_model = self.config.model.clone();

        self.current_model_index = index;
        self.config.model = model.clone();
        match self
            .session_manager
            .append_model_change(&model.provider, &model.id)
        {
            Ok(_) => Ok(model),
            Err(error) => {
                self.current_model_index = previous_index;
                self.config.model = previous_model;
                Err(error)
            }
        }
    }

    pub fn compact(
        &mut self,
        summary: &str,
        first_kept_entry_id: Option<&str>,
        tokens_before: u64,
    ) -> Result<String, String> {
        self.session_manager
            .append_compaction(summary, first_kept_entry_id, tokens_before)
    }

    pub fn compact_keep_recent(
        &mut self,
        summary: &str,
        keep_recent_messages: usize,
        tokens_before: u64,
    ) -> Result<Option<String>, String> {
        let Some(first_kept_entry_id) = self
            .session_manager
            .first_kept_entry_id_for_recent_messages(keep_recent_messages)
        else {
            return Ok(None);
        };

        self.session_manager
            .append_compaction(summary, Some(&first_kept_entry_id), tokens_before)
            .map(Some)
    }

    fn loop_config(&self) -> AgentLoopConfig {
        let fallback_models = self
            .model_catalog
            .iter()
            .enumerate()
            .filter_map(|(index, model)| {
                if index == self.current_model_index {
                    None
                } else {
                    Some(model.clone())
                }
            })
            .collect::<Vec<_>>();
        AgentLoopConfig {
            model: self.config.model.clone(),
            fallback_models,
            convert_to_llm: Arc::new(|messages: Vec<AgentMessage>| messages),
            stream_fn: self.config.stream_fn.clone(),
            retry: self.retry_config.clone(),
            get_steering_messages: None,
            get_follow_up_messages: None,
        }
    }

    fn agent_context_from_session(&self) -> AgentContext {
        let context = self.session_manager.build_session_context();
        AgentContext {
            system_prompt: self.config.system_prompt.clone(),
            messages: context.messages,
            tools: self.config.tools.clone(),
        }
    }

    async fn persist_messages_and_maybe_compact(
        &mut self,
        produced: &[AgentMessage],
    ) -> Result<(), String> {
        for message in produced {
            self.session_manager.append_message(message.clone())?;
        }
        let _ = self.maybe_auto_compact(produced).await?;
        Ok(())
    }

    async fn maybe_handle_overflow_and_retry(
        &mut self,
        produced: &[AgentMessage],
    ) -> Result<Option<Vec<AgentMessage>>, String> {
        if !self.auto_compaction.enabled {
            return Ok(None);
        }

        let Some(assistant_message) = latest_assistant_message(produced) else {
            return Ok(None);
        };

        let context_window = self.config.model.context_window as u64;
        if !is_context_overflow_message(assistant_message, context_window) {
            return Ok(None);
        }

        if !self.session_manager.rewind_leaf_if_last_assistant_error() {
            return Ok(None);
        }

        let compacted = self.auto_compact_for_overflow(assistant_message).await?;
        if !compacted {
            return Ok(None);
        }

        self.run_continue_once().await.map(Some)
    }

    async fn auto_compact_for_overflow(
        &mut self,
        assistant_message: &Message,
    ) -> Result<bool, String> {
        let context_window = self.config.model.context_window as u64;
        if context_window == 0 {
            return Ok(false);
        }

        let session_context = self.session_manager.build_session_context();
        let keep_recent_messages = self.auto_compaction.keep_recent_messages.max(1);
        if session_context.messages.len() <= keep_recent_messages {
            return Ok(false);
        }

        let summarize_upto = session_context.messages.len() - keep_recent_messages;
        let context_tokens = overflow_context_tokens(assistant_message, context_window);
        let summary = self
            .build_auto_compaction_summary_with_fallback(
                &session_context.messages[..summarize_upto],
                context_tokens,
                context_window,
            )
            .await;

        Ok(self
            .compact_keep_recent(&summary, keep_recent_messages, context_tokens)?
            .is_some())
    }

    async fn maybe_auto_compact(
        &mut self,
        produced: &[AgentMessage],
    ) -> Result<Option<String>, String> {
        if !self.auto_compaction.enabled {
            return Ok(None);
        }

        let Some(context_tokens) = latest_context_tokens_from_messages(produced) else {
            return Ok(None);
        };

        let context_window = self.config.model.context_window as u64;
        if context_window == 0 {
            return Ok(None);
        }

        let compact_threshold = context_window.saturating_sub(self.auto_compaction.reserve_tokens);
        if context_tokens <= compact_threshold {
            return Ok(None);
        }

        let session_context = self.session_manager.build_session_context();
        let keep_recent_messages = self.auto_compaction.keep_recent_messages.max(1);
        if session_context.messages.len() <= keep_recent_messages {
            return Ok(None);
        }

        let summarize_upto = session_context.messages.len() - keep_recent_messages;
        let summary = self
            .build_auto_compaction_summary_with_fallback(
                &session_context.messages[..summarize_upto],
                context_tokens,
                context_window,
            )
            .await;

        self.compact_keep_recent(&summary, keep_recent_messages, context_tokens)
    }

    async fn build_auto_compaction_summary_with_fallback(
        &self,
        messages_to_summarize: &[Message],
        context_tokens: u64,
        context_window: u64,
    ) -> String {
        match self
            .try_generate_llm_compaction_summary(
                messages_to_summarize,
                context_tokens,
                context_window,
            )
            .await
        {
            Ok(summary) if !summary.trim().is_empty() => {
                truncate_chars(summary.trim(), self.auto_compaction.max_summary_chars)
            }
            _ => build_auto_compaction_summary(
                messages_to_summarize,
                context_tokens,
                context_window,
                self.auto_compaction.max_summary_chars,
            ),
        }
    }

    async fn try_generate_llm_compaction_summary(
        &self,
        messages_to_summarize: &[Message],
        context_tokens: u64,
        context_window: u64,
    ) -> Result<String, String> {
        if messages_to_summarize.is_empty() {
            return Err("No messages available for summarization".to_string());
        }

        let conversation = serialize_messages_for_summary(messages_to_summarize);
        if conversation.trim().is_empty() {
            return Err("No textual content to summarize".to_string());
        }

        let prompt = format!(
            "Context tokens before compaction: {context_tokens}/{context_window}.\n\n<conversation>\n{conversation}\n</conversation>\n\n{AUTO_COMPACTION_SUMMARIZATION_PROMPT}"
        );
        let summary_context = LlmContext {
            system_prompt: Some(AUTO_COMPACTION_SUMMARIZATION_SYSTEM_PROMPT.to_string()),
            messages: vec![Message::User {
                content: UserContent::Text(prompt),
                timestamp: now_millis(),
            }],
            tools: None,
        };

        let stream_fn = self.config.stream_fn.clone();
        let model = self.config.model.clone();
        let response =
            tokio::task::spawn_blocking(move || (stream_fn)(model, summary_context, None))
                .await
                .map_err(|error| format!("Compaction summary stream task failed: {error}"))??;

        let summary_message = response
            .result()
            .await
            .ok_or_else(|| "Compaction summary stream ended without final result".to_string())?;

        if matches!(
            summary_message.stop_reason,
            StopReason::Error | StopReason::Aborted
        ) {
            return Err(summary_message
                .error_message
                .unwrap_or_else(|| "Compaction summary model returned an error".to_string()));
        }

        let summary = summary_message
            .content
            .iter()
            .filter_map(|block| match block {
                AssistantContentBlock::Text { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");

        if summary.trim().is_empty() {
            return Err("Compaction summary model returned empty text".to_string());
        }

        Ok(summary)
    }
}

async fn collect_agent_loop_result(
    stream: pixy_ai::EventStream<AgentEvent, Vec<AgentMessage>>,
    mut on_update: Option<&mut dyn FnMut(AgentSessionStreamUpdate)>,
) -> Result<Vec<AgentMessage>, String> {
    let mut saw_assistant_text_delta = false;

    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::MessageStart { message } => {
                if matches!(message, Message::Assistant { .. }) {
                    saw_assistant_text_delta = false;
                }
            }
            AgentEvent::ToolExecutionStart {
                tool_name, args, ..
            } => {
                if let Some(callback) = on_update.as_mut() {
                    callback(AgentSessionStreamUpdate::ToolLine(format_tool_start_line(
                        &tool_name, &args,
                    )));
                }
            }
            AgentEvent::MessageUpdate {
                assistant_message_event,
                ..
            } => {
                if let Some(callback) = on_update.as_mut() {
                    match assistant_message_event {
                        AssistantMessageEvent::TextDelta { delta, .. } => {
                            callback(AgentSessionStreamUpdate::AssistantTextDelta(delta));
                            saw_assistant_text_delta = true;
                        }
                        AssistantMessageEvent::ThinkingDelta { delta, .. } => {
                            callback(AgentSessionStreamUpdate::AssistantLine(format!(
                                "[thinking] {delta}"
                            )));
                        }
                        _ => {}
                    }
                }
            }
            AgentEvent::MessageEnd { message } => {
                if let Some(callback) = on_update.as_mut() {
                    if let Message::Assistant { .. } = &message {
                        for update in render_assistant_message_for_streaming(
                            &message,
                            saw_assistant_text_delta,
                        ) {
                            callback(update);
                        }
                    } else if let Message::ToolResult { content, .. } = &message {
                        for block in content {
                            match block {
                                ToolResultContentBlock::Text { text, .. } => {
                                    callback(AgentSessionStreamUpdate::ToolLine(text.clone()))
                                }
                                ToolResultContentBlock::Image { .. } => {
                                    callback(AgentSessionStreamUpdate::ToolLine(
                                        "(image tool result omitted)".to_string(),
                                    ))
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    stream
        .result()
        .await
        .ok_or_else(|| "Agent loop ended without a final result".to_string())
}

fn render_messages_for_streaming(messages: &[AgentMessage]) -> Vec<AgentSessionStreamUpdate> {
    let mut updates = vec![];
    for message in messages {
        match message {
            Message::Assistant { .. } => {
                updates.extend(render_assistant_message_for_streaming(message, false));
            }
            Message::ToolResult {
                tool_name,
                content,
                is_error,
                ..
            } => {
                let title = if *is_error {
                    format!("• Ran {tool_name} (error)")
                } else {
                    format!("• Ran {tool_name}")
                };
                updates.push(AgentSessionStreamUpdate::ToolLine(title));
                for block in content {
                    match block {
                        ToolResultContentBlock::Text { text, .. } => {
                            updates.push(AgentSessionStreamUpdate::ToolLine(text.clone()));
                        }
                        ToolResultContentBlock::Image { .. } => {
                            updates.push(AgentSessionStreamUpdate::ToolLine(
                                "(image tool result omitted)".to_string(),
                            ));
                        }
                    }
                }
            }
            Message::User { .. } => {}
        }
    }
    updates
}

fn format_tool_start_line(tool_name: &str, args: &Value) -> String {
    match tool_name {
        "bash" => format_bash_tool_start_line(args),
        "read" | "write" | "edit" => format_path_tool_start_line(tool_name, args),
        _ => format!("• Ran {tool_name}"),
    }
}

fn format_bash_tool_start_line(args: &Value) -> String {
    let Some(command) = args
        .get("command")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return "• Ran bash".to_string();
    };

    format!("• Ran bash -lc '{}'", shell_quote_single(command))
}

fn format_path_tool_start_line(tool_name: &str, args: &Value) -> String {
    let Some(path) = args
        .get("path")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return format!("• Ran {tool_name}");
    };
    format!("• Ran {tool_name} {path}")
}

fn shell_quote_single(value: &str) -> String {
    value.replace('\'', r"'\''")
}

fn render_assistant_message_for_streaming(
    message: &Message,
    had_text_delta: bool,
) -> Vec<AgentSessionStreamUpdate> {
    let mut updates = vec![];
    let Message::Assistant {
        content,
        stop_reason,
        error_message,
        ..
    } = message
    else {
        return updates;
    };

    if had_text_delta {
        updates.push(AgentSessionStreamUpdate::AssistantLine(String::new()));
    } else {
        for block in content {
            match block {
                AssistantContentBlock::Text { text, .. } => {
                    if !text.trim().is_empty() {
                        updates.push(AgentSessionStreamUpdate::AssistantLine(text.clone()));
                    }
                }
                AssistantContentBlock::Thinking { thinking, .. } => {
                    if !thinking.trim().is_empty() {
                        updates.push(AgentSessionStreamUpdate::AssistantLine(format!(
                            "[thinking] {thinking}"
                        )));
                    }
                }
                AssistantContentBlock::ToolCall { .. } => {}
            }
        }
    }

    if matches!(stop_reason, StopReason::Error | StopReason::Aborted) {
        if let Some(error_message) = error_message {
            updates.push(AgentSessionStreamUpdate::AssistantLine(format!(
                "[assistant_{}] {error_message}",
                stop_reason_label(stop_reason)
            )));
        }
    }

    updates
}

fn stop_reason_label(reason: &StopReason) -> &'static str {
    match reason {
        StopReason::Stop => "stop",
        StopReason::Length => "length",
        StopReason::ToolUse => "tool_use",
        StopReason::Error => "error",
        StopReason::Aborted => "aborted",
    }
}

fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0)
}

fn latest_context_tokens_from_messages(messages: &[AgentMessage]) -> Option<u64> {
    messages.iter().rev().find_map(|message| match message {
        Message::Assistant {
            usage, stop_reason, ..
        } if !matches!(stop_reason, StopReason::Error | StopReason::Aborted) => {
            Some(context_tokens_from_usage(usage))
        }
        _ => None,
    })
}

fn context_tokens_from_usage(usage: &Usage) -> u64 {
    if usage.total_tokens > 0 {
        usage.total_tokens
    } else {
        usage.input + usage.output + usage.cache_read + usage.cache_write
    }
}

fn latest_assistant_message(messages: &[AgentMessage]) -> Option<&Message> {
    messages
        .iter()
        .rev()
        .find(|message| matches!(message, Message::Assistant { .. }))
}

fn overflow_context_tokens(message: &Message, context_window: u64) -> u64 {
    match message {
        Message::Assistant { usage, .. } => {
            let tokens = context_tokens_from_usage(usage);
            if tokens > 0 {
                tokens
            } else {
                context_window.max(1)
            }
        }
        _ => context_window.max(1),
    }
}

fn is_context_overflow_message(message: &Message, context_window: u64) -> bool {
    match message {
        Message::Assistant {
            stop_reason,
            error_message,
            usage,
            ..
        } => {
            if matches!(stop_reason, StopReason::Error) {
                if let Some(error) = error_message {
                    if is_context_overflow_error_text(error) {
                        return true;
                    }
                    if is_context_overflow_status_no_body(error) {
                        return true;
                    }
                }
            }

            if matches!(stop_reason, StopReason::Stop) && context_window > 0 {
                let input_tokens = usage.input + usage.cache_read;
                return input_tokens > context_window;
            }

            false
        }
        _ => false,
    }
}

fn is_context_overflow_error_text(error: &str) -> bool {
    let normalized = error.to_ascii_lowercase();
    let patterns = [
        "prompt is too long",
        "input is too long for requested model",
        "exceeds the context window",
        "input token count",
        "maximum prompt length",
        "reduce the length of the messages",
        "maximum context length",
        "exceeds the available context size",
        "greater than the context length",
        "context window exceeds limit",
        "exceeded model token limit",
        "context length exceeded",
        "too many tokens",
        "token limit exceeded",
    ];

    patterns.iter().any(|pattern| normalized.contains(pattern))
}

fn is_context_overflow_status_no_body(error: &str) -> bool {
    let normalized = error.to_ascii_lowercase();
    normalized.contains("no body")
        && (normalized.starts_with("400")
            || normalized.starts_with("413")
            || normalized.contains("400 status code")
            || normalized.contains("413 status code"))
}

fn serialize_messages_for_summary(messages: &[Message]) -> String {
    messages
        .iter()
        .filter_map(|message| {
            let content = message_to_summary_text(message);
            if content.is_empty() {
                return None;
            }
            Some(format!("[{}]: {content}", message_role_label(message)))
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn build_auto_compaction_summary(
    messages_to_summarize: &[Message],
    context_tokens: u64,
    context_window: u64,
    max_summary_chars: usize,
) -> String {
    let mut summary = format!(
        "Auto-compaction snapshot (context tokens: {context_tokens}/{context_window}).\n\nCompacted history:\n"
    );

    let mut added_message = false;
    for message in messages_to_summarize {
        let role = message_role_label(message);
        let content = message_to_summary_text(message);
        if content.is_empty() {
            continue;
        }
        added_message = true;
        summary.push_str("- ");
        summary.push_str(role);
        summary.push_str(": ");
        summary.push_str(&content);
        summary.push('\n');
    }

    if !added_message {
        summary.push_str("- (no textual content)\n");
    }

    truncate_chars(&summary, max_summary_chars)
}

fn message_role_label(message: &Message) -> &'static str {
    match message {
        Message::User { .. } => "user",
        Message::Assistant { .. } => "assistant",
        Message::ToolResult { .. } => "tool_result",
    }
}

fn message_to_summary_text(message: &Message) -> String {
    match message {
        Message::User { content, .. } => user_content_text(content),
        Message::Assistant { content, .. } => assistant_content_text(content),
        Message::ToolResult { content, .. } => tool_result_content_text(content),
    }
}

fn user_content_text(content: &UserContent) -> String {
    match content {
        UserContent::Text(text) => normalize_text(text),
        UserContent::Blocks(blocks) => normalize_text(
            &blocks
                .iter()
                .filter_map(|block| match block {
                    UserContentBlock::Text { text, .. } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(" "),
        ),
    }
}

fn assistant_content_text(content: &[AssistantContentBlock]) -> String {
    normalize_text(
        &content
            .iter()
            .map(|block| match block {
                AssistantContentBlock::Text { text, .. } => text.clone(),
                AssistantContentBlock::Thinking { thinking, .. } => thinking.clone(),
                AssistantContentBlock::ToolCall {
                    name, arguments, ..
                } => format!(
                    "tool call `{name}` with args {}",
                    truncate_chars(&arguments.to_string(), 200)
                ),
            })
            .collect::<Vec<_>>()
            .join(" "),
    )
}

fn tool_result_content_text(content: &[ToolResultContentBlock]) -> String {
    normalize_text(
        &content
            .iter()
            .filter_map(|block| match block {
                ToolResultContentBlock::Text { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" "),
    )
}

fn normalize_text(text: &str) -> String {
    truncate_chars(&text.replace('\n', " ").trim().to_string(), 240)
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }

    if text.chars().count() <= max_chars {
        return text.to_string();
    }

    if max_chars <= 3 {
        return ".".repeat(max_chars);
    }

    let truncated = text.chars().take(max_chars - 3).collect::<String>();
    format!("{truncated}...")
}

fn resolve_resume_session_target(
    target: Option<&str>,
    current_session_file: Option<PathBuf>,
) -> Result<PathBuf, String> {
    let current_session_file = current_session_file
        .ok_or_else(|| "Current session file unavailable; cannot resume".to_string())?;
    let session_dir = current_session_file
        .parent()
        .ok_or_else(|| {
            format!(
                "Cannot determine session directory from {}",
                current_session_file.display()
            )
        })?
        .to_path_buf();

    let Some(target_value) = target.map(str::trim).filter(|value| !value.is_empty()) else {
        return latest_session_in_dir(&session_dir, Some(&current_session_file));
    };

    if target_value.eq_ignore_ascii_case("latest") {
        return latest_session_in_dir(&session_dir, None);
    }

    resolve_explicit_session_target(target_value, &session_dir)
}

fn resolve_explicit_session_target(target: &str, session_dir: &Path) -> Result<PathBuf, String> {
    let raw = PathBuf::from(target);
    if raw.is_absolute() {
        if raw.is_file() {
            return Ok(raw);
        }
        return Err(format!("Session file not found: {}", raw.display()));
    }

    let cwd = std::env::current_dir().map_err(|error| format!("read cwd failed: {error}"))?;
    let mut candidates = vec![cwd.join(&raw), session_dir.join(&raw)];
    if raw.extension().is_none() {
        let mut with_ext = raw.clone();
        with_ext.set_extension("jsonl");
        candidates.push(cwd.join(&with_ext));
        candidates.push(session_dir.join(with_ext));
    }

    if let Some(found) = candidates.into_iter().find(|path| path.is_file()) {
        return Ok(found);
    }

    if !target.contains('/') {
        let mut fuzzy_matches = list_session_files(session_dir)?
            .into_iter()
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .map(|name| name.contains(target))
                    .unwrap_or(false)
            })
            .collect::<Vec<_>>();
        fuzzy_matches.sort_by(|left, right| left.file_name().cmp(&right.file_name()));
        if let Some(found) = fuzzy_matches.pop() {
            return Ok(found);
        }
    }

    Err(format!(
        "Session not found for '{target}'. Use absolute path or a file under {}",
        session_dir.display()
    ))
}

fn latest_session_in_dir(session_dir: &Path, exclude: Option<&Path>) -> Result<PathBuf, String> {
    let mut files = list_session_files(session_dir)?;
    if let Some(excluded) = exclude {
        files.retain(|path| !paths_equal(path, excluded));
    }

    files.sort_by(|left, right| left.file_name().cmp(&right.file_name()));
    files.pop().ok_or_else(|| {
        format!(
            "No resumable sessions found under {}",
            session_dir.display()
        )
    })
}

fn list_session_files(session_dir: &Path) -> Result<Vec<PathBuf>, String> {
    let entries = std::fs::read_dir(session_dir).map_err(|error| {
        format!(
            "Read session directory failed ({}): {error}",
            session_dir.display()
        )
    })?;

    let mut files = vec![];
    for entry in entries {
        let entry = entry.map_err(|error| format!("Read session dir entry failed: {error}"))?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("jsonl"))
            .unwrap_or(false)
        {
            files.push(path);
        }
    }
    Ok(files)
}

fn paths_equal(left: &Path, right: &Path) -> bool {
    if left == right {
        return true;
    }
    match (std::fs::canonicalize(left), std::fs::canonicalize(right)) {
        (Ok(left), Ok(right)) => left == right,
        _ => false,
    }
}
