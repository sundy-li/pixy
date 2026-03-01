use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine;
use crossterm::event::{
    Event, EventStream, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEvent, MouseEventKind,
};
use futures_util::StreamExt;
use pixy_agent_core::AgentAbortController;
use pixy_ai::{Message, StopReason, UserContentBlock};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
#[cfg(test)]
use ratatui::style::Color;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, BorderType, Borders, Clear, Paragraph, Wrap};
use ratatui::{Frame, Terminal, TerminalOptions, Viewport};
use tokio::sync::mpsc;
use tokio::time::MissedTickBehavior;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

pub mod backend;
mod constants;
pub mod keybindings;
pub mod options;
mod resume;
mod runtime;
mod terminal;
pub mod theme;
mod transcript;

pub use backend::{BackendFuture, ResumeCandidate, StreamUpdate, TuiBackend};
use constants::{
    primary_input_placeholder_hint, FORCE_EXIT_SIGNAL, FORCE_EXIT_STATUS, INPUT_AREA_FIXED_HEIGHT,
    INPUT_RENDER_LEFT_PADDING, PASTED_TEXT_PREVIEW_LIMIT, RESUME_LIST_LIMIT, STATUS_HINT_LEFT,
    STATUS_HINT_RIGHT,
};
pub use keybindings::{parse_key_id, KeyBinding, TuiKeyBindings};
pub use options::TuiOptions;
use runtime::TuiRuntime;
use terminal::apply_selection_osc_colors;
#[cfg(test)]
use terminal::{
    selection_osc_reset_sequence, selection_osc_reset_sequences, selection_osc_set_sequence,
    selection_osc_set_sequences, TerminalCapabilities, TerminalMultiplexer,
};
pub use theme::TuiTheme;
use transcript::{
    is_thinking_line, is_tool_run_line, normalize_tool_line_for_display, parse_tool_name,
    render_messages, split_tool_output_lines, visible_transcript_lines, TranscriptLine,
    TranscriptLineKind,
};

#[derive(Clone, Debug, PartialEq, Eq)]
struct PendingTextAttachment {
    placeholder: String,
    content: String,
}

#[derive(Clone, Debug, PartialEq)]
struct PendingImageAttachment {
    placeholder: String,
    block: UserContentBlock,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ResumePickerState {
    candidates: Vec<ResumeCandidate>,
    selected: usize,
}

#[derive(Clone, Debug)]
struct InputHistoryStore {
    path: PathBuf,
    limit: usize,
}

impl InputHistoryStore {
    fn new(path: PathBuf, limit: usize) -> Self {
        Self {
            path,
            limit: limit.max(1),
        }
    }

    fn load(&self) -> Result<Vec<String>, String> {
        if !self.path.exists() {
            return Ok(vec![]);
        }

        let raw = fs::read_to_string(&self.path)
            .map_err(|error| format!("read {} failed: {error}", self.path.display()))?;

        let mut history = Vec::new();
        for line in raw.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let parsed =
                serde_json::from_str::<String>(trimmed).unwrap_or_else(|_| trimmed.to_string());
            if !parsed.is_empty() {
                history.push(parsed);
            }
        }

        if history.len() > self.limit {
            let keep_start = history.len() - self.limit;
            history = history.split_off(keep_start);
        }
        Ok(history)
    }

    fn persist(&self, history: &[String]) -> Result<(), String> {
        let keep = if history.len() > self.limit {
            &history[history.len() - self.limit..]
        } else {
            history
        };
        let mut encoded = String::new();
        for entry in keep {
            encoded.push_str(
                serde_json::to_string(entry)
                    .map_err(|error| format!("encode history failed: {error}"))?
                    .as_str(),
            );
            encoded.push('\n');
        }

        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("create {} failed: {error}", parent.display()))?;
        }
        fs::write(&self.path, encoded)
            .map_err(|error| format!("write {} failed: {error}", self.path.display()))?;
        Ok(())
    }
}

struct TuiApp {
    input: String,
    input_blocks: Option<Vec<UserContentBlock>>,
    pending_text_attachments: Vec<PendingTextAttachment>,
    pending_image_attachments: Vec<PendingImageAttachment>,
    cursor_pos: usize,
    input_history: Vec<String>,
    input_history_store: Option<InputHistoryStore>,
    history_nav_index: Option<usize>,
    history_stashed_input: Option<String>,
    transcript: Vec<TranscriptLine>,
    status: String,
    show_help: bool,
    show_tool_results: bool,
    show_thinking: bool,
    assistant_stream_open: bool,
    is_working: bool,
    working_message: String,
    working_tick: usize,
    working_started_at: Option<Instant>,
    interrupt_hint_label: String,
    dequeue_hint_label: String,
    last_clear_key_at_ms: i64,
    queued_follow_ups: Vec<String>,
    transcript_scroll_from_bottom: usize,
    status_top: String,
    status_left: String,
    status_right: String,
    resume_picker: Option<ResumePickerState>,
    welcome_lines: Vec<String>,
}

impl TuiApp {
    fn new(status: String, show_tool_results: bool, show_help: bool) -> Self {
        Self {
            input: String::new(),
            input_blocks: None,
            pending_text_attachments: vec![],
            pending_image_attachments: vec![],
            cursor_pos: 0,
            input_history: vec![],
            input_history_store: None,
            history_nav_index: None,
            history_stashed_input: None,
            transcript: vec![],
            status,
            show_help,
            show_tool_results,
            show_thinking: true,
            assistant_stream_open: false,
            is_working: false,
            working_message: String::new(),
            working_tick: 0,
            working_started_at: None,
            interrupt_hint_label: "esc".to_string(),
            dequeue_hint_label: "Alt+Up".to_string(),
            last_clear_key_at_ms: 0,
            queued_follow_ups: vec![],
            transcript_scroll_from_bottom: 0,
            status_top: String::new(),
            status_left: String::new(),
            status_right: String::new(),
            resume_picker: None,
            welcome_lines: vec![],
        }
    }

    fn set_status_bar_meta(&mut self, top: String, left: String, right: String) {
        self.status_top = top;
        self.status_left = left;
        self.status_right = right;
    }

    fn set_welcome_lines(&mut self, lines: Vec<String>) {
        self.welcome_lines = lines;
    }

    fn set_interrupt_hint_label(&mut self, label: String) {
        if label.trim().is_empty() {
            return;
        }
        self.interrupt_hint_label = label;
    }

    fn set_dequeue_hint_label(&mut self, label: String) {
        if label.trim().is_empty() {
            return;
        }
        self.dequeue_hint_label = label;
    }

    fn open_resume_picker(&mut self, candidates: Vec<ResumeCandidate>) {
        if candidates.is_empty() {
            self.resume_picker = None;
            return;
        }
        self.resume_picker = Some(ResumePickerState {
            candidates,
            selected: 0,
        });
    }

    fn close_resume_picker(&mut self) {
        self.resume_picker = None;
    }

    fn has_resume_picker(&self) -> bool {
        self.resume_picker.is_some()
    }

    fn set_input_history_store(&mut self, store: Option<InputHistoryStore>) {
        self.input_history_store = store;
        if let Some(store) = &self.input_history_store {
            match store.load() {
                Ok(history) => {
                    self.input_history = history;
                    self.reset_input_history_navigation();
                }
                Err(error) => {
                    self.status = format!("history load failed: {error}");
                }
            }
        }
    }

    fn trim_input_history_to_store_limit(&mut self) {
        let Some(limit) = self.input_history_store.as_ref().map(|store| store.limit) else {
            return;
        };
        if self.input_history.len() > limit {
            let overflow = self.input_history.len() - limit;
            self.input_history.drain(..overflow);
        }
    }

    fn persist_input_history(&mut self) {
        let Some(store) = &self.input_history_store else {
            return;
        };
        if let Err(error) = store.persist(&self.input_history) {
            self.status = format!("history save failed: {error}");
        }
    }

    fn maybe_update_status_right_from_backend_status(&mut self, status: &str) {
        let Some(model_path) = status.strip_prefix("model: ").map(str::trim) else {
            return;
        };
        let (provider_name, model_name) = model_path
            .split_once('/')
            .map(|(provider, model)| (provider.trim(), model.trim()))
            .unwrap_or((model_path.trim(), ""));
        if provider_name.is_empty() && model_name.is_empty() {
            return;
        }
        self.status_right = if provider_name.is_empty() {
            model_name.to_string()
        } else if model_name.is_empty() {
            provider_name.to_string()
        } else {
            format!("{provider_name}:{model_name}")
        };
    }

    fn maybe_update_status_left_from_backend_status(&mut self, status: &str) {
        let Some(permission_label) = status.strip_prefix("permission: ").map(str::trim) else {
            return;
        };
        if permission_label.is_empty() {
            return;
        }
        self.status_left = permission_label.to_string();
    }

    fn input_char_count(&self) -> usize {
        self.input.chars().count()
    }

    fn clear_input(&mut self) {
        self.input.clear();
        self.input_blocks = None;
        self.pending_text_attachments.clear();
        self.pending_image_attachments.clear();
        self.cursor_pos = 0;
        self.reset_input_history_navigation();
    }

    fn has_non_text_input_blocks(&self) -> bool {
        self.input_blocks.as_ref().is_some_and(|blocks| {
            blocks
                .iter()
                .any(|block| !matches!(block, UserContentBlock::Text { .. }))
        })
    }

    fn has_input_payload(&self) -> bool {
        !self.input.trim().is_empty() || self.has_non_text_input_blocks()
    }

    fn has_conversation_content(&self) -> bool {
        self.transcript
            .iter()
            .any(|line| line.kind != TranscriptLineKind::Overlay && !line.text.trim().is_empty())
    }

    fn should_show_input_placeholder(&self) -> bool {
        self.input.is_empty()
            && !self.has_non_text_input_blocks()
            && !self.has_conversation_content()
    }

    fn take_input_payload(&mut self) -> (String, String, Option<Vec<UserContentBlock>>) {
        self.cursor_pos = 0;
        self.reset_input_history_navigation();

        let display = self.input.trim().to_string();
        let expanded = self.expand_pasted_text_placeholders(&display);
        let text_for_blocks = self.strip_pending_image_placeholders(expanded.as_str());
        self.input.clear();

        let blocks = self.input_blocks.take().and_then(|mut blocks| {
            if text_for_blocks.is_empty() {
                blocks.retain(|block| !matches!(block, UserContentBlock::Text { .. }));
                return if blocks.is_empty() {
                    None
                } else {
                    Some(blocks)
                };
            }

            if let Some(UserContentBlock::Text { text, .. }) = blocks
                .iter_mut()
                .find(|block| matches!(block, UserContentBlock::Text { .. }))
            {
                *text = text_for_blocks.clone();
                Some(blocks)
            } else {
                let mut out = Vec::with_capacity(blocks.len() + 1);
                out.push(UserContentBlock::Text {
                    text: text_for_blocks.clone(),
                    text_signature: None,
                });
                out.extend(blocks);
                Some(out)
            }
        });

        self.pending_text_attachments.clear();
        self.pending_image_attachments.clear();
        (display, expanded, blocks)
    }

    fn insert_text(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        self.reset_input_history_navigation();
        let byte_pos = self
            .input
            .char_indices()
            .nth(self.cursor_pos)
            .map(|(i, _)| i)
            .unwrap_or(self.input.len());
        self.input.insert_str(byte_pos, text);
        self.cursor_pos += text.chars().count();
        self.scroll_transcript_to_latest();
    }

    fn insert_char(&mut self, ch: char) {
        let mut buffer = [0_u8; 4];
        let text = ch.encode_utf8(&mut buffer);
        self.insert_text(text);
    }

    fn expand_pasted_text_placeholders(&self, text: &str) -> String {
        let mut expanded = text.to_string();
        for attachment in &self.pending_text_attachments {
            expanded = expanded.replacen(&attachment.placeholder, &attachment.content, 1);
        }
        expanded
    }

    fn strip_pending_image_placeholders(&self, text: &str) -> String {
        let mut stripped = text.to_string();
        for attachment in &self.pending_image_attachments {
            stripped = stripped.replacen(&attachment.placeholder, "", 1);
        }
        stripped.trim().to_string()
    }

    fn push_pending_pasted_text(&mut self, content: String) {
        let placeholder = format!("[Pasted Content {} chars]", content.chars().count());
        self.pending_text_attachments.push(PendingTextAttachment {
            placeholder: placeholder.clone(),
            content,
        });
        self.insert_text(&placeholder);
    }

    fn push_pending_image_attachment(&mut self, placeholder: String, block: UserContentBlock) {
        self.pending_image_attachments.push(PendingImageAttachment {
            placeholder: placeholder.clone(),
            block: block.clone(),
        });
        self.insert_text(&placeholder);

        let pending_blocks = self
            .pending_image_attachments
            .iter()
            .map(|pending| pending.block.clone())
            .collect::<Vec<_>>();
        let blocks = self.input_blocks.get_or_insert_with(Vec::new);
        blocks.retain(|candidate| !pending_blocks.iter().any(|pending| pending == candidate));
        blocks.extend(pending_blocks);
    }

    fn delete_char_before_cursor(&mut self) -> bool {
        if self.cursor_pos == 0 {
            return false;
        }
        let byte_pos = self
            .input
            .char_indices()
            .nth(self.cursor_pos - 1)
            .map(|(i, _)| i)
            .unwrap_or(self.input.len());
        self.input.remove(byte_pos);
        self.cursor_pos -= 1;
        self.reset_input_history_navigation();
        self.scroll_transcript_to_latest();
        true
    }

    fn move_cursor_left(&mut self) {
        if self.cursor_pos > 0 {
            self.cursor_pos -= 1;
        }
    }

    fn move_cursor_right(&mut self) {
        if self.cursor_pos < self.input_char_count() {
            self.cursor_pos += 1;
        }
    }

    fn move_cursor_home(&mut self) {
        self.cursor_pos = 0;
    }

    fn move_cursor_end(&mut self) {
        self.cursor_pos = self.input_char_count();
    }

    fn delete_to_start(&mut self) {
        if self.cursor_pos == 0 {
            return;
        }
        self.reset_input_history_navigation();
        let byte_pos = self
            .input
            .char_indices()
            .nth(self.cursor_pos)
            .map(|(i, _)| i)
            .unwrap_or(self.input.len());
        self.input.drain(..byte_pos);
        self.cursor_pos = 0;
        self.scroll_transcript_to_latest();
    }

    fn delete_to_end(&mut self) {
        self.reset_input_history_navigation();
        let byte_pos = self
            .input
            .char_indices()
            .nth(self.cursor_pos)
            .map(|(i, _)| i)
            .unwrap_or(self.input.len());
        if byte_pos == self.input.len() {
            return;
        }
        self.input.truncate(byte_pos);
        self.scroll_transcript_to_latest();
    }

    fn delete_word_backward(&mut self) {
        if self.cursor_pos == 0 {
            return;
        }
        self.reset_input_history_navigation();
        let chars: Vec<char> = self.input.chars().collect();
        let mut new_pos = self.cursor_pos;
        while new_pos > 0 && chars[new_pos - 1].is_whitespace() {
            new_pos -= 1;
        }
        while new_pos > 0 && !chars[new_pos - 1].is_whitespace() {
            new_pos -= 1;
        }
        let start_byte = self
            .input
            .char_indices()
            .nth(new_pos)
            .map(|(i, _)| i)
            .unwrap_or(self.input.len());
        let end_byte = self
            .input
            .char_indices()
            .nth(self.cursor_pos)
            .map(|(i, _)| i)
            .unwrap_or(self.input.len());
        self.input.drain(start_byte..end_byte);
        self.cursor_pos = new_pos;
        self.scroll_transcript_to_latest();
    }

    fn push_lines(&mut self, lines: impl IntoIterator<Item = String>) {
        self.transcript.extend(
            lines
                .into_iter()
                .map(|line| TranscriptLine::new(line, TranscriptLineKind::Normal)),
        );
    }

    fn push_user_input_line(&mut self, line: String) {
        self.transcript.push(TranscriptLine::new(
            String::new(),
            TranscriptLineKind::UserInput,
        ));
        self.transcript
            .push(TranscriptLine::new(line, TranscriptLineKind::UserInput));
        self.transcript.push(TranscriptLine::new(
            String::new(),
            TranscriptLineKind::UserInput,
        ));
    }

    fn push_transcript_lines(&mut self, lines: impl IntoIterator<Item = TranscriptLine>) {
        self.transcript.extend(lines);
    }

    fn replace_transcript_with_messages(&mut self, messages: &[Message]) {
        self.assistant_stream_open = false;
        self.transcript = render_messages(messages);
        self.scroll_transcript_to_latest();
    }

    fn toggle_tool_results(&mut self) -> bool {
        self.show_tool_results = !self.show_tool_results;
        self.show_tool_results
    }

    fn toggle_thinking(&mut self) -> bool {
        self.show_thinking = !self.show_thinking;
        self.show_thinking
    }

    fn start_working(&mut self, _message: String) {
        self.is_working = true;
        self.working_message = "Thinking...".to_string();
        self.working_tick = 0;
        self.working_started_at = Some(Instant::now());
    }

    fn bump_working_tick(&mut self) {
        self.working_tick = self.working_tick.saturating_add(1);
    }

    fn stop_working(&mut self) {
        self.is_working = false;
        self.working_message.clear();
        self.working_tick = 0;
        self.working_started_at = None;
    }

    fn working_elapsed_secs(&self) -> u64 {
        self.working_started_at
            .map(|started_at| started_at.elapsed().as_secs())
            .unwrap_or(0)
    }

    fn working_elapsed_label(&self) -> String {
        let total_secs = self.working_elapsed_secs();
        let hours = total_secs / 3600;
        let mins = (total_secs % 3600) / 60;
        let secs = total_secs % 60;
        if hours > 0 {
            format!("{hours}h {mins}m {secs}s")
        } else if mins > 0 {
            format!("{mins}m {secs}s")
        } else {
            format!("{secs}s")
        }
    }

    fn status_for_render(&self) -> String {
        self.status.clone()
    }

    fn queue_follow_up(&mut self, input: String) {
        self.queued_follow_ups.push(input);
    }

    fn dequeue_follow_ups_to_editor(&mut self) -> Option<usize> {
        let count = self.queued_follow_ups.len();
        if count == 0 {
            return None;
        }
        self.input = self.queued_follow_ups.join("\n");
        self.cursor_pos = self.input_char_count();
        self.queued_follow_ups.clear();
        self.reset_input_history_navigation();
        self.scroll_transcript_to_latest();
        Some(count)
    }

    fn steering_status_lines(&self) -> Vec<String> {
        if !self.is_working || self.queued_follow_ups.is_empty() {
            return vec![];
        }

        let mut lines = self
            .queued_follow_ups
            .iter()
            .map(|queued| format!("Steering: {}", summarize_steering_message(queued)))
            .collect::<Vec<_>>();
        lines.push(format!(
            "↳ {} to edit all queued messages",
            self.dequeue_hint_label
        ));
        lines
    }

    fn pop_follow_up(&mut self) -> Option<String> {
        if self.queued_follow_ups.is_empty() {
            None
        } else {
            Some(self.queued_follow_ups.remove(0))
        }
    }

    fn queued_follow_up_count(&self) -> usize {
        self.queued_follow_ups.len()
    }

    fn scroll_transcript_up(&mut self, amount: usize) {
        self.transcript_scroll_from_bottom =
            self.transcript_scroll_from_bottom.saturating_add(amount);
    }

    fn scroll_transcript_down(&mut self, amount: usize) {
        self.transcript_scroll_from_bottom =
            self.transcript_scroll_from_bottom.saturating_sub(amount);
    }

    fn scroll_transcript_to_latest(&mut self) {
        self.transcript_scroll_from_bottom = 0;
    }

    fn reset_input_history_navigation(&mut self) {
        self.history_nav_index = None;
        self.history_stashed_input = None;
    }

    fn record_input_history(&mut self, input: &str) {
        if input.is_empty() {
            return;
        }
        if self
            .input_history
            .last()
            .is_some_and(|last| last.as_str() == input)
        {
            self.reset_input_history_navigation();
            return;
        }
        self.input_history.push(input.to_string());
        self.trim_input_history_to_store_limit();
        self.persist_input_history();
        self.reset_input_history_navigation();
    }

    fn navigate_input_history_up(&mut self) -> bool {
        if self.input_history.is_empty() {
            return false;
        }

        let next_index = match self.history_nav_index {
            Some(index) => index.saturating_sub(1),
            None => {
                self.history_stashed_input = Some(self.input.clone());
                self.input_history.len().saturating_sub(1)
            }
        };
        self.history_nav_index = Some(next_index);
        self.input = self.input_history[next_index].clone();
        self.cursor_pos = self.input_char_count();
        self.scroll_transcript_to_latest();
        true
    }

    fn navigate_input_history_down(&mut self) -> bool {
        let Some(current_index) = self.history_nav_index else {
            return false;
        };

        if current_index + 1 < self.input_history.len() {
            let next_index = current_index + 1;
            self.history_nav_index = Some(next_index);
            self.input = self.input_history[next_index].clone();
        } else {
            self.history_nav_index = None;
            self.input = self.history_stashed_input.take().unwrap_or_default();
        }
        self.cursor_pos = self.input_char_count();
        self.scroll_transcript_to_latest();
        true
    }

    fn note_working_from_update(&mut self, _app_name: &str, update: &StreamUpdate) {
        match update {
            StreamUpdate::AssistantTextDelta(_) | StreamUpdate::AssistantLine(_) => {
                self.working_message = "Streaming...".to_string();
            }
            StreamUpdate::ToolLine(line) => {
                if is_tool_run_line(line) {
                    self.working_message = "Invoking tools...".to_string();
                } else if parse_tool_name(line).is_some() {
                    self.working_message = "Invoking tools...".to_string();
                } else {
                    self.working_message = "Thinking...".to_string();
                }
            }
        }
    }

    fn working_line(&self) -> Option<TranscriptLine> {
        if !self.is_working {
            return None;
        }

        if self.status == "interrupting..." {
            return Some(TranscriptLine::new(
                "interrupting...".to_string(),
                TranscriptLineKind::Working,
            ));
        }

        let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        let spinner = frames[self.working_tick % frames.len()];
        let stop_key = if self.interrupt_hint_label.starts_with("esc") {
            "ESC".to_string()
        } else {
            self.interrupt_hint_label.to_ascii_uppercase()
        };
        let prefix = format!("{spinner} ");
        let message = if self.working_message.trim().is_empty() {
            "Thinking..."
        } else {
            self.working_message.as_str()
        };
        let suffix = format!("  (Press {stop_key} to stop)");
        let text = format!("{prefix}{message}{suffix}");

        Some(TranscriptLine::new_working_with_marquee(
            text,
            prefix.chars().count(),
            message.chars().count(),
            self.working_tick,
            4,
        ))
    }

    fn apply_stream_update(&mut self, update: StreamUpdate) {
        match update {
            StreamUpdate::AssistantTextDelta(delta) => {
                self.append_assistant_delta(&delta);
            }
            StreamUpdate::AssistantLine(line) => {
                self.assistant_stream_open = false;
                if !line.is_empty() {
                    if is_thinking_line(&line) {
                        if self.is_working
                            && matches!(
                                self.transcript.last(),
                                Some(TranscriptLine {
                                    kind: TranscriptLineKind::Thinking,
                                    ..
                                })
                            )
                        {
                            if let Some(last) = self.transcript.last_mut() {
                                last.text = line;
                            }
                        } else {
                            self.transcript
                                .push(TranscriptLine::new(line, TranscriptLineKind::Thinking));
                        }
                    } else {
                        self.transcript
                            .push(TranscriptLine::new(line, TranscriptLineKind::Assistant));
                    }
                }
            }
            StreamUpdate::ToolLine(line) => {
                self.assistant_stream_open = false;
                if !line.is_empty() {
                    for tool_line in split_tool_output_lines(&normalize_tool_line_for_display(line))
                    {
                        self.transcript
                            .push(TranscriptLine::new(tool_line, TranscriptLineKind::Tool));
                    }
                }
            }
        }
    }

    fn append_assistant_delta(&mut self, delta: &str) {
        if delta.is_empty() {
            return;
        }

        if !self.assistant_stream_open {
            self.transcript.push(TranscriptLine::new(
                String::new(),
                TranscriptLineKind::Assistant,
            ));
            self.assistant_stream_open = true;
        }

        let mut parts = delta.split('\n');
        if let Some(first) = parts.next() {
            if let Some(last) = self.transcript.last_mut() {
                last.text.push_str(first);
            }
        }
        for part in parts {
            self.transcript.push(TranscriptLine::new(
                part.to_string(),
                TranscriptLineKind::Assistant,
            ));
        }
    }
}

pub async fn run_tui<B: TuiBackend>(backend: &mut B, options: TuiOptions) -> Result<(), String> {
    TuiRuntime::new(backend, options)?.run().await
}

fn handle_editor_key_event(app: &mut TuiApp, key: KeyEvent) -> bool {
    match key.code {
        KeyCode::Left if key.modifiers == KeyModifiers::NONE => {
            let previous = app.cursor_pos;
            app.move_cursor_left();
            app.cursor_pos != previous
        }
        KeyCode::Right if key.modifiers == KeyModifiers::NONE => {
            let previous = app.cursor_pos;
            app.move_cursor_right();
            app.cursor_pos != previous
        }
        KeyCode::Home if key.modifiers == KeyModifiers::NONE => {
            let previous = app.cursor_pos;
            app.move_cursor_home();
            app.cursor_pos != previous
        }
        KeyCode::End if key.modifiers == KeyModifiers::NONE => {
            let previous = app.cursor_pos;
            app.move_cursor_end();
            app.cursor_pos != previous
        }
        KeyCode::Backspace if key.modifiers == KeyModifiers::NONE => {
            app.delete_char_before_cursor()
        }
        KeyCode::Enter if key.modifiers == KeyModifiers::SHIFT => {
            app.insert_char('\n');
            true
        }
        KeyCode::Char(c) if is_plain_char_input(key.modifiers) => {
            app.insert_char(c);
            true
        }
        KeyCode::Char(c) if is_ctrl_only(key.modifiers) => match c.to_ascii_lowercase() {
            'a' => {
                let previous = app.cursor_pos;
                app.move_cursor_home();
                app.cursor_pos != previous
            }
            'e' => {
                let previous = app.cursor_pos;
                app.move_cursor_end();
                app.cursor_pos != previous
            }
            'u' => {
                let previous_input = app.input.clone();
                app.delete_to_start();
                app.input != previous_input
            }
            'k' => {
                let previous_input = app.input.clone();
                app.delete_to_end();
                app.input != previous_input
            }
            'w' => {
                let previous_input = app.input.clone();
                app.delete_word_backward();
                app.input != previous_input
            }
            _ => false,
        },
        _ => false,
    }
}

fn handle_input_history_key_event(app: &mut TuiApp, key: KeyEvent) -> bool {
    match key.code {
        KeyCode::Up if key.modifiers == KeyModifiers::NONE => {
            if !app.input.is_empty() && app.cursor_pos > 0 {
                app.move_cursor_home();
                true
            } else {
                app.navigate_input_history_up()
            }
        }
        KeyCode::Down if key.modifiers == KeyModifiers::NONE => {
            if !app.input.is_empty() && app.cursor_pos < app.input_char_count() {
                app.move_cursor_end();
                true
            } else {
                app.navigate_input_history_down()
            }
        }
        _ => false,
    }
}

fn handle_mouse_history_event(app: &mut TuiApp, mouse: MouseEvent) -> bool {
    match mouse.kind {
        MouseEventKind::ScrollUp => {
            if app.navigate_input_history_up() {
                true
            } else {
                app.scroll_transcript_up(1);
                true
            }
        }
        MouseEventKind::ScrollDown => {
            if app.navigate_input_history_down() {
                true
            } else {
                app.scroll_transcript_down(1);
                true
            }
        }
        _ => false,
    }
}

fn handle_transcript_scroll_key(app: &mut TuiApp, key: KeyEvent) -> bool {
    match key.code {
        KeyCode::Up if key.modifiers == KeyModifiers::NONE => {
            app.scroll_transcript_up(1);
            true
        }
        KeyCode::Down if key.modifiers == KeyModifiers::NONE => {
            app.scroll_transcript_down(1);
            true
        }
        KeyCode::PageUp if key.modifiers == KeyModifiers::NONE => {
            app.scroll_transcript_up(10);
            true
        }
        KeyCode::PageDown if key.modifiers == KeyModifiers::NONE => {
            app.scroll_transcript_down(10);
            true
        }
        _ => false,
    }
}

fn is_plain_char_input(modifiers: KeyModifiers) -> bool {
    !modifiers.contains(KeyModifiers::CONTROL)
        && !modifiers.contains(KeyModifiers::ALT)
        && !modifiers.contains(KeyModifiers::SUPER)
}

fn is_ctrl_only(modifiers: KeyModifiers) -> bool {
    modifiers.contains(KeyModifiers::CONTROL)
        && !modifiers.contains(KeyModifiers::ALT)
        && !modifiers.contains(KeyModifiers::SUPER)
}

#[cfg(test)]
fn transcript_title(app_name: &str, version: &str) -> String {
    if version.trim().is_empty() {
        format!("Welcome to {app_name} Chat")
    } else {
        format!("Welcome to {app_name} Chat  v{version}")
    }
}

fn session_status_label(session_file: Option<PathBuf>) -> Option<String> {
    session_file.map(|path| format!("session: {}", path.display()))
}

fn startup_status_label<B: TuiBackend>(_backend: &B) -> String {
    "ready".to_string()
}

fn default_terminal_options() -> TerminalOptions {
    TerminalOptions {
        viewport: Viewport::Fullscreen,
    }
}

fn query_session_status_label<B: TuiBackend>(backend: &B) -> String {
    session_status_label(backend.session_file())
        .unwrap_or_else(|| "session not initialized yet".to_string())
}

fn build_welcome_banner(options: &TuiOptions) -> Vec<String> {
    let kb = &options.keybindings;

    let submit_label = keybinding_label_lower(&kb.submit).to_ascii_uppercase();
    let interrupt_label = keybinding_label_lower(&kb.interrupt).to_ascii_uppercase();

    let mut lines = pixy_ascii_logo_lines(options.app_name.as_str());
    if !options.version.trim().is_empty() {
        lines.push(String::new());
        lines.push(format!("v{}", options.version.trim()));
    }
    lines.extend([
        String::new(),
        "Let's build crazy things together".to_string(),
        String::new(),
        format!("{submit_label} to send, Shift + ENTER for a new line, @ to mention files, / for commands"),
        String::new(),
        format!("Current workspace: {}", options.status_top),
    ]);

    let startup_lines = options
        .startup_resource_lines
        .iter()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty() && !line.ends_with("SKILL.md"))
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    if !startup_lines.is_empty() {
        lines.push(String::new());
        lines.extend(startup_lines);
    }

    let _ = interrupt_label;
    lines
}

fn pixy_ascii_logo_lines(app_name: &str) -> Vec<String> {
    if !app_name.eq_ignore_ascii_case("pixy") {
        return vec![app_name.to_string()];
    }
    vec![
        "███████   ██  ██      ██  ██      ██".to_string(),
        "██    ██  ██   ██    ██    ██    ██ ".to_string(),
        "██    ██  ██    ██  ██      ██  ██  ".to_string(),
        "███████   ██     ████        ████   ".to_string(),
        "██        ██    ██  ██        ██    ".to_string(),
        "██        ██   ██    ██       ██    ".to_string(),
        "██        ██  ██      ██      ██    ".to_string(),
    ]
}

fn history_navigation_help_panel_line(enable_mouse_capture: bool) -> &'static str {
    if enable_mouse_capture {
        "  Up/Down / mouse wheel input history, PageUp/PageDown scroll messages"
    } else {
        "  Up/Down input history, PageUp/PageDown scroll messages"
    }
}

fn keybinding_label_lower(bindings: &[KeyBinding]) -> String {
    if bindings.is_empty() {
        return "(unbound)".to_string();
    }
    bindings
        .iter()
        .copied()
        .map(format_keybinding_lower)
        .collect::<Vec<_>>()
        .join("/")
}

fn primary_keybinding_label_lower(bindings: &[KeyBinding]) -> String {
    bindings
        .first()
        .copied()
        .map(format_keybinding_lower)
        .unwrap_or_else(|| "interrupt".to_string())
}

fn format_user_input_line(input: &str, input_prompt: &str) -> String {
    format!("{input_prompt} {input}")
}

fn handle_paste_event(app: &mut TuiApp, pasted: String) {
    if let Some(token) = parse_image_placeholder(pasted.as_str()) {
        match load_image_block_for_placeholder(token.as_str()) {
            Ok(block) => {
                app.push_pending_image_attachment(token.clone(), block);
                app.status = format!("attached {token}");
            }
            Err(error) => {
                app.insert_text(&token);
                app.status = format!("{token} not attached: {error}");
            }
        }
        return;
    }

    if should_shorten_pasted_text(pasted.as_str()) {
        app.push_pending_pasted_text(pasted);
        app.status = "pasted text inserted as placeholder".to_string();
        return;
    }

    app.insert_text(&pasted);
}

fn should_shorten_pasted_text(text: &str) -> bool {
    text.chars().count() > PASTED_TEXT_PREVIEW_LIMIT || text.contains('\n')
}

fn parse_image_placeholder(text: &str) -> Option<String> {
    let token = text.trim();
    if !token.starts_with("[image") || !token.ends_with(']') {
        return None;
    }
    let digits = &token[6..token.len() - 1];
    if digits.is_empty() || !digits.chars().all(|ch| ch.is_ascii_digit()) {
        return None;
    }
    Some(token.to_string())
}

fn load_image_block_for_placeholder(placeholder: &str) -> Result<UserContentBlock, String> {
    let path = resolve_image_path_for_placeholder(placeholder)?;
    let bytes =
        fs::read(&path).map_err(|error| format!("read {} failed: {error}", path.display()))?;
    let mime_type = infer_image_mime_type(&path)?;
    Ok(UserContentBlock::Image {
        data: BASE64_STANDARD.encode(bytes),
        mime_type,
    })
}

fn resolve_image_path_for_placeholder(placeholder: &str) -> Result<PathBuf, String> {
    let stem = placeholder.trim_start_matches('[').trim_end_matches(']');
    let image_dirs = candidate_image_dirs();

    for image_dir in &image_dirs {
        for extension in ["png", "jpg", "jpeg", "webp", "gif", "bmp"] {
            let candidate = image_dir.join(format!("{stem}.{extension}"));
            if candidate.exists() {
                return Ok(candidate);
            }
        }

        if let Some(path) = newest_matching_image_file(image_dir, stem)? {
            return Ok(path);
        }
    }

    let searched = image_dirs
        .iter()
        .map(|dir| dir.display().to_string())
        .collect::<Vec<_>>()
        .join(", ");
    Err(format!("no image file found in {searched}"))
}

fn candidate_image_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(configured) = env::var("PIXY_PASTED_IMAGE_DIR") {
        let configured = configured.trim();
        if !configured.is_empty() {
            dirs.push(PathBuf::from(configured));
        }
    }

    if let Ok(home) = env::var("HOME") {
        dirs.push(PathBuf::from(&home).join(".pixy/workspace/tmp"));
    }
    dirs.push(PathBuf::from("~/.pixy/workspace/tmp"));

    dedup_paths(dirs)
}

fn dedup_paths(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut deduped = Vec::new();
    for path in paths {
        if deduped.iter().any(|existing: &PathBuf| existing == &path) {
            continue;
        }
        deduped.push(path);
    }
    deduped
}

fn newest_matching_image_file(image_dir: &Path, stem: &str) -> Result<Option<PathBuf>, String> {
    let mut best_match: Option<(SystemTime, PathBuf)> = None;
    let entries = match fs::read_dir(image_dir) {
        Ok(entries) => entries,
        Err(_) => return Ok(None),
    };

    for entry in entries {
        let entry =
            entry.map_err(|error| format!("scan {} failed: {error}", image_dir.display()))?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !(file_name == stem || file_name.starts_with(&format!("{stem}."))) {
            continue;
        }
        let modified = fs::metadata(&path)
            .and_then(|metadata| metadata.modified())
            .unwrap_or(UNIX_EPOCH);
        if let Some((best_time, _)) = &best_match {
            if modified <= *best_time {
                continue;
            }
        }
        best_match = Some((modified, path));
    }

    Ok(best_match.map(|(_, path)| path))
}

fn infer_image_mime_type(path: &Path) -> Result<String, String> {
    let ext = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let mime = match ext.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "gif" => "image/gif",
        "bmp" => "image/bmp",
        "tif" | "tiff" => "image/tiff",
        _ => return Err(format!("unsupported image extension: {}", ext)),
    };
    Ok(mime.to_string())
}

fn summarize_steering_message(message: &str) -> String {
    let compacted = message.split_whitespace().collect::<Vec<_>>().join(" ");
    if compacted.is_empty() {
        "(empty)".to_string()
    } else {
        compacted
    }
}

fn format_keybinding_lower(binding: KeyBinding) -> String {
    let mut parts = Vec::new();
    if binding.modifiers.contains(KeyModifiers::CONTROL) {
        parts.push("ctrl");
    }
    if binding.modifiers.contains(KeyModifiers::SHIFT) {
        parts.push("shift");
    }
    if binding.modifiers.contains(KeyModifiers::ALT) {
        parts.push("alt");
    }
    if binding.modifiers.contains(KeyModifiers::SUPER) {
        parts.push("meta");
    }
    let key_name = match binding.code {
        KeyCode::Backspace => "backspace".to_string(),
        KeyCode::Enter => "enter".to_string(),
        KeyCode::Left => "left".to_string(),
        KeyCode::Right => "right".to_string(),
        KeyCode::Up => "up".to_string(),
        KeyCode::Down => "down".to_string(),
        KeyCode::Tab => "tab".to_string(),
        KeyCode::Esc => "escape".to_string(),
        KeyCode::F(n) => format!("f{n}"),
        KeyCode::Char(' ') => "space".to_string(),
        KeyCode::Char(ch) => ch.to_string(),
        _ => "key".to_string(),
    };
    if parts.is_empty() {
        key_name
    } else {
        format!("{}+{}", parts.join("+"), key_name)
    }
}

async fn handle_slash_command<B: TuiBackend>(
    command: &str,
    backend: &mut B,
    app: &mut TuiApp,
) -> Result<bool, String> {
    match command {
        "/help" | "?" => {
            app.show_help = !app.show_help;
            Ok(true)
        }
        "/session" => {
            app.status = query_session_status_label(backend);
            Ok(true)
        }
        "/new" => {
            app.status = backend
                .new_session()?
                .unwrap_or_else(|| "new session is not supported by this backend".to_string());
            Ok(true)
        }
        command if command.starts_with("/resume") => {
            resume::handle_slash_resume_command(command, backend, app)
        }
        "/exit" | "/quit" => Err("__EXIT__".to_string()),
        _ => Ok(false),
    }
}

fn handle_resume_picker_key_event<B: TuiBackend>(
    key: KeyEvent,
    backend: &mut B,
    app: &mut TuiApp,
) -> bool {
    resume::handle_resume_picker_key_event(key, backend, app)
}

async fn run_submitted_input<B: TuiBackend>(
    backend: &mut B,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut TuiApp,
    options: &TuiOptions,
    display_input: String,
    submitted_input: String,
    blocks: Option<Vec<UserContentBlock>>,
    events: &mut EventStream,
) -> Result<(), String> {
    app.record_input_history(&display_input);
    app.scroll_transcript_to_latest();
    persist_welcome_into_transcript(app);

    if blocks.is_none() && submitted_input == "/continue" {
        handle_continue_streaming(backend, terminal, app, options, events).await?;
        process_queued_follow_ups(backend, terminal, app, options, events).await?;
        return Ok(());
    }

    if blocks.is_none() {
        match handle_slash_command(&submitted_input, backend, app).await {
            Ok(true) => return Ok(()),
            Ok(false) => {}
            Err(error) if error == "__EXIT__" => return Err(FORCE_EXIT_SIGNAL.to_string()),
            Err(error) => return Err(error),
        }
    }

    app.push_user_input_line(format_user_input_line(
        display_input.as_str(),
        options.theme.input_prompt(),
    ));

    run_prompt_streaming(
        backend,
        terminal,
        app,
        options,
        &submitted_input,
        blocks,
        events,
    )
    .await?;

    process_queued_follow_ups(backend, terminal, app, options, events).await
}

async fn run_prompt_streaming<B: TuiBackend>(
    backend: &mut B,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut TuiApp,
    options: &TuiOptions,
    input: &str,
    blocks: Option<Vec<UserContentBlock>>,
    events: &mut EventStream,
) -> Result<(), String> {
    app.start_working(format!("{} is working...", options.app_name));
    app.status = format!("{} is working...", options.app_name);
    let _ = draw_ui_frame(terminal, app, options);

    let abort_controller = AgentAbortController::new();
    let mut interrupt_requested = false;
    let (update_tx, mut update_rx) = mpsc::unbounded_channel::<StreamUpdate>();
    let mut saw_update = false;
    let mut on_update = move |update: StreamUpdate| {
        let _ = update_tx.send(update);
    };
    let stream_future = backend.prompt_stream_with_blocks(
        input,
        blocks,
        Some(abort_controller.signal()),
        &mut on_update,
    );
    tokio::pin!(stream_future);
    let mut ticker = tokio::time::interval(Duration::from_millis(120));
    ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);

    loop {
        tokio::select! {
            maybe_event = events.next() => {
                if let Some(event_result) = maybe_event {
                    let event = event_result.map_err(|error| format!("read terminal event failed: {error}"))?;
                    let outcome = handle_streaming_event(
                        event,
                        &options.keybindings.quit,
                        &options.keybindings.interrupt,
                        &options.keybindings.continue_run,
                        &options.keybindings.dequeue,
                        &options.keybindings.newline,
                        &abort_controller,
                        app,
                    );
                    if outcome.force_exit {
                        return Err(FORCE_EXIT_SIGNAL.to_string());
                    }
                    if outcome.interrupted {
                        interrupt_requested = true;
                    }
                    if outcome.ui_changed {
                        let _ = draw_ui_frame(terminal, app, options);
                    }
                }
            }
            maybe_update = update_rx.recv() => {
                if let Some(update) = maybe_update {
                    saw_update = true;
                    app.note_working_from_update(&options.app_name, &update);
                    app.bump_working_tick();
                    app.apply_stream_update(update);
                    let _ = draw_ui_frame(terminal, app, options);
                }
            }
            _ = ticker.tick() => {
                app.bump_working_tick();
                let _ = draw_ui_frame(terminal, app, options);
            }
            result = &mut stream_future => {
                while let Ok(update) = update_rx.try_recv() {
                    saw_update = true;
                    app.note_working_from_update(&options.app_name, &update);
                    app.bump_working_tick();
                    app.apply_stream_update(update);
                }

                app.stop_working();
                match result {
                    Ok(messages) => {
                        if !saw_update {
                            app.push_transcript_lines(render_messages(&messages));
                        }
                        app.status = if interrupt_requested || has_aborted_assistant(&messages) {
                            "interrupted".to_string()
                        } else {
                            "ok".to_string()
                        };
                    }
                    Err(error) => {
                        app.push_lines([format!("[error] {error}")]);
                        app.status = format!("prompt failed: {error}");
                    }
                }
                return Ok(());
            }
        }
    }
}

async fn handle_continue_streaming<B: TuiBackend>(
    backend: &mut B,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut TuiApp,
    options: &TuiOptions,
    events: &mut EventStream,
) -> Result<(), String> {
    app.start_working(format!("{} is working...", options.app_name));
    app.status = format!("{} is working...", options.app_name);
    let _ = draw_ui_frame(terminal, app, options);

    let abort_controller = AgentAbortController::new();
    let mut interrupt_requested = false;
    let (update_tx, mut update_rx) = mpsc::unbounded_channel::<StreamUpdate>();
    let mut saw_update = false;
    let mut on_update = move |update: StreamUpdate| {
        let _ = update_tx.send(update);
    };
    let stream_future =
        backend.continue_run_stream(Some(abort_controller.signal()), &mut on_update);
    tokio::pin!(stream_future);
    let mut ticker = tokio::time::interval(Duration::from_millis(120));
    ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);

    loop {
        tokio::select! {
            maybe_event = events.next() => {
                if let Some(event_result) = maybe_event {
                    let event = event_result.map_err(|error| format!("read terminal event failed: {error}"))?;
                    let outcome = handle_streaming_event(
                        event,
                        &options.keybindings.quit,
                        &options.keybindings.interrupt,
                        &options.keybindings.continue_run,
                        &options.keybindings.dequeue,
                        &options.keybindings.newline,
                        &abort_controller,
                        app,
                    );
                    if outcome.force_exit {
                        return Err(FORCE_EXIT_SIGNAL.to_string());
                    }
                    if outcome.interrupted {
                        interrupt_requested = true;
                    }
                    if outcome.ui_changed {
                        let _ = draw_ui_frame(terminal, app, options);
                    }
                }
            }
            maybe_update = update_rx.recv() => {
                if let Some(update) = maybe_update {
                    saw_update = true;
                    app.note_working_from_update(&options.app_name, &update);
                    app.bump_working_tick();
                    app.apply_stream_update(update);
                    let _ = draw_ui_frame(terminal, app, options);
                }
            }
            _ = ticker.tick() => {
                app.bump_working_tick();
                let _ = draw_ui_frame(terminal, app, options);
            }
            result = &mut stream_future => {
                while let Ok(update) = update_rx.try_recv() {
                    saw_update = true;
                    app.note_working_from_update(&options.app_name, &update);
                    app.bump_working_tick();
                    app.apply_stream_update(update);
                }

                app.stop_working();
                match result {
                    Ok(messages) => {
                        if !saw_update {
                            app.push_transcript_lines(render_messages(&messages));
                        }
                        app.status = if interrupt_requested || has_aborted_assistant(&messages) {
                            "interrupted".to_string()
                        } else {
                            "ok".to_string()
                        };
                    }
                    Err(error) => {
                        app.push_lines([format!("[continue_error] {error}")]);
                        app.status = format!("continue failed: {error}");
                    }
                }
                return Ok(());
            }
        }
    }
}

async fn process_queued_follow_ups<B: TuiBackend>(
    backend: &mut B,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut TuiApp,
    options: &TuiOptions,
    events: &mut EventStream,
) -> Result<(), String> {
    while let Some(queued) = app.pop_follow_up() {
        app.push_user_input_line(format_user_input_line(
            queued.as_str(),
            options.theme.input_prompt(),
        ));
        run_prompt_streaming(backend, terminal, app, options, &queued, None, events).await?;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct StreamingEventOutcome {
    interrupted: bool,
    ui_changed: bool,
    force_exit: bool,
}

fn handle_streaming_event(
    event: Event,
    quit_bindings: &[KeyBinding],
    interrupt_bindings: &[KeyBinding],
    follow_up_bindings: &[KeyBinding],
    dequeue_bindings: &[KeyBinding],
    newline_bindings: &[KeyBinding],
    abort_controller: &AgentAbortController,
    app: &mut TuiApp,
) -> StreamingEventOutcome {
    if let Event::Mouse(mouse) = event {
        return StreamingEventOutcome {
            interrupted: false,
            ui_changed: handle_mouse_history_event(app, mouse),
            force_exit: false,
        };
    }

    if let Event::Paste(pasted) = event {
        handle_paste_event(app, pasted);
        return StreamingEventOutcome {
            interrupted: false,
            ui_changed: true,
            force_exit: false,
        };
    }

    let Event::Key(key) = event else {
        return StreamingEventOutcome::default();
    };
    if !matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
        return StreamingEventOutcome::default();
    }

    if matches_keybinding(quit_bindings, key) {
        app.status = FORCE_EXIT_STATUS.to_string();
        return StreamingEventOutcome {
            interrupted: false,
            ui_changed: true,
            force_exit: true,
        };
    }

    if matches_keybinding(interrupt_bindings, key) {
        if app.status == "interrupting..." || app.status == "interrupted" {
            return StreamingEventOutcome::default();
        }

        abort_controller.abort();
        app.status = "interrupting...".to_string();
        app.start_working("interrupting...".to_string());
        return StreamingEventOutcome {
            interrupted: true,
            ui_changed: true,
            force_exit: false,
        };
    }

    let plain_enter_during_streaming =
        key.code == KeyCode::Enter && key.modifiers == KeyModifiers::NONE;
    if matches_keybinding(follow_up_bindings, key) || plain_enter_during_streaming {
        let queued = app.input.trim().to_string();
        if queued.is_empty() {
            return StreamingEventOutcome::default();
        }
        app.record_input_history(&queued);
        app.clear_input();
        app.scroll_transcript_to_latest();
        app.queue_follow_up(queued);
        app.status = format!("queued follow-up ({})", app.queued_follow_up_count());
        return StreamingEventOutcome {
            interrupted: false,
            ui_changed: true,
            force_exit: false,
        };
    }

    if matches_keybinding(dequeue_bindings, key) {
        if let Some(count) = app.dequeue_follow_ups_to_editor() {
            let label = if count == 1 { "message" } else { "messages" };
            app.status = format!("editing {count} queued {label}");
            return StreamingEventOutcome {
                interrupted: false,
                ui_changed: true,
                force_exit: false,
            };
        }
        return StreamingEventOutcome::default();
    }

    if matches_keybinding(newline_bindings, key) {
        app.insert_char('\n');
        return StreamingEventOutcome {
            interrupted: false,
            ui_changed: true,
            force_exit: false,
        };
    }

    if handle_input_history_key_event(app, key) {
        return StreamingEventOutcome {
            interrupted: false,
            ui_changed: true,
            force_exit: false,
        };
    }

    if handle_transcript_scroll_key(app, key) {
        return StreamingEventOutcome {
            interrupted: false,
            ui_changed: true,
            force_exit: false,
        };
    }

    if handle_editor_key_event(app, key) {
        return StreamingEventOutcome {
            interrupted: false,
            ui_changed: true,
            force_exit: false,
        };
    }

    StreamingEventOutcome::default()
}

fn has_aborted_assistant(messages: &[Message]) -> bool {
    messages.iter().rev().any(|message| {
        matches!(
            message,
            Message::Assistant {
                stop_reason: StopReason::Aborted,
                ..
            }
        )
    })
}

fn is_force_exit_signal(error: &str) -> bool {
    error == FORCE_EXIT_SIGNAL
}

fn draw_ui_frame(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut TuiApp,
    options: &TuiOptions,
) -> io::Result<()> {
    terminal.draw(|frame| render_ui(frame, app, options))?;
    Ok(())
}

fn persist_welcome_into_transcript(app: &mut TuiApp) {
    if app.welcome_lines.is_empty() || !app.transcript.is_empty() {
        return;
    }

    for line in &app.welcome_lines {
        app.transcript.push(TranscriptLine::new(
            line.clone(),
            TranscriptLineKind::Overlay,
        ));
    }
    app.transcript.push(TranscriptLine::new(
        String::new(),
        TranscriptLineKind::Overlay,
    ));
    app.scroll_transcript_to_latest();
}

fn has_overlay_transcript_lines(lines: &[TranscriptLine]) -> bool {
    lines
        .iter()
        .any(|line| line.kind == TranscriptLineKind::Overlay)
}

fn ensure_bottom_status_separator(
    mut lines: Vec<Line<'static>>,
    target_height: usize,
) -> Vec<Line<'static>> {
    if target_height == 0 {
        return vec![];
    }

    if lines.len() > target_height {
        lines = lines[lines.len().saturating_sub(target_height)..].to_vec();
    } else if lines.len() < target_height {
        let mut padded = vec![Line::from(""); target_height.saturating_sub(lines.len())];
        padded.extend(lines);
        lines = padded;
    }

    let ends_with_blank = lines
        .last()
        .map(|line| line_text_for_status_separator(line).trim().is_empty())
        .unwrap_or(false);
    if ends_with_blank {
        return lines;
    }

    if target_height == 1 {
        return vec![Line::from("")];
    }

    if lines.len() == target_height {
        lines.remove(0);
    }
    lines.push(Line::from(""));
    if lines.len() > target_height {
        lines = lines[lines.len().saturating_sub(target_height)..].to_vec();
    }
    lines
}

fn line_text_for_status_separator(line: &Line<'_>) -> String {
    line.spans
        .iter()
        .map(|span| span.content.as_ref())
        .collect::<String>()
}

fn render_ui(frame: &mut Frame, app: &TuiApp, options: &TuiOptions) {
    let input_prompt = options.theme.input_prompt();
    let total_status_height =
        status_bar_height(app).min(frame.area().height.saturating_sub(1).max(1));
    let status_top_height = total_status_height.saturating_sub(1);
    let status_bottom_height = 1u16;
    let desired_steering_height = steering_panel_height(app);
    let steering_height = desired_steering_height.min(
        frame
            .area()
            .height
            .saturating_sub(total_status_height)
            .saturating_sub(1),
    );
    let reserved_height = total_status_height.saturating_add(steering_height);
    let input_height = input_area_height(app, frame.area(), input_prompt, reserved_height);
    let areas = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(1),
            Constraint::Length(steering_height),
            Constraint::Length(status_top_height),
            Constraint::Length(input_height),
            Constraint::Length(status_bottom_height),
        ])
        .split(frame.area());

    let transcript_area = areas[0];
    let steering_area = areas[1];
    let status_top_area = areas[2];
    let input_area = areas[3];
    let footer_area = areas[4];

    let visible_lines = visible_transcript_lines(
        &app.transcript,
        &[],
        transcript_area.height.saturating_sub(2) as usize,
        transcript_area.width.saturating_sub(2) as usize,
        app.show_tool_results,
        app.show_thinking,
        app.working_line(),
        app.transcript_scroll_from_bottom,
        options.theme,
    );

    let target_height = transcript_area.height as usize;
    let mut lines = if visible_lines.len() > target_height {
        visible_lines[visible_lines.len().saturating_sub(target_height)..].to_vec()
    } else {
        visible_lines
    };

    if lines.len() < target_height {
        let missing = target_height.saturating_sub(lines.len());
        if has_overlay_transcript_lines(&app.transcript) {
            let top_padding = missing / 2;
            let bottom_padding = missing.saturating_sub(top_padding);
            let mut padded = vec![Line::from(""); top_padding];
            padded.extend(lines);
            padded.extend(vec![Line::from(""); bottom_padding]);
            lines = padded;
        } else {
            let mut padded = vec![Line::from(""); missing];
            padded.extend(lines);
            lines = padded;
        }
    }
    lines = ensure_bottom_status_separator(lines, target_height);

    let transcript = Paragraph::new(Text::from(lines)).style(options.theme.transcript_style());
    frame.render_widget(transcript, transcript_area);

    if steering_height > 0 {
        let steering = Paragraph::new(render_steering_panel_lines(
            app,
            steering_area.width as usize,
        ))
        .style(options.theme.transcript_style());
        frame.render_widget(steering, steering_area);
    }

    let full_status = render_status_bar_lines(app, status_top_area.width as usize, options.theme);
    let mut status_lines = full_status.lines;
    let bottom_line = status_lines.pop().unwrap_or_else(|| Line::from(""));
    let status_top = Paragraph::new(Text::from(status_lines))
        .style(options.theme.footer_style())
        .wrap(Wrap { trim: false });
    frame.render_widget(status_top, status_top_area);

    let input_scroll = input_scroll_offset(app, input_area, input_prompt);
    let input = Paragraph::new(build_input_line(app, input_prompt, options.theme))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(options.theme.input_border_style()),
        )
        .wrap(Wrap { trim: false })
        .scroll((input_scroll, 0))
        .style(options.theme.input_style());
    frame.render_widget(input, input_area);

    if !app.show_help && !app.has_resume_picker() {
        let (cursor_x, cursor_y) = input_cursor_position(app, input_area, input_prompt);
        frame.set_cursor_position((cursor_x, cursor_y));
    }

    let footer = Paragraph::new(Text::from(vec![bottom_line])).style(options.theme.footer_style());
    frame.render_widget(footer, footer_area);

    if app.show_help {
        let popup = centered_rect(80, 60, frame.area());
        frame.render_widget(Clear, popup);
        let continue_key = keybinding_label(&options.keybindings.continue_run);
        let help = Paragraph::new(Text::from(vec![
            Line::from("Keybindings"),
            Line::from(format!(
                "  {:<14} submit input",
                keybinding_label(&options.keybindings.submit)
            )),
            Line::from(format!(
                "  {:<14} interrupt",
                keybinding_label(&options.keybindings.interrupt)
            )),
            Line::from(format!(
                "  {:<14} clear input / double-press exit",
                keybinding_label(&options.keybindings.clear)
            )),
            Line::from(format!(
                "  {:<14} toggle tool output",
                keybinding_label(&options.keybindings.expand_tools)
            )),
            Line::from(format!(
                "  {:<14} edit queued follow-ups",
                keybinding_label(&options.keybindings.dequeue)
            )),
            Line::from(format!(
                "  {:<14} cycle thinking level (mapped)",
                keybinding_label(&options.keybindings.cycle_thinking_level)
            )),
            Line::from(format!(
                "  {:<14} toggle thinking",
                keybinding_label(&options.keybindings.toggle_thinking)
            )),
            Line::from(format!(
                "  {:<14} cycle model forward",
                keybinding_label(&options.keybindings.cycle_model_forward)
            )),
            Line::from(format!(
                "  {:<14} cycle model backward",
                keybinding_label(&options.keybindings.cycle_model_backward)
            )),
            Line::from(format!(
                "  {:<14} cycle permission mode",
                keybinding_label(&options.keybindings.cycle_permission_mode)
            )),
            Line::from(format!(
                "  {:<14} select model",
                keybinding_label(&options.keybindings.select_model)
            )),
            Line::from(format!(
                "  {:<14} show session file",
                keybinding_label(&options.keybindings.show_session)
            )),
            Line::from(format!(
                "  {:<14} toggle help",
                keybinding_label(&options.keybindings.show_help)
            )),
            Line::from(format!(
                "  {:<14} force quit",
                keybinding_label(&options.keybindings.quit)
            )),
            Line::from(""),
            Line::from("Slash Commands"),
            Line::from(format!(
                "  /new /continue ({continue_key}) /resume [session] /session /help /exit"
            )),
            Line::from("  Ctrl+A / Ctrl+E move cursor"),
            Line::from("  Ctrl+W / Ctrl+U delete backward"),
            Line::from(format!(
                "  {} insert newline",
                keybinding_label(&options.keybindings.newline)
            )),
            Line::from(history_navigation_help_panel_line(
                options.enable_mouse_capture,
            )),
            Line::from(""),
            Line::from("Use interrupt or help key to close help."),
        ]))
        .block(
            Block::default()
                .title("Help")
                .borders(Borders::ALL)
                .border_style(options.theme.help_border_style()),
        )
        .style(options.theme.help_style())
        .wrap(Wrap { trim: false });
        frame.render_widget(help, popup);
    } else if let Some(picker) = app.resume_picker.as_ref() {
        let popup = centered_rect(88, 50, frame.area());
        frame.render_widget(Clear, popup);

        let mut lines = vec![
            Line::from("Select a session to resume"),
            Line::from("Up/Down to move, Enter to resume, Esc to cancel"),
            Line::from(""),
        ];

        for (index, candidate) in picker.candidates.iter().enumerate() {
            let indicator = if index == picker.selected { ">" } else { " " };
            let label = format!(
                "{indicator} {:>2}. {} ({})",
                index + 1,
                candidate.title,
                candidate.updated_at
            );
            let line = if index == picker.selected {
                Line::from(label).style(Style::default().add_modifier(Modifier::REVERSED))
            } else {
                Line::from(label)
            };
            lines.push(line);
        }

        if let Some(latest) = picker.candidates.first() {
            lines.push(Line::from(""));
            lines.push(Line::from(format!("0 means latest: {}", latest.title)));
        }

        let picker_popup = Paragraph::new(Text::from(lines))
            .block(
                Block::default()
                    .title("Resume Session")
                    .borders(Borders::ALL)
                    .border_style(options.theme.help_border_style()),
            )
            .style(options.theme.help_style())
            .wrap(Wrap { trim: false });
        frame.render_widget(picker_popup, popup);
    }
}

fn build_input_line(app: &TuiApp, input_prompt: &str, theme: TuiTheme) -> Line<'static> {
    if app.should_show_input_placeholder() {
        return Line::from(vec![
            Span::raw(INPUT_RENDER_LEFT_PADDING),
            Span::raw(input_prompt.to_string()),
            Span::styled(
                primary_input_placeholder_hint().to_string(),
                theme.input_placeholder_style(),
            ),
        ]);
    }
    Line::from(format!(
        "{INPUT_RENDER_LEFT_PADDING}{input_prompt}{}",
        app.input
    ))
}

fn input_cursor_position(app: &TuiApp, input_area: Rect, input_prompt: &str) -> (u16, u16) {
    let (x, y, _) = input_cursor_layout(app, input_area, input_prompt);
    (x, y)
}

fn input_scroll_offset(app: &TuiApp, input_area: Rect, input_prompt: &str) -> u16 {
    let (_, _, scroll) = input_cursor_layout(app, input_area, input_prompt);
    scroll
}

fn input_cursor_layout(app: &TuiApp, input_area: Rect, input_prompt: &str) -> (u16, u16, u16) {
    let inner_width = input_area.width.saturating_sub(2) as usize;
    let inner_height = input_area.height.saturating_sub(2) as usize;
    if inner_width == 0 || inner_height == 0 {
        let fallback_y = input_area
            .y
            .saturating_add(input_area.height.saturating_sub(1));
        return (input_area.x.saturating_add(1), fallback_y, 0);
    }

    let (row, col) = input_cursor_row_col(
        app.input.as_str(),
        app.cursor_pos,
        inner_width,
        input_prompt,
    );
    let scroll = row.saturating_sub(inner_height.saturating_sub(1));
    let visible_row = row
        .saturating_sub(scroll)
        .min(inner_height.saturating_sub(1));

    let max_x_offset = input_area.width.saturating_sub(2);
    let x = input_area
        .x
        .saturating_add(1)
        .saturating_add((col as u16).min(max_x_offset));

    let y_base = input_area.y.saturating_add(1);
    let max_y_offset = input_area.height.saturating_sub(2);
    let y = y_base.saturating_add((visible_row as u16).min(max_y_offset));

    (x, y, scroll as u16)
}

fn advance_cursor_row_col(row: &mut usize, col: &mut usize, ch: char, max_width: usize) {
    if ch == '\n' {
        *row = row.saturating_add(1);
        *col = 0;
        return;
    }

    let ch_width = UnicodeWidthChar::width(ch).unwrap_or(0);
    if ch_width == 0 {
        return;
    }

    if *col > 0 && *col + ch_width > max_width {
        *row = row.saturating_add(1);
        *col = 0;
    }

    *col += ch_width;
    if *col >= max_width {
        *row = row.saturating_add(1);
        *col = 0;
    }
}

fn input_cursor_row_col(
    input: &str,
    cursor_pos: usize,
    max_width: usize,
    input_prompt: &str,
) -> (usize, usize) {
    if max_width == 0 {
        return (0, 0);
    }

    let mut row = 0usize;
    let mut col = 0usize;

    for ch in INPUT_RENDER_LEFT_PADDING.chars() {
        advance_cursor_row_col(&mut row, &mut col, ch, max_width);
    }
    for ch in input_prompt.chars() {
        advance_cursor_row_col(&mut row, &mut col, ch, max_width);
    }
    for ch in input.chars().take(cursor_pos) {
        advance_cursor_row_col(&mut row, &mut col, ch, max_width);
    }

    (row, col)
}

fn input_area_height(
    _app: &TuiApp,
    frame_area: Rect,
    _input_prompt: &str,
    footer_height: u16,
) -> u16 {
    let max_height = frame_area.height.saturating_sub(footer_height).max(1);
    let desired = INPUT_AREA_FIXED_HEIGHT.max(3);
    desired.min(max_height)
}

fn status_bar_height(app: &TuiApp) -> u16 {
    let status = app.status_for_render();
    let has_top = if app.status_top.is_empty() {
        !status.is_empty() && status != "ok" && status != "ready" && !app.is_working
    } else {
        true
    };

    let mut lines = 3u16;
    if has_top {
        lines = lines.saturating_add(1);
    }
    lines
}

fn steering_panel_height(app: &TuiApp) -> u16 {
    app.steering_status_lines().len().min(u16::MAX as usize) as u16
}

fn render_status_bar_lines(app: &TuiApp, width: usize, theme: TuiTheme) -> Text<'static> {
    let status = app.status_for_render();

    let top = if app.status_top.is_empty() {
        if !status.is_empty() && status != "ok" && status != "ready" && !app.is_working {
            status.clone()
        } else {
            String::new()
        }
    } else if !status.is_empty() && status != "ok" && status != "ready" && !app.is_working {
        format!("{} | {status}", app.status_top)
    } else {
        app.status_top.clone()
    };

    let mut lines = Vec::new();
    if !top.is_empty() {
        lines.push(Line::from(top));
    }

    lines.push(compose_left_right_status_line_with_styles(
        format!(" {}", app.status_left).as_str(),
        app.status_right.as_str(),
        width,
        theme.status_primary_left_style(),
        theme.status_primary_right_style(),
    ));

    lines.push(compose_left_right_status_line_with_styles(
        format!(" {}", STATUS_HINT_LEFT).as_str(),
        STATUS_HINT_RIGHT,
        width,
        theme.status_hint_style(),
        theme.status_hint_style(),
    ));

    let bottom_left = if app.is_working {
        format!(" [⏱ {}]? for help", app.working_elapsed_label())
    } else {
        " ? for help".to_string()
    };
    lines.push(compose_left_right_status_line_with_styles(
        bottom_left.as_str(),
        "Pixy ◌",
        width,
        theme.status_hint_style(),
        theme.status_help_right_style(),
    ));

    Text::from(lines)
}

fn render_steering_panel_lines(app: &TuiApp, width: usize) -> Text<'static> {
    Text::from(
        app.steering_status_lines()
            .into_iter()
            .map(|line| Line::from(right_align_status_line(line.as_str(), width)))
            .collect::<Vec<_>>(),
    )
}

fn right_align_status_line(line: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }

    let line_width = UnicodeWidthStr::width(line);
    if line_width >= width {
        return line.to_string();
    }

    format!("{}{}", " ".repeat(width - line_width), line)
}

fn compose_left_right_status_line_with_styles(
    left: &str,
    right: &str,
    width: usize,
    left_style: Style,
    right_style: Style,
) -> Line<'static> {
    if width == 0 {
        return Line::from(Vec::<Span>::new());
    }

    let left_width = UnicodeWidthStr::width(left);
    let right_width = UnicodeWidthStr::width(right);
    if left_width + right_width >= width {
        if left_width >= width {
            return Line::from(vec![Span::styled(left.to_string(), left_style)]);
        }
        return Line::from(vec![
            Span::styled(left.to_string(), left_style),
            Span::raw(" "),
            Span::styled(right.to_string(), right_style),
        ]);
    }

    let gap = width.saturating_sub(left_width + right_width);
    Line::from(vec![
        Span::styled(left.to_string(), left_style),
        Span::raw(" ".repeat(gap)),
        Span::styled(right.to_string(), right_style),
    ])
}

fn matches_keybinding(bindings: &[KeyBinding], key: KeyEvent) -> bool {
    bindings.iter().copied().any(|binding| binding.matches(key))
}

fn keybinding_label(bindings: &[KeyBinding]) -> String {
    if bindings.is_empty() {
        return "(unbound)".to_string();
    }
    bindings
        .iter()
        .copied()
        .map(format_keybinding)
        .collect::<Vec<_>>()
        .join(" / ")
}

fn format_keybinding(binding: KeyBinding) -> String {
    let mut parts = Vec::new();
    if binding.modifiers.contains(KeyModifiers::CONTROL) {
        parts.push("Ctrl".to_string());
    }
    if binding.modifiers.contains(KeyModifiers::SHIFT) {
        parts.push("Shift".to_string());
    }
    if binding.modifiers.contains(KeyModifiers::ALT) {
        parts.push("Alt".to_string());
    }
    if binding.modifiers.contains(KeyModifiers::SUPER) {
        parts.push("Meta".to_string());
    }
    parts.push(match binding.code {
        KeyCode::Backspace => "Backspace".to_string(),
        KeyCode::Enter => "Enter".to_string(),
        KeyCode::Left => "Left".to_string(),
        KeyCode::Right => "Right".to_string(),
        KeyCode::Up => "Up".to_string(),
        KeyCode::Down => "Down".to_string(),
        KeyCode::Home => "Home".to_string(),
        KeyCode::End => "End".to_string(),
        KeyCode::PageUp => "PageUp".to_string(),
        KeyCode::PageDown => "PageDown".to_string(),
        KeyCode::Tab => "Tab".to_string(),
        KeyCode::Esc => "Escape".to_string(),
        KeyCode::F(number) => format!("F{number}"),
        KeyCode::Char(' ') => "Space".to_string(),
        KeyCode::Char(ch) => ch.to_ascii_uppercase().to_string(),
        _ => "Key".to_string(),
    });
    parts.join("+")
}

fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(area);
    let horizontal = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(vertical[1]);
    horizontal[1]
}

fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
#[path = "../tests/unit/lib_unit.rs"]
mod tests;
