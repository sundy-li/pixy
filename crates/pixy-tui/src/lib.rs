use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use crossterm::event::{
    DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture, Event,
    EventStream, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, KeyboardEnhancementFlags,
    MouseEvent, MouseEventKind, PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use futures_util::StreamExt;
use pixy_agent_core::AgentAbortController;
use pixy_ai::{Message, StopReason, UserContentBlock};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Text};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};
use ratatui::{Frame, Terminal};
use tokio::sync::mpsc;
use tokio::time::MissedTickBehavior;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

pub mod backend;
pub mod keybindings;
pub mod options;
mod resume;
pub mod theme;
mod transcript;

pub use backend::{BackendFuture, ResumeCandidate, StreamUpdate, TuiBackend};
pub use keybindings::{KeyBinding, TuiKeyBindings, parse_key_id};
pub use options::TuiOptions;
pub use theme::TuiTheme;
use transcript::{
    TranscriptLine, TranscriptLineKind, is_thinking_line, is_tool_run_line,
    normalize_tool_line_for_display, parse_tool_name, render_messages, split_tool_output_lines,
    visible_transcript_lines, wrap_text_by_display_width,
};

const FORCE_EXIT_SIGNAL: &str = "__FORCE_EXIT__";
const FORCE_EXIT_STATUS: &str = "force exiting...";
const PASTED_TEXT_PREVIEW_LIMIT: usize = 100;
const RESUME_LIST_LIMIT: usize = 10;

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
        }
    }

    fn set_status_bar_meta(&mut self, top: String, left: String, right: String) {
        self.status_top = top;
        self.status_left = left;
        self.status_right = right;
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
        let model_id = model_path.rsplit('/').next().unwrap_or(model_path).trim();
        if model_id.is_empty() {
            return;
        }
        let suffix = self
            .status_right
            .split_once(" • ")
            .map(|(_, value)| value.to_string())
            .unwrap_or_else(|| "medium".to_string());
        self.status_right = format!("{model_id} • {suffix}");
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

    fn toggle_tool_results(&mut self) -> bool {
        self.show_tool_results = !self.show_tool_results;
        self.show_tool_results
    }

    fn toggle_thinking(&mut self) -> bool {
        self.show_thinking = !self.show_thinking;
        self.show_thinking
    }

    fn start_working(&mut self, message: String) {
        self.is_working = true;
        self.working_message = message;
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

    fn with_working_suffix(&self, message: &str) -> String {
        format!(
            "{message} ({} • {} to interrupt)",
            self.working_elapsed_label(),
            self.interrupt_hint_label
        )
    }

    fn status_for_render(&self) -> String {
        if self.is_working {
            self.with_working_suffix(self.status.as_str())
        } else {
            self.status.clone()
        }
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

    fn note_working_from_update(&mut self, app_name: &str, update: &StreamUpdate) {
        match update {
            StreamUpdate::AssistantTextDelta(_) | StreamUpdate::AssistantLine(_) => {
                self.working_message = format!("{app_name} is reasoning...");
            }
            StreamUpdate::ToolLine(line) => {
                if is_tool_run_line(line) {
                    self.working_message = line.clone();
                } else if let Some(tool_name) = parse_tool_name(line) {
                    self.working_message = format!("• Ran {tool_name}");
                } else {
                    // Tool output lines are emitted after the tool returns.
                    // Switch back to generic "working" while waiting for next assistant token.
                    self.working_message = format!("{app_name} is working...");
                }
            }
        }
    }

    fn working_line(&self) -> Option<TranscriptLine> {
        if !self.is_working {
            return None;
        }
        let frames = if is_tool_run_line(&self.working_message) {
            ["▱", "▰", "▱", "▱"]
        } else {
            ["•", "◦", "•", "•"]
        };
        let spinner = frames[self.working_tick % frames.len()];
        let prefix = format!("{spinner} ");

        let suffix = format!(
            " ({} • {} to interrupt)",
            self.working_elapsed_label(),
            self.interrupt_hint_label
        );
        let text = format!("{prefix}{}{}", self.working_message, suffix);
        let message_char_len = self.working_message.chars().count();
        let highlight_len = message_char_len.min(4);
        let highlight_start = if message_char_len == 0 {
            0
        } else {
            self.working_tick % message_char_len
        };
        Some(TranscriptLine::new_working_with_marquee(
            text,
            prefix.chars().count(),
            message_char_len,
            highlight_start,
            highlight_len,
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
                            .push(TranscriptLine::new(line, TranscriptLineKind::Normal));
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
                TranscriptLineKind::Normal,
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
                TranscriptLineKind::Normal,
            ));
        }
    }
}

pub async fn run_tui<B: TuiBackend>(backend: &mut B, options: TuiOptions) -> Result<(), String> {
    enable_raw_mode().map_err(|error| format!("enable raw mode failed: {error}"))?;
    if options.enable_mouse_capture {
        execute!(io::stdout(), EnterAlternateScreen, EnableMouseCapture)
            .map_err(|error| format!("enter alternate screen failed: {error}"))?;
    } else {
        execute!(io::stdout(), EnterAlternateScreen)
            .map_err(|error| format!("enter alternate screen failed: {error}"))?;
    }

    let keyboard_enhancement_enabled =
        if crossterm::terminal::supports_keyboard_enhancement().unwrap_or(false) {
            execute!(
                io::stdout(),
                PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
            )
            .is_ok()
        } else {
            false
        };

    let bracketed_paste_enabled = execute!(io::stdout(), EnableBracketedPaste).is_ok();

    let mut _restore = TerminalRestore {
        keyboard_enhancement_enabled,
        mouse_capture_enabled: options.enable_mouse_capture,
        bracketed_paste_enabled,
        selection_colors_applied: false,
    };

    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))
        .map_err(|error| format!("create terminal failed: {error}"))?;
    terminal
        .clear()
        .map_err(|error| format!("clear terminal failed: {error}"))?;
    _restore.selection_colors_applied = apply_selection_osc_colors(options.theme);

    let status = backend
        .session_file()
        .map(|path| format!("session: {}", path.display()))
        .unwrap_or_else(|| "session: (none)".to_string());
    let mut app = TuiApp::new(status, options.show_tool_results, options.initial_help);
    app.set_interrupt_hint_label(primary_keybinding_label_lower(
        &options.keybindings.interrupt,
    ));
    app.set_dequeue_hint_label(keybinding_label(&options.keybindings.dequeue));
    app.set_input_history_store(
        options
            .input_history_path
            .as_ref()
            .map(|path| InputHistoryStore::new(path.clone(), options.input_history_limit)),
    );
    app.set_status_bar_meta(
        options.status_top.clone(),
        options.status_left.clone(),
        options.status_right.clone(),
    );
    app.push_lines(build_welcome_banner(&options));

    let mut events = EventStream::new();
    let mut needs_redraw = true;
    loop {
        if needs_redraw {
            terminal
                .draw(|frame| render_ui(frame, &app, &options))
                .map_err(|error| format!("draw UI failed: {error}"))?;
            needs_redraw = false;
        }

        let maybe_event = events.next().await;
        let Some(event_result) = maybe_event else {
            return Ok(());
        };
        let event = event_result.map_err(|error| format!("read terminal event failed: {error}"))?;

        if let Event::Mouse(mouse) = event {
            needs_redraw = handle_mouse_history_event(&mut app, mouse);
            continue;
        }

        if let Event::Paste(pasted) = event {
            handle_paste_event(&mut app, pasted);
            needs_redraw = true;
            continue;
        }

        if !matches!(event, Event::Key(_)) {
            // Keep resize/focus redraw responsive without repainting on every mouse drag.
            needs_redraw = true;
            continue;
        }

        let Event::Key(key) = event else {
            continue;
        };
        if !matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
            continue;
        }
        needs_redraw = true;

        if matches_keybinding(&options.keybindings.quit, key) {
            return Ok(());
        }
        if handle_resume_picker_key_event(key, backend, &mut app) {
            continue;
        }
        if matches_keybinding(&options.keybindings.interrupt, key) {
            app.clear_input();
            app.show_help = false;
            app.status = "interrupted".to_string();
            continue;
        }
        if matches_keybinding(&options.keybindings.clear, key) {
            if app.has_input_payload() || !app.pending_text_attachments.is_empty() {
                app.clear_input();
                app.last_clear_key_at_ms = now_millis();
                app.status = "input cleared".to_string();
                continue;
            }

            let now = now_millis();
            if now.saturating_sub(app.last_clear_key_at_ms) <= 500 {
                return Ok(());
            }
            app.last_clear_key_at_ms = now;
            app.status = "press clear again to exit".to_string();
            continue;
        }
        if matches_keybinding(&options.keybindings.show_help, key) {
            app.show_help = !app.show_help;
            continue;
        }
        if matches_keybinding(&options.keybindings.show_session, key) {
            app.status = backend
                .session_file()
                .map(|path| format!("session: {}", path.display()))
                .unwrap_or_else(|| "session: (none)".to_string());
            continue;
        }
        if matches_keybinding(&options.keybindings.cycle_model_forward, key) {
            app.status = match backend.cycle_model_forward() {
                Ok(Some(status)) => {
                    app.maybe_update_status_right_from_backend_status(&status);
                    status
                }
                Ok(None) => "only one model available".to_string(),
                Err(error) => format!("cycle model failed: {error}"),
            };
            continue;
        }
        if matches_keybinding(&options.keybindings.cycle_model_backward, key) {
            app.status = match backend.cycle_model_backward() {
                Ok(Some(status)) => {
                    app.maybe_update_status_right_from_backend_status(&status);
                    status
                }
                Ok(None) => "only one model available".to_string(),
                Err(error) => format!("cycle model failed: {error}"),
            };
            continue;
        }
        if matches_keybinding(&options.keybindings.select_model, key) {
            app.status = match backend.select_model() {
                Ok(Some(status)) => {
                    app.maybe_update_status_right_from_backend_status(&status);
                    status
                }
                Ok(None) => "no model selection candidates".to_string(),
                Err(error) => format!("select model failed: {error}"),
            };
            continue;
        }
        if matches_keybinding(&options.keybindings.cycle_thinking_level, key) {
            let enabled = app.toggle_thinking();
            app.status = if enabled {
                "thinking visible".to_string()
            } else {
                "thinking hidden".to_string()
            };
            continue;
        }
        if matches_keybinding(&options.keybindings.expand_tools, key) {
            let enabled = app.toggle_tool_results();
            app.status = if enabled {
                "tool output visible".to_string()
            } else {
                "tool output hidden".to_string()
            };
            continue;
        }
        if matches_keybinding(&options.keybindings.toggle_thinking, key) {
            let enabled = app.toggle_thinking();
            app.status = if enabled {
                "thinking visible".to_string()
            } else {
                "thinking hidden".to_string()
            };
            continue;
        }
        if matches_keybinding(&options.keybindings.continue_run, key) {
            if !app.has_input_payload() {
                if let Err(error) = handle_continue_streaming(
                    backend,
                    &mut terminal,
                    &mut app,
                    &options,
                    &mut events,
                )
                .await
                {
                    if is_force_exit_signal(&error) {
                        return Ok(());
                    }
                    return Err(error);
                }
                if let Err(error) = process_queued_follow_ups(
                    backend,
                    &mut terminal,
                    &mut app,
                    &options,
                    &mut events,
                )
                .await
                {
                    if is_force_exit_signal(&error) {
                        return Ok(());
                    }
                    return Err(error);
                }
            } else {
                let (display_input, submitted, blocks) = app.take_input_payload();
                if submitted.is_empty() && blocks.is_none() {
                    continue;
                }
                if let Err(error) = run_submitted_input(
                    backend,
                    &mut terminal,
                    &mut app,
                    &options,
                    display_input,
                    submitted,
                    blocks,
                    &mut events,
                )
                .await
                {
                    if is_force_exit_signal(&error) {
                        return Ok(());
                    }
                    return Err(error);
                }
            }
            continue;
        }
        if matches_keybinding(&options.keybindings.dequeue, key) {
            if let Some(count) = app.dequeue_follow_ups_to_editor() {
                let label = if count == 1 { "message" } else { "messages" };
                app.status = format!("editing {count} queued {label}");
                continue;
            }
        }
        if matches_keybinding(&options.keybindings.newline, key) {
            app.insert_char('\n');
            continue;
        }
        if matches_keybinding(&options.keybindings.submit, key) {
            let (display_input, submitted, blocks) = app.take_input_payload();
            if submitted.is_empty() && blocks.is_none() {
                continue;
            }
            if let Err(error) = run_submitted_input(
                backend,
                &mut terminal,
                &mut app,
                &options,
                display_input,
                submitted,
                blocks,
                &mut events,
            )
            .await
            {
                if is_force_exit_signal(&error) {
                    return Ok(());
                }
                return Err(error);
            }
            continue;
        }
        if handle_input_history_key_event(&mut app, key) {
            continue;
        }
        if handle_transcript_scroll_key(&mut app, key) {
            continue;
        }
        if handle_editor_key_event(&mut app, key) {
            continue;
        }
    }
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

fn transcript_title(app_name: &str, version: &str) -> String {
    if version.trim().is_empty() {
        format!("Welcome to {app_name} Chat")
    } else {
        format!("Welcome to {app_name} Chat  v{version}")
    }
}

fn build_welcome_banner(options: &TuiOptions) -> Vec<String> {
    let kb = &options.keybindings;

    let interrupt_label = keybinding_label_lower(&kb.interrupt);
    let clear_label = keybinding_label_lower(&kb.clear);
    let quit_label = keybinding_label_lower(&kb.quit);
    let cycle_thinking_label = keybinding_label_lower(&kb.cycle_thinking_level);
    let cycle_model_fwd = keybinding_label_lower(&kb.cycle_model_forward);
    let cycle_model_bwd = keybinding_label_lower(&kb.cycle_model_backward);
    let select_model_label = keybinding_label_lower(&kb.select_model);
    let expand_tools_label = keybinding_label_lower(&kb.expand_tools);
    let toggle_thinking_label = keybinding_label_lower(&kb.toggle_thinking);
    let follow_up_label = keybinding_label_lower(&kb.continue_run);
    let dequeue_label = keybinding_label_lower(&kb.dequeue);
    let newline_label = keybinding_label_lower(&kb.newline);

    let mut lines = vec![
        String::new(),
        format!(" {} to interrupt", interrupt_label),
        format!(" {} to clear", clear_label),
        format!(" {} twice to exit", clear_label),
        format!(" {} to force exit", quit_label),
        format!(" {} to cycle thinking level", cycle_thinking_label),
        format!(" {}/{} to cycle models", cycle_model_fwd, cycle_model_bwd),
        format!(" {} to select model", select_model_label),
        format!(" {} to expand tools", expand_tools_label),
        format!(" {} to expand thinking", toggle_thinking_label),
        format!(" / for commands"),
        format!(" {} to queue follow-up", follow_up_label),
        format!(" {} to edit queued follow-ups", dequeue_label),
        " ctrl+a/e: move cursor, ctrl+w/u: delete backward".to_string(),
        format!(" {}: newline", newline_label),
        history_navigation_help_line(options.enable_mouse_capture).to_string(),
    ];

    if !options.startup_resource_lines.is_empty() {
        lines.push(String::new());
        lines.extend(options.startup_resource_lines.iter().map(|line| {
            if line.is_empty() {
                String::new()
            } else {
                format!(" {line}")
            }
        }));
    }

    lines.push(String::new());
    lines
}

fn history_navigation_help_line(enable_mouse_capture: bool) -> &'static str {
    if enable_mouse_capture {
        " up/down/mouse wheel: input history, pageup/pagedown: scroll messages"
    } else {
        " up/down: input history, pageup/pagedown: scroll messages"
    }
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
    format!("{input_prompt}{input}")
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
        "/help" => {
            app.show_help = !app.show_help;
            Ok(true)
        }
        "/session" => {
            app.status = backend
                .session_file()
                .map(|path| format!("session: {}", path.display()))
                .unwrap_or_else(|| "session: (none)".to_string());
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
    let _ = terminal.draw(|frame| render_ui(frame, app, options));

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
                        let _ = terminal.draw(|frame| render_ui(frame, app, options));
                    }
                }
            }
            maybe_update = update_rx.recv() => {
                if let Some(update) = maybe_update {
                    saw_update = true;
                    app.note_working_from_update(&options.app_name, &update);
                    app.bump_working_tick();
                    app.apply_stream_update(update);
                    let _ = terminal.draw(|frame| render_ui(frame, app, options));
                }
            }
            _ = ticker.tick() => {
                app.bump_working_tick();
                let _ = terminal.draw(|frame| render_ui(frame, app, options));
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
    let _ = terminal.draw(|frame| render_ui(frame, app, options));

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
                        let _ = terminal.draw(|frame| render_ui(frame, app, options));
                    }
                }
            }
            maybe_update = update_rx.recv() => {
                if let Some(update) = maybe_update {
                    saw_update = true;
                    app.note_working_from_update(&options.app_name, &update);
                    app.bump_working_tick();
                    app.apply_stream_update(update);
                    let _ = terminal.draw(|frame| render_ui(frame, app, options));
                }
            }
            _ = ticker.tick() => {
                app.bump_working_tick();
                let _ = terminal.draw(|frame| render_ui(frame, app, options));
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

fn render_ui(frame: &mut Frame, app: &TuiApp, options: &TuiOptions) {
    let input_prompt = options.theme.input_prompt();
    let footer_height = status_bar_height().min(frame.area().height.saturating_sub(1).max(1));
    let desired_steering_height = steering_panel_height(app);
    let steering_height = desired_steering_height.min(
        frame
            .area()
            .height
            .saturating_sub(footer_height)
            .saturating_sub(1),
    );
    let reserved_height = footer_height.saturating_add(steering_height);
    let input_height = input_area_height(app, frame.area(), input_prompt, reserved_height);
    let areas = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(1),
            Constraint::Length(steering_height),
            Constraint::Length(input_height),
            Constraint::Length(footer_height),
        ])
        .split(frame.area());

    let transcript_area = areas[0];
    let steering_area = areas[1];
    let input_area = areas[2];
    let footer_area = areas[3];

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
    let transcript = Paragraph::new(Text::from(visible_lines))
        .block(
            Block::default()
                .borders(Borders::NONE)
                .title(transcript_title(
                    options.app_name.as_str(),
                    options.version.as_str(),
                )),
        )
        .style(options.theme.transcript_style());
    frame.render_widget(transcript, transcript_area);

    if steering_height > 0 {
        let steering = Paragraph::new(render_steering_panel_lines(
            app,
            steering_area.width as usize,
        ))
        .style(options.theme.transcript_style());
        frame.render_widget(steering, steering_area);
    }

    let input_scroll = input_scroll_offset(app, input_area, input_prompt);
    let input_text = format!("{input_prompt}{}", app.input);
    let input = Paragraph::new(input_text)
        .block(
            Block::default()
                .borders(Borders::TOP | Borders::BOTTOM)
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

    let footer = Paragraph::new(render_status_bar_lines(app, footer_area.width as usize))
        .style(options.theme.footer_style());
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

fn input_cursor_position(app: &TuiApp, input_area: Rect, input_prompt: &str) -> (u16, u16) {
    let (x, y, _) = input_cursor_layout(app, input_area, input_prompt);
    (x, y)
}

fn input_scroll_offset(app: &TuiApp, input_area: Rect, input_prompt: &str) -> u16 {
    let (_, _, scroll) = input_cursor_layout(app, input_area, input_prompt);
    scroll
}

fn input_cursor_layout(app: &TuiApp, input_area: Rect, input_prompt: &str) -> (u16, u16, u16) {
    let inner_width = input_area.width as usize;
    let inner_height = input_area.height.saturating_sub(2) as usize;
    if inner_width == 0 || inner_height == 0 {
        let fallback_y = input_area
            .y
            .saturating_add(input_area.height.saturating_sub(1));
        return (input_area.x, fallback_y, 0);
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

    let max_x_offset = input_area.width.saturating_sub(1);
    let x = input_area.x.saturating_add((col as u16).min(max_x_offset));

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

    for ch in input_prompt.chars() {
        advance_cursor_row_col(&mut row, &mut col, ch, max_width);
    }
    for ch in input.chars().take(cursor_pos) {
        advance_cursor_row_col(&mut row, &mut col, ch, max_width);
    }

    (row, col)
}

fn input_area_height(
    app: &TuiApp,
    frame_area: Rect,
    input_prompt: &str,
    footer_height: u16,
) -> u16 {
    // Input block has top+bottom borders only.
    let inner_width = frame_area.width as usize;
    let display_input = format!("{input_prompt}{}", app.input);
    let line_count = wrap_text_by_display_width(display_input.as_str(), inner_width)
        .len()
        .max(1);
    let desired_height = line_count.saturating_add(2) as u16;

    let max_height = frame_area.height.saturating_sub(footer_height).max(1);
    let min_height = 3u16.min(max_height);
    desired_height.max(min_height).min(max_height)
}

fn status_bar_height() -> u16 {
    2
}

fn steering_panel_height(app: &TuiApp) -> u16 {
    app.steering_status_lines().len().min(u16::MAX as usize) as u16
}

fn render_status_bar_lines(app: &TuiApp, width: usize) -> Text<'static> {
    let status = app.status_for_render();
    let mut top = if app.status_top.is_empty() {
        status.clone()
    } else {
        app.status_top.clone()
    };
    if !app.status_top.is_empty() && !status.is_empty() && status != "ok" {
        top = format!("{top} | {status}");
    }
    let bottom =
        compose_left_right_status_line(app.status_left.as_str(), app.status_right.as_str(), width);
    Text::from(vec![Line::from(top), Line::from(bottom)])
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

fn compose_left_right_status_line(left: &str, right: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }

    let left_width = UnicodeWidthStr::width(left);
    let right_width = UnicodeWidthStr::width(right);
    if left_width + right_width >= width {
        if left_width >= width {
            return left.to_string();
        }
        return format!("{left} {right}");
    }

    let gap = width.saturating_sub(left_width + right_width);
    format!("{left}{}{right}", " ".repeat(gap))
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

fn apply_selection_osc_colors(theme: TuiTheme) -> bool {
    let capabilities = detect_terminal_capabilities();
    let Some(sequences) = selection_osc_set_sequences(theme, capabilities) else {
        return false;
    };
    if sequences.is_empty() {
        return false;
    }
    let mut stdout = io::stdout();
    for sequence in sequences {
        if stdout.write_all(sequence.as_bytes()).is_err() {
            return false;
        }
    }
    let _ = stdout.flush();
    true
}

fn reset_selection_osc_colors() {
    let capabilities = detect_terminal_capabilities();
    let mut stdout = io::stdout();
    for sequence in selection_osc_reset_sequences(capabilities) {
        let _ = stdout.write_all(sequence.as_bytes());
    }
    let _ = stdout.flush();
}

#[cfg(test)]
fn selection_osc_set_sequence(theme: TuiTheme) -> Option<String> {
    selection_osc_set_sequences(theme, TerminalCapabilities::default())?
        .into_iter()
        .next()
}

fn selection_osc_set_sequences(
    theme: TuiTheme,
    capabilities: TerminalCapabilities,
) -> Option<Vec<String>> {
    let (bg, fg) = theme.selection_colors()?;
    let bg_hex = color_to_osc_hex(bg)?;
    let fg_hex = color_to_osc_hex(fg)?;
    let bg_rgb = color_to_osc_rgb(bg)?;
    let fg_rgb = color_to_osc_rgb(fg)?;

    let mut sequences = Vec::new();
    push_osc_pair(&mut sequences, "17", bg_hex.as_str(), "19", fg_hex.as_str());
    push_osc_pair(&mut sequences, "17", bg_rgb.as_str(), "19", fg_rgb.as_str());

    if capabilities.iterm2 {
        push_osc_pair(
            &mut sequences,
            "1337",
            format!("SetColors=selbg={}", bg_hex.trim_start_matches('#')).as_str(),
            "1337",
            format!("SetColors=selfg={}", fg_hex.trim_start_matches('#')).as_str(),
        );
    }

    if capabilities.kitty {
        push_osc_single(
            &mut sequences,
            "21",
            format!("selection_background={bg_hex};selection_foreground={fg_hex}").as_str(),
        );
    }

    append_multiplexer_variants(&mut sequences, capabilities.multiplexer);

    Some(sequences)
}

#[cfg(test)]
fn selection_osc_reset_sequence() -> &'static str {
    "\u{1b}]117;\u{7}\u{1b}]119;\u{7}"
}

fn selection_osc_reset_sequences(capabilities: TerminalCapabilities) -> Vec<String> {
    let mut sequences = Vec::new();
    push_osc_pair(&mut sequences, "117", "", "119", "");

    if capabilities.iterm2 {
        push_osc_pair(
            &mut sequences,
            "1337",
            "SetColors=selbg=default",
            "1337",
            "SetColors=selfg=default",
        );
    }

    if capabilities.kitty {
        push_osc_single(
            &mut sequences,
            "21",
            "selection_background;selection_foreground",
        );
    }

    append_multiplexer_variants(&mut sequences, capabilities.multiplexer);
    sequences
}

fn color_to_osc_hex(color: Color) -> Option<String> {
    let (red, green, blue) = color_to_rgb_bytes(color)?;
    Some(format!("#{red:02x}{green:02x}{blue:02x}"))
}

fn color_to_osc_rgb(color: Color) -> Option<String> {
    let (red, green, blue) = color_to_rgb_bytes(color)?;
    Some(format!("rgb:{red:02x}/{green:02x}/{blue:02x}"))
}

fn color_to_rgb_bytes(color: Color) -> Option<(u8, u8, u8)> {
    match color {
        Color::Black => Some((0x00, 0x00, 0x00)),
        Color::Red => Some((0xff, 0x00, 0x00)),
        Color::Green => Some((0x00, 0xff, 0x00)),
        Color::Yellow => Some((0xff, 0xff, 0x00)),
        Color::Blue => Some((0x00, 0x00, 0xff)),
        Color::Magenta => Some((0xff, 0x00, 0xff)),
        Color::Cyan => Some((0x00, 0xff, 0xff)),
        Color::Gray => Some((0xc0, 0xc0, 0xc0)),
        Color::DarkGray => Some((0x80, 0x80, 0x80)),
        Color::LightRed => Some((0xff, 0x55, 0x55)),
        Color::LightGreen => Some((0x55, 0xff, 0x55)),
        Color::LightYellow => Some((0xff, 0xff, 0x55)),
        Color::LightBlue => Some((0x55, 0x55, 0xff)),
        Color::LightMagenta => Some((0xff, 0x55, 0xff)),
        Color::LightCyan => Some((0x55, 0xff, 0xff)),
        Color::White => Some((0xff, 0xff, 0xff)),
        Color::Rgb(red, green, blue) => Some((red, green, blue)),
        Color::Reset | Color::Indexed(_) => None,
    }
}

fn detect_terminal_capabilities() -> TerminalCapabilities {
    let term_program = env::var("TERM_PROGRAM")
        .map(|value| value.to_ascii_lowercase())
        .unwrap_or_default();
    let term = env::var("TERM")
        .map(|value| value.to_ascii_lowercase())
        .unwrap_or_default();

    TerminalCapabilities {
        multiplexer: if env::var_os("TMUX").is_some() {
            Some(TerminalMultiplexer::Tmux)
        } else if env::var_os("STY").is_some() {
            Some(TerminalMultiplexer::Screen)
        } else {
            None
        },
        iterm2: term_program.contains("iterm"),
        kitty: env::var_os("KITTY_WINDOW_ID").is_some() || term.contains("kitty"),
    }
}

fn wrap_osc_for_multiplexer(sequence: &str, multiplexer: TerminalMultiplexer) -> String {
    match multiplexer {
        TerminalMultiplexer::Tmux => {
            let mut escaped = String::with_capacity(sequence.len() * 2);
            for ch in sequence.chars() {
                if ch == '\u{1b}' {
                    escaped.push('\u{1b}');
                }
                escaped.push(ch);
            }
            format!("\u{1b}Ptmux;{escaped}\u{1b}\\")
        }
        TerminalMultiplexer::Screen => format!("\u{1b}P{sequence}\u{1b}\\"),
    }
}

fn push_osc_single(sequences: &mut Vec<String>, code: &str, value: &str) {
    for terminator in ["\u{7}", "\u{1b}\\"] {
        sequences.push(format!("\u{1b}]{code};{value}{terminator}"));
    }
}

fn push_osc_pair(
    sequences: &mut Vec<String>,
    first_code: &str,
    first_value: &str,
    second_code: &str,
    second_value: &str,
) {
    for terminator in ["\u{7}", "\u{1b}\\"] {
        sequences.push(format!(
            "\u{1b}]{first_code};{first_value}{terminator}\u{1b}]{second_code};{second_value}{terminator}"
        ));
    }
}

fn append_multiplexer_variants(
    sequences: &mut Vec<String>,
    multiplexer: Option<TerminalMultiplexer>,
) {
    let Some(multiplexer) = multiplexer else {
        return;
    };
    sequences.extend(
        sequences
            .clone()
            .into_iter()
            .map(|sequence| wrap_osc_for_multiplexer(sequence.as_str(), multiplexer)),
    );
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TerminalMultiplexer {
    Tmux,
    Screen,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct TerminalCapabilities {
    multiplexer: Option<TerminalMultiplexer>,
    iterm2: bool,
    kitty: bool,
}

struct TerminalRestore {
    keyboard_enhancement_enabled: bool,
    mouse_capture_enabled: bool,
    bracketed_paste_enabled: bool,
    selection_colors_applied: bool,
}

impl Drop for TerminalRestore {
    fn drop(&mut self) {
        if self.keyboard_enhancement_enabled {
            let _ = execute!(io::stdout(), PopKeyboardEnhancementFlags);
        }
        let _ = disable_raw_mode();
        if self.selection_colors_applied {
            reset_selection_osc_colors();
        }
        if self.bracketed_paste_enabled {
            let _ = execute!(io::stdout(), DisableBracketedPaste);
        }
        if self.mouse_capture_enabled {
            let _ = execute!(io::stdout(), DisableMouseCapture);
        }
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
    }
}

#[cfg(test)]
#[path = "../tests/unit/lib_unit.rs"]
mod tests;
