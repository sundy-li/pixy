use std::future::Future;
use std::io;
use std::path::PathBuf;
use std::pin::Pin;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crossterm::event::{
    Event, EventStream, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, KeyboardEnhancementFlags,
    PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use crossterm::{execute, terminal};
use futures_util::StreamExt;
use pi_agent_core::{AgentAbortController, AgentAbortSignal};
use pi_ai::{AssistantContentBlock, Message, StopReason, ToolResultContentBlock};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};
use ratatui::{Frame, Terminal};
use tokio::sync::mpsc;
use tokio::time::MissedTickBehavior;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KeyBinding {
    pub code: KeyCode,
    pub modifiers: KeyModifiers,
}

impl KeyBinding {
    fn matches(self, key: KeyEvent) -> bool {
        key.code == self.code && key.modifiers == self.modifiers
    }
}

pub fn parse_key_id(key_id: &str) -> Option<KeyBinding> {
    let trimmed = key_id.trim().to_ascii_lowercase();
    if trimmed.is_empty() {
        return None;
    }

    let mut modifiers = KeyModifiers::NONE;
    let mut key_name = None;
    for segment in trimmed.split('+').filter(|part| !part.is_empty()) {
        match segment {
            "ctrl" | "control" => modifiers |= KeyModifiers::CONTROL,
            "shift" => modifiers |= KeyModifiers::SHIFT,
            "alt" => modifiers |= KeyModifiers::ALT,
            "meta" | "cmd" | "super" => modifiers |= KeyModifiers::SUPER,
            other => key_name = Some(other),
        }
    }

    let code = match key_name.unwrap_or_default() {
        "enter" => KeyCode::Enter,
        "escape" | "esc" => KeyCode::Esc,
        "tab" => KeyCode::Tab,
        "backspace" => KeyCode::Backspace,
        "up" => KeyCode::Up,
        "down" => KeyCode::Down,
        "left" => KeyCode::Left,
        "right" => KeyCode::Right,
        "space" => KeyCode::Char(' '),
        name if name.starts_with('f') => {
            let number = name[1..].parse::<u8>().ok()?;
            KeyCode::F(number)
        }
        name if name.len() == 1 => {
            let ch = name.chars().next()?;
            KeyCode::Char(ch)
        }
        _ => return None,
    };

    Some(KeyBinding { code, modifiers })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TuiKeyBindings {
    pub submit: Vec<KeyBinding>,
    pub newline: Vec<KeyBinding>,
    pub clear: Vec<KeyBinding>,
    pub continue_run: Vec<KeyBinding>,
    pub show_help: Vec<KeyBinding>,
    pub show_session: Vec<KeyBinding>,
    pub quit: Vec<KeyBinding>,
    pub interrupt: Vec<KeyBinding>,
    pub cycle_thinking_level: Vec<KeyBinding>,
    pub cycle_model_forward: Vec<KeyBinding>,
    pub cycle_model_backward: Vec<KeyBinding>,
    pub select_model: Vec<KeyBinding>,
    pub expand_tools: Vec<KeyBinding>,
    pub toggle_thinking: Vec<KeyBinding>,
}

impl Default for TuiKeyBindings {
    fn default() -> Self {
        Self {
            submit: vec![KeyBinding {
                code: KeyCode::Enter,
                modifiers: KeyModifiers::NONE,
            }],
            newline: vec![
                KeyBinding {
                    code: KeyCode::Enter,
                    modifiers: KeyModifiers::SHIFT,
                },
                KeyBinding {
                    code: KeyCode::Char('j'),
                    modifiers: KeyModifiers::CONTROL,
                },
            ],
            clear: vec![KeyBinding {
                code: KeyCode::Char('c'),
                modifiers: KeyModifiers::CONTROL,
            }],
            continue_run: vec![KeyBinding {
                code: KeyCode::Enter,
                modifiers: KeyModifiers::ALT,
            }],
            show_help: vec![KeyBinding {
                code: KeyCode::F(1),
                modifiers: KeyModifiers::NONE,
            }],
            show_session: vec![KeyBinding {
                code: KeyCode::Char('s'),
                modifiers: KeyModifiers::CONTROL,
            }],
            quit: vec![KeyBinding {
                code: KeyCode::Char('d'),
                modifiers: KeyModifiers::CONTROL,
            }],
            interrupt: vec![KeyBinding {
                code: KeyCode::Esc,
                modifiers: KeyModifiers::NONE,
            }],
            cycle_thinking_level: vec![KeyBinding {
                code: KeyCode::Tab,
                modifiers: KeyModifiers::SHIFT,
            }],
            cycle_model_forward: vec![KeyBinding {
                code: KeyCode::Char('p'),
                modifiers: KeyModifiers::CONTROL,
            }],
            cycle_model_backward: vec![KeyBinding {
                code: KeyCode::Char('p'),
                modifiers: KeyModifiers::CONTROL | KeyModifiers::SHIFT,
            }],
            select_model: vec![KeyBinding {
                code: KeyCode::Char('l'),
                modifiers: KeyModifiers::CONTROL,
            }],
            expand_tools: vec![KeyBinding {
                code: KeyCode::Char('o'),
                modifiers: KeyModifiers::CONTROL,
            }],
            toggle_thinking: vec![KeyBinding {
                code: KeyCode::Char('t'),
                modifiers: KeyModifiers::CONTROL,
            }],
        }
    }
}

#[derive(Clone, Debug)]
pub struct TuiOptions {
    pub app_name: String,
    pub version: String,
    pub show_tool_results: bool,
    pub keybindings: TuiKeyBindings,
    pub initial_help: bool,
    pub status_top: String,
    pub status_left: String,
    pub status_right: String,
}

impl Default for TuiOptions {
    fn default() -> Self {
        Self {
            app_name: "pi".to_string(),
            version: String::new(),
            show_tool_results: true,
            keybindings: TuiKeyBindings::default(),
            initial_help: false,
            status_top: String::new(),
            status_left: String::new(),
            status_right: String::new(),
        }
    }
}

struct TuiApp {
    input: String,
    cursor_pos: usize,
    input_history: Vec<String>,
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
    last_clear_key_at_ms: i64,
    queued_follow_ups: Vec<String>,
    transcript_scroll_from_bottom: usize,
    status_top: String,
    status_left: String,
    status_right: String,
}

impl TuiApp {
    fn new(status: String, show_tool_results: bool, show_help: bool) -> Self {
        Self {
            input: String::new(),
            cursor_pos: 0,
            input_history: vec![],
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
            last_clear_key_at_ms: 0,
            queued_follow_ups: vec![],
            transcript_scroll_from_bottom: 0,
            status_top: String::new(),
            status_left: String::new(),
            status_right: String::new(),
        }
    }

    fn set_status_bar_meta(&mut self, top: String, left: String, right: String) {
        self.status_top = top;
        self.status_left = left;
        self.status_right = right;
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
        self.cursor_pos = 0;
        self.reset_input_history_navigation();
    }

    fn take_input_trimmed(&mut self) -> String {
        self.cursor_pos = 0;
        self.reset_input_history_navigation();
        let trimmed = self.input.trim().to_string();
        self.input.clear();
        trimmed
    }

    fn insert_char(&mut self, ch: char) {
        self.reset_input_history_navigation();
        let byte_pos = self
            .input
            .char_indices()
            .nth(self.cursor_pos)
            .map(|(i, _)| i)
            .unwrap_or(self.input.len());
        self.input.insert(byte_pos, ch);
        self.cursor_pos += 1;
        self.scroll_transcript_to_latest();
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
    }

    fn bump_working_tick(&mut self) {
        self.working_tick = self.working_tick.saturating_add(1);
    }

    fn stop_working(&mut self) {
        self.is_working = false;
        self.working_message.clear();
        self.working_tick = 0;
    }

    fn queue_follow_up(&mut self, input: String) {
        self.queued_follow_ups.push(input);
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
                    // Keep the running tool headline while new tool output lines stream in.
                    if !is_tool_run_line(&self.working_message) {
                        self.working_message = format!("{app_name} is working...");
                    }
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
            ["-", "\\", "|", "/"]
        };
        let spinner = frames[self.working_tick % frames.len()];
        let text = if is_tool_run_line(&self.working_message) {
            format!("{spinner} {}", self.working_message)
        } else {
            format!("[{spinner}] {}", self.working_message)
        };
        Some(TranscriptLine::new(text, TranscriptLineKind::Working))
    }

    fn apply_stream_update(&mut self, update: StreamUpdate) {
        match update {
            StreamUpdate::AssistantTextDelta(delta) => {
                self.append_assistant_delta(&delta);
            }
            StreamUpdate::AssistantLine(line) => {
                self.assistant_stream_open = false;
                if !line.is_empty() {
                    let kind = if is_thinking_line(&line) {
                        TranscriptLineKind::Thinking
                    } else {
                        TranscriptLineKind::Normal
                    };
                    self.transcript.push(TranscriptLine::new(line, kind));
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

#[derive(Clone, Debug, PartialEq, Eq)]
enum TranscriptLineKind {
    Normal,
    Thinking,
    Tool,
    Working,
}

const TOOL_COMPACTION_HEAD_LINES: usize = 2;
const TOOL_COMPACTION_TAIL_LINES: usize = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
struct TranscriptLine {
    text: String,
    kind: TranscriptLineKind,
}

impl TranscriptLine {
    fn new(text: String, kind: TranscriptLineKind) -> Self {
        Self { text, kind }
    }

    fn to_line(&self, width: usize) -> Line<'static> {
        let base = match self.kind {
            TranscriptLineKind::Normal => Style::default(),
            TranscriptLineKind::Thinking => Style::default().fg(Color::DarkGray),
            TranscriptLineKind::Tool => Style::default().fg(Color::Gray),
            TranscriptLineKind::Working => Style::default()
                .fg(Color::Black)
                .bg(Color::White)
                .add_modifier(Modifier::BOLD),
        };

        let is_tool_diff_removed = matches!(self.kind, TranscriptLineKind::Tool)
            && self.text.trim_start().starts_with('-');
        let is_tool_diff_added = matches!(self.kind, TranscriptLineKind::Tool)
            && self.text.trim_start().starts_with('+');
        if is_tool_diff_removed || is_tool_diff_added {
            let style = if is_tool_diff_removed {
                base.fg(Color::Red)
            } else {
                base.fg(Color::Yellow)
            };
            return line_with_padding(vec![Span::styled(self.text.clone(), style)], width, style);
        }

        if matches!(self.kind, TranscriptLineKind::Working) {
            return line_with_padding(vec![Span::styled(self.text.clone(), base)], width, base);
        }

        let spans = highlighted_spans(self.text.as_str(), base);
        line_with_padding(spans, width, base)
    }
}

fn line_with_padding(mut spans: Vec<Span<'static>>, width: usize, style: Style) -> Line<'static> {
    if width == 0 {
        return Line::from(spans);
    }
    let text_width = spans
        .iter()
        .map(|span| UnicodeWidthStr::width(span.content.as_ref()))
        .sum::<usize>();
    if text_width < width {
        spans.push(Span::styled(" ".repeat(width - text_width), style));
    }
    Line::from(spans)
}

fn highlighted_spans(text: &str, base: Style) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let mut buffer = String::new();
    let mut buffer_is_token = None::<bool>;

    for ch in text.chars() {
        let is_token = is_token_char(ch);
        match buffer_is_token {
            None => {
                buffer.push(ch);
                buffer_is_token = Some(is_token);
            }
            Some(state) if state == is_token => {
                buffer.push(ch);
            }
            Some(state) => {
                spans.push(styled_fragment(&buffer, state, base));
                buffer.clear();
                buffer.push(ch);
                buffer_is_token = Some(is_token);
            }
        }
    }

    if let Some(state) = buffer_is_token {
        spans.push(styled_fragment(&buffer, state, base));
    }

    if spans.is_empty() {
        spans.push(Span::styled(String::new(), base));
    }

    spans
}

fn styled_fragment(fragment: &str, is_token: bool, base: Style) -> Span<'static> {
    if !is_token {
        return Span::styled(fragment.to_string(), base);
    }

    let style = if is_file_path_token(fragment) {
        base.fg(Color::Cyan).add_modifier(Modifier::BOLD)
    } else if is_key_token(fragment) {
        base.fg(Color::LightYellow).add_modifier(Modifier::BOLD)
    } else {
        base
    };
    Span::styled(fragment.to_string(), style)
}

fn is_token_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, '/' | '.' | '_' | '-' | ':' | '+' | '~')
}

fn is_file_path_token(token: &str) -> bool {
    let trimmed = token.trim_matches(|ch: char| {
        matches!(
            ch,
            ',' | ';' | '(' | ')' | '[' | ']' | '{' | '}' | '<' | '>' | '"' | '\'' | '`'
        )
    });
    if trimmed.is_empty() {
        return false;
    }
    if let Some((path, line)) = trimmed.rsplit_once(':') {
        if path.contains('/') && !line.is_empty() && line.chars().all(|ch| ch.is_ascii_digit()) {
            return true;
        }
    }
    (trimmed.contains('/') || trimmed.starts_with("~/") || trimmed.starts_with("./"))
        && (trimmed.contains('.') || trimmed.contains('/'))
}

fn is_key_token(token: &str) -> bool {
    let normalized = token
        .trim_matches(|ch: char| {
            matches!(
                ch,
                ',' | ';' | '(' | ')' | '[' | ']' | '{' | '}' | '<' | '>' | '"' | '\'' | '`' | ':'
            )
        })
        .to_ascii_lowercase();
    if normalized.contains("pageup") || normalized.contains("pagedown") {
        return true;
    }
    matches!(
        normalized.as_str(),
        "ctrl+a" | "ctrl+e" | "ctrl+w" | "ctrl+u" | "up" | "down"
    )
}

pub async fn run_tui<B: TuiBackend>(backend: &mut B, options: TuiOptions) -> Result<(), String> {
    enable_raw_mode().map_err(|error| format!("enable raw mode failed: {error}"))?;
    execute!(io::stdout(), EnterAlternateScreen)
        .map_err(|error| format!("enter alternate screen failed: {error}"))?;

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

    let _restore = TerminalRestore {
        keyboard_enhancement_enabled,
    };

    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))
        .map_err(|error| format!("create terminal failed: {error}"))?;
    terminal
        .clear()
        .map_err(|error| format!("clear terminal failed: {error}"))?;

    let status = backend
        .session_file()
        .map(|path| format!("session: {}", path.display()))
        .unwrap_or_else(|| "session: (none)".to_string());
    let mut app = TuiApp::new(status, options.show_tool_results, options.initial_help);
    app.set_status_bar_meta(
        options.status_top.clone(),
        options.status_left.clone(),
        options.status_right.clone(),
    );
    app.push_lines(build_welcome_banner(&options));

    let mut events = EventStream::new();
    loop {
        terminal
            .draw(|frame| render_ui(frame, &app, &options))
            .map_err(|error| format!("draw UI failed: {error}"))?;

        let maybe_event = events.next().await;
        let Some(event_result) = maybe_event else {
            return Ok(());
        };
        let event = event_result.map_err(|error| format!("read terminal event failed: {error}"))?;

        if let Event::Key(key) = event {
            if !matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
                continue;
            }

            if matches_keybinding(&options.keybindings.quit, key) {
                if !app.input.trim().is_empty() {
                    app.status = "input not empty; clear first".to_string();
                    continue;
                }
                return Ok(());
            }
            if matches_keybinding(&options.keybindings.interrupt, key) {
                app.clear_input();
                app.show_help = false;
                app.status = "interrupted".to_string();
                continue;
            }
            if matches_keybinding(&options.keybindings.clear, key) {
                if !app.input.is_empty() {
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
                if app.input.trim().is_empty() {
                    handle_continue_streaming(
                        backend,
                        &mut terminal,
                        &mut app,
                        &options,
                        &mut events,
                    )
                    .await?;
                    process_queued_follow_ups(
                        backend,
                        &mut terminal,
                        &mut app,
                        &options,
                        &mut events,
                    )
                    .await?;
                } else {
                    let submitted = app.take_input_trimmed();
                    app.record_input_history(&submitted);
                    app.scroll_transcript_to_latest();
                    app.push_lines([format!("{}> {}", options.app_name, submitted)]);
                    run_prompt_streaming(
                        backend,
                        &mut terminal,
                        &mut app,
                        &options,
                        &submitted,
                        &mut events,
                    )
                    .await?;
                    process_queued_follow_ups(
                        backend,
                        &mut terminal,
                        &mut app,
                        &options,
                        &mut events,
                    )
                    .await?;
                }
                continue;
            }
            if matches_keybinding(&options.keybindings.newline, key) {
                app.insert_char('\n');
                continue;
            }
            if matches_keybinding(&options.keybindings.submit, key) {
                let submitted = app.take_input_trimmed();
                if submitted.is_empty() {
                    continue;
                }
                app.record_input_history(&submitted);
                app.scroll_transcript_to_latest();
                if submitted == "/continue" {
                    handle_continue_streaming(
                        backend,
                        &mut terminal,
                        &mut app,
                        &options,
                        &mut events,
                    )
                    .await?;
                    process_queued_follow_ups(
                        backend,
                        &mut terminal,
                        &mut app,
                        &options,
                        &mut events,
                    )
                    .await?;
                    continue;
                }
                match handle_slash_command(&submitted, backend, &mut app).await {
                    Ok(true) => continue,
                    Ok(false) => {}
                    Err(error) if error == "__EXIT__" => return Ok(()),
                    Err(error) => return Err(error),
                }
                app.push_lines([format!("{}> {}", options.app_name, submitted)]);
                run_prompt_streaming(
                    backend,
                    &mut terminal,
                    &mut app,
                    &options,
                    &submitted,
                    &mut events,
                )
                .await?;
                process_queued_follow_ups(backend, &mut terminal, &mut app, &options, &mut events)
                    .await?;
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
        KeyCode::Up if key.modifiers == KeyModifiers::NONE => app.navigate_input_history_up(),
        KeyCode::Down if key.modifiers == KeyModifiers::NONE => app.navigate_input_history_down(),
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

fn build_welcome_banner(options: &TuiOptions) -> Vec<String> {
    let kb = &options.keybindings;
    let version_suffix = if options.version.is_empty() {
        String::new()
    } else {
        format!(" v{}", options.version)
    };

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
    let newline_label = keybinding_label_lower(&kb.newline);

    vec![
        String::new(),
        format!(" {}{}", options.app_name, version_suffix),
        format!(" {} to interrupt", interrupt_label),
        format!(" {} to clear", clear_label),
        format!(" {} twice to exit", clear_label),
        format!(" {} to exit (empty)", quit_label),
        format!(" {} to cycle thinking level", cycle_thinking_label),
        format!(" {}/{} to cycle models", cycle_model_fwd, cycle_model_bwd),
        format!(" {} to select model", select_model_label),
        format!(" {} to expand tools", expand_tools_label),
        format!(" {} to expand thinking", toggle_thinking_label),
        format!(" / for commands"),
        format!(" {} to queue follow-up", follow_up_label),
        " ctrl+a/e: move cursor, ctrl+w/u: delete backward".to_string(),
        format!(" {}: newline", newline_label),
        " up/down: input history, pageup/pagedown: scroll messages".to_string(),
        String::new(),
    ]
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
        command if command.starts_with("/resume") => {
            let target = command
                .strip_prefix("/resume")
                .map(str::trim)
                .filter(|value| !value.is_empty());
            match backend.resume_session(target) {
                Ok(Some(status)) => {
                    app.status = status;
                    Ok(true)
                }
                Ok(None) => {
                    app.status = "resume is not supported by this backend".to_string();
                    Ok(true)
                }
                Err(error) => {
                    app.push_lines([format!("[resume_error] {error}")]);
                    app.status = format!("resume failed: {error}");
                    Ok(true)
                }
            }
        }
        "/exit" | "/quit" => Err("__EXIT__".to_string()),
        _ => Ok(false),
    }
}

async fn run_prompt_streaming<B: TuiBackend>(
    backend: &mut B,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut TuiApp,
    options: &TuiOptions,
    input: &str,
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
        backend.prompt_stream(input, Some(abort_controller.signal()), &mut on_update);
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
                        &options.keybindings.interrupt,
                        &options.keybindings.continue_run,
                        &options.keybindings.newline,
                        &abort_controller,
                        app,
                    );
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
                        &options.keybindings.interrupt,
                        &options.keybindings.continue_run,
                        &options.keybindings.newline,
                        &abort_controller,
                        app,
                    );
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
        app.push_lines([format!("{}> {}", options.app_name, queued)]);
        run_prompt_streaming(backend, terminal, app, options, &queued, events).await?;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct StreamingEventOutcome {
    interrupted: bool,
    ui_changed: bool,
}

fn handle_streaming_event(
    event: Event,
    interrupt_bindings: &[KeyBinding],
    follow_up_bindings: &[KeyBinding],
    newline_bindings: &[KeyBinding],
    abort_controller: &AgentAbortController,
    app: &mut TuiApp,
) -> StreamingEventOutcome {
    let Event::Key(key) = event else {
        return StreamingEventOutcome::default();
    };
    if !matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
        return StreamingEventOutcome::default();
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
        };
    }

    if matches_keybinding(follow_up_bindings, key) {
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
        };
    }

    if matches_keybinding(newline_bindings, key) {
        app.insert_char('\n');
        return StreamingEventOutcome {
            interrupted: false,
            ui_changed: true,
        };
    }

    if handle_input_history_key_event(app, key) {
        return StreamingEventOutcome {
            interrupted: false,
            ui_changed: true,
        };
    }

    if handle_transcript_scroll_key(app, key) {
        return StreamingEventOutcome {
            interrupted: false,
            ui_changed: true,
        };
    }

    if handle_editor_key_event(app, key) {
        return StreamingEventOutcome {
            interrupted: false,
            ui_changed: true,
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

fn render_ui(frame: &mut Frame, app: &TuiApp, options: &TuiOptions) {
    let input_height = input_area_height(app, frame.area());
    let areas = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(1),
            Constraint::Length(input_height),
            Constraint::Length(2),
        ])
        .split(frame.area());

    let transcript_area = areas[0];
    let input_area = areas[1];
    let footer_area = areas[2];

    let visible_lines = visible_transcript_lines(
        &app.transcript,
        transcript_area.height.saturating_sub(2) as usize,
        transcript_area.width.saturating_sub(2) as usize,
        app.show_tool_results,
        app.show_thinking,
        app.working_line(),
        app.transcript_scroll_from_bottom,
    );
    let transcript = Paragraph::new(Text::from(visible_lines))
        .block(
            Block::default()
                .borders(Borders::NONE)
                .title(format!("{} Chat", options.app_name)),
        )
        .style(Style::default().fg(Color::White).bg(Color::Black));
    frame.render_widget(transcript, transcript_area);

    let input_scroll = input_scroll_offset(app, input_area);
    let input = Paragraph::new(app.input.as_str())
        .block(
            Block::default()
                .borders(Borders::TOP | Borders::BOTTOM)
                .border_style(Style::default().fg(Color::Green)),
        )
        .wrap(Wrap { trim: false })
        .scroll((input_scroll, 0))
        .style(Style::default().fg(Color::White).bg(Color::Black));
    frame.render_widget(input, input_area);

    if !app.show_help {
        let (cursor_x, cursor_y) = input_cursor_position(app, input_area);
        frame.set_cursor_position((cursor_x, cursor_y));
    }

    let footer = Paragraph::new(render_status_bar_lines(app, footer_area.width as usize)).style(
        Style::default()
            .fg(Color::DarkGray)
            .bg(Color::Black)
            .add_modifier(Modifier::DIM),
    );
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
                "  {:<14} quit",
                keybinding_label(&options.keybindings.quit)
            )),
            Line::from(""),
            Line::from("Slash Commands"),
            Line::from(format!(
                "  /continue ({continue_key}) /resume [session] /session /help /exit"
            )),
            Line::from("  Ctrl+A / Ctrl+E move cursor"),
            Line::from("  Ctrl+W / Ctrl+U delete backward"),
            Line::from(format!("  {} insert newline", keybinding_label(&options.keybindings.newline))),
            Line::from("  Up/Down input history, PageUp/PageDown scroll messages"),
            Line::from(""),
            Line::from("Use interrupt or help key to close help."),
        ]))
        .block(Block::default().title("Help").borders(Borders::ALL))
        .wrap(Wrap { trim: false });
        frame.render_widget(help, popup);
    }
}

fn input_cursor_position(app: &TuiApp, input_area: Rect) -> (u16, u16) {
    let (x, y, _) = input_cursor_layout(app, input_area);
    (x, y)
}

fn input_scroll_offset(app: &TuiApp, input_area: Rect) -> u16 {
    let (_, _, scroll) = input_cursor_layout(app, input_area);
    scroll
}

fn input_cursor_layout(app: &TuiApp, input_area: Rect) -> (u16, u16, u16) {
    let inner_width = input_area.width as usize;
    let inner_height = input_area.height.saturating_sub(2) as usize;
    if inner_width == 0 || inner_height == 0 {
        let fallback_y = input_area
            .y
            .saturating_add(input_area.height.saturating_sub(1));
        return (input_area.x, fallback_y, 0);
    }

    let (row, col) = input_cursor_row_col(app.input.as_str(), app.cursor_pos, inner_width);
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

fn input_cursor_row_col(input: &str, cursor_pos: usize, max_width: usize) -> (usize, usize) {
    if max_width == 0 {
        return (0, 0);
    }

    let mut row = 0usize;
    let mut col = 0usize;

    for ch in input.chars().take(cursor_pos) {
        if ch == '\n' {
            row = row.saturating_add(1);
            col = 0;
            continue;
        }

        let ch_width = UnicodeWidthChar::width(ch).unwrap_or(0);
        if ch_width == 0 {
            continue;
        }

        if col > 0 && col + ch_width > max_width {
            row = row.saturating_add(1);
            col = 0;
        }

        col += ch_width;
        if col >= max_width {
            row = row.saturating_add(1);
            col = 0;
        }
    }

    (row, col)
}

fn input_area_height(app: &TuiApp, frame_area: Rect) -> u16 {
    // Input block has top+bottom borders only.
    let inner_width = frame_area.width as usize;
    let line_count = wrap_text_by_display_width(app.input.as_str(), inner_width)
        .len()
        .max(1);
    let desired_height = line_count.saturating_add(2) as u16;

    let max_height = frame_area.height.saturating_sub(2).max(1);
    let min_height = 3u16.min(max_height);
    desired_height.max(min_height).min(max_height)
}

fn render_status_bar_lines(app: &TuiApp, width: usize) -> Text<'static> {
    let mut top = if app.status_top.is_empty() {
        app.status.clone()
    } else {
        app.status_top.clone()
    };
    if !app.status_top.is_empty() && !app.status.is_empty() && app.status != "ok" {
        top = format!("{top} | {}", app.status);
    }
    let bottom =
        compose_left_right_status_line(app.status_left.as_str(), app.status_right.as_str(), width);
    Text::from(vec![Line::from(top), Line::from(bottom)])
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

fn visible_transcript_lines(
    lines: &[TranscriptLine],
    max_lines: usize,
    max_width: usize,
    show_tool_results: bool,
    show_thinking: bool,
    working_line: Option<TranscriptLine>,
    scroll_from_bottom: usize,
) -> Vec<Line<'static>> {
    if max_lines == 0 || max_width == 0 {
        return vec![];
    }

    let mut filtered = lines
        .iter()
        .filter(|line| match line.kind {
            TranscriptLineKind::Normal => true,
            TranscriptLineKind::Thinking => show_thinking,
            TranscriptLineKind::Tool => show_tool_results,
            TranscriptLineKind::Working => true,
        })
        .cloned()
        .collect::<Vec<_>>();

    if let Some(working_line) = working_line {
        filtered.push(working_line);
    }

    let compacted = compact_tool_transcript_lines(&filtered);
    let wrapped = wrap_transcript_lines(&compacted, max_width);
    let max_scroll = wrapped.len().saturating_sub(max_lines);
    let scroll = scroll_from_bottom.min(max_scroll);
    let end = wrapped.len().saturating_sub(scroll);
    let start = end.saturating_sub(max_lines);
    wrapped[start..end]
        .iter()
        .map(|line| line.to_line(max_width))
        .collect()
}

fn wrap_transcript_lines(lines: &[TranscriptLine], max_width: usize) -> Vec<TranscriptLine> {
    let mut wrapped = Vec::new();
    for line in lines {
        for segment in wrap_text_by_display_width(&line.text, max_width) {
            wrapped.push(TranscriptLine::new(segment, line.kind.clone()));
        }
    }
    wrapped
}

fn compact_tool_transcript_lines(lines: &[TranscriptLine]) -> Vec<TranscriptLine> {
    let mut compacted = Vec::with_capacity(lines.len());
    let mut cursor = 0usize;
    while cursor < lines.len() {
        if lines[cursor].kind != TranscriptLineKind::Tool {
            compacted.push(lines[cursor].clone());
            cursor += 1;
            continue;
        }

        let mut block_end = cursor;
        while block_end < lines.len() && lines[block_end].kind == TranscriptLineKind::Tool {
            block_end += 1;
        }
        compacted.extend(compact_tool_block(&lines[cursor..block_end]));
        cursor = block_end;
    }
    compacted
}

fn compact_tool_block(lines: &[TranscriptLine]) -> Vec<TranscriptLine> {
    if lines.is_empty() {
        return vec![];
    }

    let mut compacted = Vec::new();
    let mut cursor = 0usize;
    while cursor < lines.len() {
        let line = &lines[cursor];
        if is_tool_run_line(line.text.as_str()) {
            compacted.push(line.clone());
            cursor += 1;
        } else if let Some((tool_name, is_error)) = parse_legacy_tool_header(line.text.as_str()) {
            let title = if is_error {
                format!("• Ran {tool_name} (error)")
            } else {
                format!("• Ran {tool_name}")
            };
            compacted.push(TranscriptLine::new(title, TranscriptLineKind::Tool));
            cursor += 1;
        } else {
            compacted.extend(compact_tool_body_lines(&lines[cursor..]));
            break;
        }

        let body_start = cursor;
        while cursor < lines.len()
            && !is_tool_run_line(lines[cursor].text.as_str())
            && parse_legacy_tool_header(lines[cursor].text.as_str()).is_none()
        {
            cursor += 1;
        }
        compacted.extend(compact_tool_body_lines(&lines[body_start..cursor]));
    }

    compacted
}

fn compact_tool_body_lines(lines: &[TranscriptLine]) -> Vec<TranscriptLine> {
    if lines.len() <= TOOL_COMPACTION_HEAD_LINES + TOOL_COMPACTION_TAIL_LINES {
        return lines.to_vec();
    }

    let hidden = lines
        .len()
        .saturating_sub(TOOL_COMPACTION_HEAD_LINES + TOOL_COMPACTION_TAIL_LINES);
    if hidden == 0 {
        return lines.to_vec();
    }

    let mut compacted = Vec::new();
    compacted.extend(lines.iter().take(TOOL_COMPACTION_HEAD_LINES).cloned());
    let suffix = if hidden == 1 { "line" } else { "lines" };
    compacted.push(TranscriptLine::new(
        format!("    … +{hidden} {suffix}"),
        TranscriptLineKind::Tool,
    ));
    compacted.extend(
        lines
            .iter()
            .skip(lines.len() - TOOL_COMPACTION_TAIL_LINES)
            .cloned(),
    );
    compacted
}

fn wrap_text_by_display_width(text: &str, max_width: usize) -> Vec<String> {
    if max_width == 0 {
        return vec![String::new()];
    }

    let mut lines = Vec::new();
    for raw_line in text.split('\n') {
        if raw_line.is_empty() {
            lines.push(String::new());
            continue;
        }

        let mut current = String::new();
        let mut current_width = 0usize;

        for ch in raw_line.chars() {
            let ch_width = UnicodeWidthChar::width(ch).unwrap_or(0);
            if current_width > 0 && current_width + ch_width > max_width {
                lines.push(current);
                current = String::new();
                current_width = 0;
            }

            current.push(ch);
            current_width += ch_width;

            if current_width >= max_width && ch_width > 0 {
                lines.push(current);
                current = String::new();
                current_width = 0;
            }
        }

        if !current.is_empty() {
            lines.push(current);
        }
    }

    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
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

fn render_messages(messages: &[Message]) -> Vec<TranscriptLine> {
    let mut lines = Vec::new();
    for message in messages {
        match message {
            Message::Assistant {
                content,
                stop_reason,
                error_message,
                ..
            } => {
                for block in content {
                    match block {
                        AssistantContentBlock::Text { text, .. } => {
                            if !text.trim().is_empty() {
                                lines.push(TranscriptLine::new(
                                    text.clone(),
                                    TranscriptLineKind::Normal,
                                ));
                            }
                        }
                        AssistantContentBlock::Thinking { thinking, .. } => {
                            if !thinking.trim().is_empty() {
                                lines.push(TranscriptLine::new(
                                    format!("[thinking] {thinking}"),
                                    TranscriptLineKind::Thinking,
                                ));
                            }
                        }
                        AssistantContentBlock::ToolCall { .. } => {}
                    }
                }
                if matches!(stop_reason, StopReason::Error | StopReason::Aborted) {
                    if let Some(error) = error_message {
                        lines.push(TranscriptLine::new(
                            format!("[assistant_{}] {error}", stop_reason_label(stop_reason)),
                            TranscriptLineKind::Normal,
                        ));
                    }
                }
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
                lines.push(TranscriptLine::new(title, TranscriptLineKind::Tool));
                for block in content {
                    match block {
                        ToolResultContentBlock::Text { text, .. } => {
                            for tool_line in split_tool_output_lines(text) {
                                lines
                                    .push(TranscriptLine::new(tool_line, TranscriptLineKind::Tool));
                            }
                        }
                        ToolResultContentBlock::Image { .. } => lines.push(TranscriptLine::new(
                            "(image tool result omitted)".to_string(),
                            TranscriptLineKind::Tool,
                        )),
                    }
                }
            }
            Message::User { .. } => {}
        }
    }
    lines
}

fn is_thinking_line(line: &str) -> bool {
    line.starts_with("[thinking]")
}

fn parse_tool_name(line: &str) -> Option<&str> {
    if let Some((name, _)) = parse_legacy_tool_header(line) {
        return Some(name);
    }
    line.strip_prefix("• Ran ")
        .and_then(|rest| rest.split_whitespace().next())
        .filter(|name| !name.is_empty())
}

fn parse_legacy_tool_header(line: &str) -> Option<(&str, bool)> {
    let rest = line.strip_prefix("[tool:")?.strip_suffix(']')?;
    let mut parts = rest.splitn(2, ':');
    let tool_name = parts.next()?.trim();
    if tool_name.is_empty() {
        return None;
    }
    let status = parts.next().unwrap_or_default();
    Some((tool_name, status == "error"))
}

fn normalize_tool_line_for_display(line: String) -> String {
    if let Some((tool_name, is_error)) = parse_legacy_tool_header(line.as_str()) {
        if is_error {
            return format!("• Ran {tool_name} (error)");
        }
        return format!("• Ran {tool_name}");
    }
    line
}

fn split_tool_output_lines(text: &str) -> Vec<String> {
    text.split('\n')
        .map(|line| line.trim_end_matches('\r').to_string())
        .collect()
}

fn is_tool_run_line(line: &str) -> bool {
    line.starts_with("• Ran ")
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

struct TerminalRestore {
    keyboard_enhancement_enabled: bool,
}

impl Drop for TerminalRestore {
    fn drop(&mut self) {
        if self.keyboard_enhancement_enabled {
            let _ = execute!(io::stdout(), PopKeyboardEnhancementFlags);
        }
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
        let _ = terminal::disable_raw_mode();
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn line_text(line: &Line<'_>) -> String {
        line.spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect()
    }

    struct TestBackend {
        resume_result: Result<Option<String>, String>,
        resume_targets: Vec<Option<String>>,
    }

    impl TuiBackend for TestBackend {
        fn prompt<'a>(&'a mut self, _input: &'a str) -> BackendFuture<'a> {
            Box::pin(async { Ok(vec![]) })
        }

        fn continue_run<'a>(&'a mut self) -> BackendFuture<'a> {
            Box::pin(async { Ok(vec![]) })
        }

        fn prompt_stream<'a>(
            &'a mut self,
            _input: &'a str,
            _abort_signal: Option<AgentAbortSignal>,
            _on_update: &'a mut dyn FnMut(StreamUpdate),
        ) -> BackendFuture<'a> {
            Box::pin(async { Ok(vec![]) })
        }

        fn continue_run_stream<'a>(
            &'a mut self,
            _abort_signal: Option<AgentAbortSignal>,
            _on_update: &'a mut dyn FnMut(StreamUpdate),
        ) -> BackendFuture<'a> {
            Box::pin(async { Ok(vec![]) })
        }

        fn resume_session(&mut self, session_ref: Option<&str>) -> Result<Option<String>, String> {
            self.resume_targets.push(session_ref.map(ToOwned::to_owned));
            self.resume_result.clone()
        }

        fn session_file(&self) -> Option<PathBuf> {
            None
        }
    }

    #[test]
    fn default_keybindings_match_expected_keys() {
        let bindings = TuiKeyBindings::default();
        assert_eq!(
            bindings.submit,
            vec![KeyBinding {
                code: KeyCode::Enter,
                modifiers: KeyModifiers::NONE
            }]
        );
        assert_eq!(
            bindings.clear,
            vec![KeyBinding {
                code: KeyCode::Char('c'),
                modifiers: KeyModifiers::CONTROL
            }]
        );
        assert_eq!(
            bindings.expand_tools,
            vec![KeyBinding {
                code: KeyCode::Char('o'),
                modifiers: KeyModifiers::CONTROL
            }]
        );
        assert_eq!(
            bindings.continue_run,
            vec![KeyBinding {
                code: KeyCode::Enter,
                modifiers: KeyModifiers::ALT
            }]
        );
        assert_eq!(
            bindings.toggle_thinking,
            vec![KeyBinding {
                code: KeyCode::Char('t'),
                modifiers: KeyModifiers::CONTROL
            }]
        );
        assert_eq!(
            bindings.interrupt,
            vec![KeyBinding {
                code: KeyCode::Esc,
                modifiers: KeyModifiers::NONE
            }]
        );
        assert_eq!(
            bindings.cycle_thinking_level,
            vec![KeyBinding {
                code: KeyCode::Tab,
                modifiers: KeyModifiers::SHIFT
            }]
        );
        assert_eq!(
            bindings.cycle_model_forward,
            vec![KeyBinding {
                code: KeyCode::Char('p'),
                modifiers: KeyModifiers::CONTROL
            }]
        );
        assert_eq!(
            bindings.cycle_model_backward,
            vec![KeyBinding {
                code: KeyCode::Char('p'),
                modifiers: KeyModifiers::CONTROL | KeyModifiers::SHIFT
            }]
        );
        assert_eq!(
            bindings.select_model,
            vec![KeyBinding {
                code: KeyCode::Char('l'),
                modifiers: KeyModifiers::CONTROL
            }]
        );
        assert_eq!(
            bindings.quit,
            vec![KeyBinding {
                code: KeyCode::Char('d'),
                modifiers: KeyModifiers::CONTROL
            }]
        );
    }

    #[test]
    fn parse_key_id_supports_common_shortcuts() {
        assert_eq!(
            parse_key_id("ctrl+o"),
            Some(KeyBinding {
                code: KeyCode::Char('o'),
                modifiers: KeyModifiers::CONTROL
            })
        );
        assert_eq!(
            parse_key_id("shift+tab"),
            Some(KeyBinding {
                code: KeyCode::Tab,
                modifiers: KeyModifiers::SHIFT
            })
        );
        assert_eq!(
            parse_key_id("shift+ctrl+p"),
            Some(KeyBinding {
                code: KeyCode::Char('p'),
                modifiers: KeyModifiers::SHIFT | KeyModifiers::CONTROL
            })
        );
        assert_eq!(
            parse_key_id("esc"),
            Some(KeyBinding {
                code: KeyCode::Esc,
                modifiers: KeyModifiers::NONE
            })
        );
        assert_eq!(parse_key_id(""), None);
        assert_eq!(parse_key_id("ctrl+unknown"), None);
    }

    #[test]
    fn editor_supports_common_emacs_shortcuts() {
        let mut app = TuiApp::new("ready".to_string(), true, false);
        app.input = "hello rust world".to_string();
        app.cursor_pos = app.input_char_count();

        assert!(handle_editor_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Char('w'), KeyModifiers::CONTROL),
        ));
        assert_eq!(app.input, "hello rust ");
        assert_eq!(app.cursor_pos, "hello rust ".chars().count());

        assert!(handle_editor_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Char('a'), KeyModifiers::CONTROL),
        ));
        assert_eq!(app.cursor_pos, 0);

        assert!(handle_editor_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Char('e'), KeyModifiers::CONTROL),
        ));
        assert_eq!(app.cursor_pos, app.input_char_count());

        assert!(handle_editor_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Char('u'), KeyModifiers::CONTROL),
        ));
        assert!(app.input.is_empty());
        assert_eq!(app.cursor_pos, 0);
    }

    #[test]
    fn editing_input_resets_transcript_scroll_to_latest() {
        let mut app = TuiApp::new("ready".to_string(), true, false);
        app.scroll_transcript_up(8);
        assert_eq!(app.transcript_scroll_from_bottom, 8);

        assert!(handle_editor_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Char('h'), KeyModifiers::NONE),
        ));
        assert_eq!(app.transcript_scroll_from_bottom, 0);
    }

    #[test]
    fn shift_enter_inserts_newline_and_moves_cursor_to_next_input_row() {
        let mut app = TuiApp::new("ready".to_string(), true, false);
        app.input = "hello".to_string();
        app.cursor_pos = app.input_char_count();

        assert!(handle_editor_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::SHIFT),
        ));
        assert_eq!(app.input, "hello\n");
        assert_eq!(app.cursor_pos, "hello\n".chars().count());

        let area = Rect {
            x: 0,
            y: 0,
            width: 10,
            height: 4,
        };
        assert_eq!(input_cursor_position(&app, area), (0, 2));
    }

    #[test]
    fn input_area_height_grows_and_shrinks_with_multiline_input() {
        let mut app = TuiApp::new("ready".to_string(), true, false);
        let frame = Rect {
            x: 0,
            y: 0,
            width: 20,
            height: 24,
        };
        assert_eq!(input_area_height(&app, frame), 3);

        app.input = "line 1\nline 2".to_string();
        assert_eq!(input_area_height(&app, frame), 4);

        app.input = "line 1".to_string();
        assert_eq!(input_area_height(&app, frame), 3);
    }

    #[test]
    fn up_down_keys_navigate_input_history() {
        let mut app = TuiApp::new("ready".to_string(), true, false);
        app.record_input_history("first");
        app.record_input_history("second");
        app.input = "draft".to_string();
        app.cursor_pos = app.input_char_count();

        assert!(handle_input_history_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Up, KeyModifiers::NONE),
        ));
        assert_eq!(app.input, "second");

        assert!(handle_input_history_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Up, KeyModifiers::NONE),
        ));
        assert_eq!(app.input, "first");

        assert!(handle_input_history_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Down, KeyModifiers::NONE),
        ));
        assert_eq!(app.input, "second");

        assert!(handle_input_history_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Down, KeyModifiers::NONE),
        ));
        assert_eq!(app.input, "draft");
        assert_eq!(app.cursor_pos, app.input_char_count());
    }

    #[test]
    fn input_cursor_position_uses_display_width_for_cjk() {
        let mut app = TuiApp::new("ready".to_string(), true, false);
        app.input = "你a".to_string();
        app.cursor_pos = 1;
        let area = Rect {
            x: 0,
            y: 0,
            width: 10,
            height: 3,
        };

        assert_eq!(input_cursor_position(&app, area), (2, 1));

        app.cursor_pos = 2;
        assert_eq!(input_cursor_position(&app, area), (3, 1));
    }

    #[test]
    fn visible_transcript_picks_latest_lines() {
        let lines = vec![
            TranscriptLine::new("l1".to_string(), TranscriptLineKind::Normal),
            TranscriptLine::new("l2".to_string(), TranscriptLineKind::Tool),
            TranscriptLine::new("l3".to_string(), TranscriptLineKind::Thinking),
            TranscriptLine::new("l4".to_string(), TranscriptLineKind::Normal),
        ];
        let visible = visible_transcript_lines(&lines, 2, 80, true, true, None, 0);
        assert_eq!(visible.len(), 2);
        assert!(line_text(&visible[0]).starts_with("l3"));
        assert!(line_text(&visible[1]).starts_with("l4"));
    }

    #[test]
    fn visible_transcript_respects_tool_and_thinking_toggles() {
        let lines = vec![
            TranscriptLine::new("normal".to_string(), TranscriptLineKind::Normal),
            TranscriptLine::new("tool".to_string(), TranscriptLineKind::Tool),
            TranscriptLine::new("thinking".to_string(), TranscriptLineKind::Thinking),
        ];
        let visible = visible_transcript_lines(&lines, 10, 80, false, false, None, 0);
        assert_eq!(visible.len(), 1);
        assert!(line_text(&visible[0]).starts_with("normal"));
    }

    #[test]
    fn visible_transcript_appends_working_line() {
        let lines = vec![TranscriptLine::new(
            "normal".to_string(),
            TranscriptLineKind::Normal,
        )];
        let visible = visible_transcript_lines(
            &lines,
            10,
            80,
            true,
            true,
            Some(TranscriptLine::new(
                "[-] pi is working...".to_string(),
                TranscriptLineKind::Working,
            )),
            0,
        );
        assert_eq!(visible.len(), 2);
        assert!(line_text(&visible[0]).starts_with("normal"));
        assert_eq!(visible[1].spans[0].content, "[-] pi is working...");
    }

    #[test]
    fn visible_transcript_wraps_and_keeps_latest_at_bottom() {
        let lines = vec![
            TranscriptLine::new("old".to_string(), TranscriptLineKind::Normal),
            TranscriptLine::new("abcdefghijklm".to_string(), TranscriptLineKind::Normal),
        ];

        let visible = visible_transcript_lines(&lines, 2, 4, true, true, None, 0);
        assert_eq!(visible.len(), 2);
        assert!(line_text(&visible[0]).starts_with("ijkl"));
        assert!(line_text(&visible[1]).starts_with("m"));
    }

    #[test]
    fn visible_transcript_supports_scrolling_up_from_bottom() {
        let lines = vec![
            TranscriptLine::new("line 1".to_string(), TranscriptLineKind::Normal),
            TranscriptLine::new("line 2".to_string(), TranscriptLineKind::Normal),
            TranscriptLine::new("line 3".to_string(), TranscriptLineKind::Normal),
            TranscriptLine::new("line 4".to_string(), TranscriptLineKind::Normal),
        ];

        let bottom = visible_transcript_lines(&lines, 2, 80, true, true, None, 0);
        assert_eq!(bottom.len(), 2);
        assert!(line_text(&bottom[0]).starts_with("line 3"));
        assert!(line_text(&bottom[1]).starts_with("line 4"));

        let scrolled = visible_transcript_lines(&lines, 2, 80, true, true, None, 1);
        assert_eq!(scrolled.len(), 2);
        assert!(line_text(&scrolled[0]).starts_with("line 2"));
        assert!(line_text(&scrolled[1]).starts_with("line 3"));
    }

    #[test]
    fn tool_output_is_compacted_with_omitted_line_marker() {
        let lines = vec![
            TranscriptLine::new(
                "• Ran bash -lc 'cargo test'".to_string(),
                TranscriptLineKind::Tool,
            ),
            TranscriptLine::new("line 1".to_string(), TranscriptLineKind::Tool),
            TranscriptLine::new("line 2".to_string(), TranscriptLineKind::Tool),
            TranscriptLine::new("line 3".to_string(), TranscriptLineKind::Tool),
            TranscriptLine::new("line 4".to_string(), TranscriptLineKind::Tool),
            TranscriptLine::new("line 5".to_string(), TranscriptLineKind::Tool),
            TranscriptLine::new("line 6".to_string(), TranscriptLineKind::Tool),
        ];

        let visible = visible_transcript_lines(&lines, 20, 120, true, true, None, 0);
        let texts = visible.iter().map(line_text).collect::<Vec<_>>();
        assert!(texts.iter().any(|line| line.contains("• Ran bash -lc")));
        assert!(texts.iter().any(|line| line.contains("… +3 lines")));
        assert!(texts.iter().any(|line| line.contains("line 6")));
    }

    #[test]
    fn multiline_tool_update_is_split_and_compacted() {
        let mut app = TuiApp::new("ready".to_string(), true, false);
        app.apply_stream_update(StreamUpdate::ToolLine(
            "• Ran read /tmp/example.txt".to_string(),
        ));
        app.apply_stream_update(StreamUpdate::ToolLine(
            "line 1\nline 2\nline 3\nline 4\nline 5\nline 6".to_string(),
        ));

        let visible = visible_transcript_lines(&app.transcript, 20, 120, true, true, None, 0);
        let texts = visible.iter().map(line_text).collect::<Vec<_>>();
        assert!(
            texts
                .iter()
                .any(|line| line.contains("• Ran read /tmp/example.txt"))
        );
        assert!(texts.iter().any(|line| line.contains("… +3 lines")));
        assert!(texts.iter().any(|line| line.contains("line 6")));
    }

    #[test]
    fn tool_diff_lines_use_expected_colors() {
        let removed = TranscriptLine::new("- old assertion".to_string(), TranscriptLineKind::Tool)
            .to_line(40);
        let added = TranscriptLine::new("+ new assertion".to_string(), TranscriptLineKind::Tool)
            .to_line(40);
        assert_eq!(removed.spans[0].style.fg, Some(Color::Red));
        assert_eq!(added.spans[0].style.fg, Some(Color::Yellow));
    }

    #[test]
    fn tool_lines_keep_default_background_and_use_light_text() {
        let tool =
            TranscriptLine::new("tool output".to_string(), TranscriptLineKind::Tool).to_line(40);
        assert_eq!(tool.spans[0].style.fg, Some(Color::Gray));
        assert_eq!(tool.spans[0].style.bg, None);
    }

    #[test]
    fn highlights_file_path_and_key_tokens() {
        let line = TranscriptLine::new(
            "See crates/pi-tui/src/lib.rs:902 and PageUp / PageDown".to_string(),
            TranscriptLineKind::Normal,
        )
        .to_line(80);

        assert!(
            line.spans.iter().any(|span| {
                span.content.contains("crates/pi-tui/src/lib.rs:902")
                    && span.style.fg == Some(Color::Cyan)
            }),
            "file path token should be highlighted"
        );
        assert!(
            line.spans
                .iter()
                .any(|span| span.content.contains("PageUp")
                    && span.style.fg == Some(Color::LightYellow)),
            "PageUp token should be highlighted"
        );
        assert!(
            line.spans
                .iter()
                .any(|span| span.content.contains("PageDown")
                    && span.style.fg == Some(Color::LightYellow)),
            "PageDown token should be highlighted"
        );
    }

    #[test]
    fn parse_tool_name_extracts_name_from_tool_header() {
        assert_eq!(parse_tool_name("[tool:read:ok]"), Some("read"));
        assert_eq!(parse_tool_name("• Ran bash -lc 'cargo test'"), Some("bash"));
        assert_eq!(parse_tool_name("tool output line"), None);
    }

    #[test]
    fn apply_stream_update_normalizes_legacy_tool_header_to_ran_title() {
        let mut app = TuiApp::new("ready".to_string(), true, false);
        app.apply_stream_update(StreamUpdate::ToolLine("[tool:read:ok]".to_string()));
        assert_eq!(app.transcript.len(), 1);
        assert_eq!(app.transcript[0].text, "• Ran read");
    }

    #[tokio::test]
    async fn slash_resume_command_updates_status_when_backend_supports_resume() {
        let mut backend = TestBackend {
            resume_result: Ok(Some("session: /tmp/resumed.jsonl".to_string())),
            resume_targets: vec![],
        };
        let mut app = TuiApp::new("ready".to_string(), true, false);

        let handled = handle_slash_command("/resume old-session", &mut backend, &mut app)
            .await
            .expect("resume command should not error");

        assert!(handled);
        assert_eq!(
            backend.resume_targets,
            vec![Some("old-session".to_string())]
        );
        assert_eq!(app.status, "session: /tmp/resumed.jsonl");
    }

    #[tokio::test]
    async fn slash_resume_command_renders_resume_error() {
        let mut backend = TestBackend {
            resume_result: Err("boom".to_string()),
            resume_targets: vec![],
        };
        let mut app = TuiApp::new("ready".to_string(), true, false);

        let handled = handle_slash_command("/resume", &mut backend, &mut app)
            .await
            .expect("resume command should be handled");

        assert!(handled);
        assert_eq!(backend.resume_targets, vec![None]);
        assert_eq!(app.status, "resume failed: boom");
        assert!(
            app.transcript
                .iter()
                .any(|line| line.text == "[resume_error] boom"
                    && line.kind == TranscriptLineKind::Normal),
            "resume error should be appended to transcript"
        );
    }

    #[test]
    fn streaming_follow_up_key_queues_input_and_clears_editor() {
        let mut app = TuiApp::new("ready".to_string(), true, false);
        app.input = "next question".to_string();
        let interrupt = vec![KeyBinding {
            code: KeyCode::Esc,
            modifiers: KeyModifiers::NONE,
        }];
        let follow_up = vec![KeyBinding {
            code: KeyCode::Enter,
            modifiers: KeyModifiers::ALT,
        }];
        let abort_controller = AgentAbortController::new();

        let outcome = handle_streaming_event(
            Event::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::ALT)),
            &interrupt,
            &follow_up,
            &[],
            &abort_controller,
            &mut app,
        );

        assert!(!outcome.interrupted);
        assert!(outcome.ui_changed);
        assert!(app.input.is_empty());
        assert_eq!(app.input_history, vec!["next question".to_string()]);
        assert_eq!(app.queued_follow_up_count(), 1);
        assert_eq!(app.status, "queued follow-up (1)");
    }

    #[test]
    fn streaming_follow_up_key_does_not_queue_empty_input() {
        let mut app = TuiApp::new("ready".to_string(), true, false);
        let interrupt = vec![KeyBinding {
            code: KeyCode::Esc,
            modifiers: KeyModifiers::NONE,
        }];
        let follow_up = vec![KeyBinding {
            code: KeyCode::Enter,
            modifiers: KeyModifiers::ALT,
        }];
        let abort_controller = AgentAbortController::new();

        let outcome = handle_streaming_event(
            Event::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::ALT)),
            &interrupt,
            &follow_up,
            &[],
            &abort_controller,
            &mut app,
        );

        assert!(!outcome.interrupted);
        assert!(!outcome.ui_changed);
        assert_eq!(app.queued_follow_up_count(), 0);
        assert_eq!(app.status, "ready");
    }
}
