use std::fs;
use std::path::PathBuf;

use pixy_agent_core::AgentAbortSignal;
use pixy_ai::{AssistantContentBlock, Cost, StopReason, ToolResultContentBlock, Usage};

use super::*;

fn line_text(line: &Line<'_>) -> String {
    line.spans
        .iter()
        .map(|span| span.content.as_ref())
        .collect()
}

fn mouse_scroll_event(kind: crossterm::event::MouseEventKind) -> crossterm::event::MouseEvent {
    crossterm::event::MouseEvent {
        kind,
        column: 0,
        row: 0,
        modifiers: KeyModifiers::NONE,
    }
}

struct TestBackend {
    resume_result: Result<Option<String>, String>,
    resume_targets: Vec<Option<String>>,
    session_messages: Option<Vec<Message>>,
    recent_sessions_result: Result<Option<Vec<ResumeCandidate>>, String>,
    recent_sessions_limits: Vec<usize>,
    new_session_result: Result<Option<String>, String>,
    new_session_calls: usize,
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

    fn prompt_stream_with_blocks<'a>(
        &'a mut self,
        _input: &'a str,
        _blocks: Option<Vec<UserContentBlock>>,
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

    fn new_session(&mut self) -> Result<Option<String>, String> {
        self.new_session_calls += 1;
        self.new_session_result.clone()
    }

    fn session_messages(&self) -> Option<Vec<Message>> {
        self.session_messages.clone()
    }

    fn recent_resumable_sessions(
        &mut self,
        limit: usize,
    ) -> Result<Option<Vec<ResumeCandidate>>, String> {
        self.recent_sessions_limits.push(limit);
        self.recent_sessions_result.clone()
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
        bindings.dequeue,
        vec![KeyBinding {
            code: KeyCode::Up,
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
            code: KeyCode::Char('n'),
            modifiers: KeyModifiers::CONTROL
        }]
    );
    assert_eq!(
        bindings.cycle_model_backward,
        vec![KeyBinding {
            code: KeyCode::Char('n'),
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
    assert_eq!(input_cursor_position(&app, area, "> "), (1, 2));
}

#[test]
fn input_area_height_is_fixed_with_multiline_input() {
    let mut app = TuiApp::new("ready".to_string(), true, false);
    let frame = Rect {
        x: 0,
        y: 0,
        width: 20,
        height: 24,
    };
    assert_eq!(
        input_area_height(&app, frame, "> ", 2),
        INPUT_AREA_FIXED_HEIGHT
    );

    app.input = "line 1\nline 2".to_string();
    assert_eq!(
        input_area_height(&app, frame, "> ", 2),
        INPUT_AREA_FIXED_HEIGHT
    );

    app.input = "line 1\nline 2\nline 3\nline 4".to_string();
    assert_eq!(
        input_area_height(&app, frame, "> ", 2),
        INPUT_AREA_FIXED_HEIGHT
    );
}

#[test]
fn terminal_defaults_to_fullscreen_viewport() {
    assert!(matches!(
        default_terminal_options().viewport,
        Viewport::Fullscreen
    ));
}

#[test]
fn welcome_banner_is_persisted_into_transcript() {
    let mut app = TuiApp::new("ready".to_string(), true, false);
    app.set_welcome_lines(vec!["Welcome line".to_string()]);

    persist_welcome_into_transcript(&mut app);
    assert!(
        app.transcript
            .iter()
            .any(|line| line.text.contains("Welcome line"))
    );
}

#[test]
fn persisted_welcome_lines_start_as_overlay_transcript_kind() {
    let mut app = TuiApp::new("ready".to_string(), true, false);
    app.set_welcome_lines(vec!["Welcome line".to_string()]);

    persist_welcome_into_transcript(&mut app);

    let welcome = app
        .transcript
        .iter()
        .find(|line| line.text == "Welcome line")
        .expect("welcome line should be present");
    assert_eq!(welcome.kind, TranscriptLineKind::Overlay);
}

#[test]
fn overlay_welcome_lines_are_center_aligned_in_transcript() {
    let mut app = TuiApp::new("ready".to_string(), true, false);
    app.set_welcome_lines(vec![
        "███████   ██  ██      ██  ██      ██".to_string(),
        "You are standing in an open terminal.".to_string(),
    ]);

    persist_welcome_into_transcript(&mut app);

    let visible = visible_transcript_lines(
        &app.transcript,
        &[],
        12,
        80,
        true,
        true,
        None,
        0,
        TuiTheme::Dark,
    );
    let centered = visible
        .iter()
        .map(line_text)
        .find(|line| line.contains("You are standing in an open terminal."))
        .expect("welcome line should be visible");

    let left_padding = centered.chars().take_while(|ch| *ch == ' ').count();
    assert!(
        left_padding > 0,
        "overlay welcome line should not be left-aligned when rendered in transcript"
    );
}

#[test]
fn overlay_logo_line_uses_reference_foreground_in_dark_theme() {
    let line = TranscriptLine::new("█ PIXY".to_string(), TranscriptLineKind::Overlay)
        .to_line(40, TuiTheme::Dark);
    assert!(
        line.spans.iter().any(|span| {
            span.content.contains("█")
                && span.style.fg == Some(Color::Rgb(192, 202, 245))
                && span
                    .style
                    .add_modifier
                    .contains(ratatui::style::Modifier::BOLD)
        }),
        "overlay logo text should use the dark theme transcript tone and bold weight"
    );
}

#[test]
fn overlay_version_line_uses_muted_tone_in_dark_theme() {
    let line = TranscriptLine::new("v0.1.0".to_string(), TranscriptLineKind::Overlay)
        .to_line(40, TuiTheme::Dark);
    assert!(
        line.spans.iter().any(|span| span.content.contains("v0.1.0")
            && span.style.fg == Some(Color::Rgb(86, 95, 137))),
        "overlay version line should use muted dark theme tone"
    );
}

#[test]
fn first_submitted_input_keeps_overlay_welcome_lines_as_overlay() {
    let mut app = TuiApp::new("ready".to_string(), true, false);
    app.set_welcome_lines(vec!["Welcome line".to_string()]);

    persist_welcome_into_transcript(&mut app);
    assert!(
        app.transcript
            .iter()
            .any(|line| line.kind == TranscriptLineKind::Overlay),
        "welcome line should start as overlay kind before first submission"
    );

    app.push_user_input_line("hello".to_string());

    assert!(
        app.transcript
            .iter()
            .any(|line| line.kind == TranscriptLineKind::Overlay),
        "welcome line should remain overlay kind after first submission"
    );
    assert!(
        app.transcript
            .iter()
            .any(|line| line.text.contains("Welcome line")),
        "welcome line should remain in transcript after first submission"
    );
    assert!(
        app.transcript.iter().any(|line| line.text == "hello"),
        "submitted user line should be appended to transcript"
    );
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
    assert_eq!(app.input, "draft");
    assert_eq!(app.cursor_pos, 0);

    assert!(handle_input_history_key_event(
        &mut app,
        KeyEvent::new(KeyCode::Up, KeyModifiers::NONE),
    ));
    assert_eq!(app.input, "second");
    assert_eq!(app.cursor_pos, app.input_char_count());

    app.move_cursor_home();
    assert!(handle_input_history_key_event(
        &mut app,
        KeyEvent::new(KeyCode::Down, KeyModifiers::NONE),
    ));
    assert_eq!(app.input, "second");
    assert_eq!(app.cursor_pos, app.input_char_count());

    assert!(handle_input_history_key_event(
        &mut app,
        KeyEvent::new(KeyCode::Down, KeyModifiers::NONE),
    ));
    assert_eq!(app.input, "draft");
    assert_eq!(app.cursor_pos, app.input_char_count());
}

#[test]
fn mouse_scroll_navigates_input_history_like_up_down() {
    let mut app = TuiApp::new("ready".to_string(), true, false);
    app.record_input_history("first");
    app.record_input_history("second");
    app.input = "draft".to_string();
    app.cursor_pos = app.input_char_count();

    assert!(handle_mouse_history_event(
        &mut app,
        mouse_scroll_event(crossterm::event::MouseEventKind::ScrollUp),
    ));
    assert_eq!(app.input, "second");

    assert!(handle_mouse_history_event(
        &mut app,
        mouse_scroll_event(crossterm::event::MouseEventKind::ScrollUp),
    ));
    assert_eq!(app.input, "first");

    assert!(handle_mouse_history_event(
        &mut app,
        mouse_scroll_event(crossterm::event::MouseEventKind::ScrollDown),
    ));
    assert_eq!(app.input, "second");

    assert!(handle_mouse_history_event(
        &mut app,
        mouse_scroll_event(crossterm::event::MouseEventKind::ScrollDown),
    ));
    assert_eq!(app.input, "draft");
}

#[test]
fn input_history_is_persisted_and_restored_with_ring_buffer_limit() {
    let history_file = std::env::temp_dir().join(format!(
        "pi-tui-history-{}-{}.jsonl",
        std::process::id(),
        now_millis()
    ));
    let _ = fs::remove_file(&history_file);

    let store = InputHistoryStore::new(history_file.clone(), 3);
    let mut app = TuiApp::new("ready".to_string(), true, false);
    app.set_input_history_store(Some(store.clone()));
    app.record_input_history("first");
    app.record_input_history("second");
    app.record_input_history("third");
    app.record_input_history("fourth");

    let persisted = fs::read_to_string(&history_file).expect("history file should exist");
    assert_eq!(persisted.lines().count(), 3);

    let mut restored = TuiApp::new("ready".to_string(), true, false);
    restored.set_input_history_store(Some(store));
    assert_eq!(
        restored.input_history,
        vec![
            "second".to_string(),
            "third".to_string(),
            "fourth".to_string(),
        ]
    );

    let _ = fs::remove_file(&history_file);
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

    assert_eq!(input_cursor_position(&app, area, "> "), (6, 1));

    app.cursor_pos = 2;
    assert_eq!(input_cursor_position(&app, area, "> "), (7, 1));
}

#[test]
fn input_cursor_starts_after_default_prompt_prefix() {
    let app = TuiApp::new("ready".to_string(), true, false);
    let area = Rect {
        x: 0,
        y: 0,
        width: 10,
        height: 3,
    };

    assert_eq!(input_cursor_position(&app, area, "> "), (4, 1));
}

#[test]
fn backspace_at_input_start_keeps_prompt_visible() {
    let mut app = TuiApp::new("ready".to_string(), true, false);
    assert!(!handle_editor_key_event(
        &mut app,
        KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE),
    ));
    assert!(app.input.is_empty());
    assert_eq!(app.cursor_pos, 0);

    let area = Rect {
        x: 0,
        y: 0,
        width: 10,
        height: 3,
    };
    assert_eq!(input_cursor_position(&app, area, "> "), (4, 1));
}

#[test]
fn input_line_shows_placeholder_before_first_submission() {
    let mut app = TuiApp::new("ready".to_string(), true, false);
    app.set_welcome_lines(vec!["Welcome line".to_string()]);
    persist_welcome_into_transcript(&mut app);

    let line = build_input_line(&app, "> ", TuiTheme::Dark);
    let rendered = line_text(&line);
    let expected = crate::constants::INPUT_PLACEHOLDER_HINTS
        .first()
        .copied()
        .expect("placeholder hints should not be empty");
    assert!(rendered.contains(expected));
    assert!(
        line.spans.iter().any(|span| {
            span.content.contains(expected) && span.style.fg == Some(Color::Rgb(86, 95, 137))
        }),
        "placeholder hint should be rendered with muted foreground color"
    );
}

#[test]
fn input_line_hides_placeholder_after_first_submission() {
    let mut app = TuiApp::new("ready".to_string(), true, false);
    app.push_user_input_line(">  hello".to_string());

    let line = build_input_line(&app, "> ", TuiTheme::Dark);
    let rendered = line_text(&line);
    assert!(
        !rendered.contains("Try \"Search the documentation for this library\""),
        "placeholder should be hidden once conversation starts"
    );
    assert_eq!(rendered, " > ");
}

#[test]
fn visible_transcript_picks_latest_lines() {
    let lines = vec![
        TranscriptLine::new("l1".to_string(), TranscriptLineKind::Normal),
        TranscriptLine::new("l2".to_string(), TranscriptLineKind::Tool),
        TranscriptLine::new("l3".to_string(), TranscriptLineKind::Thinking),
        TranscriptLine::new("l4".to_string(), TranscriptLineKind::Normal),
    ];
    let visible = visible_transcript_lines(&lines, &[], 2, 80, true, true, None, 0, TuiTheme::Dark);
    assert_eq!(visible.len(), 2);
    assert!(line_text(&visible[0]).trim().is_empty());
    assert!(line_text(&visible[1]).starts_with("l4"));
}

#[test]
fn visible_transcript_uses_single_spacing_before_tool_block() {
    let lines = vec![
        TranscriptLine::new("normal".to_string(), TranscriptLineKind::Normal),
        TranscriptLine::new("tool".to_string(), TranscriptLineKind::Tool),
    ];
    let visible =
        visible_transcript_lines(&lines, &[], 10, 80, true, true, None, 0, TuiTheme::Dark);
    assert_eq!(visible.len(), 3);
    assert!(line_text(&visible[0]).starts_with("normal"));
    assert!(line_text(&visible[1]).trim().is_empty());
    assert!(line_text(&visible[2]).starts_with("tool"));
}

#[test]
fn visible_transcript_reuses_existing_block_padding_when_present() {
    let lines = vec![
        TranscriptLine::new(String::new(), TranscriptLineKind::UserInput),
        TranscriptLine::new("hello".to_string(), TranscriptLineKind::UserInput),
        TranscriptLine::new(String::new(), TranscriptLineKind::UserInput),
        TranscriptLine::new("reply".to_string(), TranscriptLineKind::Normal),
    ];
    let visible =
        visible_transcript_lines(&lines, &[], 10, 80, true, true, None, 0, TuiTheme::Dark);
    assert_eq!(visible.len(), 5);
    assert!(line_text(&visible[0]).trim().is_empty());
    assert!(line_text(&visible[1]).starts_with("hello"));
    assert!(line_text(&visible[2]).trim().is_empty());
    assert!(line_text(&visible[3]).trim().is_empty());
    assert!(line_text(&visible[4]).starts_with("reply"));
}

#[test]
fn visible_transcript_respects_tool_and_thinking_toggles() {
    let lines = vec![
        TranscriptLine::new("normal".to_string(), TranscriptLineKind::Normal),
        TranscriptLine::new("tool".to_string(), TranscriptLineKind::Tool),
        TranscriptLine::new("thinking".to_string(), TranscriptLineKind::Thinking),
    ];
    let visible =
        visible_transcript_lines(&lines, &[], 10, 80, false, false, None, 0, TuiTheme::Dark);
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
        &[],
        10,
        80,
        true,
        true,
        Some(TranscriptLine::new(
            "[-] pi is working...".to_string(),
            TranscriptLineKind::Working,
        )),
        0,
        TuiTheme::Dark,
    );
    assert_eq!(visible.len(), 3);
    assert!(line_text(&visible[0]).starts_with("normal"));
    assert!(line_text(&visible[1]).trim().is_empty());
    assert_eq!(visible[2].spans[0].content, "[-] pi is working...");
}

#[test]
fn status_separator_keeps_one_blank_line_between_transcript_and_status() {
    let lines = vec![
        Line::from("first"),
        Line::from("second"),
        Line::from("third"),
    ];
    let separated = ensure_bottom_status_separator(lines, 3);

    assert_eq!(separated.len(), 3);
    assert_eq!(line_text(&separated[0]).trim(), "second");
    assert_eq!(line_text(&separated[1]).trim(), "third");
    assert!(
        line_text(separated.last().expect("last line"))
            .trim()
            .is_empty(),
        "last transcript row should be blank so status area is visually separated"
    );
}

#[test]
fn status_separator_keeps_existing_blank_tail_unchanged() {
    let lines = vec![Line::from("first"), Line::from(""), Line::from("")];
    let separated = ensure_bottom_status_separator(lines.clone(), 3);

    assert_eq!(separated, lines);
}

#[test]
fn status_bar_keeps_top_line_stable_while_working() {
    let mut app = TuiApp::new("pi is working...".to_string(), true, false);
    app.set_status_bar_meta(
        String::new(),
        "Auto (High) - allow all commands".to_string(),
        "Databend GPT-5.3 Codex [custom]".to_string(),
    );
    app.start_working("pi is working...".to_string());
    app.status_left = "Auto (High) - allow all commands".to_string();

    let status = render_status_bar_lines(&app, 120, TuiTheme::Dark);
    let middle = line_text(&status.lines[0]);
    let hints = line_text(&status.lines[1]);
    let bottom = line_text(&status.lines[2]);

    assert!(middle.contains("Auto (High) - allow all commands"));
    assert!(middle.contains("Databend GPT-5.3 Codex [custom]"));
    assert!(hints.contains("shift+tab to cycle modes"));
    assert!(bottom.contains("[⏱"));
}

#[test]
fn tool_output_line_resets_working_message_to_generic_state() {
    let mut app = TuiApp::new("ready".to_string(), true, false);
    app.start_working("pixy is working...".to_string());

    app.note_working_from_update(
        "pixy",
        &StreamUpdate::ToolLine("• Ran bash -lc 'mkdir -p /tmp/snake_v5'".to_string()),
    );
    assert_eq!(app.working_message, "Invoking tools...");

    app.note_working_from_update("pixy", &StreamUpdate::ToolLine("(no output)".to_string()));
    assert_eq!(app.working_message, "Thinking...");
}

#[test]
fn assistant_stream_update_switches_working_message_to_streaming() {
    let mut app = TuiApp::new("ready".to_string(), true, false);
    app.start_working("pixy is working...".to_string());

    app.note_working_from_update("pixy", &StreamUpdate::AssistantTextDelta("h".to_string()));
    assert_eq!(app.working_message, "Streaming...");
}

#[test]
fn working_line_uses_single_space_after_spinner_prefix() {
    let mut app = TuiApp::new("ready".to_string(), true, false);
    app.start_working("pixy is working...".to_string());
    app.working_tick = 0;

    let line = app.working_line().expect("working line should be present");
    assert!(
        line.text.starts_with("⠋ Thinking..."),
        "spinner prefix should have exactly one space before status text"
    );
}

#[test]
fn working_line_marquee_highlights_spinner_text_with_dark_theme_colors() {
    let mut app = TuiApp::new("ready".to_string(), true, false);
    app.start_working("Checking file git status flags".to_string());
    app.working_tick = 0;

    let line0 = app
        .working_line()
        .expect("working line should be present")
        .to_line(200, TuiTheme::Dark);
    assert!(
        line0.spans.iter().any(|span| {
            span.content.contains("Thin")
                && span.style.fg == Some(Color::Rgb(224, 175, 104))
                && span.style.bg == Some(Color::Rgb(26, 27, 38))
        }),
        "tick=0 should highlight first 4 chars with amber-on-black"
    );

    app.working_tick = 1;
    let line1 = app
        .working_line()
        .expect("working line should be present")
        .to_line(200, TuiTheme::Dark);
    assert!(
        line1.spans.iter().any(|span| {
            span.content.contains("hink")
                && span.style.fg == Some(Color::Rgb(224, 175, 104))
                && span.style.bg == Some(Color::Rgb(26, 27, 38))
        }),
        "tick=1 should move highlight window forward by one char"
    );
}

#[test]
fn working_line_marquee_uses_light_theme_colors() {
    let mut app = TuiApp::new("ready".to_string(), true, false);
    app.start_working("Checking file git status flags".to_string());

    let line = app
        .working_line()
        .expect("working line should be present")
        .to_line(200, TuiTheme::Light);

    assert!(
        line.spans
            .iter()
            .any(|span| span.style.bg == Some(Color::Rgb(208, 208, 224))),
        "light theme should use configured working background"
    );
    assert!(
        line.spans
            .iter()
            .any(|span| { span.content.contains("Thin") && span.style.fg == Some(Color::Black) }),
        "highlight text should use light theme highlight color"
    );
    assert!(
        line.spans
            .iter()
            .any(|span| span.style.fg == Some(Color::Rgb(108, 108, 108))),
        "non-highlight text should use light theme working text color"
    );
}

#[test]
fn visible_transcript_wraps_and_keeps_latest_at_bottom() {
    let lines = vec![
        TranscriptLine::new("old".to_string(), TranscriptLineKind::Normal),
        TranscriptLine::new("abcdefghijklm".to_string(), TranscriptLineKind::Normal),
    ];

    let visible = visible_transcript_lines(&lines, &[], 2, 4, true, true, None, 0, TuiTheme::Dark);
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

    let bottom = visible_transcript_lines(&lines, &[], 2, 80, true, true, None, 0, TuiTheme::Dark);
    assert_eq!(bottom.len(), 2);
    assert!(line_text(&bottom[0]).starts_with("line 3"));
    assert!(line_text(&bottom[1]).starts_with("line 4"));

    let scrolled =
        visible_transcript_lines(&lines, &[], 2, 80, true, true, None, 1, TuiTheme::Dark);
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

    let visible =
        visible_transcript_lines(&lines, &[], 20, 120, true, true, None, 0, TuiTheme::Dark);
    let texts = visible.iter().map(line_text).collect::<Vec<_>>();
    assert!(texts.iter().any(|line| line.contains("• Ran bash -lc")));
    assert!(texts.iter().any(|line| line.contains("… +3 lines")));
    assert!(texts.iter().any(|line| line.contains("line 6")));
}

#[test]
fn tool_call_groups_have_blank_line_separator() {
    let lines = vec![
        TranscriptLine::new(
            "• Ran edit crates/pixy-tui/src/lib.rs".to_string(),
            TranscriptLineKind::Tool,
        ),
        TranscriptLine::new(
            "crates/pixy-tui/src/lib.rs | 2 ++--".to_string(),
            TranscriptLineKind::Tool,
        ),
        TranscriptLine::new(
            "• Ran bash -lc 'cargo test -p pixy-tui'".to_string(),
            TranscriptLineKind::Tool,
        ),
        TranscriptLine::new("running 1 test".to_string(), TranscriptLineKind::Tool),
    ];

    let visible =
        visible_transcript_lines(&lines, &[], 20, 120, true, true, None, 0, TuiTheme::Dark);
    let texts = visible.iter().map(line_text).collect::<Vec<_>>();

    let first_tool_idx = texts
        .iter()
        .position(|line| line.contains("• Ran edit crates/pixy-tui/src/lib.rs"))
        .expect("first tool title should exist");
    let second_tool_idx = texts
        .iter()
        .position(|line| line.contains("• Ran bash -lc 'cargo test -p pixy-tui'"))
        .expect("second tool title should exist");

    assert!(second_tool_idx > first_tool_idx + 1);
    assert!(texts[second_tool_idx - 1].trim().is_empty());
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

    let visible = visible_transcript_lines(
        &app.transcript,
        &[],
        20,
        120,
        true,
        true,
        None,
        0,
        TuiTheme::Dark,
    );
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
        .to_line(40, TuiTheme::Dark);
    let added = TranscriptLine::new("+ new assertion".to_string(), TranscriptLineKind::Tool)
        .to_line(40, TuiTheme::Dark);
    assert_eq!(removed.spans[0].style.fg, Some(Color::Rgb(247, 118, 142)));
    assert_eq!(added.spans[0].style.fg, Some(Color::Rgb(158, 206, 106)));
}

#[test]
fn tool_diff_stat_line_colors_plus_and_minus_segments() {
    let line = TranscriptLine::new(
        "crates/pixy-coding-agent/tests/tools.rs | 124 ++++++++++---------".to_string(),
        TranscriptLineKind::Tool,
    )
    .to_line(80, TuiTheme::Dark);

    let plus_span = line
        .spans
        .iter()
        .find(|span| span.content.contains('+'))
        .expect("plus span");
    let minus_span = line
        .spans
        .iter()
        .find(|span| span.content.contains('-'))
        .expect("minus span");
    assert_eq!(plus_span.style.fg, Some(Color::Rgb(158, 206, 106)));
    assert_eq!(minus_span.style.fg, Some(Color::Rgb(247, 118, 142)));
}

#[test]
fn tool_lines_keep_default_background_and_use_light_text() {
    let tool = TranscriptLine::new("tool output".to_string(), TranscriptLineKind::Tool)
        .to_line(40, TuiTheme::Dark);
    assert_eq!(tool.spans[0].style.fg, Some(Color::Rgb(169, 177, 214)));
    assert_eq!(tool.spans[0].style.bg, None);
}

#[test]
fn user_input_line_uses_orange_foreground_with_default_background() {
    let input = TranscriptLine::new("hello".to_string(), TranscriptLineKind::UserInput)
        .to_line(40, TuiTheme::Dark);
    assert_eq!(input.spans[0].style.fg, Some(Color::Rgb(227, 153, 42)));
    assert_eq!(input.spans[0].style.bg, Some(Color::Rgb(26, 27, 38)));
}

#[test]
fn assistant_output_lines_use_sigil_prefix() {
    let lines = vec![TranscriptLine::new(
        "Hello there".to_string(),
        TranscriptLineKind::Assistant,
    )];
    let visible = visible_transcript_lines(&lines, &[], 3, 80, true, true, None, 0, TuiTheme::Dark);
    let rendered = line_text(&visible[0]);
    let output_prompt = TuiTheme::Dark.output_prompt();
    assert!(
        rendered.contains(&format!("{output_prompt}Hello there")),
        "assistant output should use the configured output prompt prefix"
    );
}

#[test]
fn assistant_multiline_block_prefixes_only_first_non_empty_line() {
    let lines = vec![
        TranscriptLine::new("first line".to_string(), TranscriptLineKind::Assistant),
        TranscriptLine::new("second line".to_string(), TranscriptLineKind::Assistant),
    ];
    let visible = visible_transcript_lines(&lines, &[], 4, 80, true, true, None, 0, TuiTheme::Dark);
    let rendered = visible.iter().map(line_text).collect::<Vec<_>>();
    let output_prompt = TuiTheme::Dark.output_prompt();

    assert!(
        rendered
            .iter()
            .any(|line| line.contains(&format!("{output_prompt}first line")))
    );
    assert!(rendered.iter().any(|line| line.contains("second line")));
    assert!(
        !rendered
            .iter()
            .any(|line| line.contains(&format!("{output_prompt}second line"))),
        "only the first line in an assistant block should have the prefix"
    );
}

#[test]
fn format_user_input_line_prefixes_prompt() {
    assert_eq!(format_user_input_line("hello", "> "), ">  hello");
    assert_eq!(
        format_user_input_line("this is second question", "pi> "),
        "pi>  this is second question"
    );
}

#[test]
fn push_user_input_line_wraps_content_with_blank_padding_lines() {
    let mut app = TuiApp::new("ready".to_string(), true, false);
    app.push_user_input_line("hello".to_string());

    assert_eq!(app.transcript.len(), 3);
    assert_eq!(app.transcript[0].text, "");
    assert_eq!(app.transcript[0].kind, TranscriptLineKind::UserInput);
    assert_eq!(app.transcript[1].text, "hello");
    assert_eq!(app.transcript[1].kind, TranscriptLineKind::UserInput);
    assert_eq!(app.transcript[2].text, "");
    assert_eq!(app.transcript[2].kind, TranscriptLineKind::UserInput);
}

#[test]
fn user_input_prompt_prefix_is_not_rendered_as_markdown_quote() {
    let lines = vec![TranscriptLine::new(
        ">  hello".to_string(),
        TranscriptLineKind::UserInput,
    )];

    let visible = visible_transcript_lines(&lines, &[], 3, 60, true, true, None, 0, TuiTheme::Dark);
    let rendered = line_text(&visible[0]);
    assert!(
        rendered.contains(">  hello"),
        "user input prompt prefix should be rendered verbatim"
    );
}

#[test]
fn first_submitted_input_keeps_welcome_banner_in_transcript() {
    let mut app = TuiApp::new("ready".to_string(), true, false);
    app.set_welcome_lines(vec!["Welcome line".to_string()]);

    persist_welcome_into_transcript(&mut app);
    app.push_user_input_line("hello".to_string());

    assert!(
        app.transcript
            .iter()
            .any(|line| line.text.contains("Welcome line"))
    );
    assert!(app.transcript.iter().any(|line| line.text == "hello"));
}

#[test]
fn parse_tui_theme_names_case_insensitive() {
    assert_eq!(TuiTheme::from_name("dark"), Some(TuiTheme::Dark));
    assert_eq!(TuiTheme::from_name("LIGHT"), Some(TuiTheme::Light));
    assert_eq!(TuiTheme::from_name(""), None);
    assert_eq!(TuiTheme::from_name("solarized"), None);
}

#[test]
fn dark_theme_selection_osc_sequence_uses_reference_selection_colors() {
    assert_eq!(
        selection_osc_set_sequence(TuiTheme::Dark),
        Some("\u{1b}]17;#3b4261\u{7}\u{1b}]19;#c0caf5\u{7}".to_string())
    );
}

#[test]
fn selection_osc_reset_sequence_resets_both_colors() {
    assert_eq!(
        selection_osc_reset_sequence(),
        "\u{1b}]117;\u{7}\u{1b}]119;\u{7}"
    );
}

#[test]
fn dark_theme_selection_osc_sequences_include_rgb_and_st_variants() {
    let sequences = selection_osc_set_sequences(TuiTheme::Dark, TerminalCapabilities::default())
        .expect("dark theme should provide selection sequences");
    assert!(
        sequences
            .iter()
            .any(|sequence| sequence.contains("rgb:3b/42/61") && sequence.contains("\u{7}"))
    );
    assert!(
        sequences
            .iter()
            .any(|sequence| sequence.contains("rgb:c0/ca/f5") && sequence.contains("\u{1b}\\"))
    );
}

#[test]
fn selection_osc_sequences_are_wrapped_for_tmux() {
    let sequences = selection_osc_set_sequences(
        TuiTheme::Dark,
        TerminalCapabilities {
            multiplexer: Some(TerminalMultiplexer::Tmux),
            ..TerminalCapabilities::default()
        },
    )
    .expect("dark theme should provide selection sequences");
    assert!(
        sequences
            .iter()
            .any(|sequence| sequence.starts_with("\u{1b}Ptmux;"))
    );
}

#[test]
fn selection_osc_reset_sequences_are_wrapped_for_tmux() {
    let sequences = selection_osc_reset_sequences(TerminalCapabilities {
        multiplexer: Some(TerminalMultiplexer::Tmux),
        ..TerminalCapabilities::default()
    });
    assert!(
        sequences
            .iter()
            .any(|sequence| sequence.starts_with("\u{1b}Ptmux;"))
    );
}

#[test]
fn light_theme_uses_light_palette_for_tokens_and_tool_lines() {
    let tool = TranscriptLine::new("tool output".to_string(), TranscriptLineKind::Tool)
        .to_line(40, TuiTheme::Light);
    assert_eq!(tool.spans[0].style.fg, Some(Color::Rgb(108, 108, 108)));

    let line = TranscriptLine::new(
        "See crates/pixy-tui/src/lib.rs:902 and PageUp".to_string(),
        TranscriptLineKind::Normal,
    )
    .to_line(80, TuiTheme::Light);

    assert!(
        line.spans.iter().any(|span| {
            span.content.contains("crates/pixy-tui/src/lib.rs:902")
                && span.style.fg == Some(Color::Rgb(84, 125, 167))
        }),
        "file path token should be highlighted in light theme"
    );
    assert!(
        line.spans.iter().any(|span| span.content.contains("PageUp")
            && span.style.fg == Some(Color::Rgb(90, 128, 128))),
        "PageUp token should be highlighted in light theme"
    );
}

#[test]
fn highlights_file_path_and_key_tokens() {
    let line = TranscriptLine::new(
        "See crates/pixy-tui/src/lib.rs:902 and PageUp / PageDown / ctrl+c / shift+tab / alt+enter"
            .to_string(),
        TranscriptLineKind::Normal,
    )
    .to_line(120, TuiTheme::Dark);

    assert!(
        line.spans.iter().any(|span| {
            span.content.contains("crates/pixy-tui/src/lib.rs:902")
                && span.style.fg == Some(Color::Rgb(122, 162, 247))
        }),
        "file path token should be highlighted"
    );
    assert!(
        line.spans.iter().any(|span| {
            span.content.contains("PageUp") && span.style.fg == Some(Color::Rgb(125, 207, 255))
        }),
        "PageUp token should be highlighted"
    );
    assert!(
        line.spans.iter().any(|span| {
            span.content.contains("PageDown") && span.style.fg == Some(Color::Rgb(125, 207, 255))
        }),
        "PageDown token should be highlighted"
    );
    assert!(
        line.spans.iter().any(|span| {
            span.content.contains("ctrl+c") && span.style.fg == Some(Color::Rgb(125, 207, 255))
        }),
        "ctrl+c token should be highlighted"
    );
    assert!(
        line.spans.iter().any(|span| {
            span.content.contains("shift+tab") && span.style.fg == Some(Color::Rgb(125, 207, 255))
        }),
        "shift+tab token should be highlighted"
    );
    assert!(
        line.spans.iter().any(|span| {
            span.content.contains("alt+enter") && span.style.fg == Some(Color::Rgb(125, 207, 255))
        }),
        "alt+enter token should be highlighted"
    );
}

#[test]
fn markdown_fenced_code_blocks_are_rendered_without_fence_and_with_syntax_highlight() {
    let lines = vec![
        TranscriptLine::new("Before".to_string(), TranscriptLineKind::Normal),
        TranscriptLine::new("```rust".to_string(), TranscriptLineKind::Normal),
        TranscriptLine::new("fn main() {".to_string(), TranscriptLineKind::Normal),
        TranscriptLine::new("    // comment".to_string(), TranscriptLineKind::Normal),
        TranscriptLine::new(
            "    println!(\"hello\", 42);".to_string(),
            TranscriptLineKind::Normal,
        ),
        TranscriptLine::new("}".to_string(), TranscriptLineKind::Normal),
        TranscriptLine::new("```".to_string(), TranscriptLineKind::Normal),
        TranscriptLine::new("After".to_string(), TranscriptLineKind::Normal),
    ];

    let visible =
        visible_transcript_lines(&lines, &[], 20, 120, true, true, None, 0, TuiTheme::Dark);
    let texts = visible.iter().map(line_text).collect::<Vec<_>>();

    assert!(!texts.iter().any(|line| line.contains("```")));
    assert!(texts.iter().any(|line| line.contains("fn main() {")));
    assert!(texts.iter().any(|line| line.contains("After")));

    let fn_line = visible
        .iter()
        .find(|line| line_text(line).contains("fn main() {"))
        .expect("code line should exist");
    assert!(
        fn_line
            .spans
            .iter()
            .any(|span| span.content == "fn" && span.style.fg == Some(Color::Rgb(129, 161, 193))),
        "rust keyword should be highlighted"
    );

    let comment_line = visible
        .iter()
        .find(|line| line_text(line).contains("// comment"))
        .expect("comment line should exist");
    assert!(
        comment_line
            .spans
            .iter()
            .any(|span| span.content.contains("// comment")
                && span.style.fg == Some(Color::Rgb(125, 130, 140))),
        "comment should be highlighted"
    );

    let print_line = visible
        .iter()
        .find(|line| line_text(line).contains("println!"))
        .expect("string/number line should exist");
    assert!(
        print_line.spans.iter().any(|span| {
            span.content.contains("\"hello\"") && span.style.fg == Some(Color::Rgb(163, 190, 140))
        }),
        "string should be highlighted"
    );
    assert!(
        print_line
            .spans
            .iter()
            .any(|span| span.content.contains("42")
                && span.style.fg == Some(Color::Rgb(208, 135, 112))),
        "number should be highlighted"
    );
}

#[test]
fn markdown_tables_are_rendered_as_aligned_box_drawing_tables() {
    let lines = vec![
        TranscriptLine::new("| col1 | col2 |".to_string(), TranscriptLineKind::Normal),
        TranscriptLine::new("| ---- | ---- |".to_string(), TranscriptLineKind::Normal),
        TranscriptLine::new("| A    | BBB  |".to_string(), TranscriptLineKind::Normal),
        TranscriptLine::new("| CC   | D    |".to_string(), TranscriptLineKind::Normal),
    ];

    let visible =
        visible_transcript_lines(&lines, &[], 20, 120, true, true, None, 0, TuiTheme::Dark);
    let texts = visible
        .iter()
        .map(line_text)
        .map(|line| line.trim_end().to_string())
        .collect::<Vec<_>>();

    assert!(texts.iter().any(|line| line == "┌──────┬──────┐"));
    assert!(texts.iter().any(|line| line == "│ col1 │ col2 │"));
    assert!(texts.iter().any(|line| line == "├──────┼──────┤"));
    assert!(texts.iter().any(|line| line == "│ A    │ BBB  │"));
    assert!(texts.iter().any(|line| line == "│ CC   │ D    │"));
    assert!(texts.iter().any(|line| line == "└──────┴──────┘"));
    assert!(
        !texts.iter().any(|line| line.contains("| ---- |")),
        "markdown separator rows should not leak into rendered output"
    );
}

#[test]
fn multiline_assistant_markdown_tables_are_rendered_in_transcript() {
    let messages = vec![Message::Assistant {
        content: vec![AssistantContentBlock::Text {
            text: "before\n| col1 | col2 |\n| ---- | ---- |\n| A    | BBB  |\nafter".to_string(),
            text_signature: None,
        }],
        api: "openai-responses".to_string(),
        provider: "openai".to_string(),
        model: "gpt-4o-mini".to_string(),
        usage: Usage {
            input: 0,
            output: 0,
            cache_read: 0,
            cache_write: 0,
            total_tokens: 0,
            cost: Cost {
                input: 0.0,
                output: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
                total: 0.0,
            },
        },
        stop_reason: StopReason::Stop,
        error_message: None,
        timestamp: 1_700_000_000_000,
    }];

    let rendered = render_messages(&messages);
    let visible =
        visible_transcript_lines(&rendered, &[], 30, 120, true, true, None, 0, TuiTheme::Dark);
    let texts = visible
        .iter()
        .map(line_text)
        .map(|line| line.trim_end().to_string())
        .collect::<Vec<_>>();

    assert!(
        texts
            .iter()
            .any(|line| line == &format!("{}before", TuiTheme::Dark.output_prompt()))
    );
    assert!(texts.iter().any(|line| line == "┌──────┬──────┐"));
    assert!(texts.iter().any(|line| line == "│ A    │ BBB  │"));
    assert!(texts.iter().any(|line| line == "after"));
}

#[test]
fn streaming_delta_with_chinese_markdown_table_is_rendered_as_table() {
    let mut app = TuiApp::new("ready".to_string(), true, false);
    app.apply_stream_update(StreamUpdate::AssistantTextDelta(
        "给你一个 markdown 表格：\n\n| 列1 | 列2 | 列3 |\n| --- | --- | --- |\n| A1  | B1  | C1  |\n| A2  | B2  | C2  |\n| A3  | B3  | C3  |\n".to_string(),
    ));

    let visible = visible_transcript_lines(
        &app.transcript,
        &[],
        40,
        120,
        app.show_tool_results,
        app.show_thinking,
        app.working_line(),
        app.transcript_scroll_from_bottom,
        TuiTheme::Dark,
    );
    let texts = visible
        .iter()
        .map(line_text)
        .map(|line| line.trim_end().to_string())
        .collect::<Vec<_>>();

    assert!(
        texts
            .iter()
            .any(|line| line
                == &format!("{}给你一个 markdown 表格：", TuiTheme::Dark.output_prompt()))
    );
    assert!(texts.iter().any(|line| line == "┌─────┬─────┬─────┐"));
    assert!(texts.iter().any(|line| line == "│ 列1 │ 列2 │ 列3 │"));
    assert!(texts.iter().any(|line| line == "│ A1  │ B1  │ C1  │"));
    assert!(texts.iter().any(|line| line == "│ A3  │ B3  │ C3  │"));
    assert!(
        !texts
            .iter()
            .any(|line| line.contains("| 列1 | 列2 | 列3 |")),
        "raw markdown table rows should be replaced by rendered table output"
    );
}

#[test]
fn user_input_markdown_table_is_rendered_as_table() {
    let mut app = TuiApp::new("ready".to_string(), true, false);
    let input = "给我一个 markdown 表格，三行三列\n\n| 列1 | 列2 | 列3 |\n| --- | --- | --- |\n| A1  | B1  | C1  |\n| A2  | B2  | C2  |\n| A3  | B3  | C3  |";
    app.push_user_input_line(format_user_input_line(input, TuiTheme::Dark.input_prompt()));

    let visible = visible_transcript_lines(
        &app.transcript,
        &[],
        40,
        120,
        app.show_tool_results,
        app.show_thinking,
        app.working_line(),
        app.transcript_scroll_from_bottom,
        TuiTheme::Dark,
    );
    let texts = visible
        .iter()
        .map(line_text)
        .map(|line| line.trim_end().to_string())
        .collect::<Vec<_>>();

    assert!(texts.iter().any(|line| line == "┌─────┬─────┬─────┐"));
    assert!(texts.iter().any(|line| line == "│ 列1 │ 列2 │ 列3 │"));
    assert!(texts.iter().any(|line| line == "│ A3  │ B3  │ C3  │"));
    assert!(
        !texts
            .iter()
            .any(|line| line.contains("| 列1 | 列2 | 列3 |")),
        "raw user markdown table rows should be replaced by rendered table output"
    );
}

#[test]
fn assistant_line_with_multiline_markdown_table_is_rendered_as_table() {
    let mut app = TuiApp::new("ready".to_string(), true, false);
    app.apply_stream_update(StreamUpdate::AssistantLine(
        "这里是表格：\n\n| 列1 | 列2 | 列3 |\n| --- | --- | --- |\n| A1  | B1  | C1  |\n| A2  | B2  | C2  |\n| A3  | B3  | C3  |".to_string(),
    ));

    let visible = visible_transcript_lines(
        &app.transcript,
        &[],
        40,
        120,
        app.show_tool_results,
        app.show_thinking,
        app.working_line(),
        app.transcript_scroll_from_bottom,
        TuiTheme::Dark,
    );
    let texts = visible
        .iter()
        .map(line_text)
        .map(|line| line.trim_end().to_string())
        .collect::<Vec<_>>();

    assert!(
        texts
            .iter()
            .any(|line| line == &format!("{}这里是表格：", TuiTheme::Dark.output_prompt()))
    );
    assert!(texts.iter().any(|line| line == "┌─────┬─────┬─────┐"));
    assert!(texts.iter().any(|line| line == "│ 列1 │ 列2 │ 列3 │"));
    assert!(texts.iter().any(|line| line == "│ A3  │ B3  │ C3  │"));
    assert!(
        !texts
            .iter()
            .any(|line| line.contains("| 列1 | 列2 | 列3 |")),
        "raw assistant markdown table rows should be replaced by rendered table output"
    );
}

#[test]
fn markdown_fenced_markdown_table_is_rendered_as_table_not_code() {
    let mut app = TuiApp::new("ready".to_string(), true, false);
    app.apply_stream_update(StreamUpdate::AssistantLine(
        "```markdown\n| 列1 | 列2 | 列3 |\n| --- | --- | --- |\n| A1  | B1  | C1  |\n| A2  | B2  | C2  |\n| A3  | B3  | C3  |\n```".to_string(),
    ));

    let visible = visible_transcript_lines(
        &app.transcript,
        &[],
        40,
        120,
        app.show_tool_results,
        app.show_thinking,
        app.working_line(),
        app.transcript_scroll_from_bottom,
        TuiTheme::Dark,
    );
    let texts = visible
        .iter()
        .map(line_text)
        .map(|line| line.trim_end().to_string())
        .collect::<Vec<_>>();

    assert!(texts.iter().any(|line| line == "┌─────┬─────┬─────┐"));
    assert!(texts.iter().any(|line| line == "│ 列1 │ 列2 │ 列3 │"));
    assert!(
        !texts.iter().any(|line| line.contains("```")),
        "markdown fences should be hidden in transcript rendering"
    );
}

#[test]
fn inline_markdown_bold_italic_and_code_are_rendered_without_markers() {
    let line = TranscriptLine::new(
        "Use **bold**, *italic*, and `code`".to_string(),
        TranscriptLineKind::Normal,
    )
    .to_line(120, TuiTheme::Dark);
    let text = line_text(&line);

    assert!(text.contains("Use bold, italic, and code"));
    assert!(!text.contains("**"));
    assert!(!text.contains("`"));
    assert!(
        line.spans.iter().any(|span| {
            span.content.contains("bold")
                && span
                    .style
                    .add_modifier
                    .contains(ratatui::style::Modifier::BOLD)
        }),
        "bold markdown should add bold style"
    );
    assert!(
        line.spans.iter().any(|span| {
            span.content.contains("italic")
                && span
                    .style
                    .add_modifier
                    .contains(ratatui::style::Modifier::ITALIC)
        }),
        "italic markdown should add italic style"
    );
    assert!(
        line.spans.iter().any(|span| {
            span.content.contains("code") && span.style.bg == Some(Color::Rgb(40, 44, 52))
        }),
        "inline code should use markdown code background color"
    );
}

#[test]
fn inline_markdown_extended_styles_and_links_are_rendered_without_markers() {
    let line = TranscriptLine::new(
        "Try __bold__, _italic_, ~~done~~, and [Pixy](https://example.com)".to_string(),
        TranscriptLineKind::Normal,
    )
    .to_line(200, TuiTheme::Dark);
    let text = line_text(&line);

    assert!(text.contains("Try bold, italic, done, and Pixy (https://example.com)"));
    assert!(!text.contains("__"));
    assert!(!text.contains("~~"));
    assert!(!text.contains("[Pixy]("));
    assert!(
        line.spans.iter().any(|span| {
            span.content.contains("bold")
                && span
                    .style
                    .add_modifier
                    .contains(ratatui::style::Modifier::BOLD)
        }),
        "double underscore bold markdown should add bold style"
    );
    assert!(
        line.spans.iter().any(|span| {
            span.content.contains("italic")
                && span
                    .style
                    .add_modifier
                    .contains(ratatui::style::Modifier::ITALIC)
        }),
        "underscore italic markdown should add italic style"
    );
    assert!(
        line.spans.iter().any(|span| {
            span.content.contains("done")
                && span
                    .style
                    .add_modifier
                    .contains(ratatui::style::Modifier::CROSSED_OUT)
        }),
        "strikethrough markdown should add crossed-out style"
    );
    assert!(
        line.spans.iter().any(|span| {
            span.content.contains("Pixy")
                && span
                    .style
                    .add_modifier
                    .contains(ratatui::style::Modifier::UNDERLINED)
        }),
        "markdown links should be rendered with underline style"
    );
}

#[test]
fn inline_markdown_underscore_markers_do_not_break_snake_case_tokens() {
    let line = TranscriptLine::new(
        "Keep snake_case and config_value unchanged".to_string(),
        TranscriptLineKind::Normal,
    )
    .to_line(120, TuiTheme::Dark);
    let text = line_text(&line);

    assert!(text.contains("snake_case"));
    assert!(text.contains("config_value"));
    assert!(
        !line.spans.iter().any(|span| {
            span.content.contains("snake_case")
                && span
                    .style
                    .add_modifier
                    .contains(ratatui::style::Modifier::ITALIC)
        }),
        "plain snake_case tokens should not be parsed as italic markdown"
    );
}

#[test]
fn markdown_heading_quote_list_and_rule_are_structurally_rendered() {
    let lines = vec![
        TranscriptLine::new("# Heading".to_string(), TranscriptLineKind::Normal),
        TranscriptLine::new("- item one".to_string(), TranscriptLineKind::Normal),
        TranscriptLine::new("- [x] done".to_string(), TranscriptLineKind::Normal),
        TranscriptLine::new("> quote".to_string(), TranscriptLineKind::Normal),
        TranscriptLine::new("---".to_string(), TranscriptLineKind::Normal),
    ];

    let visible =
        visible_transcript_lines(&lines, &[], 30, 120, true, true, None, 0, TuiTheme::Dark);
    let texts = visible
        .iter()
        .map(line_text)
        .map(|line| line.trim_end().to_string())
        .collect::<Vec<_>>();

    assert!(texts.iter().any(|line| line == "Heading"));
    assert!(texts.iter().any(|line| line == "• item one"));
    assert!(texts.iter().any(|line| line == "☑ done"));
    assert!(texts.iter().any(|line| line == "│ quote"));
    assert!(texts.iter().any(|line| line == "────────────────────────"));
    assert!(
        !texts.iter().any(|line| line == "# Heading"),
        "heading marker should be removed"
    );
}

#[test]
fn markdown_ordered_and_indented_lists_are_structurally_rendered() {
    let lines = vec![
        TranscriptLine::new("1. first".to_string(), TranscriptLineKind::Normal),
        TranscriptLine::new("2) second".to_string(), TranscriptLineKind::Normal),
        TranscriptLine::new("  - child".to_string(), TranscriptLineKind::Normal),
    ];

    let visible =
        visible_transcript_lines(&lines, &[], 20, 120, true, true, None, 0, TuiTheme::Dark);
    let texts = visible
        .iter()
        .map(line_text)
        .map(|line| line.trim_end().to_string())
        .collect::<Vec<_>>();

    assert!(texts.iter().any(|line| line == "1. first"));
    assert!(texts.iter().any(|line| line == "2) second"));
    assert!(texts.iter().any(|line| line == "  • child"));
}

#[test]
fn slash_compound_shortcuts_use_key_token_color_instead_of_path_color() {
    let token = "ctrl+p/ctrl+shift+p";
    let line = TranscriptLine::new(
        format!("Use {token} to cycle models"),
        TranscriptLineKind::Normal,
    )
    .to_line(80, TuiTheme::Dark);

    assert!(
        line.spans.iter().any(|span| {
            span.content.contains(token) && span.style.fg == Some(Color::Rgb(125, 207, 255))
        }),
        "slash compound shortcuts should use key token color"
    );
    assert!(
        !line.spans.iter().any(|span| {
            span.content.contains(token) && span.style.fg == Some(Color::Rgb(122, 162, 247))
        }),
        "slash compound shortcuts should not be treated as file paths"
    );
}

#[test]
fn named_single_key_shortcuts_are_highlighted() {
    let line = TranscriptLine::new(
        "Use escape to interrupt, enter to submit".to_string(),
        TranscriptLineKind::Normal,
    )
    .to_line(80, TuiTheme::Dark);

    assert!(
        line.spans.iter().any(|span| {
            span.content.contains("escape") && span.style.fg == Some(Color::Rgb(125, 207, 255))
        }),
        "escape should be highlighted as a key token"
    );
    assert!(
        line.spans.iter().any(|span| {
            span.content.contains("enter") && span.style.fg == Some(Color::Rgb(125, 207, 255))
        }),
        "enter should be highlighted as a key token"
    );
}

#[test]
fn slash_command_hint_token_is_highlighted() {
    let line = TranscriptLine::new("Use / for commands".to_string(), TranscriptLineKind::Normal)
        .to_line(40, TuiTheme::Dark);

    assert!(
        line.spans
            .iter()
            .any(|span| span.content == "/" && span.style.fg == Some(Color::Rgb(125, 207, 255))),
        "slash command hint should use key token color"
    );
}

#[test]
fn section_headers_and_user_group_use_reference_colors() {
    let skills_header = TranscriptLine::new("[Skills]".to_string(), TranscriptLineKind::Normal)
        .to_line(40, TuiTheme::Dark);
    assert!(
        skills_header
            .spans
            .iter()
            .any(|span| span.content == "[" && span.style.fg == Some(Color::Rgb(224, 175, 104))),
        "skills header left bracket should use reference accent color"
    );
    assert!(
        skills_header.spans.iter().any(|span| {
            span.content.contains("Skills") && span.style.fg == Some(Color::Rgb(224, 175, 104))
        }),
        "skills header text should use reference accent color"
    );
    assert!(
        skills_header
            .spans
            .iter()
            .any(|span| span.content == "]" && span.style.fg == Some(Color::Rgb(224, 175, 104))),
        "skills header right bracket should use reference accent color"
    );

    let context_header = TranscriptLine::new("[Context]".to_string(), TranscriptLineKind::Normal)
        .to_line(40, TuiTheme::Dark);
    assert!(
        context_header
            .spans
            .iter()
            .any(|span| span.content == "[" && span.style.fg == Some(Color::Rgb(224, 175, 104))),
        "context header left bracket should use reference accent color"
    );
    assert!(
        context_header.spans.iter().any(|span| {
            span.content.contains("Context") && span.style.fg == Some(Color::Rgb(224, 175, 104))
        }),
        "context header text should use reference accent color"
    );
    assert!(
        context_header
            .spans
            .iter()
            .any(|span| span.content == "]" && span.style.fg == Some(Color::Rgb(224, 175, 104))),
        "context header right bracket should use reference accent color"
    );

    let user_group = TranscriptLine::new("  user".to_string(), TranscriptLineKind::Normal)
        .to_line(40, TuiTheme::Dark);
    assert!(
        user_group.spans.iter().any(|span| {
            span.content.contains("user") && span.style.fg == Some(Color::Rgb(115, 218, 202))
        }),
        "skills user group should use reference accent color"
    );
}

#[test]
fn parse_tool_name_extracts_name_from_tool_header() {
    assert_eq!(parse_tool_name("[tool:read:ok]"), Some("read"));
    assert_eq!(parse_tool_name("• Ran bash -lc 'cargo test'"), Some("bash"));
    assert_eq!(parse_tool_name("tool output line"), None);
}

#[test]
fn transcript_module_exposes_tool_helpers() {
    assert!(crate::transcript::is_tool_run_line("• Ran read /tmp/file"));
    assert_eq!(
        crate::transcript::parse_tool_name("[tool:write:error]"),
        Some("write")
    );
}

#[test]
fn transcript_render_messages_hides_read_tool_text_blocks() {
    let messages = vec![Message::ToolResult {
        tool_call_id: "call-read-1".to_string(),
        tool_name: "read".to_string(),
        content: vec![pixy_ai::ToolResultContentBlock::Text {
            text: "secret file body".to_string(),
            text_signature: None,
        }],
        details: None,
        is_error: false,
        timestamp: 1_700_000_000_010,
    }];

    let lines = render_messages(&messages);
    assert_eq!(lines.len(), 1, "read tool should only render the run line");
    assert_eq!(lines[0].text, "• Ran read");
    assert_eq!(lines[0].kind, TranscriptLineKind::Tool);
}

#[test]
fn apply_stream_update_normalizes_legacy_tool_header_to_ran_title() {
    let mut app = TuiApp::new("ready".to_string(), true, false);
    app.apply_stream_update(StreamUpdate::ToolLine("[tool:read:ok]".to_string()));
    assert_eq!(app.transcript.len(), 1);
    assert_eq!(app.transcript[0].text, "• Ran read");
}

#[test]
fn apply_stream_update_updates_thinking_line_in_place_while_streaming() {
    let mut app = TuiApp::new("ready".to_string(), true, false);
    app.start_working("pixy is reasoning...".to_string());
    app.apply_stream_update(StreamUpdate::AssistantLine("[thinking] a".to_string()));
    app.apply_stream_update(StreamUpdate::AssistantLine("[thinking] ab".to_string()));

    assert_eq!(app.transcript.len(), 1);
    assert_eq!(app.transcript[0].text, "[thinking] ab");
    assert_eq!(app.transcript[0].kind, TranscriptLineKind::Thinking);

    app.stop_working();
    app.apply_stream_update(StreamUpdate::AssistantLine("[thinking] done".to_string()));
    assert_eq!(app.transcript.len(), 2);
}

#[tokio::test]
async fn slash_resume_command_updates_status_when_backend_supports_resume() {
    let mut backend = TestBackend {
        resume_result: Ok(Some("session: /tmp/resumed.jsonl".to_string())),
        resume_targets: vec![],
        session_messages: None,
        recent_sessions_result: Ok(None),
        recent_sessions_limits: vec![],
        new_session_result: Ok(None),
        new_session_calls: 0,
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
async fn slash_resume_command_renders_resumed_session_messages() {
    let resumed_messages = vec![
        Message::Assistant {
            content: vec![AssistantContentBlock::Text {
                text: "restored answer".to_string(),
                text_signature: None,
            }],
            api: "openai-responses".to_string(),
            provider: "openai".to_string(),
            model: "gpt-4o-mini".to_string(),
            usage: Usage {
                input: 12,
                output: 4,
                cache_read: 0,
                cache_write: 0,
                total_tokens: 16,
                cost: Cost {
                    input: 0.0,
                    output: 0.0,
                    cache_read: 0.0,
                    cache_write: 0.0,
                    total: 0.0,
                },
            },
            stop_reason: StopReason::Stop,
            error_message: None,
            timestamp: 1_700_000_000_000,
        },
        Message::ToolResult {
            tool_call_id: "call_1".to_string(),
            tool_name: "bash".to_string(),
            content: vec![ToolResultContentBlock::Text {
                text: "ok".to_string(),
                text_signature: None,
            }],
            details: None,
            is_error: false,
            timestamp: 1_700_000_000_010,
        },
    ];
    let mut backend = TestBackend {
        resume_result: Ok(Some("session: /tmp/resumed.jsonl".to_string())),
        resume_targets: vec![],
        session_messages: Some(resumed_messages.clone()),
        recent_sessions_result: Ok(None),
        recent_sessions_limits: vec![],
        new_session_result: Ok(None),
        new_session_calls: 0,
    };
    let mut app = TuiApp::new("ready".to_string(), true, false);
    app.push_lines(["welcome".to_string()]);

    let handled = handle_slash_command("/resume old-session", &mut backend, &mut app)
        .await
        .expect("resume command should not error");

    assert!(handled);
    assert_eq!(app.status, "session: /tmp/resumed.jsonl");
    assert_eq!(app.transcript, render_messages(&resumed_messages));
}

#[tokio::test]
async fn slash_new_command_starts_new_session() {
    let mut backend = TestBackend {
        resume_result: Ok(None),
        resume_targets: vec![],
        session_messages: None,
        recent_sessions_result: Ok(None),
        recent_sessions_limits: vec![],
        new_session_result: Ok(Some("session: /tmp/new-session.jsonl".to_string())),
        new_session_calls: 0,
    };
    let mut app = TuiApp::new("ready".to_string(), true, false);

    let handled = handle_slash_command("/new", &mut backend, &mut app)
        .await
        .expect("/new should be handled");

    assert!(handled);
    assert_eq!(backend.new_session_calls, 1);
    assert_eq!(app.status, "session: /tmp/new-session.jsonl");
}

#[tokio::test]
async fn slash_session_command_avoids_none_placeholder_when_uninitialized() {
    let mut backend = TestBackend {
        resume_result: Ok(None),
        resume_targets: vec![],
        session_messages: None,
        recent_sessions_result: Ok(None),
        recent_sessions_limits: vec![],
        new_session_result: Ok(None),
        new_session_calls: 0,
    };
    let mut app = TuiApp::new("ready".to_string(), true, false);

    let handled = handle_slash_command("/session", &mut backend, &mut app)
        .await
        .expect("/session should be handled");

    assert!(handled);
    assert_eq!(app.status, "session not initialized yet");
}

#[tokio::test]
async fn slash_resume_command_renders_resume_error() {
    let mut backend = TestBackend {
        resume_result: Err("boom".to_string()),
        resume_targets: vec![],
        session_messages: None,
        recent_sessions_result: Ok(None),
        recent_sessions_limits: vec![],
        new_session_result: Ok(None),
        new_session_calls: 0,
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

#[tokio::test]
async fn slash_resume_without_target_lists_recent_sessions() {
    let mut backend = TestBackend {
        resume_result: Ok(Some("session: /tmp/resumed.jsonl".to_string())),
        resume_targets: vec![],
        session_messages: None,
        recent_sessions_result: Ok(Some(vec![
            ResumeCandidate {
                session_ref: "/tmp/session-2.jsonl".to_string(),
                title: "first task".to_string(),
                updated_at: "2026-02-25 12:10".to_string(),
            },
            ResumeCandidate {
                session_ref: "/tmp/session-1.jsonl".to_string(),
                title: "older task".to_string(),
                updated_at: "2026-02-25 11:03".to_string(),
            },
        ])),
        recent_sessions_limits: vec![],
        new_session_result: Ok(None),
        new_session_calls: 0,
    };
    let mut app = TuiApp::new("ready".to_string(), true, false);

    let handled = handle_slash_command("/resume", &mut backend, &mut app)
        .await
        .expect("resume list command should be handled");

    assert!(handled);
    assert!(backend.resume_targets.is_empty());
    assert_eq!(backend.recent_sessions_limits, vec![RESUME_LIST_LIMIT]);
    assert!(app.resume_picker.is_some(), "resume picker should open");
    assert_eq!(app.status, "select session and press Enter to resume");
}

#[tokio::test]
async fn slash_resume_numeric_selection_resumes_selected_candidate() {
    let mut backend = TestBackend {
        resume_result: Ok(Some("session: /tmp/session-1.jsonl".to_string())),
        resume_targets: vec![],
        session_messages: None,
        recent_sessions_result: Ok(Some(vec![
            ResumeCandidate {
                session_ref: "/tmp/session-2.jsonl".to_string(),
                title: "first task".to_string(),
                updated_at: "2026-02-25 12:10".to_string(),
            },
            ResumeCandidate {
                session_ref: "/tmp/session-1.jsonl".to_string(),
                title: "older task".to_string(),
                updated_at: "2026-02-25 11:03".to_string(),
            },
        ])),
        recent_sessions_limits: vec![],
        new_session_result: Ok(None),
        new_session_calls: 0,
    };
    let mut app = TuiApp::new("ready".to_string(), true, false);

    let handled = handle_slash_command("/resume 2", &mut backend, &mut app)
        .await
        .expect("resume selection command should be handled");

    assert!(handled);
    assert_eq!(
        backend.resume_targets,
        vec![Some("/tmp/session-1.jsonl".to_string())]
    );
    assert_eq!(backend.recent_sessions_limits, vec![RESUME_LIST_LIMIT]);
    assert_eq!(app.status, "session: /tmp/session-1.jsonl");
}

#[tokio::test]
async fn slash_resume_numeric_selection_rejects_out_of_range_index() {
    let mut backend = TestBackend {
        resume_result: Ok(Some("session: /tmp/session-1.jsonl".to_string())),
        resume_targets: vec![],
        session_messages: None,
        recent_sessions_result: Ok(Some(vec![ResumeCandidate {
            session_ref: "/tmp/session-2.jsonl".to_string(),
            title: "first task".to_string(),
            updated_at: "2026-02-25 12:10".to_string(),
        }])),
        recent_sessions_limits: vec![],
        new_session_result: Ok(None),
        new_session_calls: 0,
    };
    let mut app = TuiApp::new("ready".to_string(), true, false);

    let handled = handle_slash_command("/resume 2", &mut backend, &mut app)
        .await
        .expect("resume selection command should be handled");

    assert!(handled);
    assert!(backend.resume_targets.is_empty());
    assert_eq!(backend.recent_sessions_limits, vec![RESUME_LIST_LIMIT]);
    assert!(app.status.starts_with("resume failed:"));
    assert!(
        app.transcript
            .iter()
            .any(|line| line.text.contains("[resume_error] selection out of range")),
        "out-of-range error should be shown in transcript"
    );
}

#[test]
fn resume_picker_enter_resumes_selected_item() {
    let mut backend = TestBackend {
        resume_result: Ok(Some("session: /tmp/session-1.jsonl".to_string())),
        resume_targets: vec![],
        session_messages: None,
        recent_sessions_result: Ok(Some(vec![])),
        recent_sessions_limits: vec![],
        new_session_result: Ok(None),
        new_session_calls: 0,
    };
    let mut app = TuiApp::new("ready".to_string(), true, false);
    app.open_resume_picker(vec![
        ResumeCandidate {
            session_ref: "/tmp/session-2.jsonl".to_string(),
            title: "first task".to_string(),
            updated_at: "2026-02-25 12:10".to_string(),
        },
        ResumeCandidate {
            session_ref: "/tmp/session-1.jsonl".to_string(),
            title: "older task".to_string(),
            updated_at: "2026-02-25 11:03".to_string(),
        },
    ]);
    app.resume_picker.as_mut().expect("picker").selected = 1;

    let handled = handle_resume_picker_key_event(
        KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
        &mut backend,
        &mut app,
    );

    assert!(handled);
    assert_eq!(
        backend.resume_targets,
        vec![Some("/tmp/session-1.jsonl".to_string())]
    );
    assert!(
        app.resume_picker.is_none(),
        "picker should close after resume"
    );
}

#[test]
fn resume_picker_escape_cancels_picker() {
    let mut backend = TestBackend {
        resume_result: Ok(Some("session: /tmp/session-1.jsonl".to_string())),
        resume_targets: vec![],
        session_messages: None,
        recent_sessions_result: Ok(Some(vec![])),
        recent_sessions_limits: vec![],
        new_session_result: Ok(None),
        new_session_calls: 0,
    };
    let mut app = TuiApp::new("ready".to_string(), true, false);
    app.open_resume_picker(vec![ResumeCandidate {
        session_ref: "/tmp/session-2.jsonl".to_string(),
        title: "first task".to_string(),
        updated_at: "2026-02-25 12:10".to_string(),
    }]);

    let handled = handle_resume_picker_key_event(
        KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE),
        &mut backend,
        &mut app,
    );

    assert!(handled);
    assert!(backend.resume_targets.is_empty());
    assert!(app.resume_picker.is_none());
    assert_eq!(app.status, "resume cancelled");
}

#[test]
fn streaming_follow_up_key_queues_input_and_clears_editor() {
    let mut app = TuiApp::new("ready".to_string(), true, false);
    app.input = "next question".to_string();
    let quit = vec![KeyBinding {
        code: KeyCode::Char('d'),
        modifiers: KeyModifiers::CONTROL,
    }];
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
        &quit,
        &interrupt,
        &follow_up,
        &[],
        &[],
        &abort_controller,
        &mut app,
    );

    assert!(!outcome.interrupted);
    assert!(outcome.ui_changed);
    assert!(!outcome.force_exit);
    assert!(app.input.is_empty());
    assert_eq!(app.input_history, vec!["next question".to_string()]);
    assert_eq!(app.queued_follow_up_count(), 1);
    assert_eq!(app.status, "queued follow-up (1)");
}

#[test]
fn streaming_plain_enter_queues_input_and_clears_editor() {
    let mut app = TuiApp::new("ready".to_string(), true, false);
    app.input = "queued by enter".to_string();
    let quit = vec![KeyBinding {
        code: KeyCode::Char('d'),
        modifiers: KeyModifiers::CONTROL,
    }];
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
        Event::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
        &quit,
        &interrupt,
        &follow_up,
        &[],
        &[],
        &abort_controller,
        &mut app,
    );

    assert!(!outcome.interrupted);
    assert!(outcome.ui_changed);
    assert!(!outcome.force_exit);
    assert!(app.input.is_empty());
    assert_eq!(app.input_history, vec!["queued by enter".to_string()]);
    assert_eq!(app.queued_follow_up_count(), 1);
    assert_eq!(app.status, "queued follow-up (1)");
}

#[test]
fn streaming_follow_up_key_does_not_queue_empty_input() {
    let mut app = TuiApp::new("ready".to_string(), true, false);
    let quit = vec![KeyBinding {
        code: KeyCode::Char('d'),
        modifiers: KeyModifiers::CONTROL,
    }];
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
        &quit,
        &interrupt,
        &follow_up,
        &[],
        &[],
        &abort_controller,
        &mut app,
    );

    assert!(!outcome.interrupted);
    assert!(!outcome.ui_changed);
    assert!(!outcome.force_exit);
    assert_eq!(app.queued_follow_up_count(), 0);
    assert_eq!(app.status, "ready");
}

#[test]
fn streaming_ctrl_d_sets_force_exit_status() {
    let mut app = TuiApp::new("ready".to_string(), true, false);
    let quit = vec![KeyBinding {
        code: KeyCode::Char('d'),
        modifiers: KeyModifiers::CONTROL,
    }];
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
        Event::Key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL)),
        &quit,
        &interrupt,
        &follow_up,
        &[],
        &[],
        &abort_controller,
        &mut app,
    );

    assert!(!outcome.interrupted);
    assert!(outcome.ui_changed);
    assert!(outcome.force_exit);
    assert_eq!(app.status, FORCE_EXIT_STATUS);
}

#[test]
fn welcome_banner_includes_block_pixy_logo() {
    let lines = build_welcome_banner(&TuiOptions::default());
    let logo_rows = lines.iter().take_while(|line| line.contains('█')).count();
    assert_eq!(logo_rows, 7, "PIXY logo should use 7 block rows");
    assert!(
        lines.iter().any(|line| line.contains("████")),
        "welcome banner should render the PIXY logo with block glyphs"
    );
    assert!(
        lines.iter().all(|line| !line.contains("PPPP")),
        "welcome banner should not fall back to letter-based art"
    );
}

#[test]
fn welcome_banner_includes_version_line_when_present() {
    let mut options = TuiOptions::default();
    options.version = "0.1.0".to_string();
    let lines = build_welcome_banner(&options);
    assert!(
        lines.iter().any(|line| line.contains("v0.1.0")),
        "version should be shown in concise welcome card"
    );
}

#[test]
fn welcome_banner_omits_skill_file_names() {
    let mut options = TuiOptions::default();
    options.startup_resource_lines = vec![
        "Loaded skills: 2".to_string(),
        "/workspace/.agents/skills/demo/SKILL.md".to_string(),
    ];
    let lines = build_welcome_banner(&options);
    assert!(
        lines.iter().any(|line| line == "Loaded skills: 2"),
        "banner should include loaded skills summary"
    );
    assert!(
        lines.iter().all(|line| !line.trim().ends_with("SKILL.md")),
        "banner should not enumerate individual skill file names"
    );
}

#[test]
fn transcript_title_includes_version_when_present() {
    assert_eq!(
        transcript_title("pixy", "0.1.0"),
        "Welcome to pixy Chat  v0.1.0"
    );
    assert_eq!(transcript_title("pixy", ""), "Welcome to pixy Chat");
}

#[test]
fn welcome_banner_includes_startup_resource_lines() {
    let mut options = TuiOptions::default();
    options.startup_resource_lines = vec!["Loaded skills: 4".to_string()];

    let lines = build_welcome_banner(&options);
    assert!(lines.contains(&"Loaded skills: 4".to_string()));
}

#[test]
fn startup_status_defaults_to_ready_even_when_session_exists() {
    let backend = TestBackend {
        resume_result: Ok(None),
        resume_targets: vec![],
        session_messages: None,
        recent_sessions_result: Ok(None),
        recent_sessions_limits: vec![],
        new_session_result: Ok(None),
        new_session_calls: 0,
    };

    assert_eq!(startup_status_label(&backend), "ready");
}

#[test]
fn status_bar_shows_working_without_steering_rows() {
    let mut app = TuiApp::new("pixy is working...".to_string(), true, false);
    app.set_status_bar_meta(
        String::new(),
        "Auto (High) - allow all commands".to_string(),
        "Databend GPT-5.3 Codex [custom]".to_string(),
    );
    app.start_working("pixy is working...".to_string());
    app.queue_follow_up("43".to_string());
    app.queue_follow_up("434".to_string());

    let status = render_status_bar_lines(&app, 40, TuiTheme::Dark);
    assert_eq!(status.lines.len(), 3);
    assert!(!line_text(&status.lines[0]).contains("Steering:"));
    assert!(line_text(&status.lines[0]).contains("Auto (High)"));
    assert!(line_text(&status.lines[1]).contains("shift+tab"));
    assert!(line_text(&status.lines[2]).contains("? for help"));
}

#[test]
fn status_bar_hides_ready_suffix_when_status_top_present() {
    let mut app = TuiApp::new("ready".to_string(), true, false);
    app.set_status_bar_meta(
        "/data/work/pixy (main)".to_string(),
        "Auto (High) - allow all commands".to_string(),
        "databend/gpt-5.3-codex".to_string(),
    );

    let status = render_status_bar_lines(&app, 80, TuiTheme::Dark);
    assert_eq!(line_text(&status.lines[0]), "/data/work/pixy (main)");
}

#[test]
fn steering_panel_lines_are_right_aligned_above_input() {
    let mut app = TuiApp::new("pixy is working...".to_string(), true, false);
    app.start_working("pixy is working...".to_string());
    app.queue_follow_up("43".to_string());
    app.queue_follow_up("434".to_string());

    let steering = render_steering_panel_lines(&app, 40);
    assert_eq!(steering.lines.len(), 3);
    let steering_1 = line_text(&steering.lines[0]);
    let steering_2 = line_text(&steering.lines[1]);
    let steering_hint = line_text(&steering.lines[2]);
    assert_eq!(steering_1.trim_start(), "Steering: 43");
    assert_eq!(steering_2.trim_start(), "Steering: 434");
    assert_eq!(
        steering_hint.trim_start(),
        "↳ Alt+Up to edit all queued messages"
    );
    assert!(steering_1.starts_with(' '));
    assert!(steering_2.starts_with(' '));
    assert!(steering_hint.starts_with(' '));
}

#[test]
fn dequeue_key_moves_queued_follow_ups_into_editor_and_clears_queue() {
    let mut app = TuiApp::new("ready".to_string(), true, false);
    app.start_working("pixy is working...".to_string());
    app.queue_follow_up("first".to_string());
    app.queue_follow_up("second".to_string());
    let quit = vec![KeyBinding {
        code: KeyCode::Char('d'),
        modifiers: KeyModifiers::CONTROL,
    }];
    let interrupt = vec![KeyBinding {
        code: KeyCode::Esc,
        modifiers: KeyModifiers::NONE,
    }];
    let follow_up = vec![KeyBinding {
        code: KeyCode::Enter,
        modifiers: KeyModifiers::ALT,
    }];
    let dequeue = vec![KeyBinding {
        code: KeyCode::Up,
        modifiers: KeyModifiers::ALT,
    }];
    let abort_controller = AgentAbortController::new();

    let outcome = handle_streaming_event(
        Event::Key(KeyEvent::new(KeyCode::Up, KeyModifiers::ALT)),
        &quit,
        &interrupt,
        &follow_up,
        &dequeue,
        &[],
        &abort_controller,
        &mut app,
    );

    assert!(!outcome.interrupted);
    assert!(outcome.ui_changed);
    assert!(!outcome.force_exit);
    assert_eq!(app.input, "first\nsecond");
    assert_eq!(app.queued_follow_up_count(), 0);
    assert!(
        app.steering_status_lines().is_empty(),
        "dequeue should clear steering panel lines"
    );
    assert_eq!(app.status, "editing 2 queued messages");
}

#[test]
fn dequeue_key_accepts_alt_shift_up_for_terminal_compatibility() {
    let mut app = TuiApp::new("ready".to_string(), true, false);
    app.start_working("pixy is working...".to_string());
    app.queue_follow_up("first".to_string());
    let quit = vec![KeyBinding {
        code: KeyCode::Char('d'),
        modifiers: KeyModifiers::CONTROL,
    }];
    let interrupt = vec![KeyBinding {
        code: KeyCode::Esc,
        modifiers: KeyModifiers::NONE,
    }];
    let follow_up = vec![KeyBinding {
        code: KeyCode::Enter,
        modifiers: KeyModifiers::ALT,
    }];
    let dequeue = vec![KeyBinding {
        code: KeyCode::Up,
        modifiers: KeyModifiers::ALT,
    }];
    let abort_controller = AgentAbortController::new();

    let outcome = handle_streaming_event(
        Event::Key(KeyEvent::new(
            KeyCode::Up,
            KeyModifiers::ALT | KeyModifiers::SHIFT,
        )),
        &quit,
        &interrupt,
        &follow_up,
        &dequeue,
        &[],
        &abort_controller,
        &mut app,
    );

    assert!(outcome.ui_changed);
    assert_eq!(app.input, "first");
    assert_eq!(app.queued_follow_up_count(), 0);
}
