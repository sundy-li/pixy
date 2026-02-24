use std::fs;
use std::path::PathBuf;

use pixy_agent_core::AgentAbortSignal;

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
    recent_sessions_result: Result<Option<Vec<ResumeCandidate>>, String>,
    recent_sessions_limits: Vec<usize>,
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
    assert_eq!(input_cursor_position(&app, area, "› "), (0, 2));
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
    assert_eq!(input_area_height(&app, frame, "› ", 2), 3);

    app.input = "line 1\nline 2".to_string();
    assert_eq!(input_area_height(&app, frame, "› ", 2), 4);

    app.input = "line 1".to_string();
    assert_eq!(input_area_height(&app, frame, "› ", 2), 3);
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

    assert_eq!(input_cursor_position(&app, area, "› "), (4, 1));

    app.cursor_pos = 2;
    assert_eq!(input_cursor_position(&app, area, "› "), (5, 1));
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

    assert_eq!(input_cursor_position(&app, area, "› "), (2, 1));
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
    assert_eq!(input_cursor_position(&app, area, "› "), (2, 1));
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
fn working_line_includes_elapsed_and_interrupt_hint_when_running_tool() {
    let mut app = TuiApp::new("ready".to_string(), true, false);
    app.start_working("• Ran read".to_string());

    let working = app
        .working_line()
        .expect("working line should be present while working");

    assert!(working.text.contains("• Ran read"));
    assert!(working.text.contains("s •"));
    assert!(working.text.contains("to interrupt"));
}

#[test]
fn status_bar_shows_elapsed_and_interrupt_hint_while_working() {
    let mut app = TuiApp::new("pi is working...".to_string(), true, false);
    app.start_working("pi is working...".to_string());

    let status = render_status_bar_lines(&app, 120);
    let top = line_text(&status.lines[0]);

    assert!(top.contains("pi is working..."));
    assert!(top.contains("s •"));
    assert!(top.contains("to interrupt"));
}

#[test]
fn tool_output_line_resets_working_message_to_generic_state() {
    let mut app = TuiApp::new("ready".to_string(), true, false);
    app.start_working("pixy is working...".to_string());

    app.note_working_from_update(
        "pixy",
        &StreamUpdate::ToolLine("• Ran bash -lc 'mkdir -p /tmp/snake_v5'".to_string()),
    );
    assert_eq!(
        app.working_message,
        "• Ran bash -lc 'mkdir -p /tmp/snake_v5'"
    );

    app.note_working_from_update("pixy", &StreamUpdate::ToolLine("(no output)".to_string()));
    assert_eq!(app.working_message, "pixy is working...");
}

#[test]
fn working_line_formats_elapsed_time_as_minutes_and_seconds() {
    let mut app = TuiApp::new("ready".to_string(), true, false);
    app.start_working("Checking file git status flags".to_string());
    app.working_started_at = Some(Instant::now() - Duration::from_secs(510));

    let working = app
        .working_line()
        .expect("working line should be present while working");

    assert!(working.text.contains("(8m 30s •"));
}

#[test]
fn working_line_marquee_highlights_four_chars_and_rotates() {
    let mut app = TuiApp::new("ready".to_string(), true, false);
    app.start_working("Checking file git status flags".to_string());
    app.working_tick = 0;

    let line0 = app
        .working_line()
        .expect("working line should be present")
        .to_line(200, TuiTheme::Dark);
    assert!(
        line0.spans.iter().any(|span| {
            span.content.contains("Chec")
                && span.style.fg == Some(Color::White)
                && span.style.bg == Some(Color::Rgb(40, 44, 52))
        }),
        "tick=0 should highlight first 4 chars with white-on-black"
    );
    assert!(
        line0.spans.iter().any(|span| {
            span.style.fg == Some(Color::Rgb(148, 150, 153))
                && span.style.bg == Some(Color::Rgb(40, 44, 52))
        }),
        "non-highlight text should use gray-on-black"
    );

    app.working_tick = 1;
    let line1 = app
        .working_line()
        .expect("working line should be present")
        .to_line(200, TuiTheme::Dark);
    assert!(
        line1.spans.iter().any(|span| {
            span.content.contains("heck")
                && span.style.fg == Some(Color::White)
                && span.style.bg == Some(Color::Rgb(40, 44, 52))
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
            .any(|span| { span.content.contains("Chec") && span.style.fg == Some(Color::Black) }),
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
        TranscriptLine::new("• Ran edit crates/pixy-tui/src/lib.rs".to_string(), TranscriptLineKind::Tool),
        TranscriptLine::new("crates/pixy-tui/src/lib.rs | 2 ++--".to_string(), TranscriptLineKind::Tool),
        TranscriptLine::new("• Ran bash -lc 'cargo test -p pixy-tui'".to_string(), TranscriptLineKind::Tool),
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
    assert_eq!(removed.spans[0].style.fg, Some(Color::Red));
    assert_eq!(added.spans[0].style.fg, Some(Color::Green));
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
    assert_eq!(plus_span.style.fg, Some(Color::Green));
    assert_eq!(minus_span.style.fg, Some(Color::Red));
}

#[test]
fn tool_lines_keep_default_background_and_use_light_text() {
    let tool = TranscriptLine::new("tool output".to_string(), TranscriptLineKind::Tool)
        .to_line(40, TuiTheme::Dark);
    assert_eq!(tool.spans[0].style.fg, Some(Color::Gray));
    assert_eq!(tool.spans[0].style.bg, None);
}

#[test]
fn user_input_line_uses_input_block_background_from_theme() {
    let input = TranscriptLine::new("hello".to_string(), TranscriptLineKind::UserInput)
        .to_line(40, TuiTheme::Dark);
    assert_eq!(input.spans[0].style.bg, Some(Color::Rgb(52, 53, 65)));
}

#[test]
fn format_user_input_line_prefixes_prompt() {
    assert_eq!(format_user_input_line("hello", "› "), "› hello");
    assert_eq!(
        format_user_input_line("this is second question", "pi> "),
        "pi> this is second question"
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
fn parse_tui_theme_names_case_insensitive() {
    assert_eq!(TuiTheme::from_name("dark"), Some(TuiTheme::Dark));
    assert_eq!(TuiTheme::from_name("LIGHT"), Some(TuiTheme::Light));
    assert_eq!(TuiTheme::from_name(""), None);
    assert_eq!(TuiTheme::from_name("solarized"), None);
}

#[test]
fn dark_theme_selection_osc_sequence_is_white_bg_black_fg() {
    assert_eq!(
        selection_osc_set_sequence(TuiTheme::Dark),
        Some("\u{1b}]17;#ffffff\u{7}\u{1b}]19;#000000\u{7}".to_string())
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
            .any(|sequence| sequence.contains("rgb:ff/ff/ff") && sequence.contains("\u{7}"))
    );
    assert!(
        sequences
            .iter()
            .any(|sequence| sequence.contains("rgb:ff/ff/ff") && sequence.contains("\u{1b}\\"))
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
                && span.style.fg == Some(Color::Cyan)
        }),
        "file path token should be highlighted"
    );
    assert!(
        line.spans.iter().any(
            |span| span.content.contains("PageUp") && span.style.fg == Some(Color::LightYellow)
        ),
        "PageUp token should be highlighted"
    );
    assert!(
        line.spans
            .iter()
            .any(|span| span.content.contains("PageDown")
                && span.style.fg == Some(Color::LightYellow)),
        "PageDown token should be highlighted"
    );
    assert!(
        line.spans.iter().any(
            |span| span.content.contains("ctrl+c") && span.style.fg == Some(Color::LightYellow)
        ),
        "ctrl+c token should be highlighted"
    );
    assert!(
        line.spans
            .iter()
            .any(|span| span.content.contains("shift+tab")
                && span.style.fg == Some(Color::LightYellow)),
        "shift+tab token should be highlighted"
    );
    assert!(
        line.spans
            .iter()
            .any(|span| span.content.contains("alt+enter")
                && span.style.fg == Some(Color::LightYellow)),
        "alt+enter token should be highlighted"
    );
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
        line.spans
            .iter()
            .any(|span| span.content.contains(token) && span.style.fg == Some(Color::LightYellow)),
        "slash compound shortcuts should use key token color"
    );
    assert!(
        !line
            .spans
            .iter()
            .any(|span| span.content.contains(token) && span.style.fg == Some(Color::Cyan)),
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
        line.spans.iter().any(
            |span| span.content.contains("escape") && span.style.fg == Some(Color::LightYellow)
        ),
        "escape should be highlighted as a key token"
    );
    assert!(
        line.spans.iter().any(
            |span| span.content.contains("enter") && span.style.fg == Some(Color::LightYellow)
        ),
        "enter should be highlighted as a key token"
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
        recent_sessions_result: Ok(None),
        recent_sessions_limits: vec![],
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
        recent_sessions_result: Ok(None),
        recent_sessions_limits: vec![],
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
        recent_sessions_result: Ok(Some(vec![ResumeCandidate {
            session_ref: "/tmp/session-2.jsonl".to_string(),
            title: "first task".to_string(),
            updated_at: "2026-02-25 12:10".to_string(),
        }])),
        recent_sessions_limits: vec![],
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
        recent_sessions_result: Ok(Some(vec![])),
        recent_sessions_limits: vec![],
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
        recent_sessions_result: Ok(Some(vec![])),
        recent_sessions_limits: vec![],
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
fn welcome_banner_mentions_force_exit_for_quit_key() {
    let lines = build_welcome_banner(&TuiOptions::default());
    assert!(
        lines.iter().any(|line| line.contains("to force exit")),
        "welcome banner should explicitly advertise force-exit behavior"
    );
}

#[test]
fn welcome_banner_omits_standalone_version_line() {
    let lines = build_welcome_banner(&TuiOptions::default());
    assert!(
        !lines
            .iter()
            .any(|line| line.trim_start().starts_with("pixy v")),
        "version should be rendered in the title instead of a standalone welcome line"
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
    options.startup_resource_lines = vec![
        "[Context]".to_string(),
        "  /workspace/AGENTS.md".to_string(),
        String::new(),
        "[Skills]".to_string(),
        "  user".to_string(),
        "    /workspace/.agents/skills/demo/SKILL.md".to_string(),
    ];

    let lines = build_welcome_banner(&options);
    assert!(lines.contains(&" [Context]".to_string()));
    assert!(lines.contains(&"   /workspace/AGENTS.md".to_string()));
    assert!(lines.contains(&" [Skills]".to_string()));
    assert!(lines.contains(&"   user".to_string()));
    assert!(lines.contains(&"     /workspace/.agents/skills/demo/SKILL.md".to_string()));
}

#[test]
fn status_bar_shows_working_without_steering_rows() {
    let mut app = TuiApp::new("pixy is working...".to_string(), true, false);
    app.start_working("pixy is working...".to_string());
    app.queue_follow_up("43".to_string());
    app.queue_follow_up("434".to_string());

    let status = render_status_bar_lines(&app, 40);
    assert_eq!(status.lines.len(), 2);
    assert!(line_text(&status.lines[0]).contains("pixy is working..."));
    assert!(!line_text(&status.lines[0]).contains("Steering:"));
    assert!(!line_text(&status.lines[1]).contains("Steering:"));
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
