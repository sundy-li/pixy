use std::io;

use crossterm::event::{
    EnableBracketedPaste, EnableMouseCapture, Event, EventStream, KeyEvent, KeyEventKind,
    KeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use crossterm::execute;
use crossterm::terminal::enable_raw_mode;
use futures_util::StreamExt;
use pixy_ai::UserContentBlock;
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

use super::terminal::TerminalRestore;
use super::{
    apply_selection_osc_colors, build_welcome_banner, default_terminal_options, draw_ui_frame,
    handle_continue_streaming as handle_continue_streaming_impl, handle_editor_key_event,
    handle_input_history_key_event, handle_mouse_history_event, handle_paste_event,
    handle_resume_picker_key_event as handle_resume_picker_key_event_impl,
    handle_transcript_scroll_key, is_force_exit_signal, keybinding_label, matches_keybinding,
    now_millis, persist_welcome_into_transcript,
    primary_keybinding_label_lower as primary_keybinding_label_lower_impl,
    process_queued_follow_ups as process_queued_follow_ups_impl, query_session_status_label,
    run_submitted_input as run_submitted_input_impl, startup_status_label, InputHistoryStore,
    TuiApp, TuiBackend, TuiOptions,
};

pub(crate) struct TuiRuntime<'a, B: TuiBackend> {
    backend: &'a mut B,
    options: TuiOptions,
    app: TuiApp,
    terminal: Terminal<CrosstermBackend<io::Stdout>>,
    events: EventStream,
    _restore: TerminalRestore,
}

enum RuntimeControl {
    Continue,
    Exit,
}

fn normalize_backend_model_status(status: String) -> String {
    if status.starts_with("model: ") {
        "ready".to_string()
    } else {
        status
    }
}

impl<'a, B: TuiBackend> TuiRuntime<'a, B> {
    pub(crate) fn new(backend: &'a mut B, options: TuiOptions) -> Result<Self, String> {
        enable_raw_mode().map_err(|error| format!("enable raw mode failed: {error}"))?;
        let mouse_capture_enabled = if options.enable_mouse_capture {
            execute!(io::stdout(), EnableMouseCapture)
                .map_err(|error| format!("enable mouse capture failed: {error}"))?;
            true
        } else {
            false
        };

        let keyboard_enhancement_enabled = if crossterm::terminal::supports_keyboard_enhancement()
            .unwrap_or(false)
        {
            execute!(
                io::stdout(),
                PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
            )
            .is_ok()
        } else {
            false
        };

        let bracketed_paste_enabled = execute!(io::stdout(), EnableBracketedPaste).is_ok();

        let mut restore = TerminalRestore {
            keyboard_enhancement_enabled,
            mouse_capture_enabled,
            bracketed_paste_enabled,
            selection_colors_applied: false,
            alternate_screen_enabled: false,
        };

        let status = startup_status_label(backend);
        let mut app = TuiApp::new(status, options.show_tool_results, options.initial_help);
        app.set_interrupt_hint_label(primary_keybinding_label_lower_impl(
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
            String::new(),
            options.status_left.clone(),
            options.status_right.clone(),
        );
        app.set_welcome_lines(build_welcome_banner(&options));
        persist_welcome_into_transcript(&mut app);

        let mut fullscreen_init_error: Option<String> = None;
        let mut terminal = match Terminal::with_options(
            CrosstermBackend::new(io::stdout()),
            default_terminal_options(),
        ) {
            Ok(terminal) => terminal,
            Err(error) => {
                fullscreen_init_error = Some(error.to_string());
                Terminal::new(CrosstermBackend::new(io::stdout())).map_err(|fallback_error| {
                    format!(
                        "create terminal failed: {error}; fallback terminal failed: {fallback_error}"
                    )
                })?
            }
        };
        terminal
            .clear()
            .map_err(|error| format!("clear terminal failed: {error}"))?;
        restore.selection_colors_applied = apply_selection_osc_colors(options.theme);

        if let Some(error) = fullscreen_init_error {
            app.status = format!("fullscreen viewport unavailable: {error}");
        }

        Ok(Self {
            backend,
            options,
            app,
            terminal,
            events: EventStream::new(),
            _restore: restore,
        })
    }

    pub(crate) async fn run(&mut self) -> Result<(), String> {
        let mut needs_redraw = true;
        loop {
            if needs_redraw {
                self.draw_ui()?;
                needs_redraw = false;
            }

            let maybe_event = self.events.next().await;
            let Some(event_result) = maybe_event else {
                return Ok(());
            };
            let event =
                event_result.map_err(|error| format!("read terminal event failed: {error}"))?;

            if let Event::Mouse(mouse) = event {
                needs_redraw = handle_mouse_history_event(&mut self.app, mouse);
                continue;
            }

            if let Event::Paste(pasted) = event {
                handle_paste_event(&mut self.app, pasted);
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

            if let RuntimeControl::Exit = self.dispatch_key_event(key).await? {
                return Ok(());
            }
        }
    }

    fn draw_ui(&mut self) -> Result<(), String> {
        draw_ui_frame(&mut self.terminal, &mut self.app, &self.options)
            .map_err(|error| format!("draw UI failed: {error}"))
    }

    async fn dispatch_key_event(&mut self, key: KeyEvent) -> Result<RuntimeControl, String> {
        if matches_keybinding(&self.options.keybindings.quit, key) {
            return Ok(RuntimeControl::Exit);
        }
        if self.handle_resume_picker_key_event(key) {
            return Ok(RuntimeControl::Continue);
        }
        if matches_keybinding(&self.options.keybindings.interrupt, key) {
            self.app.clear_input();
            self.app.show_help = false;
            self.app.status = "interrupted".to_string();
            return Ok(RuntimeControl::Continue);
        }
        if matches_keybinding(&self.options.keybindings.clear, key) {
            if self.app.has_input_payload() || !self.app.pending_text_attachments.is_empty() {
                self.app.clear_input();
                self.app.last_clear_key_at_ms = now_millis();
                self.app.status = "input cleared".to_string();
                return Ok(RuntimeControl::Continue);
            }

            let now = now_millis();
            if now.saturating_sub(self.app.last_clear_key_at_ms) <= 500 {
                return Ok(RuntimeControl::Exit);
            }
            self.app.last_clear_key_at_ms = now;
            self.app.status = "press clear again to exit".to_string();
            return Ok(RuntimeControl::Continue);
        }
        if matches_keybinding(&self.options.keybindings.show_help, key) {
            self.app.show_help = !self.app.show_help;
            return Ok(RuntimeControl::Continue);
        }
        if matches_keybinding(&self.options.keybindings.show_session, key) {
            self.app.status = query_session_status_label(self.backend);
            return Ok(RuntimeControl::Continue);
        }
        if matches_keybinding(&self.options.keybindings.cycle_model_forward, key) {
            self.app.status = match self.backend.cycle_model_forward() {
                Ok(Some(status)) => {
                    self.app
                        .maybe_update_status_right_from_backend_status(&status);
                    normalize_backend_model_status(status)
                }
                Ok(None) => "".to_string(),
                Err(error) => format!("cycle model failed: {error}"),
            };
            return Ok(RuntimeControl::Continue);
        }
        if matches_keybinding(&self.options.keybindings.cycle_model_backward, key) {
            self.app.status = match self.backend.cycle_model_backward() {
                Ok(Some(status)) => {
                    self.app
                        .maybe_update_status_right_from_backend_status(&status);
                    normalize_backend_model_status(status)
                }
                Ok(None) => "".to_string(),
                Err(error) => format!("cycle model failed: {error}"),
            };
            return Ok(RuntimeControl::Continue);
        }
        if matches_keybinding(&self.options.keybindings.select_model, key) {
            self.app.status = match self.backend.select_model() {
                Ok(Some(status)) => {
                    self.app
                        .maybe_update_status_right_from_backend_status(&status);
                    normalize_backend_model_status(status)
                }
                Ok(None) => "".to_string(),
                Err(error) => format!("select model failed: {error}"),
            };
            return Ok(RuntimeControl::Continue);
        }
        if matches_keybinding(&self.options.keybindings.cycle_thinking_level, key) {
            self.app.status = match self.backend.cycle_mode() {
                Ok(Some(status)) => {
                    self.app
                        .maybe_update_status_left_from_backend_status(&status);
                    if status.starts_with("mode: ") {
                        "ready".to_string()
                    } else {
                        status
                    }
                }
                Ok(None) => "".to_string(),
                Err(error) => format!("cycle mode failed: {error}"),
            };
            return Ok(RuntimeControl::Continue);
        }
        if matches_keybinding(&self.options.keybindings.expand_tools, key) {
            let enabled = self.app.toggle_tool_results();
            self.app.status = if enabled {
                "tool output visible".to_string()
            } else {
                "tool output hidden".to_string()
            };
            return Ok(RuntimeControl::Continue);
        }
        if matches_keybinding(&self.options.keybindings.continue_run, key) {
            if !self.app.has_input_payload() {
                if let Err(error) = self.handle_continue_streaming().await {
                    if is_force_exit_signal(&error) {
                        return Ok(RuntimeControl::Exit);
                    }
                    return Err(error);
                }
                if let Err(error) = self.process_queued_follow_ups().await {
                    if is_force_exit_signal(&error) {
                        return Ok(RuntimeControl::Exit);
                    }
                    return Err(error);
                }
            } else {
                let (display_input, submitted, blocks) = self.app.take_input_payload();
                if submitted.is_empty() && blocks.is_none() {
                    return Ok(RuntimeControl::Continue);
                }
                if let Err(error) = self
                    .run_submitted_input(display_input, submitted, blocks)
                    .await
                {
                    if is_force_exit_signal(&error) {
                        return Ok(RuntimeControl::Exit);
                    }
                    return Err(error);
                }
            }
            return Ok(RuntimeControl::Continue);
        }
        if matches_keybinding(&self.options.keybindings.dequeue, key) {
            if let Some(count) = self.app.dequeue_follow_ups_to_editor() {
                let label = if count == 1 { "message" } else { "messages" };
                self.app.status = format!("editing {count} queued {label}");
                return Ok(RuntimeControl::Continue);
            }
        }
        if matches_keybinding(&self.options.keybindings.newline, key) {
            self.app.insert_char('\n');
            return Ok(RuntimeControl::Continue);
        }
        if matches_keybinding(&self.options.keybindings.submit, key) {
            let (display_input, submitted, blocks) = self.app.take_input_payload();
            if submitted.is_empty() && blocks.is_none() {
                return Ok(RuntimeControl::Continue);
            }
            if let Err(error) = self
                .run_submitted_input(display_input, submitted, blocks)
                .await
            {
                if is_force_exit_signal(&error) {
                    return Ok(RuntimeControl::Exit);
                }
                return Err(error);
            }
            return Ok(RuntimeControl::Continue);
        }
        if handle_input_history_key_event(&mut self.app, key) {
            return Ok(RuntimeControl::Continue);
        }
        if handle_transcript_scroll_key(&mut self.app, key) {
            return Ok(RuntimeControl::Continue);
        }
        if handle_editor_key_event(&mut self.app, key) {
            return Ok(RuntimeControl::Continue);
        }

        Ok(RuntimeControl::Continue)
    }

    fn handle_resume_picker_key_event(&mut self, key: KeyEvent) -> bool {
        handle_resume_picker_key_event_impl(key, self.backend, &mut self.app)
    }

    async fn run_submitted_input(
        &mut self,
        display_input: String,
        submitted_input: String,
        blocks: Option<Vec<UserContentBlock>>,
    ) -> Result<(), String> {
        run_submitted_input_impl(
            self.backend,
            &mut self.terminal,
            &mut self.app,
            &self.options,
            display_input,
            submitted_input,
            blocks,
            &mut self.events,
        )
        .await
    }

    async fn handle_continue_streaming(&mut self) -> Result<(), String> {
        handle_continue_streaming_impl(
            self.backend,
            &mut self.terminal,
            &mut self.app,
            &self.options,
            &mut self.events,
        )
        .await
    }

    async fn process_queued_follow_ups(&mut self) -> Result<(), String> {
        process_queued_follow_ups_impl(
            self.backend,
            &mut self.terminal,
            &mut self.app,
            &self.options,
            &mut self.events,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::normalize_backend_model_status;

    #[test]
    fn normalize_backend_model_status_hides_ephemeral_model_line() {
        assert_eq!(
            normalize_backend_model_status("model: anthropic/claude-3-5-sonnet-latest".to_string()),
            "ready"
        );
        assert_eq!(
            normalize_backend_model_status("cycle model failed: boom".to_string()),
            "cycle model failed: boom"
        );
    }
}
