use crossterm::event::{KeyCode, KeyEvent};

use crate::{TuiApp, TuiBackend, RESUME_LIST_LIMIT};

pub(super) fn handle_slash_resume_command<B: TuiBackend>(
    command: &str,
    backend: &mut B,
    app: &mut TuiApp,
) -> Result<bool, String> {
    let target = command
        .strip_prefix("/resume")
        .map(str::trim)
        .filter(|value| !value.is_empty());
    if let Some(target) = target {
        if is_resume_cancel_command(target) {
            app.close_resume_picker();
            app.status = "resume cancelled".to_string();
            return Ok(true);
        }
        if let Some(index) = parse_resume_selection_index(target) {
            return match resolve_resume_target_by_index(backend, index) {
                Ok(Some(session_ref)) => {
                    app.close_resume_picker();
                    let result = backend.resume_session(Some(session_ref.as_str()));
                    Ok(apply_resume_result(backend, result, app))
                }
                Ok(None) => {
                    let result = backend.resume_session(Some(target));
                    Ok(apply_resume_result(backend, result, app))
                }
                Err(error) => {
                    app.push_lines([format!("[resume_error] {error}")]);
                    app.status = format!("resume failed: {error}");
                    Ok(true)
                }
            };
        }
        app.close_resume_picker();
        let result = backend.resume_session(Some(target));
        return Ok(apply_resume_result(backend, result, app));
    }

    match backend.recent_resumable_sessions(RESUME_LIST_LIMIT) {
        Ok(Some(candidates)) => {
            if candidates.is_empty() {
                app.status = "resume failed: no historical sessions found".to_string();
                return Ok(true);
            }
            app.open_resume_picker(candidates);
            app.status = "select session and press Enter to resume".to_string();
            Ok(true)
        }
        Ok(None) => {
            let result = backend.resume_session(None);
            Ok(apply_resume_result(backend, result, app))
        }
        Err(error) => {
            app.push_lines([format!("[resume_error] {error}")]);
            app.status = format!("resume failed: {error}");
            Ok(true)
        }
    }
}

pub(super) fn handle_resume_picker_key_event<B: TuiBackend>(
    key: KeyEvent,
    backend: &mut B,
    app: &mut TuiApp,
) -> bool {
    let Some(picker) = app.resume_picker.as_mut() else {
        return false;
    };

    match key.code {
        KeyCode::Esc => {
            app.close_resume_picker();
            app.status = "resume cancelled".to_string();
            true
        }
        KeyCode::Up => {
            picker.selected = picker.selected.saturating_sub(1);
            true
        }
        KeyCode::Down => {
            let last_index = picker.candidates.len().saturating_sub(1);
            picker.selected = (picker.selected + 1).min(last_index);
            true
        }
        KeyCode::Enter => {
            let selected = picker
                .candidates
                .get(picker.selected)
                .map(|candidate| candidate.session_ref.clone());
            app.close_resume_picker();
            match selected {
                Some(session_ref) => {
                    let result = backend.resume_session(Some(session_ref.as_str()));
                    apply_resume_result(backend, result, app)
                }
                None => {
                    app.status = "resume failed: selection unavailable".to_string();
                    true
                }
            }
        }
        _ => true,
    }
}

fn apply_resume_result<B: TuiBackend>(
    backend: &B,
    result: Result<Option<String>, String>,
    app: &mut TuiApp,
) -> bool {
    match result {
        Ok(Some(status)) => {
            app.status = status;
            if let Some(messages) = backend.session_messages() {
                app.replace_transcript_with_messages(&messages);
            }
            true
        }
        Ok(None) => {
            app.status = "resume is not supported by this backend".to_string();
            true
        }
        Err(error) => {
            app.push_lines([format!("[resume_error] {error}")]);
            app.status = format!("resume failed: {error}");
            true
        }
    }
}

fn is_resume_cancel_command(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "q" | "quit" | "cancel"
    )
}

fn parse_resume_selection_index(value: &str) -> Option<usize> {
    value.trim().parse::<usize>().ok()
}

fn resolve_resume_target_by_index<B: TuiBackend>(
    backend: &mut B,
    index: usize,
) -> Result<Option<String>, String> {
    let Some(candidates) = backend.recent_resumable_sessions(RESUME_LIST_LIMIT)? else {
        return Ok(None);
    };
    if candidates.is_empty() {
        return Err("no historical sessions found".to_string());
    }
    if index == 0 {
        return Ok(Some(candidates[0].session_ref.clone()));
    }
    if index > candidates.len() {
        return Err(format!(
            "selection out of range: {index}, expected 0..{}",
            candidates.len()
        ));
    }
    Ok(Some(candidates[index - 1].session_ref.clone()))
}
