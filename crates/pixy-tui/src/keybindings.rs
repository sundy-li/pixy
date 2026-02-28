use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KeyBinding {
    pub code: KeyCode,
    pub modifiers: KeyModifiers,
}

impl KeyBinding {
    pub(crate) fn matches(self, key: KeyEvent) -> bool {
        if key.code != self.code {
            return false;
        }

        let expected = normalize_modifiers(self.modifiers);
        let actual = normalize_modifiers(key.modifiers);
        if actual == expected {
            return true;
        }

        // Some terminals report Alt+arrows as Alt+Shift+arrows.
        if !expected.contains(KeyModifiers::SHIFT)
            && is_navigation_key(self.code)
            && actual == (expected | KeyModifiers::SHIFT)
        {
            return true;
        }

        false
    }
}

fn normalize_modifiers(modifiers: KeyModifiers) -> KeyModifiers {
    let mut normalized = KeyModifiers::NONE;

    if modifiers.contains(KeyModifiers::SHIFT) {
        normalized |= KeyModifiers::SHIFT;
    }
    if modifiers.contains(KeyModifiers::CONTROL) {
        normalized |= KeyModifiers::CONTROL;
    }
    if modifiers.intersects(KeyModifiers::ALT | KeyModifiers::META) {
        normalized |= KeyModifiers::ALT;
    }
    if modifiers.intersects(KeyModifiers::SUPER | KeyModifiers::HYPER) {
        normalized |= KeyModifiers::SUPER;
    }

    normalized
}

fn is_navigation_key(code: KeyCode) -> bool {
    matches!(
        code,
        KeyCode::Up
            | KeyCode::Down
            | KeyCode::Left
            | KeyCode::Right
            | KeyCode::Home
            | KeyCode::End
            | KeyCode::PageUp
            | KeyCode::PageDown
    )
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
    pub dequeue: Vec<KeyBinding>,
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
            dequeue: vec![KeyBinding {
                code: KeyCode::Up,
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
                code: KeyCode::Char('n'),
                modifiers: KeyModifiers::CONTROL,
            }],
            cycle_model_backward: vec![KeyBinding {
                code: KeyCode::Char('n'),
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
