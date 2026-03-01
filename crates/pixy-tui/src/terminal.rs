use std::env;
use std::io::{self, Write};

use crossterm::event::{DisableBracketedPaste, DisableMouseCapture, PopKeyboardEnhancementFlags};
use crossterm::execute;
use crossterm::terminal::disable_raw_mode;
use ratatui::style::Color;

use crate::TuiTheme;

pub(crate) fn apply_selection_osc_colors(theme: TuiTheme) -> bool {
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

pub(crate) fn reset_selection_osc_colors() {
    let capabilities = detect_terminal_capabilities();
    let mut stdout = io::stdout();
    for sequence in selection_osc_reset_sequences(capabilities) {
        let _ = stdout.write_all(sequence.as_bytes());
    }
    let _ = stdout.flush();
}

#[cfg(test)]
pub(crate) fn selection_osc_set_sequence(theme: TuiTheme) -> Option<String> {
    selection_osc_set_sequences(theme, TerminalCapabilities::default())?
        .into_iter()
        .next()
}

pub(crate) fn selection_osc_set_sequences(
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
pub(crate) fn selection_osc_reset_sequence() -> &'static str {
    "\u{1b}]117;\u{7}\u{1b}]119;\u{7}"
}

pub(crate) fn selection_osc_reset_sequences(capabilities: TerminalCapabilities) -> Vec<String> {
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
pub(crate) enum TerminalMultiplexer {
    Tmux,
    Screen,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct TerminalCapabilities {
    pub(crate) multiplexer: Option<TerminalMultiplexer>,
    pub(crate) iterm2: bool,
    pub(crate) kitty: bool,
}

pub(crate) struct TerminalRestore {
    pub(crate) keyboard_enhancement_enabled: bool,
    pub(crate) mouse_capture_enabled: bool,
    pub(crate) bracketed_paste_enabled: bool,
    pub(crate) selection_colors_applied: bool,
    pub(crate) alternate_screen_enabled: bool,
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
        if self.alternate_screen_enabled {
            let _ = execute!(io::stdout(), crossterm::terminal::LeaveAlternateScreen);
        }
    }
}
