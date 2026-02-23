use std::sync::OnceLock;

use ratatui::style::{Color, Modifier, Style};
use serde::Deserialize;

use crate::transcript::TranscriptLineKind;

const DARK_THEME_JSON: &str = include_str!("../themes/dark.json");
const LIGHT_THEME_JSON: &str = include_str!("../themes/light.json");
const DEFAULT_INPUT_PROMPT: &str = "› ";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TuiTheme {
    Dark,
    Light,
}

impl TuiTheme {
    pub fn from_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "dark" => Some(Self::Dark),
            "light" => Some(Self::Light),
            _ => None,
        }
    }

    fn theme_name(self) -> &'static str {
        match self {
            Self::Dark => "dark",
            Self::Light => "light",
        }
    }

    fn palette(self) -> &'static ThemePalette {
        static DARK: OnceLock<ThemePalette> = OnceLock::new();
        static LIGHT: OnceLock<ThemePalette> = OnceLock::new();

        match self {
            Self::Dark => DARK.get_or_init(|| {
                ThemePalette::from_json(self.theme_name(), DARK_THEME_JSON)
                    .unwrap_or_else(|error| panic!("load built-in dark theme failed: {error}"))
            }),
            Self::Light => LIGHT.get_or_init(|| {
                ThemePalette::from_json(self.theme_name(), LIGHT_THEME_JSON)
                    .unwrap_or_else(|error| panic!("load built-in light theme failed: {error}"))
            }),
        }
    }

    pub(crate) fn transcript_style(self) -> Style {
        let palette = self.palette();
        Style::default()
            .fg(palette.colors.transcript_fg)
            .bg(palette.colors.transcript_bg)
    }

    pub(crate) fn input_style(self) -> Style {
        self.transcript_style()
    }

    pub(crate) fn input_prompt(self) -> &'static str {
        self.palette().input_prompt.as_str()
    }

    pub(crate) fn input_border_style(self) -> Style {
        let palette = self.palette();
        Style::default().fg(palette.colors.input_border)
    }

    pub(crate) fn footer_style(self) -> Style {
        let palette = self.palette();
        Style::default()
            .fg(palette.colors.footer_fg)
            .bg(palette.colors.footer_bg)
            .add_modifier(Modifier::DIM)
    }

    pub(crate) fn help_style(self) -> Style {
        self.transcript_style()
    }

    pub(crate) fn help_border_style(self) -> Style {
        let palette = self.palette();
        if let Some(color) = palette.colors.help_border {
            Style::default().fg(color)
        } else {
            Style::default()
        }
    }

    pub(crate) fn line_style(self, kind: TranscriptLineKind) -> Style {
        let palette = self.palette();
        match kind {
            TranscriptLineKind::Normal => Style::default(),
            TranscriptLineKind::UserInput => Style::default()
                .fg(palette.colors.transcript_fg)
                .bg(palette.colors.input_block_bg),
            TranscriptLineKind::Thinking => Style::default().fg(palette.colors.thinking_fg),
            TranscriptLineKind::Tool => Style::default().fg(palette.colors.tool_fg),
            TranscriptLineKind::Working => Style::default()
                .fg(palette.colors.working_fg)
                .bg(palette.colors.working_bg)
                .add_modifier(Modifier::BOLD),
        }
    }

    pub(crate) fn tool_diff_added(self) -> Color {
        self.palette().colors.tool_diff_added
    }

    pub(crate) fn tool_diff_removed(self) -> Color {
        self.palette().colors.tool_diff_removed
    }

    pub(crate) fn file_path_style(self, base: Style) -> Style {
        base.fg(self.palette().colors.file_path_fg)
            .add_modifier(Modifier::BOLD)
    }

    pub(crate) fn key_token_style(self, base: Style) -> Style {
        base.fg(self.palette().colors.key_token_fg)
            .add_modifier(Modifier::BOLD)
    }
}

impl Default for TuiTheme {
    fn default() -> Self {
        Self::Dark
    }
}

#[derive(Clone, Copy, Debug)]
struct ThemeColors {
    transcript_fg: Color,
    transcript_bg: Color,
    input_block_bg: Color,
    input_border: Color,
    footer_fg: Color,
    footer_bg: Color,
    help_border: Option<Color>,
    thinking_fg: Color,
    tool_fg: Color,
    working_fg: Color,
    working_bg: Color,
    tool_diff_added: Color,
    tool_diff_removed: Color,
    file_path_fg: Color,
    key_token_fg: Color,
}

#[derive(Clone, Debug)]
struct ThemePalette {
    colors: ThemeColors,
    input_prompt: String,
}

impl ThemePalette {
    fn from_json(expected_name: &str, raw_json: &str) -> Result<Self, String> {
        let parsed: ThemeFile = serde_json::from_str(raw_json)
            .map_err(|error| format!("invalid theme json: {error}"))?;
        let ThemeFile {
            name,
            colors,
            input_prompt,
        } = parsed;

        if name.trim().to_ascii_lowercase() != expected_name {
            return Err(format!(
                "theme name mismatch, expected '{expected_name}' got '{}'",
                name
            ));
        }

        Ok(Self {
            colors: ThemeColors {
                transcript_fg: parse_color(&colors.transcript_fg)
                    .map_err(|error| format!("invalid transcriptFg: {error}"))?,
                transcript_bg: parse_color(&colors.transcript_bg)
                    .map_err(|error| format!("invalid transcriptBg: {error}"))?,
                input_block_bg: parse_color(&colors.input_block_bg)
                    .map_err(|error| format!("invalid inputBlockBg: {error}"))?,
                input_border: parse_color(&colors.input_border)
                    .map_err(|error| format!("invalid inputBorder: {error}"))?,
                footer_fg: parse_color(&colors.footer_fg)
                    .map_err(|error| format!("invalid footerFg: {error}"))?,
                footer_bg: parse_color(&colors.footer_bg)
                    .map_err(|error| format!("invalid footerBg: {error}"))?,
                help_border: colors
                    .help_border
                    .map(|value| {
                        parse_color(value.as_str())
                            .map_err(|error| format!("invalid helpBorder: {error}"))
                    })
                    .transpose()?,
                thinking_fg: parse_color(&colors.thinking_fg)
                    .map_err(|error| format!("invalid thinkingFg: {error}"))?,
                tool_fg: parse_color(&colors.tool_fg)
                    .map_err(|error| format!("invalid toolFg: {error}"))?,
                working_fg: parse_color(&colors.working_fg)
                    .map_err(|error| format!("invalid workingFg: {error}"))?,
                working_bg: parse_color(&colors.working_bg)
                    .map_err(|error| format!("invalid workingBg: {error}"))?,
                tool_diff_added: parse_color(&colors.tool_diff_added)
                    .map_err(|error| format!("invalid toolDiffAdded: {error}"))?,
                tool_diff_removed: parse_color(&colors.tool_diff_removed)
                    .map_err(|error| format!("invalid toolDiffRemoved: {error}"))?,
                file_path_fg: parse_color(&colors.file_path_fg)
                    .map_err(|error| format!("invalid filePathFg: {error}"))?,
                key_token_fg: parse_color(&colors.key_token_fg)
                    .map_err(|error| format!("invalid keyTokenFg: {error}"))?,
            },
            input_prompt,
        })
    }
}

#[derive(Debug, Deserialize)]
struct ThemeFile {
    name: String,
    #[serde(
        default = "default_input_prompt",
        rename = "inputPrompt",
        alias = "input_prompt"
    )]
    input_prompt: String,
    colors: ThemeFileColors,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ThemeFileColors {
    transcript_fg: String,
    transcript_bg: String,
    #[serde(alias = "input_block_bg")]
    input_block_bg: String,
    input_border: String,
    footer_fg: String,
    footer_bg: String,
    help_border: Option<String>,
    thinking_fg: String,
    tool_fg: String,
    working_fg: String,
    working_bg: String,
    tool_diff_added: String,
    tool_diff_removed: String,
    file_path_fg: String,
    key_token_fg: String,
}

fn default_input_prompt() -> String {
    DEFAULT_INPUT_PROMPT.to_string()
}

fn parse_color(raw: &str) -> Result<Color, String> {
    let normalized = raw.trim();
    if normalized.is_empty() {
        return Err("empty color value".to_string());
    }

    if let Some(hex) = normalized.strip_prefix('#') {
        return parse_hex_color(hex);
    }

    match normalized.to_ascii_lowercase().as_str() {
        "black" => Ok(Color::Black),
        "white" => Ok(Color::White),
        "green" => Ok(Color::Green),
        "darkgray" | "dark_gray" => Ok(Color::DarkGray),
        "gray" => Ok(Color::Gray),
        "red" => Ok(Color::Red),
        "yellow" => Ok(Color::Yellow),
        "cyan" => Ok(Color::Cyan),
        "lightyellow" | "light_yellow" => Ok(Color::LightYellow),
        other => Err(format!("unsupported named color '{other}'")),
    }
}

fn parse_hex_color(hex: &str) -> Result<Color, String> {
    if hex.len() != 6 {
        return Err(format!("expected 6 hex digits, got '{}': {hex}", hex.len()));
    }

    let red = u8::from_str_radix(&hex[0..2], 16)
        .map_err(|error| format!("invalid red channel '{}': {error}", &hex[0..2]))?;
    let green = u8::from_str_radix(&hex[2..4], 16)
        .map_err(|error| format!("invalid green channel '{}': {error}", &hex[2..4]))?;
    let blue = u8::from_str_radix(&hex[4..6], 16)
        .map_err(|error| format!("invalid blue channel '{}': {error}", &hex[4..6]))?;
    Ok(Color::Rgb(red, green, blue))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_color_supports_named_and_hex_values() {
        assert_eq!(parse_color("gray"), Ok(Color::Gray));
        assert_eq!(parse_color("#547da7"), Ok(Color::Rgb(84, 125, 167)));
    }

    #[test]
    fn built_in_themes_default_input_prompt_is_supported() {
        assert_eq!(TuiTheme::Dark.input_prompt(), "› ");
        assert_eq!(TuiTheme::Light.input_prompt(), "› ");
    }

    #[test]
    fn parse_theme_file_uses_configured_input_prompt() {
        let raw = r##"
        {
          "name": "dark",
          "inputPrompt": "pi> ",
          "colors": {
            "transcriptFg": "white",
            "transcriptBg": "black",
            "inputBlockBg": "#343541",
            "inputBorder": "green",
            "footerFg": "darkGray",
            "footerBg": "black",
            "helpBorder": null,
            "thinkingFg": "darkGray",
            "toolFg": "gray",
            "workingFg": "black",
            "workingBg": "white",
            "toolDiffAdded": "yellow",
            "toolDiffRemoved": "red",
            "filePathFg": "cyan",
            "keyTokenFg": "lightYellow"
          }
        }
        "##;
        let palette = ThemePalette::from_json("dark", raw).expect("theme should parse");
        assert_eq!(palette.input_prompt, "pi> ");
    }

    #[test]
    fn parse_theme_file_defaults_input_prompt_when_not_present() {
        let raw = r##"
        {
          "name": "dark",
          "colors": {
            "transcriptFg": "white",
            "transcriptBg": "black",
            "inputBlockBg": "#343541",
            "inputBorder": "green",
            "footerFg": "darkGray",
            "footerBg": "black",
            "helpBorder": null,
            "thinkingFg": "darkGray",
            "toolFg": "gray",
            "workingFg": "black",
            "workingBg": "white",
            "toolDiffAdded": "yellow",
            "toolDiffRemoved": "red",
            "filePathFg": "cyan",
            "keyTokenFg": "lightYellow"
          }
        }
        "##;
        let palette = ThemePalette::from_json("dark", raw).expect("theme should parse");
        assert_eq!(palette.input_prompt, "› ");
    }

    #[test]
    fn parse_theme_file_accepts_legacy_input_block_bg_alias() {
        let raw = r##"
        {
          "name": "dark",
          "colors": {
            "transcriptFg": "white",
            "transcriptBg": "black",
            "input_block_bg": "#343541",
            "inputBorder": "green",
            "footerFg": "darkGray",
            "footerBg": "black",
            "helpBorder": null,
            "thinkingFg": "darkGray",
            "toolFg": "gray",
            "workingFg": "black",
            "workingBg": "white",
            "toolDiffAdded": "yellow",
            "toolDiffRemoved": "red",
            "filePathFg": "cyan",
            "keyTokenFg": "lightYellow"
          }
        }
        "##;
        let palette = ThemePalette::from_json("dark", raw).expect("theme should parse");
        assert_eq!(palette.colors.input_block_bg, Color::Rgb(52, 53, 65));
    }
}
