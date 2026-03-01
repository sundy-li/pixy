use std::sync::OnceLock;

use ratatui::style::{Color, Modifier, Style};
use serde::Deserialize;

use crate::transcript::TranscriptLineKind;

const DARK_THEME_JSON: &str = include_str!("../themes/dark.json");
const LIGHT_THEME_JSON: &str = include_str!("../themes/light.json");
const DEFAULT_INPUT_PROMPT: &str = "> ";
const DEFAULT_OUTPUT_PROMPT: &str = "⛬  ";

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
        let palette = self.palette();
        Style::default()
            .fg(palette.colors.user_input_fg)
            .bg(palette.colors.input_block_bg)
    }

    pub(crate) fn input_placeholder_style(self) -> Style {
        Style::default().fg(self.palette().colors.input_placeholder_fg)
    }

    pub(crate) fn input_prompt(self) -> &'static str {
        self.palette().input_prompt.as_str()
    }

    pub(crate) fn output_prompt(self) -> &'static str {
        self.palette().output_prompt.as_str()
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
    }

    pub(crate) fn status_primary_left_style(self) -> Style {
        Style::default().fg(self.palette().colors.status_primary_left_fg)
    }

    pub(crate) fn status_primary_right_style(self) -> Style {
        Style::default().fg(self.palette().colors.status_primary_right_fg)
    }

    pub(crate) fn status_hint_style(self) -> Style {
        Style::default().fg(self.palette().colors.status_hint_fg)
    }

    pub(crate) fn status_help_right_style(self) -> Style {
        Style::default().fg(self.palette().colors.status_help_right_fg)
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
            TranscriptLineKind::Assistant => Style::default(),
            TranscriptLineKind::Overlay => Style::default(),
            TranscriptLineKind::Code => self.code_block_style(),
            TranscriptLineKind::UserInput => Style::default()
                .fg(palette.colors.user_input_fg)
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

    pub(crate) fn skills_header_style(self, base: Style) -> Style {
        base.fg(self.palette().colors.skills_header_fg)
    }

    pub(crate) fn skills_group_style(self, base: Style) -> Style {
        base.fg(self.palette().colors.skills_group_fg)
    }

    pub(crate) fn code_block_style(self) -> Style {
        let palette = self.palette();
        Style::default()
            .fg(palette.colors.code_block_fg)
            .bg(palette.colors.code_block_bg)
    }

    pub(crate) fn code_keyword_style(self, base: Style) -> Style {
        base.fg(self.palette().colors.code_keyword_fg)
            .add_modifier(Modifier::BOLD)
    }

    pub(crate) fn code_string_style(self, base: Style) -> Style {
        base.fg(self.palette().colors.code_string_fg)
    }

    pub(crate) fn code_comment_style(self, base: Style) -> Style {
        base.fg(self.palette().colors.code_comment_fg)
            .add_modifier(Modifier::ITALIC)
    }

    pub(crate) fn code_number_style(self, base: Style) -> Style {
        base.fg(self.palette().colors.code_number_fg)
    }

    pub(crate) fn overlay_logo_style(self, base: Style) -> Style {
        base.fg(self.palette().colors.overlay_logo_fg)
            .add_modifier(Modifier::BOLD)
    }

    pub(crate) fn overlay_version_style(self, base: Style) -> Style {
        base.fg(self.palette().colors.overlay_version_fg)
    }

    pub(crate) fn selection_colors(self) -> Option<(Color, Color)> {
        let palette = self.palette();
        match (palette.colors.selection_bg, palette.colors.selection_fg) {
            (Some(bg), Some(fg)) => Some((bg, fg)),
            _ => None,
        }
    }

    pub(crate) fn working_marquee_base_style(self) -> Style {
        let palette = self.palette();
        Style::default()
            .fg(palette.colors.working_fg)
            .bg(palette.colors.working_bg)
    }

    pub(crate) fn working_marquee_highlight_style(self) -> Style {
        let palette = self.palette();
        Style::default()
            .fg(palette.colors.working_highlight_fg)
            .bg(palette.colors.working_bg)
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
    user_input_fg: Color,
    input_placeholder_fg: Color,
    input_border: Color,
    footer_fg: Color,
    footer_bg: Color,
    status_primary_left_fg: Color,
    status_primary_right_fg: Color,
    status_hint_fg: Color,
    status_help_right_fg: Color,
    help_border: Option<Color>,
    thinking_fg: Color,
    tool_fg: Color,
    working_fg: Color,
    working_bg: Color,
    working_highlight_fg: Color,
    tool_diff_added: Color,
    tool_diff_removed: Color,
    file_path_fg: Color,
    key_token_fg: Color,
    skills_header_fg: Color,
    skills_group_fg: Color,
    code_block_fg: Color,
    code_block_bg: Color,
    code_keyword_fg: Color,
    code_string_fg: Color,
    code_comment_fg: Color,
    code_number_fg: Color,
    overlay_logo_fg: Color,
    overlay_version_fg: Color,
    selection_bg: Option<Color>,
    selection_fg: Option<Color>,
}

#[derive(Clone, Debug)]
struct ThemePalette {
    colors: ThemeColors,
    input_prompt: String,
    output_prompt: String,
}

impl ThemePalette {
    fn from_json(expected_name: &str, raw_json: &str) -> Result<Self, String> {
        let parsed: ThemeFile = serde_json::from_str(raw_json)
            .map_err(|error| format!("invalid theme json: {error}"))?;
        let ThemeFile {
            name,
            colors,
            input_prompt,
            output_prompt,
        } = parsed;

        if name.trim().to_ascii_lowercase() != expected_name {
            return Err(format!(
                "theme name mismatch, expected '{expected_name}' got '{}'",
                name
            ));
        }

        let selection_bg = colors
            .selection_bg
            .as_ref()
            .map(|value| {
                parse_color(value.as_str()).map_err(|error| format!("invalid selectionBg: {error}"))
            })
            .transpose()?;
        let selection_fg = colors
            .selection_fg
            .as_ref()
            .map(|value| {
                parse_color(value.as_str()).map_err(|error| format!("invalid selectionFg: {error}"))
            })
            .transpose()?;
        if selection_bg.is_some() ^ selection_fg.is_some() {
            return Err("selectionBg and selectionFg must be configured together".to_string());
        }

        let transcript_fg = parse_color(&colors.transcript_fg)
            .map_err(|error| format!("invalid transcriptFg: {error}"))?;
        let transcript_bg = parse_color(&colors.transcript_bg)
            .map_err(|error| format!("invalid transcriptBg: {error}"))?;
        let input_block_bg = parse_color(&colors.input_block_bg)
            .map_err(|error| format!("invalid inputBlockBg: {error}"))?;
        let user_input_fg = colors
            .user_input_fg
            .as_ref()
            .map(|value| {
                parse_color(value.as_str()).map_err(|error| format!("invalid userInputFg: {error}"))
            })
            .transpose()?
            .unwrap_or(transcript_fg);
        let input_placeholder_fg = colors
            .input_placeholder_fg
            .as_ref()
            .map(|value| {
                parse_color(value.as_str())
                    .map_err(|error| format!("invalid inputPlaceholderFg: {error}"))
            })
            .transpose()?;
        let input_border = parse_color(&colors.input_border)
            .map_err(|error| format!("invalid inputBorder: {error}"))?;
        let footer_fg =
            parse_color(&colors.footer_fg).map_err(|error| format!("invalid footerFg: {error}"))?;
        let footer_bg =
            parse_color(&colors.footer_bg).map_err(|error| format!("invalid footerBg: {error}"))?;
        let status_primary_left_fg = colors
            .status_primary_left_fg
            .as_ref()
            .map(|value| {
                parse_color(value.as_str())
                    .map_err(|error| format!("invalid statusPrimaryLeftFg: {error}"))
            })
            .transpose()?
            .unwrap_or(footer_fg);
        let status_primary_right_fg = colors
            .status_primary_right_fg
            .as_ref()
            .map(|value| {
                parse_color(value.as_str())
                    .map_err(|error| format!("invalid statusPrimaryRightFg: {error}"))
            })
            .transpose()?
            .unwrap_or(transcript_fg);
        let status_hint_fg = colors
            .status_hint_fg
            .as_ref()
            .map(|value| {
                parse_color(value.as_str())
                    .map_err(|error| format!("invalid statusHintFg: {error}"))
            })
            .transpose()?;
        let status_help_right_fg = colors
            .status_help_right_fg
            .as_ref()
            .map(|value| {
                parse_color(value.as_str())
                    .map_err(|error| format!("invalid statusHelpRightFg: {error}"))
            })
            .transpose()?;
        let help_border = colors
            .help_border
            .map(|value| {
                parse_color(value.as_str()).map_err(|error| format!("invalid helpBorder: {error}"))
            })
            .transpose()?;
        let thinking_fg = parse_color(&colors.thinking_fg)
            .map_err(|error| format!("invalid thinkingFg: {error}"))?;
        let input_placeholder_fg = input_placeholder_fg.unwrap_or(thinking_fg);
        let status_hint_fg = status_hint_fg.unwrap_or(thinking_fg);
        let status_help_right_fg = status_help_right_fg.unwrap_or(status_primary_left_fg);
        let tool_fg =
            parse_color(&colors.tool_fg).map_err(|error| format!("invalid toolFg: {error}"))?;
        let working_fg = parse_color(&colors.working_fg)
            .map_err(|error| format!("invalid workingFg: {error}"))?;
        let working_bg = parse_color(&colors.working_bg)
            .map_err(|error| format!("invalid workingBg: {error}"))?;
        let working_highlight_fg = colors
            .working_highlight_fg
            .as_ref()
            .map(|value| {
                parse_color(value.as_str())
                    .map_err(|error| format!("invalid workingHighlightFg: {error}"))
            })
            .transpose()?
            .unwrap_or(transcript_fg);
        let tool_diff_added = parse_color(&colors.tool_diff_added)
            .map_err(|error| format!("invalid toolDiffAdded: {error}"))?;
        let tool_diff_removed = parse_color(&colors.tool_diff_removed)
            .map_err(|error| format!("invalid toolDiffRemoved: {error}"))?;
        let file_path_fg = parse_color(&colors.file_path_fg)
            .map_err(|error| format!("invalid filePathFg: {error}"))?;
        let key_token_fg = parse_color(&colors.key_token_fg)
            .map_err(|error| format!("invalid keyTokenFg: {error}"))?;
        let skills_header_fg = colors
            .skills_header_fg
            .as_ref()
            .map(|value| {
                parse_color(value.as_str())
                    .map_err(|error| format!("invalid skillsHeaderFg: {error}"))
            })
            .transpose()?
            .unwrap_or(key_token_fg);
        let skills_group_fg = colors
            .skills_group_fg
            .as_ref()
            .map(|value| {
                parse_color(value.as_str())
                    .map_err(|error| format!("invalid skillsGroupFg: {error}"))
            })
            .transpose()?
            .unwrap_or(key_token_fg);
        let code_block_fg = colors
            .code_block_fg
            .as_ref()
            .map(|value| {
                parse_color(value.as_str()).map_err(|error| format!("invalid codeBlockFg: {error}"))
            })
            .transpose()?
            .unwrap_or(transcript_fg);
        let code_block_bg = colors
            .code_block_bg
            .as_ref()
            .map(|value| {
                parse_color(value.as_str()).map_err(|error| format!("invalid codeBlockBg: {error}"))
            })
            .transpose()?
            .unwrap_or(input_block_bg);
        let code_keyword_fg = colors
            .code_keyword_fg
            .as_ref()
            .map(|value| {
                parse_color(value.as_str())
                    .map_err(|error| format!("invalid codeKeywordFg: {error}"))
            })
            .transpose()?
            .unwrap_or(file_path_fg);
        let code_string_fg = colors
            .code_string_fg
            .as_ref()
            .map(|value| {
                parse_color(value.as_str())
                    .map_err(|error| format!("invalid codeStringFg: {error}"))
            })
            .transpose()?
            .unwrap_or(skills_group_fg);
        let code_comment_fg = colors
            .code_comment_fg
            .as_ref()
            .map(|value| {
                parse_color(value.as_str())
                    .map_err(|error| format!("invalid codeCommentFg: {error}"))
            })
            .transpose()?
            .unwrap_or(file_path_fg);
        let code_number_fg = colors
            .code_number_fg
            .as_ref()
            .map(|value| {
                parse_color(value.as_str())
                    .map_err(|error| format!("invalid codeNumberFg: {error}"))
            })
            .transpose()?
            .unwrap_or(skills_header_fg);
        let overlay_logo_fg = colors
            .overlay_logo_fg
            .as_ref()
            .map(|value| {
                parse_color(value.as_str())
                    .map_err(|error| format!("invalid overlayLogoFg: {error}"))
            })
            .transpose()?
            .unwrap_or(transcript_fg);
        let overlay_version_fg = colors
            .overlay_version_fg
            .as_ref()
            .map(|value| {
                parse_color(value.as_str())
                    .map_err(|error| format!("invalid overlayVersionFg: {error}"))
            })
            .transpose()?
            .unwrap_or(thinking_fg);

        Ok(Self {
            colors: ThemeColors {
                transcript_fg,
                transcript_bg,
                input_block_bg,
                user_input_fg,
                input_placeholder_fg,
                input_border,
                footer_fg,
                footer_bg,
                status_primary_left_fg,
                status_primary_right_fg,
                status_hint_fg,
                status_help_right_fg,
                help_border,
                thinking_fg,
                tool_fg,
                working_fg,
                working_bg,
                working_highlight_fg,
                tool_diff_added,
                tool_diff_removed,
                file_path_fg,
                key_token_fg,
                skills_header_fg,
                skills_group_fg,
                code_block_fg,
                code_block_bg,
                code_keyword_fg,
                code_string_fg,
                code_comment_fg,
                code_number_fg,
                overlay_logo_fg,
                overlay_version_fg,
                selection_bg,
                selection_fg,
            },
            input_prompt,
            output_prompt,
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
    #[serde(
        default = "default_output_prompt",
        rename = "outputPrompt",
        alias = "output_prompt"
    )]
    output_prompt: String,
    colors: ThemeFileColors,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ThemeFileColors {
    transcript_fg: String,
    transcript_bg: String,
    #[serde(alias = "input_block_bg")]
    input_block_bg: String,
    user_input_fg: Option<String>,
    input_placeholder_fg: Option<String>,
    input_border: String,
    footer_fg: String,
    footer_bg: String,
    status_primary_left_fg: Option<String>,
    status_primary_right_fg: Option<String>,
    status_hint_fg: Option<String>,
    status_help_right_fg: Option<String>,
    help_border: Option<String>,
    thinking_fg: String,
    tool_fg: String,
    working_fg: String,
    working_bg: String,
    working_highlight_fg: Option<String>,
    tool_diff_added: String,
    tool_diff_removed: String,
    file_path_fg: String,
    key_token_fg: String,
    skills_header_fg: Option<String>,
    skills_group_fg: Option<String>,
    code_block_fg: Option<String>,
    code_block_bg: Option<String>,
    code_keyword_fg: Option<String>,
    code_string_fg: Option<String>,
    code_comment_fg: Option<String>,
    code_number_fg: Option<String>,
    overlay_logo_fg: Option<String>,
    overlay_version_fg: Option<String>,
    #[serde(alias = "selection_bg")]
    selection_bg: Option<String>,
    #[serde(alias = "selection_fg")]
    selection_fg: Option<String>,
}

fn default_input_prompt() -> String {
    DEFAULT_INPUT_PROMPT.to_string()
}

fn default_output_prompt() -> String {
    DEFAULT_OUTPUT_PROMPT.to_string()
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
        assert!(!TuiTheme::Dark.input_prompt().trim().is_empty());
        assert!(!TuiTheme::Light.input_prompt().trim().is_empty());
    }

    #[test]
    fn built_in_themes_default_output_prompt_is_supported() {
        assert_eq!(TuiTheme::Dark.output_prompt(), "⛬  ");
        assert_eq!(TuiTheme::Light.output_prompt(), "⛬  ");
    }

    #[test]
    fn built_in_dark_theme_includes_selection_colors() {
        assert_eq!(
            TuiTheme::Dark.selection_colors(),
            Some((Color::Rgb(59, 66, 97), Color::Rgb(192, 202, 245)))
        );
        assert_eq!(TuiTheme::Light.selection_colors(), None);
    }

    #[test]
    fn built_in_dark_theme_matches_reference_palette_basics() {
        assert_eq!(
            TuiTheme::Dark.transcript_style(),
            Style::default().fg(Color::White).bg(Color::Rgb(26, 27, 38))
        );
        assert_eq!(
            TuiTheme::Dark.input_border_style(),
            Style::default().fg(Color::Rgb(59, 66, 97))
        );
        assert_eq!(
            TuiTheme::Dark.footer_style(),
            Style::default()
                .fg(Color::Rgb(122, 162, 247))
                .bg(Color::Rgb(26, 27, 38))
        );
        assert_eq!(
            TuiTheme::Dark.line_style(TranscriptLineKind::Thinking),
            Style::default().fg(Color::Rgb(86, 95, 137))
        );
        assert_eq!(
            TuiTheme::Dark.line_style(TranscriptLineKind::Tool),
            Style::default().fg(Color::Rgb(169, 177, 214))
        );
        assert_eq!(
            TuiTheme::Dark.input_style(),
            Style::default()
                .fg(Color::Rgb(227, 153, 42))
                .bg(Color::Rgb(26, 27, 38))
        );
        assert_eq!(
            TuiTheme::Dark.line_style(TranscriptLineKind::UserInput),
            Style::default()
                .fg(Color::Rgb(227, 153, 42))
                .bg(Color::Rgb(26, 27, 38))
        );
    }

    #[test]
    fn built_in_dark_theme_status_styles_match_reference_palette() {
        assert_eq!(
            TuiTheme::Dark.status_primary_left_style(),
            Style::default().fg(Color::Rgb(224, 175, 104))
        );
        assert_eq!(
            TuiTheme::Dark.status_primary_right_style(),
            Style::default().fg(Color::Rgb(192, 202, 245))
        );
        assert_eq!(
            TuiTheme::Dark.status_hint_style(),
            Style::default().fg(Color::Rgb(86, 95, 137))
        );
        assert_eq!(
            TuiTheme::Dark.status_help_right_style(),
            Style::default().fg(Color::Rgb(187, 154, 247))
        );
        assert_eq!(
            TuiTheme::Dark.input_placeholder_style(),
            Style::default().fg(Color::Rgb(86, 95, 137))
        );
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
    fn parse_theme_file_uses_configured_output_prompt() {
        let raw = r##"
        {
          "name": "dark",
          "outputPrompt": "bot> ",
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
        assert_eq!(palette.output_prompt, "bot> ");
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
        assert_eq!(palette.input_prompt, "> ");
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

    #[test]
    fn parse_theme_file_requires_selection_bg_and_fg_together() {
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
            "keyTokenFg": "lightYellow",
            "selectionBg": "white"
          }
        }
        "##;

        let error = ThemePalette::from_json("dark", raw).expect_err("theme should fail");
        assert!(error.contains("selectionBg and selectionFg"));
    }

    #[test]
    fn parse_theme_file_defaults_working_highlight_to_transcript_fg() {
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
            "workingFg": "#949699",
            "workingBg": "#282c34",
            "toolDiffAdded": "yellow",
            "toolDiffRemoved": "red",
            "filePathFg": "cyan",
            "keyTokenFg": "lightYellow"
          }
        }
        "##;
        let palette = ThemePalette::from_json("dark", raw).expect("theme should parse");
        assert_eq!(palette.colors.working_highlight_fg, Color::White);
    }

    #[test]
    fn parse_theme_file_accepts_configured_working_highlight_fg() {
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
            "workingFg": "#949699",
            "workingBg": "#282c34",
            "workingHighlightFg": "#123456",
            "toolDiffAdded": "yellow",
            "toolDiffRemoved": "red",
            "filePathFg": "cyan",
            "keyTokenFg": "lightYellow"
          }
        }
        "##;
        let palette = ThemePalette::from_json("dark", raw).expect("theme should parse");
        assert_eq!(palette.colors.working_highlight_fg, Color::Rgb(18, 52, 86));
    }
}
