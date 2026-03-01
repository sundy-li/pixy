use pixy_ai::{AssistantContentBlock, Message, StopReason, ToolResultContentBlock};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::TuiTheme;
use crate::keybindings::parse_key_id;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum TranscriptLineKind {
    Normal,
    Assistant,
    Overlay,
    Code,
    UserInput,
    Thinking,
    Tool,
    Working,
}

const TOOL_COMPACTION_HEAD_LINES: usize = 2;
const TOOL_COMPACTION_TAIL_LINES: usize = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TranscriptLine {
    pub(crate) text: String,
    pub(crate) kind: TranscriptLineKind,
    code_language: Option<String>,
    markdown_line_style: Option<MarkdownLineStyle>,
    working_marquee: Option<WorkingMarquee>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum MarkdownLineStyle {
    Heading,
    Quote,
    Rule,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct WorkingMarquee {
    message_char_start: usize,
    message_char_len: usize,
    highlight_start: usize,
    highlight_len: usize,
}

impl TranscriptLine {
    pub(crate) fn new(text: String, kind: TranscriptLineKind) -> Self {
        Self {
            text,
            kind,
            code_language: None,
            markdown_line_style: None,
            working_marquee: None,
        }
    }

    pub(crate) fn new_code(text: String, language: Option<String>) -> Self {
        Self {
            text,
            kind: TranscriptLineKind::Code,
            code_language: language,
            markdown_line_style: None,
            working_marquee: None,
        }
    }

    fn new_markdown(
        text: String,
        kind: TranscriptLineKind,
        markdown_line_style: MarkdownLineStyle,
    ) -> Self {
        Self {
            text,
            kind,
            code_language: None,
            markdown_line_style: Some(markdown_line_style),
            working_marquee: None,
        }
    }

    pub(crate) fn new_working_with_marquee(
        text: String,
        message_char_start: usize,
        message_char_len: usize,
        highlight_start: usize,
        highlight_len: usize,
    ) -> Self {
        Self {
            text,
            kind: TranscriptLineKind::Working,
            code_language: None,
            markdown_line_style: None,
            working_marquee: Some(WorkingMarquee {
                message_char_start,
                message_char_len,
                highlight_start,
                highlight_len,
            }),
        }
    }

    pub(crate) fn to_line(&self, width: usize, theme: TuiTheme) -> Line<'static> {
        let mut base = theme.line_style(self.kind.clone());
        if let Some(markdown_line_style) = &self.markdown_line_style {
            base = apply_markdown_line_style(base, markdown_line_style);
        }

        let is_tool_diff_removed = matches!(self.kind, TranscriptLineKind::Tool)
            && self.text.trim_start().starts_with('-');
        let is_tool_diff_added = matches!(self.kind, TranscriptLineKind::Tool)
            && self.text.trim_start().starts_with('+');
        if is_tool_diff_removed || is_tool_diff_added {
            let style = if is_tool_diff_removed {
                base.fg(theme.tool_diff_removed())
            } else {
                base.fg(theme.tool_diff_added())
            };
            return line_with_padding(vec![Span::styled(self.text.clone(), style)], width, style);
        }

        if matches!(self.kind, TranscriptLineKind::Tool) {
            if let Some(spans) = tool_diff_stat_spans(self.text.as_str(), base, theme) {
                return line_with_padding(spans, width, base);
            }
        }

        if matches!(self.kind, TranscriptLineKind::Code) {
            let code_base = theme.code_block_style();
            let spans = code_highlighted_spans(
                self.text.as_str(),
                code_base,
                theme,
                self.code_language.as_deref(),
            );
            return line_with_padding(spans, width, code_base);
        }

        if matches!(self.kind, TranscriptLineKind::Working) {
            let working_base = working_base_style(theme);
            let working_highlight = working_highlight_style(theme);
            if let Some(marquee) = &self.working_marquee {
                let spans = working_marquee_spans(
                    self.text.as_str(),
                    marquee,
                    working_base,
                    working_highlight,
                );
                return line_with_padding(spans, width, working_base);
            }
            return line_with_padding(
                vec![Span::styled(self.text.clone(), working_base)],
                width,
                working_base,
            );
        }

        if matches!(self.kind, TranscriptLineKind::Overlay) {
            let text_style = overlay_welcome_style(self.text.as_str(), base, theme);
            return centered_line_with_padding(self.text.as_str(), width, text_style, base);
        }

        let spans = highlighted_spans(
            self.text.as_str(),
            base,
            theme,
            matches!(
                self.kind,
                TranscriptLineKind::Normal | TranscriptLineKind::UserInput
            ),
        );
        line_with_padding(spans, width, base)
    }
}

fn overlay_welcome_style(text: &str, base: Style, theme: TuiTheme) -> Style {
    let trimmed = text.trim();
    if trimmed.contains('█') {
        return theme.overlay_logo_style(base);
    }

    if trimmed.starts_with('v') {
        return theme.overlay_version_style(base);
    }

    base
}

fn apply_markdown_line_style(base: Style, line_style: &MarkdownLineStyle) -> Style {
    match line_style {
        MarkdownLineStyle::Heading => base.add_modifier(Modifier::BOLD),
        MarkdownLineStyle::Quote => base.add_modifier(Modifier::ITALIC),
        MarkdownLineStyle::Rule => base.add_modifier(Modifier::DIM),
    }
}

fn working_base_style(theme: TuiTheme) -> Style {
    theme.working_marquee_base_style()
}

fn working_highlight_style(theme: TuiTheme) -> Style {
    theme.working_marquee_highlight_style()
}

fn working_marquee_spans(
    text: &str,
    marquee: &WorkingMarquee,
    base: Style,
    highlight: Style,
) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let mut chunk = String::new();
    let mut current_is_highlighted = None::<bool>;

    for (idx, ch) in text.chars().enumerate() {
        let is_highlighted = is_marquee_highlighted(idx, marquee);
        match current_is_highlighted {
            None => {
                chunk.push(ch);
                current_is_highlighted = Some(is_highlighted);
            }
            Some(state) if state == is_highlighted => {
                chunk.push(ch);
            }
            Some(state) => {
                let style = if state { highlight } else { base };
                spans.push(Span::styled(std::mem::take(&mut chunk), style));
                chunk.push(ch);
                current_is_highlighted = Some(is_highlighted);
            }
        }
    }

    if let Some(state) = current_is_highlighted {
        let style = if state { highlight } else { base };
        spans.push(Span::styled(chunk, style));
    }

    if spans.is_empty() {
        spans.push(Span::styled(String::new(), base));
    }
    spans
}

fn is_marquee_highlighted(idx: usize, marquee: &WorkingMarquee) -> bool {
    if marquee.message_char_len == 0 || marquee.highlight_len == 0 {
        return false;
    }
    if idx < marquee.message_char_start
        || idx >= marquee.message_char_start + marquee.message_char_len
    {
        return false;
    }

    let local_idx = idx - marquee.message_char_start;
    let start = marquee.highlight_start % marquee.message_char_len;
    let window_len = marquee.highlight_len.min(marquee.message_char_len);
    let end = start + window_len;
    if end <= marquee.message_char_len {
        local_idx >= start && local_idx < end
    } else {
        local_idx >= start || local_idx < (end % marquee.message_char_len)
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

fn centered_line_with_padding(
    text: &str,
    width: usize,
    text_style: Style,
    padding_style: Style,
) -> Line<'static> {
    if width == 0 {
        return Line::from(vec![Span::styled(text.to_string(), text_style)]);
    }
    let text_width = UnicodeWidthStr::width(text);
    if text_width >= width {
        return line_with_padding(
            vec![Span::styled(text.to_string(), text_style)],
            width,
            text_style,
        );
    }

    let left_padding = (width - text_width) / 2;
    let right_padding = width - left_padding - text_width;
    let mut spans = Vec::with_capacity(3);
    if left_padding > 0 {
        spans.push(Span::styled(" ".repeat(left_padding), padding_style));
    }
    spans.push(Span::styled(text.to_string(), text_style));
    if right_padding > 0 {
        spans.push(Span::styled(" ".repeat(right_padding), padding_style));
    }
    Line::from(spans)
}

fn tool_diff_stat_spans(text: &str, base: Style, theme: TuiTheme) -> Option<Vec<Span<'static>>> {
    if !looks_like_tool_diff_stat_line(text) {
        return None;
    }

    let mut spans = Vec::new();
    let mut buffer = String::new();
    let mut current = SegmentKind::Normal;

    for ch in text.chars() {
        let next = if ch == '+' {
            SegmentKind::Added
        } else if ch == '-' {
            SegmentKind::Removed
        } else {
            SegmentKind::Normal
        };

        if buffer.is_empty() {
            current = next;
            buffer.push(ch);
            continue;
        }

        if next == current {
            buffer.push(ch);
            continue;
        }

        spans.push(Span::styled(
            std::mem::take(&mut buffer),
            segment_style(current, base, theme),
        ));
        buffer.push(ch);
        current = next;
    }

    if !buffer.is_empty() {
        spans.push(Span::styled(buffer, segment_style(current, base, theme)));
    }

    Some(spans)
}

fn looks_like_tool_diff_stat_line(text: &str) -> bool {
    let Some((_, right)) = text.split_once('|') else {
        return false;
    };
    let suffix = right.trim();
    suffix.contains('+') || suffix.contains('-')
}

fn segment_style(segment: SegmentKind, base: Style, theme: TuiTheme) -> Style {
    match segment {
        SegmentKind::Normal => base,
        SegmentKind::Added => base.fg(theme.tool_diff_added()),
        SegmentKind::Removed => base.fg(theme.tool_diff_removed()),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SegmentKind {
    Normal,
    Added,
    Removed,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct InlineMarkdownStyle {
    bold: bool,
    italic: bool,
    code: bool,
    strikethrough: bool,
    link: bool,
}

#[derive(Clone, Debug)]
struct StyledFragment {
    text: String,
    is_token: bool,
    style: InlineMarkdownStyle,
}

fn highlighted_spans(
    text: &str,
    base: Style,
    theme: TuiTheme,
    enable_inline_markdown: bool,
) -> Vec<Span<'static>> {
    let mut fragments = Vec::new();
    if enable_inline_markdown {
        for (segment, markdown_style) in parse_inline_markdown_segments(text) {
            let segment_fragments = tokenize_text_fragments(segment.as_str(), markdown_style.code);
            for (text, is_token) in segment_fragments {
                fragments.push(StyledFragment {
                    text,
                    is_token,
                    style: markdown_style,
                });
            }
        }
    } else {
        for (text, is_token) in tokenize_text_fragments(text, false) {
            fragments.push(StyledFragment {
                text,
                is_token,
                style: InlineMarkdownStyle::default(),
            });
        }
    }

    let mut spans = Vec::with_capacity(fragments.len().max(1));
    for (index, fragment) in fragments.iter().enumerate() {
        let previous = index
            .checked_sub(1)
            .and_then(|previous_index| fragments.get(previous_index))
            .map(|fragment| fragment.text.as_str());
        let next = fragments
            .get(index + 1)
            .map(|fragment| fragment.text.as_str());
        let fragment_base = apply_inline_markdown_style(base, fragment.style, theme);
        spans.push(styled_fragment(
            fragment.text.as_str(),
            fragment.is_token,
            previous,
            next,
            fragment_base,
            theme,
        ));
    }

    if spans.is_empty() {
        spans.push(Span::styled(String::new(), base));
    }

    spans
}

fn tokenize_text_fragments(text: &str, disable_token_detection: bool) -> Vec<(String, bool)> {
    let mut fragments = Vec::new();
    let mut buffer = String::new();
    let mut buffer_is_token = None::<bool>;

    for ch in text.chars() {
        let is_token = !disable_token_detection && is_token_char(ch);
        match buffer_is_token {
            None => {
                buffer.push(ch);
                buffer_is_token = Some(is_token);
            }
            Some(state) if state == is_token => {
                buffer.push(ch);
            }
            Some(state) => {
                fragments.push((std::mem::take(&mut buffer), state));
                buffer.push(ch);
                buffer_is_token = Some(is_token);
            }
        }
    }

    if let Some(state) = buffer_is_token {
        fragments.push((buffer, state));
    }

    fragments
}

fn parse_inline_markdown_segments(text: &str) -> Vec<(String, InlineMarkdownStyle)> {
    if text.is_empty() {
        return vec![(String::new(), InlineMarkdownStyle::default())];
    }

    let chars = text.chars().collect::<Vec<_>>();
    let mut segments = Vec::new();
    let mut current = String::new();
    let mut style = InlineMarkdownStyle::default();
    let mut idx = 0usize;

    while idx < chars.len() {
        if chars[idx] == '\\' && idx + 1 < chars.len() && is_markdown_escape_char(chars[idx + 1]) {
            current.push(chars[idx + 1]);
            idx += 2;
            continue;
        }

        if !style.code && chars[idx] == '[' {
            if let Some((consumed, label, url)) = parse_markdown_link(&chars, idx) {
                flush_markdown_segment(&mut segments, &mut current, style);
                let mut link_style = style;
                link_style.link = true;
                if url.is_empty() {
                    segments.push((label, link_style));
                } else {
                    segments.push((format!("{label} ({url})"), link_style));
                }
                idx += consumed;
                continue;
            }
        }

        if chars[idx] == '`' && (style.code || has_future_unescaped_char(&chars, idx + 1, '`')) {
            flush_markdown_segment(&mut segments, &mut current, style);
            style.code = !style.code;
            idx += 1;
            continue;
        }

        if !style.code
            && idx + 1 < chars.len()
            && chars[idx] == '~'
            && chars[idx + 1] == '~'
            && (style.strikethrough || has_future_unescaped_pair(&chars, idx + 2, '~', '~'))
        {
            flush_markdown_segment(&mut segments, &mut current, style);
            style.strikethrough = !style.strikethrough;
            idx += 2;
            continue;
        }

        if !style.code
            && idx + 1 < chars.len()
            && chars[idx] == '*'
            && chars[idx + 1] == '*'
            && (style.bold || has_future_unescaped_pair(&chars, idx + 2, '*', '*'))
        {
            flush_markdown_segment(&mut segments, &mut current, style);
            style.bold = !style.bold;
            idx += 2;
            continue;
        }

        if !style.code
            && idx + 1 < chars.len()
            && chars[idx] == '_'
            && chars[idx + 1] == '_'
            && can_toggle_underscore_marker(&chars, idx, 2, style.bold)
            && (style.bold || has_future_unescaped_pair(&chars, idx + 2, '_', '_'))
        {
            flush_markdown_segment(&mut segments, &mut current, style);
            style.bold = !style.bold;
            idx += 2;
            continue;
        }

        if !style.code
            && chars[idx] == '*'
            && chars.get(idx + 1).is_none_or(|next| *next != '*')
            && (style.italic || has_future_unescaped_char(&chars, idx + 1, '*'))
        {
            flush_markdown_segment(&mut segments, &mut current, style);
            style.italic = !style.italic;
            idx += 1;
            continue;
        }

        if !style.code
            && chars[idx] == '_'
            && chars.get(idx + 1).is_none_or(|next| *next != '_')
            && can_toggle_underscore_marker(&chars, idx, 1, style.italic)
            && (style.italic || has_future_unescaped_char(&chars, idx + 1, '_'))
        {
            flush_markdown_segment(&mut segments, &mut current, style);
            style.italic = !style.italic;
            idx += 1;
            continue;
        }

        current.push(chars[idx]);
        idx += 1;
    }

    if !current.is_empty() || segments.is_empty() {
        segments.push((current, style));
    }

    segments
}

fn flush_markdown_segment(
    segments: &mut Vec<(String, InlineMarkdownStyle)>,
    current: &mut String,
    style: InlineMarkdownStyle,
) {
    if current.is_empty() {
        return;
    }
    segments.push((std::mem::take(current), style));
}

fn is_markdown_escape_char(ch: char) -> bool {
    matches!(ch, '\\' | '*' | '`' | '_' | '~' | '[' | ']' | '(' | ')')
}

fn parse_markdown_link(chars: &[char], start: usize) -> Option<(usize, String, String)> {
    if chars.get(start).copied() != Some('[') {
        return None;
    }

    let mut index = start + 1;
    let mut label = String::new();
    let mut escaped = false;
    while index < chars.len() {
        let ch = chars[index];
        if escaped {
            label.push(ch);
            escaped = false;
            index += 1;
            continue;
        }
        if ch == '\\' {
            escaped = true;
            index += 1;
            continue;
        }
        if ch == ']' {
            break;
        }
        label.push(ch);
        index += 1;
    }

    if index >= chars.len() || chars[index] != ']' || label.trim().is_empty() {
        return None;
    }
    index += 1;
    if index >= chars.len() || chars[index] != '(' {
        return None;
    }
    index += 1;

    let mut url = String::new();
    escaped = false;
    while index < chars.len() {
        let ch = chars[index];
        if escaped {
            url.push(ch);
            escaped = false;
            index += 1;
            continue;
        }
        if ch == '\\' {
            escaped = true;
            index += 1;
            continue;
        }
        if ch == ')' {
            break;
        }
        url.push(ch);
        index += 1;
    }

    if index >= chars.len() || chars[index] != ')' {
        return None;
    }

    Some((index + 1 - start, label, url.trim().to_string()))
}

fn can_toggle_underscore_marker(
    chars: &[char],
    idx: usize,
    marker_len: usize,
    is_active: bool,
) -> bool {
    let previous = idx.checked_sub(1).and_then(|i| chars.get(i)).copied();
    let next = chars.get(idx + marker_len).copied();
    if is_active {
        next.is_none_or(|ch| !is_markdown_word_char(ch))
    } else {
        previous.is_none_or(|ch| !is_markdown_word_char(ch))
    }
}

fn is_markdown_word_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_'
}

fn has_future_unescaped_char(chars: &[char], mut idx: usize, target: char) -> bool {
    let mut escaped = false;
    while idx < chars.len() {
        let ch = chars[idx];
        if escaped {
            escaped = false;
            idx += 1;
            continue;
        }
        if ch == '\\' {
            escaped = true;
            idx += 1;
            continue;
        }
        if ch == target {
            return true;
        }
        idx += 1;
    }
    false
}

fn has_future_unescaped_pair(chars: &[char], mut idx: usize, first: char, second: char) -> bool {
    let mut escaped = false;
    while idx < chars.len() {
        let ch = chars[idx];
        if escaped {
            escaped = false;
            idx += 1;
            continue;
        }
        if ch == '\\' {
            escaped = true;
            idx += 1;
            continue;
        }
        if idx + 1 < chars.len() && ch == first && chars[idx + 1] == second {
            return true;
        }
        idx += 1;
    }
    false
}

fn apply_inline_markdown_style(base: Style, style: InlineMarkdownStyle, theme: TuiTheme) -> Style {
    let mut rendered = base;
    if style.code {
        let code_base = theme.code_block_style();
        if let Some(fg) = code_base.fg {
            rendered = rendered.fg(fg);
        }
        if let Some(bg) = code_base.bg {
            rendered = rendered.bg(bg);
        }
    }
    if style.bold {
        rendered = rendered.add_modifier(Modifier::BOLD);
    }
    if style.italic {
        rendered = rendered.add_modifier(Modifier::ITALIC);
    }
    if style.strikethrough {
        rendered = rendered.add_modifier(Modifier::CROSSED_OUT);
    }
    if style.link {
        rendered = theme
            .file_path_style(rendered)
            .add_modifier(Modifier::UNDERLINED);
    }
    rendered
}

fn code_highlighted_spans(
    text: &str,
    base: Style,
    theme: TuiTheme,
    language: Option<&str>,
) -> Vec<Span<'static>> {
    let chars = text.chars().collect::<Vec<_>>();
    if chars.is_empty() {
        return vec![Span::styled(String::new(), base)];
    }

    let mut spans = Vec::new();
    let mut idx = 0usize;
    while idx < chars.len() {
        if idx + 1 < chars.len() && chars[idx] == '/' && chars[idx + 1] == '/' {
            let fragment = chars[idx..].iter().collect::<String>();
            spans.push(Span::styled(fragment, theme.code_comment_style(base)));
            break;
        }

        if chars[idx] == '"' || chars[idx] == '\'' {
            let quote = chars[idx];
            let start = idx;
            idx += 1;
            let mut escaped = false;
            while idx < chars.len() {
                let ch = chars[idx];
                if escaped {
                    escaped = false;
                    idx += 1;
                    continue;
                }
                if ch == '\\' {
                    escaped = true;
                    idx += 1;
                    continue;
                }
                idx += 1;
                if ch == quote {
                    break;
                }
            }
            let fragment = chars[start..idx].iter().collect::<String>();
            spans.push(Span::styled(fragment, theme.code_string_style(base)));
            continue;
        }

        if chars[idx].is_ascii_digit() {
            let start = idx;
            idx += 1;
            while idx < chars.len() {
                let ch = chars[idx];
                if ch.is_ascii_digit() || ch == '_' || ch == '.' {
                    idx += 1;
                    continue;
                }
                break;
            }
            let fragment = chars[start..idx].iter().collect::<String>();
            spans.push(Span::styled(fragment, theme.code_number_style(base)));
            continue;
        }

        if is_identifier_start(chars[idx]) {
            let start = idx;
            idx += 1;
            while idx < chars.len() && is_identifier_continue(chars[idx]) {
                idx += 1;
            }
            let token = chars[start..idx].iter().collect::<String>();
            let style = if is_code_keyword(token.as_str(), language) {
                theme.code_keyword_style(base)
            } else {
                base
            };
            spans.push(Span::styled(token, style));
            continue;
        }

        let start = idx;
        idx += 1;
        while idx < chars.len()
            && !is_identifier_start(chars[idx])
            && !chars[idx].is_ascii_digit()
            && chars[idx] != '"'
            && chars[idx] != '\''
            && !(idx + 1 < chars.len() && chars[idx] == '/' && chars[idx + 1] == '/')
        {
            idx += 1;
        }
        spans.push(Span::styled(
            chars[start..idx].iter().collect::<String>(),
            base,
        ));
    }

    if spans.is_empty() {
        spans.push(Span::styled(String::new(), base));
    }
    spans
}

fn is_identifier_start(ch: char) -> bool {
    ch.is_ascii_alphabetic() || ch == '_'
}

fn is_identifier_continue(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_'
}

fn is_code_keyword(token: &str, language: Option<&str>) -> bool {
    let normalized = token.to_ascii_lowercase();
    match language.map(|value| value.to_ascii_lowercase()) {
        Some(language) if matches!(language.as_str(), "python" | "py") => matches!(
            normalized.as_str(),
            "def"
                | "class"
                | "if"
                | "elif"
                | "else"
                | "for"
                | "while"
                | "return"
                | "import"
                | "from"
                | "try"
                | "except"
                | "finally"
                | "with"
                | "as"
                | "lambda"
                | "yield"
                | "pass"
                | "break"
                | "continue"
                | "raise"
                | "async"
                | "await"
                | "true"
                | "false"
                | "none"
        ),
        Some(language)
            if matches!(language.as_str(), "javascript" | "js" | "typescript" | "ts") =>
        {
            matches!(
                normalized.as_str(),
                "const"
                    | "let"
                    | "var"
                    | "function"
                    | "class"
                    | "if"
                    | "else"
                    | "for"
                    | "while"
                    | "return"
                    | "import"
                    | "export"
                    | "from"
                    | "async"
                    | "await"
                    | "try"
                    | "catch"
                    | "finally"
                    | "switch"
                    | "case"
                    | "break"
                    | "continue"
                    | "new"
                    | "null"
                    | "undefined"
                    | "true"
                    | "false"
            )
        }
        _ => matches!(
            normalized.as_str(),
            "fn" | "let"
                | "mut"
                | "pub"
                | "impl"
                | "struct"
                | "enum"
                | "trait"
                | "use"
                | "mod"
                | "crate"
                | "self"
                | "super"
                | "const"
                | "static"
                | "match"
                | "if"
                | "else"
                | "loop"
                | "while"
                | "for"
                | "in"
                | "return"
                | "break"
                | "continue"
                | "where"
                | "as"
                | "type"
                | "async"
                | "await"
                | "move"
                | "ref"
                | "unsafe"
                | "dyn"
                | "true"
                | "false"
                | "none"
        ),
    }
}

fn parse_markdown_fence(text: &str) -> Option<Option<String>> {
    let trimmed = text.trim();
    let rest = trimmed.strip_prefix("```")?;
    let language = rest
        .split_whitespace()
        .next()
        .filter(|token| !token.is_empty())
        .map(|token| token.to_ascii_lowercase());
    Some(language)
}

fn explode_multiline_transcript_lines(lines: &[TranscriptLine]) -> Vec<TranscriptLine> {
    let mut exploded = Vec::with_capacity(lines.len());
    for line in lines {
        if !line.text.contains('\n') || matches!(line.kind, TranscriptLineKind::Working) {
            exploded.push(line.clone());
            continue;
        }

        for part in line.text.split('\n') {
            let segment = part.trim_end_matches('\r').to_string();
            if matches!(line.kind, TranscriptLineKind::Code) {
                exploded.push(TranscriptLine::new_code(
                    segment,
                    line.code_language.clone(),
                ));
            } else {
                exploded.push(TranscriptLine::new(segment, line.kind.clone()));
            }
        }
    }
    exploded
}

fn render_markdown(lines: &[TranscriptLine]) -> Vec<TranscriptLine> {
    let expanded = explode_multiline_transcript_lines(lines);
    let mut rendered = Vec::with_capacity(expanded.len());
    let mut code_block_kind: Option<TranscriptLineKind> = None;
    let mut markdown_fence_kind: Option<TranscriptLineKind> = None;
    let mut current_language: Option<String> = None;
    let mut cursor = 0usize;

    while cursor < expanded.len() {
        let line = &expanded[cursor];
        if code_block_kind
            .as_ref()
            .is_some_and(|kind| *kind != line.kind)
        {
            code_block_kind = None;
            current_language = None;
        }
        if markdown_fence_kind
            .as_ref()
            .is_some_and(|kind| *kind != line.kind)
        {
            markdown_fence_kind = None;
        }

        if supports_markdown_render(line.kind.clone()) {
            if let Some(language) = parse_markdown_fence(line.text.as_str()) {
                if code_block_kind.is_some() {
                    code_block_kind = None;
                    current_language = None;
                } else if markdown_fence_kind.is_some() {
                    markdown_fence_kind = None;
                } else if is_markdown_fence_language(language.as_deref()) {
                    // Treat ```markdown fenced blocks as normal markdown content.
                    markdown_fence_kind = Some(line.kind.clone());
                } else {
                    code_block_kind = Some(line.kind.clone());
                    current_language = language;
                }
                cursor += 1;
                continue;
            }

            if code_block_kind
                .as_ref()
                .is_some_and(|kind| *kind == line.kind)
            {
                rendered.push(TranscriptLine::new_code(
                    line.text.clone(),
                    current_language.clone(),
                ));
                cursor += 1;
                continue;
            }

            if let Some((consumed, table_lines)) =
                render_markdown_table_block(&expanded[cursor..], line.kind.clone())
            {
                rendered.extend(table_lines);
                cursor += consumed;
                continue;
            }

            if let Some(structural_line) = render_markdown_structural_line(line) {
                rendered.push(structural_line);
                cursor += 1;
                continue;
            }
        }

        rendered.push(line.clone());
        cursor += 1;
    }

    rendered
}

fn supports_markdown_render(kind: TranscriptLineKind) -> bool {
    matches!(
        kind,
        TranscriptLineKind::Normal | TranscriptLineKind::Assistant | TranscriptLineKind::UserInput
    )
}

fn is_markdown_fence_language(language: Option<&str>) -> bool {
    matches!(language, Some("markdown" | "md"))
}

fn render_markdown_structural_line(line: &TranscriptLine) -> Option<TranscriptLine> {
    if !supports_markdown_render(line.kind.clone()) {
        return None;
    }

    let indent_columns = line
        .text
        .chars()
        .take_while(|ch| ch.is_whitespace())
        .map(|ch| if ch == '\t' { 4 } else { 1 })
        .sum::<usize>();
    let trimmed = line.text.trim_start();
    if trimmed.is_empty() {
        return None;
    }

    if matches!(line.kind, TranscriptLineKind::UserInput)
        && looks_like_prompt_prefixed_user_input(trimmed)
    {
        return None;
    }

    if let Some(heading) = parse_markdown_heading(trimmed) {
        return Some(TranscriptLine::new_markdown(
            heading,
            line.kind.clone(),
            MarkdownLineStyle::Heading,
        ));
    }

    if let Some(quote) = parse_markdown_quote(trimmed) {
        return Some(TranscriptLine::new_markdown(
            quote,
            line.kind.clone(),
            MarkdownLineStyle::Quote,
        ));
    }

    if let Some(list_item) = parse_markdown_list_item(trimmed, indent_columns / 2) {
        return Some(TranscriptLine::new(list_item, line.kind.clone()));
    }

    if is_markdown_horizontal_rule(trimmed) {
        return Some(TranscriptLine::new_markdown(
            "────────────────────────".to_string(),
            line.kind.clone(),
            MarkdownLineStyle::Rule,
        ));
    }

    None
}

fn looks_like_prompt_prefixed_user_input(trimmed: &str) -> bool {
    trimmed.starts_with(">  ")
}

fn parse_markdown_heading(line: &str) -> Option<String> {
    let marker_len = line.chars().take_while(|ch| *ch == '#').count();
    if !(1..=6).contains(&marker_len) {
        return None;
    }

    let rest = line.chars().skip(marker_len).collect::<String>();
    let heading = rest.trim_start();
    if heading.is_empty() {
        return None;
    }
    Some(heading.to_string())
}

fn parse_markdown_quote(line: &str) -> Option<String> {
    let mut depth = 0usize;
    let mut rest = line;

    loop {
        let Some(stripped) = rest.strip_prefix('>') else {
            break;
        };
        depth += 1;
        rest = stripped.trim_start();
    }

    if depth == 0 {
        return None;
    }

    let prefix = "│ ".repeat(depth);
    if rest.is_empty() {
        return Some(prefix.trim_end().to_string());
    }
    Some(format!("{prefix}{rest}"))
}

fn parse_markdown_list_item(line: &str, indent_level: usize) -> Option<String> {
    let indent = "  ".repeat(indent_level);

    if let Some(body) = line
        .strip_prefix("- ")
        .or_else(|| line.strip_prefix("* "))
        .or_else(|| line.strip_prefix("+ "))
    {
        if let Some(task) = parse_markdown_task_item(body) {
            return Some(format!("{indent}{task}"));
        }
        return Some(format!("{indent}• {}", body.trim_start()));
    }

    let bytes = line.as_bytes();
    let mut index = 0usize;
    while index < bytes.len() && bytes[index].is_ascii_digit() {
        index += 1;
    }
    if index == 0 || index + 1 >= bytes.len() {
        return None;
    }
    if !matches!(bytes[index], b'.' | b')') || bytes[index + 1] != b' ' {
        return None;
    }

    let marker = &line[..=index];
    let body = line[index + 2..].trim_start();
    Some(format!("{indent}{marker} {body}"))
}

fn parse_markdown_task_item(body: &str) -> Option<String> {
    if let Some(rest) = body.strip_prefix("[ ] ") {
        return Some(format!("☐ {}", rest.trim_start()));
    }
    if let Some(rest) = body
        .strip_prefix("[x] ")
        .or_else(|| body.strip_prefix("[X] "))
    {
        return Some(format!("☑ {}", rest.trim_start()));
    }
    None
}

fn is_markdown_horizontal_rule(line: &str) -> bool {
    let compact = line
        .chars()
        .filter(|ch| !ch.is_ascii_whitespace())
        .collect::<Vec<_>>();
    if compact.len() < 3 {
        return false;
    }
    let marker = compact[0];
    if !matches!(marker, '-' | '*' | '_') {
        return false;
    }
    compact.iter().all(|ch| *ch == marker)
}

fn render_markdown_table_block(
    lines: &[TranscriptLine],
    line_kind: TranscriptLineKind,
) -> Option<(usize, Vec<TranscriptLine>)> {
    if lines.len() < 2 {
        return None;
    }

    let header_line = lines.first()?;
    if header_line.kind != line_kind {
        return None;
    }
    let header = parse_markdown_table_row(header_line.text.as_str())?;
    if header.is_empty() {
        return None;
    }

    let separator_line = lines.get(1)?;
    if separator_line.kind != line_kind {
        return None;
    }
    let separator = parse_markdown_table_row(separator_line.text.as_str())?;
    if separator.len() != header.len() || !is_markdown_table_separator_row(&separator) {
        return None;
    }

    let mut rows = Vec::new();
    let mut consumed = 2usize;
    while let Some(line) = lines.get(consumed) {
        if line.kind != line_kind || line.text.trim().is_empty() {
            break;
        }
        let Some(cells) = parse_markdown_table_row(line.text.as_str()) else {
            break;
        };
        if cells.len() != header.len() || is_markdown_table_separator_row(&cells) {
            break;
        }
        rows.push(cells);
        consumed += 1;
    }

    let rendered = render_box_table_lines(header.as_slice(), rows.as_slice())
        .into_iter()
        .map(|line| TranscriptLine::new(line, line_kind.clone()))
        .collect::<Vec<_>>();
    Some((consumed, rendered))
}

fn parse_markdown_table_row(line: &str) -> Option<Vec<String>> {
    let trimmed = line.trim();
    if trimmed.is_empty() || !trimmed.contains('|') {
        return None;
    }

    let mut cells = split_markdown_table_cells(trimmed);
    if trimmed.starts_with('|') && !cells.is_empty() {
        cells.remove(0);
    }
    if trimmed.ends_with('|') && !cells.is_empty() {
        cells.pop();
    }

    if cells.is_empty() {
        return None;
    }

    Some(
        cells
            .into_iter()
            .map(|cell| cell.trim().to_string())
            .collect(),
    )
}

fn split_markdown_table_cells(row: &str) -> Vec<String> {
    let mut cells = Vec::new();
    let mut current = String::new();
    let mut chars = row.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '\\' {
            if chars.peek().is_some_and(|next| *next == '|') {
                current.push('|');
                chars.next();
            } else {
                current.push(ch);
            }
            continue;
        }

        if ch == '|' {
            cells.push(std::mem::take(&mut current));
            continue;
        }

        current.push(ch);
    }

    cells.push(current);
    cells
}

fn is_markdown_table_separator_row(cells: &[String]) -> bool {
    if cells.is_empty() {
        return false;
    }
    cells.iter().all(|cell| {
        let trimmed = cell.trim();
        !trimmed.is_empty()
            && trimmed.chars().all(|ch| matches!(ch, '-' | ':'))
            && trimmed.contains('-')
    })
}

fn render_box_table_lines(header: &[String], rows: &[Vec<String>]) -> Vec<String> {
    let mut widths = vec![0usize; header.len()];
    for row in std::iter::once(header).chain(rows.iter().map(|row| row.as_slice())) {
        for (index, cell) in row.iter().enumerate() {
            widths[index] = widths[index].max(UnicodeWidthStr::width(cell.as_str()));
        }
    }

    let mut rendered = Vec::new();
    rendered.push(render_table_border('┌', '┬', '┐', widths.as_slice()));
    rendered.push(render_table_row(header, widths.as_slice()));
    rendered.push(render_table_border('├', '┼', '┤', widths.as_slice()));
    for row in rows {
        rendered.push(render_table_row(row.as_slice(), widths.as_slice()));
    }
    rendered.push(render_table_border('└', '┴', '┘', widths.as_slice()));
    rendered
}

fn render_table_border(left: char, mid: char, right: char, widths: &[usize]) -> String {
    let mut line = String::new();
    line.push(left);
    for (index, width) in widths.iter().enumerate() {
        line.push_str("─".repeat(*width + 2).as_str());
        if index + 1 < widths.len() {
            line.push(mid);
        }
    }
    line.push(right);
    line
}

fn render_table_row(row: &[String], widths: &[usize]) -> String {
    let mut line = String::new();
    line.push('│');
    for (index, width) in widths.iter().enumerate() {
        let cell = row.get(index).map(String::as_str).unwrap_or("");
        let cell_width = UnicodeWidthStr::width(cell);
        line.push(' ');
        line.push_str(cell);
        if cell_width < *width {
            line.push_str(" ".repeat(*width - cell_width).as_str());
        }
        line.push(' ');
        line.push('│');
    }
    line
}

fn styled_fragment(
    fragment: &str,
    is_token: bool,
    previous: Option<&str>,
    next: Option<&str>,
    base: Style,
    theme: TuiTheme,
) -> Span<'static> {
    let style = if is_section_header_bracket_fragment(fragment, previous, next) {
        theme.skills_header_style(base)
    } else if !is_token {
        base
    } else if is_skills_header_token(fragment, previous, next) {
        theme.skills_header_style(base)
    } else if is_skills_user_group_token(fragment, previous, next) {
        theme.skills_group_style(base)
    } else if is_key_token(fragment) {
        theme.key_token_style(base)
    } else if is_file_path_token(fragment) {
        theme.file_path_style(base)
    } else {
        base
    };
    Span::styled(fragment.to_string(), style)
}

fn is_section_header_bracket_fragment(
    fragment: &str,
    previous: Option<&str>,
    next: Option<&str>,
) -> bool {
    let normalized = fragment.trim_matches(char::is_whitespace);
    if normalized == "[" {
        return next.is_some_and(is_section_header_name_fragment);
    }
    if normalized == "]" {
        return previous.is_some_and(is_section_header_name_fragment);
    }
    false
}

fn is_section_header_name_fragment(fragment: &str) -> bool {
    let normalized = fragment
        .trim_matches(char::is_whitespace)
        .to_ascii_lowercase();
    matches!(normalized.as_str(), "skills" | "context")
}

fn is_skills_header_token(token: &str, previous: Option<&str>, next: Option<&str>) -> bool {
    if !is_section_header_name_fragment(token) {
        return false;
    }

    previous.is_some_and(|fragment| fragment.trim_end().ends_with('['))
        && next.is_some_and(|fragment| fragment.trim_start().starts_with(']'))
}

fn is_skills_user_group_token(token: &str, previous: Option<&str>, next: Option<&str>) -> bool {
    let normalized = token
        .trim_matches(|ch: char| ch.is_ascii_whitespace())
        .to_ascii_lowercase();
    if normalized != "user" {
        return false;
    }

    let previous_is_whitespace = previous
        .map(|fragment| fragment.chars().all(char::is_whitespace))
        .unwrap_or(true);
    let next_is_whitespace = next
        .map(|fragment| fragment.chars().all(char::is_whitespace))
        .unwrap_or(true);
    previous_is_whitespace && next_is_whitespace
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

    if normalized.is_empty() {
        return false;
    }

    if is_compound_key_token(normalized.as_str()) {
        return true;
    }

    if normalized.len() > 1 && parse_key_id(normalized.as_str()).is_some() {
        return true;
    }

    matches!(
        normalized.as_str(),
        "/" | "up" | "down" | "pageup" | "pagedown"
    )
}

fn is_compound_key_token(normalized: &str) -> bool {
    if !normalized.contains('/') {
        // Highlight generic modifier+key labels like ctrl+c, ctrl+shift+p, shift+tab, alt+enter, etc.
        // This keeps startup/help shortcut hints visually consistent even when keymaps are customized.
        return normalized.contains('+') && parse_key_id(normalized).is_some();
    }

    if normalized.split('/').all(|part| {
        matches!(
            part,
            "up" | "down" | "left" | "right" | "pageup" | "pagedown"
        )
    }) {
        return true;
    }

    if !normalized.contains('+') {
        return false;
    }

    // Support compact forms like ctrl+a/e, where trailing segments inherit modifiers.
    let mut modifier_prefix: Option<&str> = None;
    for segment in normalized.split('/') {
        if segment.is_empty() {
            return false;
        }

        if parse_key_id(segment).is_some() {
            modifier_prefix = segment.rsplit_once('+').map(|(prefix, _)| prefix);
            continue;
        }

        let Some(prefix) = modifier_prefix else {
            return false;
        };
        let inherited = format!("{prefix}+{segment}");
        if parse_key_id(inherited.as_str()).is_none() {
            return false;
        }
    }

    true
}

pub(crate) fn visible_transcript_lines(
    lines: &[TranscriptLine],
    supplemental_lines: &[TranscriptLine],
    max_lines: usize,
    max_width: usize,
    show_tool_results: bool,
    show_thinking: bool,
    working_line: Option<TranscriptLine>,
    scroll_from_bottom: usize,
    theme: TuiTheme,
) -> Vec<Line<'static>> {
    if max_lines == 0 || max_width == 0 {
        return vec![];
    }

    let mut filtered = lines
        .iter()
        .filter(|line| match line.kind {
            TranscriptLineKind::Normal => true,
            TranscriptLineKind::Assistant => true,
            TranscriptLineKind::Overlay => true,
            TranscriptLineKind::Code => true,
            TranscriptLineKind::UserInput => true,
            TranscriptLineKind::Thinking => show_thinking,
            TranscriptLineKind::Tool => show_tool_results,
            TranscriptLineKind::Working => true,
        })
        .cloned()
        .collect::<Vec<_>>();

    filtered.extend(supplemental_lines.iter().cloned());

    if let Some(working_line) = working_line {
        filtered.push(working_line);
    }

    let markdown_rendered = render_markdown(&filtered);
    let compacted = compact_tool_transcript_lines(&markdown_rendered);
    let spaced = pad_transcript_block_boundaries(&compacted);
    let wrapped = wrap_transcript_lines(&spaced, max_width);
    let prefixed = decorate_assistant_output_prefix(&wrapped, theme.output_prompt());
    let max_scroll = prefixed.len().saturating_sub(max_lines);
    let scroll = scroll_from_bottom.min(max_scroll);
    let end = prefixed.len().saturating_sub(scroll);
    let start = end.saturating_sub(max_lines);
    prefixed[start..end]
        .iter()
        .map(|line| line.to_line(max_width, theme))
        .collect()
}

fn decorate_assistant_output_prefix(
    lines: &[TranscriptLine],
    output_prompt: &str,
) -> Vec<TranscriptLine> {
    let mut decorated = Vec::with_capacity(lines.len());
    let mut should_prefix_current_block = true;

    for line in lines {
        if line.kind != TranscriptLineKind::Assistant {
            should_prefix_current_block = true;
            decorated.push(line.clone());
            continue;
        }

        let mut decorated_line = line.clone();
        if should_prefix_current_block
            && should_prefix_assistant_output_line(decorated_line.text.as_str())
            && !decorated_line.text.starts_with(output_prompt)
        {
            decorated_line.text =
                format_assistant_output_line(decorated_line.text.as_str(), output_prompt);
            should_prefix_current_block = false;
        }
        decorated.push(decorated_line);
    }

    decorated
}

fn format_assistant_output_line(line: &str, output_prompt: &str) -> String {
    format!("{output_prompt}{line}")
}

fn should_prefix_assistant_output_line(line: &str) -> bool {
    let trimmed = line.trim_start();
    if trimmed.is_empty() {
        return false;
    }
    !trimmed.starts_with(['┌', '├', '│', '└'])
}

fn wrap_transcript_lines(lines: &[TranscriptLine], max_width: usize) -> Vec<TranscriptLine> {
    let mut wrapped = Vec::new();
    for line in lines {
        let segments = wrap_text_by_display_width(&line.text, max_width);
        if segments.len() == 1 {
            let mut single = line.clone();
            single.text = segments[0].clone();
            wrapped.push(single);
            continue;
        }
        for segment in segments {
            wrapped.push(TranscriptLine::new(segment, line.kind.clone()));
        }
    }
    wrapped
}

fn pad_transcript_block_boundaries(lines: &[TranscriptLine]) -> Vec<TranscriptLine> {
    if lines.is_empty() {
        return vec![];
    }

    let mut blocks: Vec<Vec<TranscriptLine>> = Vec::new();
    let mut current_block: Vec<TranscriptLine> = Vec::new();
    let mut current_kind: Option<TranscriptLineKind> = None;

    for line in lines {
        match current_kind.as_ref() {
            Some(kind) if *kind == line.kind => current_block.push(line.clone()),
            Some(_) => {
                blocks.push(current_block);
                current_block = vec![line.clone()];
                current_kind = Some(line.kind.clone());
            }
            None => {
                current_block.push(line.clone());
                current_kind = Some(line.kind.clone());
            }
        }
    }
    if !current_block.is_empty() {
        blocks.push(current_block);
    }

    let total_blocks = blocks.len();
    for (index, block) in blocks.iter_mut().enumerate() {
        if block.is_empty() {
            continue;
        }
        let kind = block[0].kind.clone();
        let is_first_block = index == 0;
        let is_last_block = index + 1 == total_blocks;

        if !is_first_block && should_insert_leading_block_padding(&kind) {
            let has_leading_blank = block
                .first()
                .map(|line| line.text.trim().is_empty())
                .unwrap_or(false);
            if !has_leading_blank {
                block.insert(0, TranscriptLine::new(String::new(), kind.clone()));
            }
        }

        if !is_last_block {
            let has_trailing_blank = block
                .last()
                .map(|line| line.text.trim().is_empty())
                .unwrap_or(false);
            if !has_trailing_blank {
                block.push(TranscriptLine::new(String::new(), kind));
            }
        }
    }

    blocks.into_iter().flatten().collect()
}

fn should_insert_leading_block_padding(kind: &TranscriptLineKind) -> bool {
    !matches!(kind, TranscriptLineKind::Tool | TranscriptLineKind::Working)
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

    let mut compacted: Vec<TranscriptLine> = Vec::new();
    let mut cursor = 0usize;
    let mut saw_tool_invocation = false;
    while cursor < lines.len() {
        let line = &lines[cursor];
        let tool_title = if is_tool_run_line(line.text.as_str()) {
            Some(line.text.clone())
        } else if let Some((tool_name, is_error)) = parse_legacy_tool_header(line.text.as_str()) {
            let title = if is_error {
                format!("• Ran {tool_name} (error)")
            } else {
                format!("• Ran {tool_name}")
            };
            Some(title)
        } else {
            None
        };

        if let Some(title) = tool_title {
            if saw_tool_invocation
                && compacted
                    .last()
                    .map(|last| !last.text.trim().is_empty())
                    .unwrap_or(false)
            {
                compacted.push(TranscriptLine::new(String::new(), TranscriptLineKind::Tool));
            }

            compacted.push(TranscriptLine::new(title, TranscriptLineKind::Tool));
            saw_tool_invocation = true;
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

pub(crate) fn wrap_text_by_display_width(text: &str, max_width: usize) -> Vec<String> {
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

pub(crate) fn render_messages(messages: &[Message]) -> Vec<TranscriptLine> {
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
                                    TranscriptLineKind::Assistant,
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
                            TranscriptLineKind::Assistant,
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
                if should_render_tool_result_content(tool_name) {
                    for block in content {
                        match block {
                            ToolResultContentBlock::Text { text, .. } => {
                                for tool_line in split_tool_output_lines(text) {
                                    lines.push(TranscriptLine::new(
                                        tool_line,
                                        TranscriptLineKind::Tool,
                                    ));
                                }
                            }
                            ToolResultContentBlock::Image { .. } => {
                                lines.push(TranscriptLine::new(
                                    "(image tool result omitted)".to_string(),
                                    TranscriptLineKind::Tool,
                                ))
                            }
                        }
                    }
                }
            }
            Message::User { .. } => {}
        }
    }
    lines
}

pub(crate) fn is_thinking_line(line: &str) -> bool {
    line.starts_with("[thinking]")
}

pub(crate) fn parse_tool_name(line: &str) -> Option<&str> {
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

pub(crate) fn normalize_tool_line_for_display(line: String) -> String {
    if let Some((tool_name, is_error)) = parse_legacy_tool_header(line.as_str()) {
        if is_error {
            return format!("• Ran {tool_name} (error)");
        }
        return format!("• Ran {tool_name}");
    }
    line
}

pub(crate) fn split_tool_output_lines(text: &str) -> Vec<String> {
    text.split('\n')
        .map(|line| line.trim_end_matches('\r').to_string())
        .collect()
}

pub(crate) fn is_tool_run_line(line: &str) -> bool {
    line.starts_with("• Ran ")
}

fn should_render_tool_result_content(tool_name: &str) -> bool {
    tool_name != "read"
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
