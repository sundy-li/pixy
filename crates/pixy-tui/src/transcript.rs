use pixy_ai::{AssistantContentBlock, Message, StopReason, ToolResultContentBlock};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::TuiTheme;
use crate::keybindings::parse_key_id;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum TranscriptLineKind {
    Normal,
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
    working_marquee: Option<WorkingMarquee>,
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
            working_marquee: None,
        }
    }

    pub(crate) fn new_code(text: String, language: Option<String>) -> Self {
        Self {
            text,
            kind: TranscriptLineKind::Code,
            code_language: language,
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
            working_marquee: Some(WorkingMarquee {
                message_char_start,
                message_char_len,
                highlight_start,
                highlight_len,
            }),
        }
    }

    pub(crate) fn to_line(&self, width: usize, theme: TuiTheme) -> Line<'static> {
        let base = theme.line_style(self.kind.clone());

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

        let spans = highlighted_spans(self.text.as_str(), base, theme);
        line_with_padding(spans, width, base)
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

fn highlighted_spans(text: &str, base: Style, theme: TuiTheme) -> Vec<Span<'static>> {
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
                spans.push(styled_fragment(&buffer, state, base, theme));
                buffer.clear();
                buffer.push(ch);
                buffer_is_token = Some(is_token);
            }
        }
    }

    if let Some(state) = buffer_is_token {
        spans.push(styled_fragment(&buffer, state, base, theme));
    }

    if spans.is_empty() {
        spans.push(Span::styled(String::new(), base));
    }

    spans
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

fn render_markdown_code_blocks(lines: &[TranscriptLine]) -> Vec<TranscriptLine> {
    let mut rendered = Vec::with_capacity(lines.len());
    let mut in_code_block = false;
    let mut current_language: Option<String> = None;

    for line in lines {
        if matches!(line.kind, TranscriptLineKind::Normal) {
            if let Some(language) = parse_markdown_fence(line.text.as_str()) {
                if in_code_block {
                    in_code_block = false;
                    current_language = None;
                } else {
                    in_code_block = true;
                    current_language = language;
                }
                continue;
            }

            if in_code_block {
                rendered.push(TranscriptLine::new_code(
                    line.text.clone(),
                    current_language.clone(),
                ));
                continue;
            }
        }

        rendered.push(line.clone());
    }

    rendered
}

fn styled_fragment(fragment: &str, is_token: bool, base: Style, theme: TuiTheme) -> Span<'static> {
    if !is_token {
        return Span::styled(fragment.to_string(), base);
    }

    let style = if is_key_token(fragment) {
        theme.key_token_style(base)
    } else if is_file_path_token(fragment) {
        theme.file_path_style(base)
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

    let markdown_rendered = render_markdown_code_blocks(&filtered);
    let compacted = compact_tool_transcript_lines(&markdown_rendered);
    let spaced = pad_transcript_block_boundaries(&compacted);
    let wrapped = wrap_transcript_lines(&spaced, max_width);
    let max_scroll = wrapped.len().saturating_sub(max_lines);
    let scroll = scroll_from_bottom.min(max_scroll);
    let end = wrapped.len().saturating_sub(scroll);
    let start = end.saturating_sub(max_lines);
    wrapped[start..end]
        .iter()
        .map(|line| line.to_line(max_width, theme))
        .collect()
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

fn stop_reason_label(reason: &StopReason) -> &'static str {
    match reason {
        StopReason::Stop => "stop",
        StopReason::Length => "length",
        StopReason::ToolUse => "tool_use",
        StopReason::Error => "error",
        StopReason::Aborted => "aborted",
    }
}
