use std::collections::{HashMap, HashSet};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use clap::{Args, Parser};
use pixy_ai::{
    AssistantContentBlock, Cost, DEFAULT_TRANSPORT_RETRY_COUNT, Message, Model,
    SimpleStreamOptions, StopReason, ToolResultContentBlock,
};
use pixy_coding_agent::{
    AgentSession, AgentSessionConfig, AgentSessionStreamUpdate, LoadSkillsOptions, SessionManager,
    Skill, SkillSource, create_coding_tools, load_skills,
};
use pixy_tui::{KeyBinding, TuiKeyBindings, TuiOptions, TuiTheme, parse_key_id};
use serde::Deserialize;
use system_prompt::build_system_prompt;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

mod system_prompt;

const RESUME_PICKER_LIMIT: usize = 10;

#[derive(Parser, Debug)]
#[command(name = "pixy", version, about = "pixy interactive CLI")]
struct Cli {
    #[command(flatten)]
    chat: ChatArgs,
}

#[derive(Args, Debug, Clone)]
struct ChatArgs {
    #[arg(long)]
    api: Option<String>,
    #[arg(long)]
    provider: Option<String>,
    #[arg(long)]
    model: Option<String>,
    #[arg(long)]
    base_url: Option<String>,
    #[arg(long)]
    context_window: Option<u32>,
    #[arg(long)]
    max_tokens: Option<u32>,
    #[arg(long)]
    agent_dir: Option<PathBuf>,
    #[arg(long)]
    cwd: Option<PathBuf>,
    #[arg(long)]
    session_dir: Option<PathBuf>,
    #[arg(long)]
    session_file: Option<PathBuf>,
    #[arg(long)]
    system_prompt: Option<String>,
    #[arg(long)]
    prompt: Option<String>,
    #[arg(long, default_value_t = false)]
    continue_first: bool,
    #[arg(long, default_value_t = false)]
    no_tools: bool,
    #[arg(long = "skill")]
    skills: Vec<String>,
    #[arg(long, default_value_t = false)]
    no_skills: bool,
    #[arg(long, default_value_t = false)]
    hide_tool_results: bool,
    #[arg(long, default_value_t = false)]
    no_tui: bool,
    #[arg(long)]
    theme: Option<String>,
}

#[tokio::main]
async fn main() {
    init_tracing();
    let cli = Cli::parse();
    if let Err(error) = run(cli.chat).await {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn init_tracing() {
    static TRACE_GUARD: OnceLock<WorkerGuard> = OnceLock::new();

    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let log_dir = PathBuf::from(home).join(".pixy");
    if let Err(error) = std::fs::create_dir_all(&log_dir) {
        eprintln!(
            "warning: failed to create log dir {}: {error}",
            log_dir.display()
        );
        return;
    }

    let appender = tracing_appender::rolling::never(&log_dir, "pixy.log");
    let (non_blocking, guard) = tracing_appender::non_blocking(appender);
    let _ = TRACE_GUARD.set(guard);

    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::registry()
        .with(env_filter)
        .with(
            tracing_subscriber::fmt::layer()
                .with_ansi(false)
                .with_writer(non_blocking),
        )
        .try_init();
}

async fn run(args: ChatArgs) -> Result<(), String> {
    let process_cwd =
        std::env::current_dir().map_err(|error| format!("read cwd failed: {error}"))?;
    let cwd = args
        .cwd
        .as_ref()
        .map(|path| resolve_path(&process_cwd, path))
        .unwrap_or(process_cwd);

    let agent_dir = args
        .agent_dir
        .as_ref()
        .map(|path| resolve_path(&cwd, path))
        .unwrap_or_else(default_agent_dir);
    let local_config = load_agent_local_config(&agent_dir)?;
    pixy_ai::set_transport_retry_count(resolve_transport_retry_count(&local_config.settings));
    let runtime = resolve_runtime_config(&args, &local_config)?;
    let discovered_skills = load_runtime_skills(&args, &local_config, &cwd, &agent_dir);

    let session_dir = args
        .session_dir
        .as_ref()
        .map(|path| resolve_path(&cwd, path))
        .unwrap_or_else(|| default_agent_dir().join("sessions"));
    let mut session = create_session(&args, &runtime, &cwd, &session_dir, &discovered_skills)?;
    let use_tui = args.prompt.is_none() && !args.no_tui;

    if !use_tui {
        if let Some(session_file) = session.session_file() {
            println!("session: {}", session_file.display());
        }
        println!("cwd: {}", cwd.display());
        println!(
            "model: {}/{}/{}",
            runtime.model.api, runtime.model.provider, runtime.model.id
        );
    }

    if let Some(prompt) = args.prompt.as_deref() {
        run_prompt_streaming_cli(&mut session, prompt, !args.hide_tool_results).await?;
        return Ok(());
    }

    if use_tui {
        let theme_name = resolve_tui_theme_name(
            args.theme.as_deref(),
            local_config.settings.theme.as_deref(),
        )?;
        let theme = TuiTheme::from_name(theme_name.as_str())
            .ok_or_else(|| format!("unsupported theme '{theme_name}', expected dark or light"))?;
        let status_top = build_status_top_line(&cwd);
        let status_left = format!(
            "0.0%/{} (auto)",
            format_token_window(runtime.model.context_window)
        );
        let status_right = format!("{} • medium", runtime.model.id);
        let enable_mouse_capture = std::env::var("PI_TUI_MOUSE_CAPTURE")
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes"
                )
            })
            .unwrap_or(false);
        let startup_resource_lines =
            build_startup_resource_lines(&cwd, &agent_dir, &discovered_skills);
        let mut tui_options = TuiOptions {
            app_name: "pixy".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            show_tool_results: !args.hide_tool_results,
            status_top,
            status_left,
            status_right,
            theme,
            input_history_path: Some(agent_dir.join("input_history.jsonl")),
            enable_mouse_capture,
            startup_resource_lines,
            ..TuiOptions::default()
        };
        if let Some(keybindings) = load_tui_keybindings(&agent_dir) {
            tui_options.keybindings = keybindings;
        }
        return pixy_tui::run_tui(&mut session, tui_options).await;
    }

    if args.continue_first {
        run_continue_streaming_cli(&mut session, !args.hide_tool_results).await?;
    }

    println!("commands: /continue, /resume [session], /session, /help, /exit");
    repl_loop(&mut session, !args.hide_tool_results).await
}

fn build_status_top_line(cwd: &Path) -> String {
    if let Some(branch) = detect_git_branch(cwd) {
        format!("{} ({branch})", cwd.display())
    } else {
        cwd.display().to_string()
    }
}

fn detect_git_branch(cwd: &Path) -> Option<String> {
    let output = Command::new("git")
        .arg("rev-parse")
        .arg("--abbrev-ref")
        .arg("HEAD")
        .current_dir(cwd)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let branch = String::from_utf8(output.stdout).ok()?.trim().to_string();
    if branch.is_empty() || branch == "HEAD" {
        None
    } else {
        Some(branch)
    }
}

fn format_token_window(value: u32) -> String {
    if value >= 1_000 {
        if value % 1_000 == 0 {
            return format!("{}k", value / 1_000);
        }
        let formatted = format!("{:.1}", value as f64 / 1_000.0);
        format!("{}k", formatted.trim_end_matches('0').trim_end_matches('.'))
    } else {
        value.to_string()
    }
}

fn build_startup_resource_lines(cwd: &Path, agent_dir: &Path, skills: &[Skill]) -> Vec<String> {
    let mut lines = vec![];
    let contexts = collect_startup_context_files(cwd, agent_dir);
    if !contexts.is_empty() {
        lines.push("[Context]".to_string());
        lines.extend(
            contexts
                .iter()
                .map(|path| format!("  {}", format_display_path(path))),
        );
    }

    let mut user_skills = vec![];
    let mut project_skills = vec![];
    let mut path_skills = vec![];
    for skill in skills {
        let rendered = format_display_path(&skill.file_path);
        match skill.source {
            SkillSource::User => user_skills.push(rendered),
            SkillSource::Project => project_skills.push(rendered),
            SkillSource::Path => path_skills.push(rendered),
        }
    }

    for group in [&mut user_skills, &mut project_skills, &mut path_skills] {
        group.sort();
        group.dedup();
    }

    let has_skills =
        !user_skills.is_empty() || !project_skills.is_empty() || !path_skills.is_empty();
    if has_skills {
        if !lines.is_empty() {
            lines.push(String::new());
        }
        lines.push("[Skills]".to_string());
        append_skill_group(&mut lines, "user", &user_skills);
        append_skill_group(&mut lines, "project", &project_skills);
        append_skill_group(&mut lines, "path", &path_skills);
    }

    lines
}

fn append_skill_group(lines: &mut Vec<String>, label: &str, values: &[String]) {
    if values.is_empty() {
        return;
    }
    lines.push(format!("  {label}"));
    lines.extend(values.iter().map(|value| format!("    {value}")));
}

fn collect_startup_context_files(cwd: &Path, agent_dir: &Path) -> Vec<PathBuf> {
    let mut contexts = vec![];
    let mut seen = HashSet::<PathBuf>::new();

    if let Some(global_context) = find_context_file_in_dir(agent_dir) {
        push_unique_path(&mut contexts, &mut seen, global_context);
    }

    let mut ancestor_contexts = vec![];
    let mut current = cwd.to_path_buf();
    loop {
        if let Some(context_file) = find_context_file_in_dir(&current) {
            push_unique_path(&mut ancestor_contexts, &mut seen, context_file);
        }
        let Some(parent) = current.parent() else {
            break;
        };
        if parent == current {
            break;
        }
        current = parent.to_path_buf();
    }
    ancestor_contexts.reverse();
    contexts.extend(ancestor_contexts);

    contexts
}

fn push_unique_path(out: &mut Vec<PathBuf>, seen: &mut HashSet<PathBuf>, path: PathBuf) {
    let key = std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
    if seen.insert(key) {
        out.push(path);
    }
}

fn find_context_file_in_dir(dir: &Path) -> Option<PathBuf> {
    for name in ["AGENTS.md", "CLAUDE.md"] {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn format_display_path(path: &Path) -> String {
    let Some(home_dir) = std::env::var_os("HOME").map(PathBuf::from) else {
        return path.display().to_string();
    };
    let Ok(stripped) = path.strip_prefix(&home_dir) else {
        return path.display().to_string();
    };
    if stripped.as_os_str().is_empty() {
        "~".to_string()
    } else {
        format!("~/{}", stripped.display())
    }
}

fn create_session(
    args: &ChatArgs,
    runtime: &ResolvedRuntimeConfig,
    cwd: &Path,
    session_dir: &Path,
    skills: &[Skill],
) -> Result<AgentSession, String> {
    let session_manager = if let Some(session_file) = &args.session_file {
        let resolved = resolve_path(cwd, session_file);
        SessionManager::load(&resolved)?
    } else {
        let cwd_text = cwd
            .to_str()
            .ok_or_else(|| format!("cwd is not valid UTF-8: {}", cwd.display()))?;
        SessionManager::create(cwd_text, session_dir)?
    };

    let tools = if args.no_tools {
        vec![]
    } else {
        create_coding_tools(cwd)
    };

    let runtime_api_key = runtime.api_key.clone();
    let stream_fn = Arc::new(
        move |model: Model, context: pixy_ai::Context, options: Option<SimpleStreamOptions>| {
            let mut resolved_options = options.unwrap_or_default();
            if resolved_options.stream.api_key.is_none() {
                resolved_options.stream.api_key = runtime_api_key.clone();
            }
            pixy_ai::stream_simple(model, context, Some(resolved_options))
        },
    );

    let config = AgentSessionConfig {
        model: runtime.model.clone(),
        system_prompt: build_system_prompt(args.system_prompt.as_deref(), cwd, &tools, skills),
        stream_fn,
        tools,
    };
    let mut session = AgentSession::new(session_manager, config);
    session.set_model_catalog(runtime.model_catalog.clone());
    Ok(session)
}

fn load_runtime_skills(
    args: &ChatArgs,
    local: &AgentLocalConfig,
    cwd: &Path,
    agent_dir: &Path,
) -> Vec<Skill> {
    let mut options = LoadSkillsOptions::new(cwd.to_path_buf(), agent_dir.to_path_buf());
    options.include_defaults = !args.no_skills;
    options.skill_paths = local.settings.skills.clone();
    options.skill_paths.extend(args.skills.clone());

    let result = load_skills(options);
    for diagnostic in &result.diagnostics {
        eprintln!(
            "warning: skill {}: {}",
            diagnostic.path.display(),
            diagnostic.message
        );
    }
    result.skills
}

async fn repl_loop(session: &mut AgentSession, show_tool_results: bool) -> Result<(), String> {
    loop {
        print!("pixy> ");
        io::stdout()
            .flush()
            .map_err(|error| format!("stdout flush failed: {error}"))?;

        let mut line = String::new();
        let read = io::stdin()
            .read_line(&mut line)
            .map_err(|error| format!("stdin read failed: {error}"))?;
        if read == 0 {
            println!();
            return Ok(());
        }

        let input = line.trim();
        if input.is_empty() {
            continue;
        }

        if input.starts_with("/resume") {
            let explicit_target = input
                .strip_prefix("/resume")
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned);
            let target = if let Some(target) = explicit_target {
                Some(target)
            } else {
                match prompt_resume_target_selection(session, RESUME_PICKER_LIMIT) {
                    Ok(Some(path)) => Some(path.display().to_string()),
                    Ok(None) => {
                        println!("resume cancelled");
                        continue;
                    }
                    Err(error) => {
                        eprintln!("resume failed: {error}");
                        continue;
                    }
                }
            };

            match session.resume(target.as_deref()) {
                Ok(path) => println!("resumed: {}", path.display()),
                Err(error) => eprintln!("resume failed: {error}"),
            }
            continue;
        }

        match input {
            "/exit" | "/quit" => return Ok(()),
            "/help" => {
                println!("commands:");
                println!(
                    "  /continue  continue from current context without adding a user message"
                );
                println!(
                    "  /resume [session]  choose from recent sessions (or pass a session file directly)"
                );
                println!("  /session   print current session file path");
                println!("  /help      show this help");
                println!("  /exit      quit");
            }
            "/session" => {
                if let Some(path) = session.session_file() {
                    println!("{}", path.display());
                } else {
                    println!("(no session file)");
                }
            }
            "/continue" => {
                if let Err(error) = run_continue_streaming_cli(session, show_tool_results).await {
                    eprintln!("continue failed: {error}");
                }
            }
            _ => {
                if let Err(error) =
                    run_prompt_streaming_cli(session, input, show_tool_results).await
                {
                    eprintln!("prompt failed: {error}");
                }
            }
        }
    }
}

fn prompt_resume_target_selection(
    session: &AgentSession,
    limit: usize,
) -> Result<Option<PathBuf>, String> {
    let candidates = session.recent_resumable_sessions(limit)?;
    if candidates.is_empty() {
        return Err("No historical sessions available to resume".to_string());
    }

    println!("recent sessions:");
    for (index, candidate) in candidates.iter().enumerate() {
        println!(
            "  {}. {}",
            index + 1,
            format_resume_candidate_name(candidate)
        );
    }
    println!(
        "  0. latest ({})",
        format_resume_candidate_name(&candidates[0])
    );

    loop {
        print!(
            "select session [0-{}, Enter=0, q=cancel]: ",
            candidates.len()
        );
        io::stdout()
            .flush()
            .map_err(|error| format!("stdout flush failed: {error}"))?;

        let mut selection = String::new();
        io::stdin()
            .read_line(&mut selection)
            .map_err(|error| format!("stdin read failed: {error}"))?;

        match resolve_resume_picker_selection(&candidates, &selection) {
            Ok(value) => return Ok(value),
            Err(error) => eprintln!("{error}"),
        }
    }
}

fn format_resume_candidate_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| path.display().to_string())
}

fn resolve_resume_picker_selection(
    candidates: &[PathBuf],
    raw_input: &str,
) -> Result<Option<PathBuf>, String> {
    if candidates.is_empty() {
        return Err("No historical sessions available to resume".to_string());
    }

    let trimmed = raw_input.trim();
    if trimmed.is_empty() || trimmed == "0" {
        return Ok(Some(candidates[0].clone()));
    }

    let normalized = trimmed.to_ascii_lowercase();
    if matches!(normalized.as_str(), "q" | "quit" | "cancel") {
        return Ok(None);
    }

    let choice = trimmed.parse::<usize>().map_err(|_| {
        format!(
            "Invalid selection '{trimmed}'. Enter a number between 0 and {}, or q to cancel.",
            candidates.len()
        )
    })?;
    if choice == 0 || choice > candidates.len() {
        return Err(format!(
            "Invalid selection '{trimmed}'. Enter a number between 1 and {}, or q to cancel.",
            candidates.len()
        ));
    }

    Ok(Some(candidates[choice - 1].clone()))
}

async fn run_prompt_streaming_cli(
    session: &mut AgentSession,
    input: &str,
    show_tool_results: bool,
) -> Result<(), String> {
    let mut renderer = CliStreamRenderer::new(io::stdout(), show_tool_results);
    let mut render_error: Option<String> = None;
    let produced = session
        .prompt_streaming(input, |update| {
            if render_error.is_none() {
                if let Err(error) = renderer.on_update(update) {
                    render_error = Some(error);
                }
            }
        })
        .await?;

    if let Some(error) = render_error {
        return Err(error);
    }
    renderer.finish()?;

    if !renderer.saw_updates() {
        render_messages(&produced, show_tool_results);
    }
    Ok(())
}

async fn run_continue_streaming_cli(
    session: &mut AgentSession,
    show_tool_results: bool,
) -> Result<(), String> {
    let mut renderer = CliStreamRenderer::new(io::stdout(), show_tool_results);
    let mut render_error: Option<String> = None;
    let produced = session
        .continue_run_streaming(|update| {
            if render_error.is_none() {
                if let Err(error) = renderer.on_update(update) {
                    render_error = Some(error);
                }
            }
        })
        .await?;

    if let Some(error) = render_error {
        return Err(error);
    }
    renderer.finish()?;

    if !renderer.saw_updates() {
        render_messages(&produced, show_tool_results);
    }
    Ok(())
}

struct CliStreamRenderer<W: Write> {
    writer: W,
    show_tool_results: bool,
    saw_updates: bool,
    assistant_delta_open: bool,
    thinking_line_open: bool,
    thinking_visual_lines: usize,
}

impl<W: Write> CliStreamRenderer<W> {
    fn new(writer: W, show_tool_results: bool) -> Self {
        Self {
            writer,
            show_tool_results,
            saw_updates: false,
            assistant_delta_open: false,
            thinking_line_open: false,
            thinking_visual_lines: 0,
        }
    }

    fn saw_updates(&self) -> bool {
        self.saw_updates
    }

    fn on_update(&mut self, update: AgentSessionStreamUpdate) -> Result<(), String> {
        self.saw_updates = true;
        match update {
            AgentSessionStreamUpdate::AssistantTextDelta(delta) => {
                if delta.is_empty() {
                    return Ok(());
                }
                if self.thinking_line_open {
                    writeln!(self.writer)
                        .map_err(|error| format!("stdout write failed: {error}"))?;
                    self.thinking_line_open = false;
                }
                write!(self.writer, "{delta}")
                    .and_then(|_| self.writer.flush())
                    .map_err(|error| format!("stdout write failed: {error}"))?;
                self.assistant_delta_open = true;
            }
            AgentSessionStreamUpdate::AssistantLine(line) => {
                if is_thinking_line(&line) {
                    if self.assistant_delta_open {
                        writeln!(self.writer)
                            .map_err(|error| format!("stdout write failed: {error}"))?;
                        self.assistant_delta_open = false;
                    }
                    if self.thinking_line_open {
                        self.clear_thinking_block()?;
                    }
                    if !line.is_empty() {
                        write!(self.writer, "{line}")
                            .and_then(|_| self.writer.flush())
                            .map_err(|error| format!("stdout write failed: {error}"))?;
                    }
                    if !line.is_empty() {
                        self.thinking_visual_lines = estimate_visual_line_count(&line);
                        self.thinking_line_open = true;
                    } else {
                        self.thinking_visual_lines = 0;
                        self.thinking_line_open = false;
                    }
                    return Ok(());
                }

                if self.assistant_delta_open || self.thinking_line_open {
                    writeln!(self.writer)
                        .map_err(|error| format!("stdout write failed: {error}"))?;
                    self.assistant_delta_open = false;
                    self.thinking_line_open = false;
                    self.thinking_visual_lines = 0;
                }
                if !line.is_empty() {
                    writeln!(self.writer, "{line}")
                        .map_err(|error| format!("stdout write failed: {error}"))?;
                }
            }
            AgentSessionStreamUpdate::ToolLine(line) => {
                if !self.show_tool_results {
                    return Ok(());
                }
                if self.assistant_delta_open || self.thinking_line_open {
                    writeln!(self.writer)
                        .map_err(|error| format!("stdout write failed: {error}"))?;
                    self.assistant_delta_open = false;
                    self.thinking_line_open = false;
                    self.thinking_visual_lines = 0;
                }
                writeln!(self.writer, "{line}")
                    .map_err(|error| format!("stdout write failed: {error}"))?;
            }
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<(), String> {
        if self.assistant_delta_open || self.thinking_line_open {
            writeln!(self.writer).map_err(|error| format!("stdout write failed: {error}"))?;
            self.assistant_delta_open = false;
            self.thinking_line_open = false;
            self.thinking_visual_lines = 0;
        }
        Ok(())
    }

    fn clear_thinking_block(&mut self) -> Result<(), String> {
        let lines = self.thinking_visual_lines.max(1);
        for idx in 0..lines {
            write!(self.writer, "\r\x1b[2K")
                .map_err(|error| format!("stdout write failed: {error}"))?;
            if idx + 1 < lines {
                write!(self.writer, "\x1b[1A")
                    .map_err(|error| format!("stdout write failed: {error}"))?;
            }
        }
        Ok(())
    }

    #[cfg(test)]
    fn into_inner(self) -> W {
        self.writer
    }
}

fn is_thinking_line(line: &str) -> bool {
    line.starts_with("[thinking]")
}

fn estimate_visual_line_count(text: &str) -> usize {
    let width = terminal_columns().max(1) as usize;
    let mut total = 0usize;
    for segment in text.split('\n') {
        let chars = segment.chars().count();
        let rows = if chars == 0 {
            1
        } else {
            ((chars - 1) / width) + 1
        };
        total += rows;
    }
    total.max(1)
}

fn terminal_columns() -> u16 {
    std::env::var("COLUMNS")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(80)
}

fn render_messages(messages: &[Message], show_tool_results: bool) {
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
                                println!("{text}");
                            }
                        }
                        AssistantContentBlock::Thinking { thinking, .. } => {
                            if !thinking.trim().is_empty() {
                                println!("[thinking] {thinking}");
                            }
                        }
                        AssistantContentBlock::ToolCall { .. } => {}
                    }
                }

                if matches!(stop_reason, StopReason::Error | StopReason::Aborted) {
                    if let Some(error_message) = error_message {
                        eprintln!(
                            "[assistant_{}] {error_message}",
                            stop_reason_label(stop_reason)
                        );
                    }
                }
            }
            Message::ToolResult {
                tool_name,
                content,
                is_error,
                ..
            } => {
                if !show_tool_results {
                    continue;
                }
                let status = if *is_error { "error" } else { "ok" };
                println!("[tool:{tool_name}:{status}]");
                for block in content {
                    match block {
                        ToolResultContentBlock::Text { text, .. } => println!("{text}"),
                        ToolResultContentBlock::Image { .. } => {
                            println!("(image tool result omitted)")
                        }
                    }
                }
            }
            Message::User { .. } => {}
        }
    }
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

fn resolve_path(cwd: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    }
}

fn default_pixy_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".pixy")
}

fn default_pixy_config_path() -> PathBuf {
    default_pixy_dir().join("pixy.toml")
}

fn default_agent_dir() -> PathBuf {
    default_pixy_dir().join("agent")
}

fn resolve_tui_theme_name(
    cli_theme: Option<&str>,
    settings_theme: Option<&str>,
) -> Result<String, String> {
    let selected = first_non_empty([
        cli_theme.map(str::to_string),
        settings_theme.map(str::to_string),
        Some("dark".to_string()),
    ])
    .expect("default theme candidate is always available");
    let normalized = selected.trim().to_ascii_lowercase();
    if TuiTheme::from_name(normalized.as_str()).is_some() {
        Ok(normalized)
    } else {
        Err(format!(
            "unsupported theme '{selected}', expected dark or light"
        ))
    }
}

fn load_tui_keybindings(agent_dir: &Path) -> Option<TuiKeyBindings> {
    let config_path = agent_dir.join("keybindings.json");
    let content = std::fs::read_to_string(config_path).ok()?;
    let parsed = serde_json::from_str::<serde_json::Value>(&content).ok()?;
    let object = parsed.as_object()?;

    let mut keybindings = TuiKeyBindings::default();
    let mut changed = false;

    if let Some(bindings) = object.get("clear").and_then(parse_keybinding_values) {
        keybindings.clear = bindings;
        changed = true;
    }
    if let Some(bindings) = object.get("exit").and_then(parse_keybinding_values) {
        keybindings.quit = bindings;
        changed = true;
    }
    if let Some(bindings) = object.get("interrupt").and_then(parse_keybinding_values) {
        keybindings.interrupt = bindings;
        changed = true;
    }
    if let Some(bindings) = object
        .get("cycleThinkingLevel")
        .and_then(parse_keybinding_values)
    {
        keybindings.cycle_thinking_level = bindings;
        changed = true;
    }
    if let Some(bindings) = object.get("expandTools").and_then(parse_keybinding_values) {
        keybindings.expand_tools = bindings;
        changed = true;
    }
    if let Some(bindings) = object
        .get("cycleModelForward")
        .and_then(parse_keybinding_values)
    {
        keybindings.cycle_model_forward = bindings;
        changed = true;
    }
    if let Some(bindings) = object
        .get("cycleModelBackward")
        .and_then(parse_keybinding_values)
    {
        keybindings.cycle_model_backward = bindings;
        changed = true;
    }
    if let Some(bindings) = object.get("selectModel").and_then(parse_keybinding_values) {
        keybindings.select_model = bindings;
        changed = true;
    }
    if let Some(bindings) = object
        .get("toggleThinking")
        .and_then(parse_keybinding_values)
    {
        keybindings.toggle_thinking = bindings;
        changed = true;
    }
    if let Some(bindings) = object.get("followUp").and_then(parse_keybinding_values) {
        keybindings.continue_run = bindings;
        changed = true;
    }
    if let Some(bindings) = object.get("dequeue").and_then(parse_keybinding_values) {
        keybindings.dequeue = bindings;
        changed = true;
    }
    if let Some(bindings) = object.get("newline").and_then(parse_keybinding_values) {
        keybindings.newline = bindings;
        changed = true;
    }

    if changed { Some(keybindings) } else { None }
}

fn parse_keybinding_values(value: &serde_json::Value) -> Option<Vec<KeyBinding>> {
    match value {
        serde_json::Value::String(key_id) => parse_key_id(key_id).map(|binding| vec![binding]),
        serde_json::Value::Array(values) => {
            let bindings = values
                .iter()
                .filter_map(|item| match item {
                    serde_json::Value::String(key_id) => parse_key_id(key_id),
                    _ => None,
                })
                .collect::<Vec<_>>();
            Some(bindings)
        }
        _ => None,
    }
}

#[derive(Debug, Clone, Default)]
struct AgentSettingsFile {
    default_provider: Option<String>,
    theme: Option<String>,
    transport_retry_count: Option<usize>,
    skills: Vec<String>,
    env: HashMap<String, String>,
}

#[derive(Debug, Clone, Default)]
struct ModelsFile {
    providers: HashMap<String, ProviderConfig>,
}

#[derive(Debug, Clone, Default)]
struct ProviderConfig {
    provider: Option<String>,
    kind: Option<String>,
    api: Option<String>,
    base_url: Option<String>,
    api_key: Option<String>,
    default_model: Option<String>,
    weight: u8,
    models: Vec<ProviderModelConfig>,
}

fn default_provider_weight() -> u8 {
    1
}

#[derive(Debug, Clone, Default, Deserialize)]
struct ProviderModelConfig {
    id: String,
    api: Option<String>,
    #[serde(rename = "baseUrl")]
    base_url: Option<String>,
    #[serde(rename = "reasoning", default)]
    reasoning: Option<bool>,
    #[serde(rename = "reasoningEffort", default)]
    reasoning_effort: Option<pixy_ai::ThinkingLevel>,
    #[serde(rename = "contextWindow")]
    context_window: Option<u32>,
    #[serde(rename = "maxTokens")]
    max_tokens: Option<u32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct PixyTomlFile {
    #[serde(default)]
    llm: PixyTomlLlm,
    theme: Option<String>,
    #[serde(default)]
    transport_retry_count: Option<usize>,
    #[serde(default)]
    skills: Vec<String>,
    #[serde(default)]
    env: HashMap<String, String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct PixyTomlLlm {
    #[serde(default)]
    default_provider: Option<String>,
    #[serde(default)]
    providers: Vec<PixyTomlProvider>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct PixyTomlProvider {
    name: String,
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    provider: Option<String>,
    #[serde(default)]
    api: Option<String>,
    #[serde(default)]
    base_url: Option<String>,
    #[serde(default)]
    api_key: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default = "default_provider_weight")]
    weight: u8,
    #[serde(default)]
    reasoning: Option<bool>,
    #[serde(default)]
    reasoning_effort: Option<pixy_ai::ThinkingLevel>,
    #[serde(default)]
    context_window: Option<u32>,
    #[serde(default)]
    max_tokens: Option<u32>,
}

#[derive(Debug, Clone, Default)]
struct AgentLocalConfig {
    settings: AgentSettingsFile,
    models: ModelsFile,
}

#[derive(Debug, Clone)]
struct ResolvedRuntimeConfig {
    model: Model,
    model_catalog: Vec<Model>,
    api_key: Option<String>,
}

#[derive(Debug, Clone)]
pub struct LLMRouter {
    slots: Vec<Slot>,
}

#[derive(Debug, Clone)]
struct Slot {
    provider: String,
    weight: u8,
}

impl LLMRouter {
    fn from_provider_configs(providers: &HashMap<String, ProviderConfig>) -> Result<Self, String> {
        let mut slots = Vec::with_capacity(providers.len());
        for (provider, config) in providers {
            if !is_chat_provider(config) {
                continue;
            }
            if config.weight >= 100 {
                return Err(format!(
                    "Provider '{provider}' has invalid weight {}, expected value < 100",
                    config.weight
                ));
            }
            slots.push(Slot {
                provider: provider.clone(),
                weight: config.weight,
            });
        }
        slots.sort_by(|left, right| left.provider.cmp(&right.provider));
        Ok(Self { slots })
    }

    fn select_provider(&self, seed: u64) -> Option<String> {
        let total_weight: u64 = self.slots.iter().map(|slot| slot.weight as u64).sum();
        if total_weight == 0 {
            return None;
        }

        let mut cursor = seed % total_weight;
        for slot in &self.slots {
            let weight = slot.weight as u64;
            if weight == 0 {
                continue;
            }
            if cursor < weight {
                return Some(slot.provider.clone());
            }
            cursor -= weight;
        }
        None
    }
}

fn load_agent_local_config(_agent_dir: &Path) -> Result<AgentLocalConfig, String> {
    let config_path = default_pixy_config_path();
    let config = read_toml_if_exists::<PixyTomlFile>(&config_path)?;
    Ok(config
        .map(convert_pixy_toml_to_local_config)
        .unwrap_or_default())
}

fn convert_pixy_toml_to_local_config(config: PixyTomlFile) -> AgentLocalConfig {
    let mut providers = HashMap::new();
    for provider in config.llm.providers {
        if provider.name.trim().is_empty() {
            continue;
        }

        let provider_key = provider.name.trim().to_string();
        let model_id = provider.model.clone().map(|value| value.trim().to_string());
        let provider_model =
            model_id
                .clone()
                .filter(|id| !id.is_empty())
                .map(|id| ProviderModelConfig {
                    id,
                    api: provider.api.clone(),
                    base_url: provider.base_url.clone(),
                    reasoning: provider.reasoning,
                    reasoning_effort: provider.reasoning_effort.clone(),
                    context_window: provider.context_window,
                    max_tokens: provider.max_tokens,
                });

        let provider_config = ProviderConfig {
            provider: provider
                .provider
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string),
            kind: provider.kind,
            api: provider.api,
            base_url: provider.base_url,
            api_key: provider.api_key,
            default_model: model_id.filter(|id| !id.is_empty()),
            weight: provider.weight,
            models: provider_model.into_iter().collect(),
        };
        providers.insert(provider_key, provider_config);
    }

    AgentLocalConfig {
        settings: AgentSettingsFile {
            default_provider: config.llm.default_provider,
            theme: config.theme,
            transport_retry_count: config.transport_retry_count,
            skills: config.skills,
            env: config.env,
        },
        models: ModelsFile { providers },
    }
}

fn resolve_transport_retry_count(settings: &AgentSettingsFile) -> usize {
    settings
        .transport_retry_count
        .unwrap_or(DEFAULT_TRANSPORT_RETRY_COUNT)
}

fn read_toml_if_exists<T>(path: &Path) -> Result<Option<T>, String>
where
    T: for<'de> Deserialize<'de>,
{
    if !path.exists() {
        return Ok(None);
    }

    let content = std::fs::read_to_string(path)
        .map_err(|error| format!("read {} failed: {error}", path.display()))?;
    let parsed = toml::from_str::<T>(&content)
        .map_err(|error| format!("parse {} failed: {error}", path.display()))?;
    Ok(Some(parsed))
}

fn resolve_runtime_config(
    args: &ChatArgs,
    local: &AgentLocalConfig,
) -> Result<ResolvedRuntimeConfig, String> {
    resolve_runtime_config_with_seed(args, local, runtime_router_seed())
}

fn runtime_router_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0)
}

fn resolve_runtime_config_with_seed(
    args: &ChatArgs,
    local: &AgentLocalConfig,
    router_seed: u64,
) -> Result<ResolvedRuntimeConfig, String> {
    let cli_model_parts = split_provider_model(args.model.as_deref());
    let explicit_provider = first_non_empty([args.provider.clone(), cli_model_parts.0.clone()]);
    let routed_provider = if explicit_provider.is_none() {
        resolve_weighted_provider_selection(local, router_seed)?
    } else {
        None
    };
    let settings_provider = local
        .settings
        .default_provider
        .clone()
        .filter(|value| value.trim() != "*");

    let provider = first_non_empty([
        explicit_provider,
        routed_provider,
        settings_provider,
        infer_single_provider(local),
        Some("openai".to_string()),
    ])
    .ok_or_else(|| "Unable to resolve provider".to_string())?;

    let provider_config = local.models.providers.get(&provider);
    if let Some(config) = provider_config {
        if config.weight >= 100 {
            return Err(format!(
                "Provider '{provider}' has invalid weight {}, expected value < 100",
                config.weight
            ));
        }
        if !is_chat_provider(config) {
            let kind = config.kind.as_deref().unwrap_or("unknown");
            return Err(format!(
                "Provider '{provider}' has kind '{kind}', expected 'chat' for coding sessions"
            ));
        }
    }
    let provider_name = resolve_provider_name(&provider, provider_config);
    let model_id = first_non_empty([
        cli_model_parts.1.clone(),
        provider_config_default_model(provider_config),
        default_model_for_provider(&provider_name),
    ])
    .ok_or_else(|| format!("Unable to resolve model for provider '{provider}'"))?;

    let selected_model_cfg = provider_config.and_then(|provider_config| {
        provider_config
            .models
            .iter()
            .find(|model| model.id == model_id)
    });

    let api = first_non_empty([
        args.api.clone(),
        selected_model_cfg.and_then(|model| model.api.clone()),
        provider_config.and_then(|provider| provider.api.clone()),
        infer_api_for_provider(&provider_name),
    ])
    .ok_or_else(|| format!("Unable to resolve API for provider '{provider}'"))?;

    let cli_base_url = args
        .base_url
        .as_ref()
        .and_then(|value| resolve_config_value(value, &local.settings.env));
    let model_base_url = selected_model_cfg
        .and_then(|model| model.base_url.as_ref())
        .and_then(|value| resolve_config_value(value, &local.settings.env));
    let provider_base_url = provider_config
        .and_then(|provider| provider.base_url.as_ref())
        .and_then(|value| resolve_config_value(value, &local.settings.env));
    let base_url = first_non_empty([
        cli_base_url,
        model_base_url,
        provider_base_url,
        default_base_url_for_api(&api),
    ])
    .ok_or_else(|| format!("Unable to resolve base URL for api '{api}'"))?;

    let context_window = args
        .context_window
        .or_else(|| selected_model_cfg.and_then(|model| model.context_window))
        .unwrap_or(200_000);
    let max_tokens = args
        .max_tokens
        .or_else(|| selected_model_cfg.and_then(|model| model.max_tokens))
        .unwrap_or(8_192);

    let api_key = provider_config
        .and_then(|provider_cfg| provider_cfg.api_key.as_ref())
        .and_then(|value| resolve_config_value(value, &local.settings.env))
        .or_else(|| infer_api_key_from_settings(&provider_name, &local.settings.env))
        .or_else(|| std::env::var(primary_env_key_for_provider(&provider_name)).ok());

    let reasoning = selected_model_cfg
        .and_then(|cfg| cfg.reasoning)
        .unwrap_or_else(|| default_reasoning_enabled_for_api(&api));
    let reasoning_effort = selected_model_cfg
        .and_then(|cfg| cfg.reasoning_effort.clone())
        .or_else(|| {
            if reasoning {
                Some(pixy_ai::ThinkingLevel::Medium)
            } else {
                None
            }
        });

    let model = Model {
        id: model_id.clone(),
        name: model_id,
        api,
        provider: provider_name.clone(),
        base_url,
        reasoning,
        reasoning_effort,
        input: vec!["text".to_string()],
        cost: Cost {
            input: 0.0,
            output: 0.0,
            cache_read: 0.0,
            cache_write: 0.0,
            total: 0.0,
        },
        context_window,
        max_tokens,
    };

    let mut model_catalog = provider_config
        .map(|provider_cfg| {
            provider_cfg
                .models
                .iter()
                .map(|entry| {
                    model_from_config(
                        entry,
                        &provider_name,
                        &model.api,
                        &model.base_url,
                        model.context_window,
                        model.max_tokens,
                    )
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    dedupe_models(&mut model_catalog);

    if let Some(position) = model_catalog
        .iter()
        .position(|entry| entry.provider == model.provider && entry.id == model.id)
    {
        model_catalog.remove(position);
    }
    model_catalog.insert(0, model.clone());

    Ok(ResolvedRuntimeConfig {
        model,
        model_catalog,
        api_key,
    })
}

fn resolve_weighted_provider_selection(
    local: &AgentLocalConfig,
    router_seed: u64,
) -> Result<Option<String>, String> {
    let use_weighted_provider = local
        .settings
        .default_provider
        .as_deref()
        .map(str::trim)
        .is_some_and(|provider| provider == "*");
    if !use_weighted_provider {
        return Ok(None);
    }

    let router = LLMRouter::from_provider_configs(&local.models.providers)?;
    let provider = router.select_provider(router_seed).ok_or_else(|| {
        "default_provider='*' requires at least one chat provider with non-zero weight in ~/.pixy/pixy.toml".to_string()
    })?;
    Ok(Some(provider))
}

fn provider_config_default_model(provider_config: Option<&ProviderConfig>) -> Option<String> {
    provider_config.and_then(|config| {
        first_non_empty([
            config.default_model.clone(),
            config.models.first().map(|entry| entry.id.clone()),
        ])
    })
}

fn resolve_provider_name(provider_key: &str, provider_config: Option<&ProviderConfig>) -> String {
    provider_config
        .and_then(|config| config.provider.clone())
        .and_then(|value| {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        })
        .unwrap_or_else(|| provider_key.to_string())
}

fn infer_single_provider(local: &AgentLocalConfig) -> Option<String> {
    let chat_providers = local
        .models
        .providers
        .iter()
        .filter(|(_, config)| is_chat_provider(config))
        .map(|(provider, _)| provider.clone())
        .collect::<Vec<_>>();
    if chat_providers.len() == 1 {
        chat_providers.into_iter().next()
    } else {
        None
    }
}

fn is_chat_provider(config: &ProviderConfig) -> bool {
    config
        .kind
        .as_deref()
        .map(str::trim)
        .map_or(true, |kind| kind.eq_ignore_ascii_case("chat"))
}

fn model_from_config(
    config: &ProviderModelConfig,
    provider: &str,
    default_api: &str,
    default_base_url: &str,
    default_context_window: u32,
    default_max_tokens: u32,
) -> Model {
    let api = config
        .api
        .clone()
        .unwrap_or_else(|| default_api.to_string());
    let base_url = config
        .base_url
        .clone()
        .or_else(|| default_base_url_for_api(&api))
        .unwrap_or_else(|| default_base_url.to_string());
    let reasoning = config
        .reasoning
        .unwrap_or_else(|| default_reasoning_enabled_for_api(&api));
    let reasoning_effort = config.reasoning_effort.clone().or_else(|| {
        if reasoning {
            Some(pixy_ai::ThinkingLevel::Medium)
        } else {
            None
        }
    });

    Model {
        id: config.id.clone(),
        name: config.id.clone(),
        api,
        provider: provider.to_string(),
        base_url,
        reasoning,
        reasoning_effort,
        input: vec!["text".to_string()],
        cost: Cost {
            input: 0.0,
            output: 0.0,
            cache_read: 0.0,
            cache_write: 0.0,
            total: 0.0,
        },
        context_window: config.context_window.unwrap_or(default_context_window),
        max_tokens: config.max_tokens.unwrap_or(default_max_tokens),
    }
}

fn default_reasoning_enabled_for_api(api: &str) -> bool {
    matches!(
        api,
        "openai-responses"
            | "openai-completions"
            | "openai-codex-responses"
            | "azure-openai-responses"
    )
}

fn dedupe_models(models: &mut Vec<Model>) {
    let mut deduped = Vec::new();
    for model in std::mem::take(models) {
        if deduped
            .iter()
            .any(|existing: &Model| existing.provider == model.provider && existing.id == model.id)
        {
            continue;
        }
        deduped.push(model);
    }
    *models = deduped;
}

fn split_provider_model(input: Option<&str>) -> (Option<String>, Option<String>) {
    let Some(raw) = input.map(str::trim).filter(|value| !value.is_empty()) else {
        return (None, None);
    };

    if let Some((provider, model)) = raw.split_once('/') {
        if !provider.is_empty() && !model.is_empty() {
            return (Some(provider.to_string()), Some(model.to_string()));
        }
    }

    (None, Some(raw.to_string()))
}

fn first_non_empty<const N: usize>(candidates: [Option<String>; N]) -> Option<String> {
    for candidate in candidates {
        if let Some(value) = candidate {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

fn infer_api_for_provider(provider: &str) -> Option<String> {
    match provider {
        "openai" => Some("openai-responses".to_string()),
        "openai-completions" => Some("openai-completions".to_string()),
        "openai-responses" => Some("openai-responses".to_string()),
        "openai-codex-responses" | "codex" => Some("openai-codex-responses".to_string()),
        "azure-openai" | "azure-openai-responses" => Some("azure-openai-responses".to_string()),
        "anthropic" | "anthropic-messages" => Some("anthropic-messages".to_string()),
        "google" | "google-generative-ai" => Some("google-generative-ai".to_string()),
        "google-gemini-cli" => Some("google-gemini-cli".to_string()),
        "google-vertex" => Some("google-vertex".to_string()),
        "bedrock" | "amazon-bedrock" | "bedrock-converse-stream" => {
            Some("bedrock-converse-stream".to_string())
        }
        _ => None,
    }
}

fn default_model_for_provider(provider: &str) -> Option<String> {
    match provider {
        "openai" | "openai-completions" => Some("gpt-5.3-codex".to_string()),
        "openai-responses" | "azure-openai" | "azure-openai-responses" => {
            Some("gpt-5.3-codex".to_string())
        }
        "openai-codex-responses" | "codex" => Some("codex-mini-latest".to_string()),
        "anthropic" | "anthropic-messages" => Some("claude-3-5-sonnet-latest".to_string()),
        "google" | "google-generative-ai" | "google-gemini-cli" | "google-vertex" => {
            Some("gemini-2.5-flash".to_string())
        }
        "bedrock" | "amazon-bedrock" | "bedrock-converse-stream" => {
            Some("anthropic.claude-3-5-sonnet-20241022-v2:0".to_string())
        }
        _ => None,
    }
}

fn default_base_url_for_api(api: &str) -> Option<String> {
    match api {
        "openai-completions" | "openai-responses" | "openai-codex-responses" => {
            Some("https://api.openai.com/v1".to_string())
        }
        "anthropic-messages" => Some("https://api.anthropic.com/v1".to_string()),
        "google-generative-ai" => {
            Some("https://generativelanguage.googleapis.com/v1beta".to_string())
        }
        _ => None,
    }
}

fn primary_env_key_for_provider(provider: &str) -> &'static str {
    match provider {
        "anthropic" | "anthropic-messages" => "ANTHROPIC_API_KEY",
        "openai"
        | "openai-completions"
        | "openai-responses"
        | "openai-codex-responses"
        | "azure-openai"
        | "azure-openai-responses"
        | "codex" => "OPENAI_API_KEY",
        "google" | "google-generative-ai" | "google-gemini-cli" | "google-vertex" => {
            "GOOGLE_API_KEY"
        }
        "bedrock" | "amazon-bedrock" | "bedrock-converse-stream" => "AWS_ACCESS_KEY_ID",
        _ => "OPENAI_API_KEY",
    }
}

fn infer_api_key_from_settings(
    provider: &str,
    env_map: &HashMap<String, String>,
) -> Option<String> {
    let provider_upper = provider.to_uppercase().replace('-', "_");
    let provider_api_key = format!("{provider_upper}_API_KEY");
    let provider_auth_token = format!("{provider_upper}_AUTH_TOKEN");

    [
        env_map.get(&provider_api_key).cloned(),
        env_map.get(&provider_auth_token).cloned(),
        env_map.get("OPENAI_API_KEY").cloned(),
        env_map.get("ANTHROPIC_API_KEY").cloned(),
        env_map.get("ANTHROPIC_AUTH_TOKEN").cloned(),
        env_map.get("GOOGLE_API_KEY").cloned(),
    ]
    .into_iter()
    .flatten()
    .find(|value| !value.trim().is_empty())
}

fn resolve_config_value(value: &str, env_map: &HashMap<String, String>) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Some(env_key) = trimmed.strip_prefix('$') {
        return env_map
            .get(env_key)
            .cloned()
            .or_else(|| std::env::var(env_key).ok())
            .filter(|resolved| !resolved.trim().is_empty());
    }
    Some(trimmed.to_string())
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use tempfile::tempdir;

    fn test_chat_args() -> ChatArgs {
        ChatArgs {
            api: None,
            provider: None,
            model: None,
            base_url: None,
            context_window: None,
            max_tokens: None,
            agent_dir: None,
            cwd: None,
            session_dir: None,
            session_file: None,
            system_prompt: Some("test".to_string()),
            prompt: None,
            continue_first: false,
            no_tools: false,
            skills: vec![],
            no_skills: false,
            hide_tool_results: false,
            no_tui: false,
            theme: None,
        }
    }

    #[test]
    fn resolves_default_provider_model_from_settings_and_models() {
        let args = ChatArgs {
            api: None,
            provider: None,
            model: None,
            base_url: None,
            context_window: None,
            max_tokens: None,
            agent_dir: None,
            cwd: None,
            session_dir: None,
            session_file: None,
            system_prompt: Some("test".to_string()),
            prompt: None,
            continue_first: false,
            no_tools: false,
            skills: vec![],
            no_skills: false,
            hide_tool_results: false,
            no_tui: false,
            theme: None,
        };

        let local = AgentLocalConfig {
            settings: AgentSettingsFile {
                default_provider: Some("anthropic".to_string()),
                theme: None,
                transport_retry_count: None,
                skills: vec![],
                env: HashMap::from([(
                    "ANTHROPIC_AUTH_TOKEN".to_string(),
                    "token-from-settings".to_string(),
                )]),
            },
            models: ModelsFile {
                providers: HashMap::from([(
                    "anthropic".to_string(),
                    ProviderConfig {
                        provider: None,
                        kind: None,
                        api: None,
                        base_url: Some("https://custom.anthropic.local".to_string()),
                        api_key: Some("model-json-token".to_string()),
                        default_model: Some("claude-opus-4-6".to_string()),
                        weight: 1,
                        models: vec![],
                    },
                )]),
            },
        };

        let resolved =
            resolve_runtime_config(&args, &local).expect("runtime config should resolve");
        assert_eq!(resolved.model.provider, "anthropic");
        assert_eq!(resolved.model.api, "anthropic-messages");
        assert_eq!(resolved.model.id, "claude-opus-4-6");
        assert_eq!(resolved.model.base_url, "https://custom.anthropic.local");
        assert_eq!(resolved.api_key.as_deref(), Some("model-json-token"));
    }

    #[test]
    fn wildcard_default_provider_selects_provider_by_weight() {
        let args = test_chat_args();
        let local = AgentLocalConfig {
            settings: AgentSettingsFile {
                default_provider: Some("*".to_string()),
                ..AgentSettingsFile::default()
            },
            models: ModelsFile {
                providers: HashMap::from([
                    (
                        "openai".to_string(),
                        ProviderConfig {
                            provider: None,
                            kind: None,
                            api: Some("openai-responses".to_string()),
                            base_url: Some("https://api.openai.com/v1".to_string()),
                            api_key: None,
                            default_model: Some("gpt-5.3-codex".to_string()),
                            weight: 80,
                            models: vec![],
                        },
                    ),
                    (
                        "anthropic".to_string(),
                        ProviderConfig {
                            provider: None,
                            kind: None,
                            api: Some("anthropic-messages".to_string()),
                            base_url: Some("https://api.anthropic.com/v1".to_string()),
                            api_key: None,
                            default_model: Some("claude-3-5-sonnet-latest".to_string()),
                            weight: 20,
                            models: vec![],
                        },
                    ),
                ]),
            },
        };

        let weighted_anthropic = resolve_runtime_config_with_seed(&args, &local, 10)
            .expect("runtime config should resolve");
        assert_eq!(weighted_anthropic.model.provider, "anthropic");

        let weighted_openai = resolve_runtime_config_with_seed(&args, &local, 90)
            .expect("runtime config should resolve");
        assert_eq!(weighted_openai.model.provider, "openai");
    }

    #[test]
    fn wildcard_default_provider_rejects_invalid_provider_weight() {
        let args = test_chat_args();
        let local = AgentLocalConfig {
            settings: AgentSettingsFile {
                default_provider: Some("*".to_string()),
                ..AgentSettingsFile::default()
            },
            models: ModelsFile {
                providers: HashMap::from([(
                    "openai".to_string(),
                    ProviderConfig {
                        provider: None,
                        kind: None,
                        api: Some("openai-responses".to_string()),
                        base_url: Some("https://api.openai.com/v1".to_string()),
                        api_key: None,
                        default_model: Some("gpt-5.3-codex".to_string()),
                        weight: 100,
                        models: vec![],
                    },
                )]),
            },
        };

        let error = resolve_runtime_config_with_seed(&args, &local, 0)
            .expect_err("invalid weight should be rejected");
        assert!(error.contains("expected value < 100"));
    }

    #[test]
    fn fixed_provider_rejects_invalid_provider_weight() {
        let mut args = test_chat_args();
        args.provider = Some("openai".to_string());

        let local = AgentLocalConfig {
            settings: AgentSettingsFile::default(),
            models: ModelsFile {
                providers: HashMap::from([(
                    "openai".to_string(),
                    ProviderConfig {
                        provider: None,
                        kind: None,
                        api: Some("openai-responses".to_string()),
                        base_url: Some("https://api.openai.com/v1".to_string()),
                        api_key: None,
                        default_model: Some("gpt-5.3-codex".to_string()),
                        weight: 100,
                        models: vec![],
                    },
                )]),
            },
        };

        let error = resolve_runtime_config_with_seed(&args, &local, 0)
            .expect_err("invalid weight should be rejected");
        assert!(error.contains("expected value < 100"));
    }

    #[test]
    fn explicit_provider_bypasses_weighted_router_validation() {
        let mut args = test_chat_args();
        args.provider = Some("openai".to_string());
        args.model = Some("gpt-5.3-codex".to_string());

        let local = AgentLocalConfig {
            settings: AgentSettingsFile {
                default_provider: Some("*".to_string()),
                ..AgentSettingsFile::default()
            },
            models: ModelsFile {
                providers: HashMap::from([
                    (
                        "openai".to_string(),
                        ProviderConfig {
                            provider: None,
                            kind: None,
                            api: Some("openai-responses".to_string()),
                            base_url: Some("https://api.openai.com/v1".to_string()),
                            api_key: None,
                            default_model: Some("gpt-5.3-codex".to_string()),
                            weight: 10,
                            models: vec![],
                        },
                    ),
                    (
                        "anthropic".to_string(),
                        ProviderConfig {
                            provider: None,
                            kind: None,
                            api: Some("anthropic-messages".to_string()),
                            base_url: Some("https://api.anthropic.com/v1".to_string()),
                            api_key: None,
                            default_model: Some("claude-3-5-sonnet-latest".to_string()),
                            weight: 100,
                            models: vec![],
                        },
                    ),
                ]),
            },
        };

        let resolved = resolve_runtime_config_with_seed(&args, &local, 0)
            .expect("runtime config should resolve");
        assert_eq!(resolved.model.provider, "openai");
        assert_eq!(resolved.model.id, "gpt-5.3-codex");
    }

    #[test]
    fn fixed_default_provider_bypasses_weighted_router() {
        let args = test_chat_args();
        let local = AgentLocalConfig {
            settings: AgentSettingsFile {
                default_provider: Some("openai".to_string()),
                ..AgentSettingsFile::default()
            },
            models: ModelsFile {
                providers: HashMap::from([
                    (
                        "openai".to_string(),
                        ProviderConfig {
                            provider: None,
                            kind: None,
                            api: Some("openai-responses".to_string()),
                            base_url: Some("https://api.openai.com/v1".to_string()),
                            api_key: None,
                            default_model: Some("gpt-5.3-codex".to_string()),
                            weight: 1,
                            models: vec![],
                        },
                    ),
                    (
                        "anthropic".to_string(),
                        ProviderConfig {
                            provider: None,
                            kind: None,
                            api: Some("anthropic-messages".to_string()),
                            base_url: Some("https://api.anthropic.com/v1".to_string()),
                            api_key: None,
                            default_model: Some("claude-3-5-sonnet-latest".to_string()),
                            weight: 99,
                            models: vec![],
                        },
                    ),
                ]),
            },
        };

        let resolved = resolve_runtime_config_with_seed(&args, &local, 0)
            .expect("runtime config should resolve");
        assert_eq!(resolved.model.provider, "openai");
    }

    #[test]
    fn resolve_runtime_config_uses_first_provider_model_when_default_model_missing() {
        let args = test_chat_args();
        let local = AgentLocalConfig {
            settings: AgentSettingsFile {
                default_provider: Some("openai".to_string()),
                ..AgentSettingsFile::default()
            },
            models: ModelsFile {
                providers: HashMap::from([(
                    "openai".to_string(),
                    ProviderConfig {
                        provider: None,
                        kind: None,
                        api: Some("openai-responses".to_string()),
                        base_url: Some("https://api.openai.com/v1".to_string()),
                        api_key: None,
                        default_model: None,
                        weight: 1,
                        models: vec![
                            ProviderModelConfig {
                                id: "gpt-4.1".to_string(),
                                api: None,
                                base_url: None,
                                reasoning: None,
                                reasoning_effort: None,
                                context_window: None,
                                max_tokens: None,
                            },
                            ProviderModelConfig {
                                id: "gpt-5.3-codex".to_string(),
                                api: None,
                                base_url: None,
                                reasoning: None,
                                reasoning_effort: None,
                                context_window: None,
                                max_tokens: None,
                            },
                        ],
                    },
                )]),
            },
        };

        let resolved =
            resolve_runtime_config(&args, &local).expect("runtime config should resolve");
        assert_eq!(resolved.model.id, "gpt-4.1");
    }

    #[test]
    fn cli_model_with_provider_prefix_overrides_settings_provider() {
        let args = ChatArgs {
            api: None,
            provider: None,
            model: Some("openai/gpt-4o-mini".to_string()),
            base_url: None,
            context_window: None,
            max_tokens: None,
            agent_dir: None,
            cwd: None,
            session_dir: None,
            session_file: None,
            system_prompt: Some("test".to_string()),
            prompt: None,
            continue_first: false,
            no_tools: false,
            skills: vec![],
            no_skills: false,
            hide_tool_results: false,
            no_tui: false,
            theme: None,
        };

        let local = AgentLocalConfig {
            settings: AgentSettingsFile {
                default_provider: Some("anthropic".to_string()),
                theme: None,
                transport_retry_count: None,
                skills: vec![],
                env: HashMap::new(),
            },
            models: ModelsFile::default(),
        };

        let resolved =
            resolve_runtime_config(&args, &local).expect("runtime config should resolve");
        assert_eq!(resolved.model.provider, "openai");
        assert_eq!(resolved.model.api, "openai-responses");
        assert_eq!(resolved.model.id, "gpt-4o-mini");
    }

    #[test]
    fn provider_api_key_can_be_resolved_from_settings_env() {
        let args = ChatArgs {
            api: None,
            provider: Some("anthropic".to_string()),
            model: Some("claude-opus-4-6".to_string()),
            base_url: Some("https://api.anthropic.com/v1".to_string()),
            context_window: None,
            max_tokens: None,
            agent_dir: None,
            cwd: None,
            session_dir: None,
            session_file: None,
            system_prompt: Some("test".to_string()),
            prompt: None,
            continue_first: false,
            no_tools: false,
            skills: vec![],
            no_skills: false,
            hide_tool_results: false,
            no_tui: false,
            theme: None,
        };

        let local = AgentLocalConfig {
            settings: AgentSettingsFile {
                default_provider: None,
                theme: None,
                transport_retry_count: None,
                skills: vec![],
                env: HashMap::from([(
                    "ANTHROPIC_AUTH_TOKEN".to_string(),
                    "anthropic-token".to_string(),
                )]),
            },
            models: ModelsFile::default(),
        };

        let resolved =
            resolve_runtime_config(&args, &local).expect("runtime config should resolve");
        assert_eq!(resolved.api_key.as_deref(), Some("anthropic-token"));
    }

    #[test]
    fn provider_base_url_can_be_resolved_from_settings_env() {
        let mut args = test_chat_args();
        args.provider = Some("openai".to_string());
        args.model = Some("gpt-5.3-codex".to_string());

        let local = AgentLocalConfig {
            settings: AgentSettingsFile {
                env: HashMap::from([(
                    "LLM_BASE_URL".to_string(),
                    "https://codex.databend.cloud".to_string(),
                )]),
                ..AgentSettingsFile::default()
            },
            models: ModelsFile {
                providers: HashMap::from([(
                    "openai".to_string(),
                    ProviderConfig {
                        provider: Some("openai".to_string()),
                        kind: Some("chat".to_string()),
                        api: Some("openai-responses".to_string()),
                        base_url: Some("$LLM_BASE_URL".to_string()),
                        api_key: None,
                        default_model: Some("gpt-5.3-codex".to_string()),
                        weight: 1,
                        models: vec![],
                    },
                )]),
            },
        };

        let resolved =
            resolve_runtime_config(&args, &local).expect("runtime config should resolve");
        assert_eq!(resolved.model.base_url, "https://codex.databend.cloud");
    }

    #[test]
    fn resolve_runtime_config_builds_model_catalog_with_selected_model_first() {
        let args = ChatArgs {
            api: None,
            provider: Some("openai".to_string()),
            model: Some("gpt-5.3-codex".to_string()),
            base_url: None,
            context_window: None,
            max_tokens: None,
            agent_dir: None,
            cwd: None,
            session_dir: None,
            session_file: None,
            system_prompt: Some("test".to_string()),
            prompt: None,
            continue_first: false,
            no_tools: false,
            skills: vec![],
            no_skills: false,
            hide_tool_results: false,
            no_tui: false,
            theme: None,
        };

        let local = AgentLocalConfig {
            settings: AgentSettingsFile::default(),
            models: ModelsFile {
                providers: HashMap::from([(
                    "openai".to_string(),
                    ProviderConfig {
                        provider: None,
                        kind: None,
                        api: Some("openai-completions".to_string()),
                        base_url: Some("https://api.openai.com/v1".to_string()),
                        api_key: None,
                        default_model: None,
                        weight: 1,
                        models: vec![
                            ProviderModelConfig {
                                id: "gpt-4.1".to_string(),
                                api: None,
                                base_url: None,
                                reasoning: None,
                                reasoning_effort: None,
                                context_window: None,
                                max_tokens: None,
                            },
                            ProviderModelConfig {
                                id: "gpt-5.3-codex".to_string(),
                                api: None,
                                base_url: None,
                                reasoning: None,
                                reasoning_effort: None,
                                context_window: None,
                                max_tokens: None,
                            },
                        ],
                    },
                )]),
            },
        };

        let resolved =
            resolve_runtime_config(&args, &local).expect("runtime config should resolve");
        assert_eq!(resolved.model_catalog.len(), 2);
        assert_eq!(resolved.model_catalog[0].id, "gpt-5.3-codex");
        assert_eq!(resolved.model_catalog[1].id, "gpt-4.1");
    }

    #[test]
    fn load_tui_keybindings_reads_supported_actions() {
        let dir = tempdir().expect("tempdir");
        let config_path = dir.path().join("keybindings.json");
        std::fs::write(
            &config_path,
            r#"{
  "clear": "ctrl+l",
  "exit": "ctrl+q",
  "interrupt": ["escape", "ctrl+c"],
  "cycleThinkingLevel": "shift+tab",
  "cycleModelForward": "ctrl+p",
  "cycleModelBackward": "shift+ctrl+p",
  "selectModel": "ctrl+k",
  "expandTools": ["invalid", "ctrl+e"],
  "toggleThinking": "ctrl+y",
  "followUp": "alt+enter",
  "dequeue": "alt+up"
}"#,
        )
        .expect("write keybindings");

        let bindings = load_tui_keybindings(dir.path()).expect("bindings should parse");
        assert_eq!(
            bindings.clear,
            vec![parse_key_id("ctrl+l").expect("parse ctrl+l")]
        );
        assert_eq!(
            bindings.quit,
            vec![parse_key_id("ctrl+q").expect("parse ctrl+q")]
        );
        assert_eq!(
            bindings.interrupt,
            vec![
                parse_key_id("escape").expect("parse escape"),
                parse_key_id("ctrl+c").expect("parse ctrl+c")
            ]
        );
        assert_eq!(
            bindings.cycle_thinking_level,
            vec![parse_key_id("shift+tab").expect("parse shift+tab")]
        );
        assert_eq!(
            bindings.cycle_model_forward,
            vec![parse_key_id("ctrl+p").expect("parse ctrl+p")]
        );
        assert_eq!(
            bindings.cycle_model_backward,
            vec![parse_key_id("shift+ctrl+p").expect("parse shift+ctrl+p")]
        );
        assert_eq!(
            bindings.select_model,
            vec![parse_key_id("ctrl+k").expect("parse ctrl+k")]
        );
        assert_eq!(
            bindings.expand_tools,
            vec![parse_key_id("ctrl+e").expect("parse ctrl+e")]
        );
        assert_eq!(
            bindings.toggle_thinking,
            vec![parse_key_id("ctrl+y").expect("parse ctrl+y")]
        );
        assert_eq!(
            bindings.continue_run,
            vec![parse_key_id("alt+enter").expect("parse alt+enter")]
        );
        assert_eq!(
            bindings.dequeue,
            vec![parse_key_id("alt+up").expect("parse alt+up")]
        );
    }

    #[test]
    fn load_tui_keybindings_ignores_invalid_json() {
        let dir = tempdir().expect("tempdir");
        let config_path = dir.path().join("keybindings.json");
        std::fs::write(&config_path, "{").expect("write invalid keybindings");

        assert!(load_tui_keybindings(dir.path()).is_none());
    }

    #[test]
    fn resolve_tui_theme_name_prefers_cli_then_settings_then_default() {
        assert_eq!(
            resolve_tui_theme_name(Some("light"), Some("dark")).expect("cli theme should win"),
            "light"
        );
        assert_eq!(
            resolve_tui_theme_name(None, Some("light")).expect("settings theme should be used"),
            "light"
        );
        assert_eq!(
            resolve_tui_theme_name(None, None).expect("default theme should be dark"),
            "dark"
        );
    }

    #[test]
    fn resolve_tui_theme_name_rejects_unknown_values() {
        let error = resolve_tui_theme_name(Some("solarized"), None).expect_err("invalid theme");
        assert!(error.contains("unsupported theme"));
        assert!(error.contains("dark"));
        assert!(error.contains("light"));
    }

    #[test]
    fn cli_stream_renderer_formats_deltas_and_tool_lines() {
        let mut renderer = CliStreamRenderer::new(Vec::<u8>::new(), true);

        renderer
            .on_update(AgentSessionStreamUpdate::AssistantTextDelta(
                "hello".to_string(),
            ))
            .expect("delta write succeeds");
        renderer
            .on_update(AgentSessionStreamUpdate::AssistantTextDelta(
                " world".to_string(),
            ))
            .expect("delta write succeeds");
        renderer
            .on_update(AgentSessionStreamUpdate::AssistantLine(String::new()))
            .expect("assistant line succeeds");
        renderer
            .on_update(AgentSessionStreamUpdate::ToolLine(
                "[tool:read:ok]".to_string(),
            ))
            .expect("tool line succeeds");
        renderer
            .on_update(AgentSessionStreamUpdate::ToolLine(
                "tool output".to_string(),
            ))
            .expect("tool line succeeds");
        renderer.finish().expect("finish succeeds");

        let output = String::from_utf8(renderer.into_inner()).expect("utf-8 output");
        assert_eq!(output, "hello world\n[tool:read:ok]\ntool output\n");
    }

    #[test]
    fn cli_stream_renderer_hides_tool_lines_when_disabled() {
        let mut renderer = CliStreamRenderer::new(Vec::<u8>::new(), false);

        renderer
            .on_update(AgentSessionStreamUpdate::AssistantTextDelta(
                "hi".to_string(),
            ))
            .expect("delta write succeeds");
        renderer
            .on_update(AgentSessionStreamUpdate::AssistantLine(String::new()))
            .expect("assistant line succeeds");
        renderer
            .on_update(AgentSessionStreamUpdate::ToolLine(
                "[tool:read:ok]".to_string(),
            ))
            .expect("tool line ignored");
        renderer.finish().expect("finish succeeds");

        let output = String::from_utf8(renderer.into_inner()).expect("utf-8 output");
        assert_eq!(output, "hi\n");
    }

    #[test]
    fn cli_stream_renderer_updates_thinking_line_in_place() {
        let mut renderer = CliStreamRenderer::new(Vec::<u8>::new(), true);

        renderer
            .on_update(AgentSessionStreamUpdate::AssistantLine(
                "[thinking] a".to_string(),
            ))
            .expect("thinking line succeeds");
        renderer
            .on_update(AgentSessionStreamUpdate::AssistantLine(
                "[thinking] ab".to_string(),
            ))
            .expect("thinking update succeeds");
        renderer
            .on_update(AgentSessionStreamUpdate::AssistantLine(String::new()))
            .expect("flush line succeeds");
        renderer.finish().expect("finish succeeds");

        let output = String::from_utf8(renderer.into_inner()).expect("utf-8 output");
        assert_eq!(output, "[thinking] a\r\u{1b}[2K[thinking] ab\n");
    }

    #[test]
    fn cli_stream_renderer_replaces_multiline_thinking_block() {
        let mut renderer = CliStreamRenderer::new(Vec::<u8>::new(), true);

        renderer
            .on_update(AgentSessionStreamUpdate::AssistantLine(
                "[thinking] line1\nline2".to_string(),
            ))
            .expect("initial multiline thinking succeeds");
        renderer
            .on_update(AgentSessionStreamUpdate::AssistantLine(
                "[thinking] line1\nline2 updated".to_string(),
            ))
            .expect("multiline thinking update succeeds");
        renderer
            .on_update(AgentSessionStreamUpdate::AssistantLine(String::new()))
            .expect("flush line succeeds");
        renderer.finish().expect("finish succeeds");

        let output = String::from_utf8(renderer.into_inner()).expect("utf-8 output");
        assert_eq!(
            output,
            "[thinking] line1\nline2\r\u{1b}[2K\u{1b}[1A\r\u{1b}[2K[thinking] line1\nline2 updated\n"
        );
    }

    #[test]
    fn resolve_runtime_config_supports_google_provider_inference() {
        let args = ChatArgs {
            api: None,
            provider: Some("google".to_string()),
            model: Some("gemini-2.5-flash".to_string()),
            base_url: None,
            context_window: None,
            max_tokens: None,
            agent_dir: None,
            cwd: None,
            session_dir: None,
            session_file: None,
            system_prompt: Some("test".to_string()),
            prompt: None,
            continue_first: false,
            no_tools: false,
            skills: vec![],
            no_skills: false,
            hide_tool_results: false,
            no_tui: false,
            theme: None,
        };
        let local = AgentLocalConfig::default();

        let resolved =
            resolve_runtime_config(&args, &local).expect("runtime config should resolve");
        assert_eq!(resolved.model.api, "google-generative-ai");
        assert_eq!(
            resolved.model.base_url,
            "https://generativelanguage.googleapis.com/v1beta"
        );
    }

    #[test]
    fn resolve_runtime_config_defaults_reasoning_to_medium_when_unconfigured() {
        let args = ChatArgs {
            api: None,
            provider: Some("openai".to_string()),
            model: Some("gpt-5.3-codex".to_string()),
            base_url: None,
            context_window: None,
            max_tokens: None,
            agent_dir: None,
            cwd: None,
            session_dir: None,
            session_file: None,
            system_prompt: Some("test".to_string()),
            prompt: None,
            continue_first: false,
            no_tools: false,
            skills: vec![],
            no_skills: false,
            hide_tool_results: false,
            no_tui: false,
            theme: None,
        };
        let local = AgentLocalConfig::default();

        let resolved =
            resolve_runtime_config(&args, &local).expect("runtime config should resolve");
        assert!(resolved.model.reasoning);
        assert_eq!(
            resolved.model.reasoning_effort,
            Some(pixy_ai::ThinkingLevel::Medium)
        );
    }

    #[test]
    fn default_agent_dir_uses_pixy_home_directory() {
        unsafe {
            std::env::set_var("HOME", "/tmp/pixy-home");
        }
        assert_eq!(
            default_agent_dir(),
            PathBuf::from("/tmp/pixy-home/.pixy/agent")
        );
    }

    #[test]
    fn load_agent_local_config_reads_pixy_toml() {
        let dir = tempdir().expect("tempdir");
        let pixy_dir = dir.path().join(".pixy");
        std::fs::create_dir_all(&pixy_dir).expect("create .pixy");
        std::fs::write(
            pixy_dir.join("pixy.toml"),
            r#"
theme = "light"
transport_retry_count = 7
skills = ["./skills"]

[env]
OPENAI_KEY = "sk-test"

[llm]
default_provider = "*"

[[llm.providers]]
name = "openai"
kind = "chat"
provider = "openai"
api = "openai-responses"
base_url = "https://api.openai.com/v1"
api_key = "$OPENAI_KEY"
model = "gpt-5.3-codex"
weight = 30
"#,
        )
        .expect("write pixy.toml");

        unsafe {
            std::env::set_var("HOME", dir.path());
        }

        let local = load_agent_local_config(Path::new(".")).expect("load config");
        assert_eq!(local.settings.theme.as_deref(), Some("light"));
        assert_eq!(local.settings.transport_retry_count, Some(7));
        assert_eq!(local.settings.default_provider.as_deref(), Some("*"));
        assert_eq!(
            local.settings.env.get("OPENAI_KEY").map(String::as_str),
            Some("sk-test")
        );

        let provider = local
            .models
            .providers
            .get("openai")
            .expect("openai provider exists");
        assert_eq!(provider.provider.as_deref(), Some("openai"));
        assert_eq!(provider.kind.as_deref(), Some("chat"));
        assert_eq!(provider.default_model.as_deref(), Some("gpt-5.3-codex"));
        assert_eq!(provider.weight, 30);
        assert_eq!(provider.models.len(), 1);
        assert_eq!(provider.models[0].id, "gpt-5.3-codex");
    }

    #[test]
    fn wildcard_router_ignores_embedding_provider_kind() {
        let args = test_chat_args();
        let local = AgentLocalConfig {
            settings: AgentSettingsFile {
                default_provider: Some("*".to_string()),
                ..AgentSettingsFile::default()
            },
            models: ModelsFile {
                providers: HashMap::from([
                    (
                        "embed".to_string(),
                        ProviderConfig {
                            provider: Some("openai".to_string()),
                            kind: Some("embedding".to_string()),
                            api: Some("openai-responses".to_string()),
                            base_url: Some("https://api.openai.com/v1".to_string()),
                            api_key: None,
                            default_model: Some("text-embedding-3-small".to_string()),
                            weight: 99,
                            models: vec![],
                        },
                    ),
                    (
                        "chat".to_string(),
                        ProviderConfig {
                            provider: Some("openai".to_string()),
                            kind: Some("chat".to_string()),
                            api: Some("openai-responses".to_string()),
                            base_url: Some("https://api.openai.com/v1".to_string()),
                            api_key: None,
                            default_model: Some("gpt-5.3-codex".to_string()),
                            weight: 1,
                            models: vec![],
                        },
                    ),
                ]),
            },
        };

        let resolved = resolve_runtime_config_with_seed(&args, &local, 0)
            .expect("runtime config should resolve");
        assert_eq!(resolved.model.provider, "openai");
        assert_eq!(resolved.model.id, "gpt-5.3-codex");
    }

    #[test]
    fn resolve_transport_retry_count_defaults_to_five() {
        assert_eq!(
            resolve_transport_retry_count(&AgentSettingsFile::default()),
            DEFAULT_TRANSPORT_RETRY_COUNT
        );
    }

    #[test]
    fn resolve_transport_retry_count_reads_settings_value() {
        let settings = AgentSettingsFile {
            transport_retry_count: Some(2),
            ..AgentSettingsFile::default()
        };
        assert_eq!(resolve_transport_retry_count(&settings), 2);
    }

    #[test]
    fn load_runtime_skills_merges_settings_and_cli_paths() {
        let dir = tempdir().expect("temp dir");
        let settings_skill = dir.path().join("settings-skill");
        let cli_skill = dir.path().join("cli-skill");
        std::fs::create_dir_all(&settings_skill).expect("create settings skill dir");
        std::fs::create_dir_all(&cli_skill).expect("create cli skill dir");
        std::fs::write(
            settings_skill.join("SKILL.md"),
            r#"---
name: settings-skill
description: from settings
---
"#,
        )
        .expect("write settings skill");
        std::fs::write(
            cli_skill.join("SKILL.md"),
            r#"---
name: cli-skill
description: from cli
---
"#,
        )
        .expect("write cli skill");

        let mut args = test_chat_args();
        args.no_skills = true;
        args.skills = vec!["cli-skill".to_string()];

        let local = AgentLocalConfig {
            settings: AgentSettingsFile {
                skills: vec!["settings-skill".to_string()],
                ..AgentSettingsFile::default()
            },
            models: ModelsFile::default(),
        };
        let skills =
            load_runtime_skills(&args, &local, dir.path(), &dir.path().join(".pixy/agent"));
        let names = skills
            .iter()
            .map(|skill| skill.name.clone())
            .collect::<HashSet<_>>();
        assert!(names.contains("settings-skill"));
        assert!(names.contains("cli-skill"));
    }

    #[test]
    fn load_runtime_skills_honors_no_skills_for_default_discovery() {
        let dir = tempdir().expect("temp dir");
        let agent_dir = dir.path().join(".pixy/agent");
        let default_skill = agent_dir.join("skills").join("default-skill");
        std::fs::create_dir_all(&default_skill).expect("create default skill dir");
        std::fs::write(
            default_skill.join("SKILL.md"),
            r#"---
name: default-skill
description: default discovered skill
---
"#,
        )
        .expect("write default skill");

        let mut args = test_chat_args();
        args.no_skills = true;

        let local = AgentLocalConfig::default();
        let skills = load_runtime_skills(&args, &local, dir.path(), &agent_dir);
        assert!(skills.is_empty());
    }

    #[test]
    fn resolve_runtime_config_reads_reasoning_and_effort_from_model_config() {
        let mut args = test_chat_args();
        args.provider = Some("openai".to_string());
        args.model = Some("gpt-4.1".to_string());

        let local = AgentLocalConfig {
            settings: AgentSettingsFile::default(),
            models: ModelsFile {
                providers: HashMap::from([(
                    "openai".to_string(),
                    ProviderConfig {
                        provider: None,
                        kind: None,
                        api: Some("openai-completions".to_string()),
                        base_url: Some("https://api.openai.com/v1".to_string()),
                        api_key: None,
                        default_model: None,
                        weight: 1,
                        models: vec![ProviderModelConfig {
                            id: "gpt-4.1".to_string(),
                            api: None,
                            base_url: None,
                            reasoning: Some(true),
                            reasoning_effort: Some(pixy_ai::ThinkingLevel::High),
                            context_window: None,
                            max_tokens: None,
                        }],
                    },
                )]),
            },
        };

        let resolved =
            resolve_runtime_config(&args, &local).expect("runtime config should resolve");
        assert!(resolved.model.reasoning);
        assert_eq!(
            resolved.model.reasoning_effort,
            Some(pixy_ai::ThinkingLevel::High)
        );
    }

    #[test]
    fn collect_startup_context_files_prefers_agents_and_orders_scopes() {
        let dir = tempdir().expect("temp dir");
        let agent_dir = dir.path().join(".pixy/agent");
        let project_root = dir.path().join("repo");
        let nested = project_root.join("nested/workspace");
        std::fs::create_dir_all(&agent_dir).expect("create agent dir");
        std::fs::create_dir_all(&nested).expect("create nested dir");

        std::fs::write(agent_dir.join("AGENTS.md"), "global").expect("write global agents");
        std::fs::write(agent_dir.join("CLAUDE.md"), "global claude").expect("write global claude");
        std::fs::write(project_root.join("AGENTS.md"), "project").expect("write project agents");
        std::fs::write(nested.join("CLAUDE.md"), "nested").expect("write nested claude");

        let contexts = collect_startup_context_files(&nested, &agent_dir);
        assert_eq!(
            contexts,
            vec![
                agent_dir.join("AGENTS.md"),
                project_root.join("AGENTS.md"),
                nested.join("CLAUDE.md"),
            ]
        );
    }

    #[test]
    fn build_startup_resource_lines_lists_context_and_grouped_skills() {
        let dir = tempdir().expect("temp dir");
        let agent_dir = dir.path().join(".pixy/agent");
        let cwd = dir.path().join("workspace");
        std::fs::create_dir_all(&agent_dir).expect("create agent dir");
        std::fs::create_dir_all(&cwd).expect("create cwd");
        std::fs::write(cwd.join("AGENTS.md"), "project").expect("write project agents");

        let skills = vec![
            Skill {
                name: "project-skill".to_string(),
                description: "project".to_string(),
                file_path: cwd.join(".agents/skills/project/SKILL.md"),
                base_dir: cwd.join(".agents/skills/project"),
                source: pixy_coding_agent::SkillSource::Project,
                disable_model_invocation: false,
            },
            Skill {
                name: "path-skill".to_string(),
                description: "path".to_string(),
                file_path: cwd.join("custom/path-skill/SKILL.md"),
                base_dir: cwd.join("custom/path-skill"),
                source: pixy_coding_agent::SkillSource::Path,
                disable_model_invocation: false,
            },
            Skill {
                name: "user-skill".to_string(),
                description: "user".to_string(),
                file_path: agent_dir.join("skills/user/SKILL.md"),
                base_dir: agent_dir.join("skills/user"),
                source: pixy_coding_agent::SkillSource::User,
                disable_model_invocation: false,
            },
        ];

        let lines = build_startup_resource_lines(&cwd, &agent_dir, &skills);
        assert!(lines.contains(&"[Context]".to_string()));
        assert!(lines.contains(&"[Skills]".to_string()));
        assert!(lines.contains(&"  user".to_string()));
        assert!(lines.contains(&"  project".to_string()));
        assert!(lines.contains(&"  path".to_string()));
        assert!(
            lines
                .iter()
                .any(|line| line.ends_with(".agents/skills/project/SKILL.md")),
            "project skill path should be rendered"
        );
    }

    #[test]
    fn resolve_resume_picker_selection_defaults_to_latest() {
        let candidates = vec![PathBuf::from("/tmp/session-a.jsonl")];
        let selected = resolve_resume_picker_selection(&candidates, "")
            .expect("selection should parse")
            .expect("selection should not cancel");
        assert_eq!(selected, candidates[0]);
    }

    #[test]
    fn resolve_resume_picker_selection_supports_numeric_choice() {
        let candidates = vec![
            PathBuf::from("/tmp/session-a.jsonl"),
            PathBuf::from("/tmp/session-b.jsonl"),
        ];
        let selected = resolve_resume_picker_selection(&candidates, "2")
            .expect("selection should parse")
            .expect("selection should not cancel");
        assert_eq!(selected, candidates[1]);
    }

    #[test]
    fn resolve_resume_picker_selection_supports_cancel() {
        let candidates = vec![PathBuf::from("/tmp/session-a.jsonl")];
        let selected =
            resolve_resume_picker_selection(&candidates, "q").expect("selection should parse");
        assert!(selected.is_none());
    }

    #[test]
    fn resolve_resume_picker_selection_rejects_out_of_range_choice() {
        let candidates = vec![PathBuf::from("/tmp/session-a.jsonl")];
        let error =
            resolve_resume_picker_selection(&candidates, "2").expect_err("selection should fail");
        assert!(error.contains("between 1 and 1"));
    }
}
