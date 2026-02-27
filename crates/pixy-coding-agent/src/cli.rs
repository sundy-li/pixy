use std::collections::{HashMap, HashSet};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use crate::cli_app::{
    CliSession, CliSessionFactory, CliSessionRequest, ReplCommand, ReplCommandParser,
};
use crate::{AgentSession, AgentSessionStreamUpdate, RuntimeOverrides, Skill, SkillSource};
use clap::{Args, Parser, Subcommand};
use pixy_ai::{AssistantContentBlock, Message, StopReason, ToolResultContentBlock};
use pixy_tui::{KeyBinding, TuiKeyBindings, TuiOptions, TuiTheme, parse_key_id};
use serde::Deserialize;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

#[cfg(test)]
use crate::{LoadSkillsOptions, RuntimeLoadOptions, load_skills};
#[cfg(test)]
use pixy_ai::{Cost, DEFAULT_TRANSPORT_RETRY_COUNT, Model};
#[cfg(test)]
use std::time::{SystemTime, UNIX_EPOCH};

const RESUME_PICKER_LIMIT: usize = 10;
const DEFAULT_PIXY_HOME_DIR_NAME: &str = ".pixy";
const DEFAULT_LOG_LEVEL: &str = "info";
const DEFAULT_LOG_ROTATE_SIZE_MB: u64 = 100;
const DEFAULT_LOG_STDOUT: bool = false;
const MARKDOWN_CODE_BASE_STYLE: &str = "\x1b[48;5;236m\x1b[38;5;252m";
const MARKDOWN_CODE_KEYWORD_STYLE: &str = "\x1b[48;5;236m\x1b[38;5;111m";
const MARKDOWN_CODE_STRING_STYLE: &str = "\x1b[48;5;236m\x1b[38;5;150m";
const MARKDOWN_CODE_COMMENT_STYLE: &str = "\x1b[48;5;236m\x1b[38;5;245m";
const MARKDOWN_CODE_NUMBER_STYLE: &str = "\x1b[48;5;236m\x1b[38;5;215m";
const ANSI_STYLE_RESET: &str = "\x1b[0m";
static CONF_DIR: OnceLock<PathBuf> = OnceLock::new();

#[derive(Parser, Debug)]
#[command(name = "pixy", version, about = "pixy interactive CLI")]
struct Cli {
    #[arg(long, global = true)]
    conf_dir: Option<PathBuf>,
    #[command(subcommand)]
    command: Option<RootCommand>,
    #[command(flatten)]
    chat: ChatArgs,
}

#[derive(Subcommand, Debug, Clone)]
enum RootCommand {
    Cli(ChatArgs),
    Gateway(GatewayArgs),
}

#[derive(Args, Debug, Clone)]
pub struct ChatArgs {
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

#[derive(Args, Debug, Clone)]
struct GatewayArgs {
    #[command(subcommand)]
    command: GatewaySubcommand,
}

#[derive(Subcommand, Debug, Clone)]
enum GatewaySubcommand {
    Start(GatewayStartArgs),
    Stop,
    Restart,
    #[command(hide = true)]
    Serve,
}

#[derive(Args, Debug, Clone)]
struct GatewayStartArgs {
    #[arg(long, default_value_t = false)]
    daemon: bool,
}

pub async fn run_cli_process() -> Result<(), String> {
    let cli = Cli::parse();
    run_parsed_cli(cli).await
}

pub async fn run_chat_with_conf(args: ChatArgs, conf_dir: Option<PathBuf>) -> Result<(), String> {
    init_conf_dir(conf_dir.as_deref());
    init_tracing();
    run(args).await
}

async fn run_parsed_cli(cli: Cli) -> Result<(), String> {
    let conf_dir = cli.conf_dir.clone();
    init_conf_dir(conf_dir.as_deref());
    init_tracing();
    match cli.command {
        Some(RootCommand::Cli(args)) => run(args).await,
        Some(RootCommand::Gateway(args)) => run_gateway(args, conf_dir.as_deref()).await,
        None => run(cli.chat).await,
    }
}

async fn run_gateway(args: GatewayArgs, conf_dir: Option<&Path>) -> Result<(), String> {
    let mut command = Command::new("pixy-gateway");
    for token in gateway_command_tokens(&args.command, conf_dir) {
        command.arg(token);
    }
    let status = command
        .status()
        .map_err(|error| format!("failed to launch pixy-gateway: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("pixy-gateway exited with status {status}"))
    }
}

fn gateway_command_tokens(command: &GatewaySubcommand, conf_dir: Option<&Path>) -> Vec<String> {
    let mut tokens = Vec::new();
    if let Some(path) = conf_dir {
        tokens.push("--conf-dir".to_string());
        tokens.push(path.display().to_string());
    }
    match command {
        GatewaySubcommand::Start(start) => {
            tokens.push("start".to_string());
            if start.daemon {
                tokens.push("--daemon".to_string());
            }
            tokens
        }
        GatewaySubcommand::Stop => {
            tokens.push("stop".to_string());
            tokens
        }
        GatewaySubcommand::Restart => {
            tokens.push("restart".to_string());
            tokens
        }
        GatewaySubcommand::Serve => {
            tokens.push("serve".to_string());
            tokens
        }
    }
}

fn init_tracing() {
    static TRACE_GUARD: OnceLock<WorkerGuard> = OnceLock::new();

    let config = load_runtime_log_config("pixy.log");
    let file_writer =
        match SizeRotatingFileWriter::new(config.file_path.clone(), config.rotate_size_bytes) {
            Ok(writer) => writer,
            Err(error) => {
                eprintln!("warning: failed to initialize tracing writer: {error}");
                return;
            }
        };
    let (non_blocking, guard) = tracing_appender::non_blocking(file_writer);
    let _ = TRACE_GUARD.set(guard);

    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(config.level.clone()));
    let file_layer = tracing_subscriber::fmt::layer()
        .with_ansi(false)
        .with_writer(non_blocking);
    let mut stdout_layer = tracing_subscriber::fmt::layer();
    stdout_layer.set_ansi(false);

    let init_result = if config.stdout {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(file_layer)
            .with(stdout_layer)
            .try_init()
    } else {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(file_layer)
            .try_init()
    };
    if let Err(error) = init_result {
        eprintln!(
            "warning: failed to initialize tracing subscriber for {}: {error}",
            config.file_path.display()
        );
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
struct PixyTomlLogFile {
    #[serde(default)]
    log: PixyTomlLog,
    #[serde(default)]
    env: HashMap<String, String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct PixyTomlLog {
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    level: Option<String>,
    #[serde(default)]
    rotate_size_mb: Option<u64>,
    #[serde(default)]
    stdout: Option<bool>,
}

#[derive(Debug, Clone)]
struct RuntimeLogConfig {
    file_path: PathBuf,
    level: String,
    rotate_size_bytes: u64,
    stdout: bool,
}

fn load_runtime_log_config(file_name: &str) -> RuntimeLogConfig {
    let config_path = default_pixy_config_path();
    let parsed = read_toml_if_exists::<PixyTomlLogFile>(&config_path)
        .ok()
        .flatten()
        .unwrap_or_default();
    build_runtime_log_config(&parsed.log, &parsed.env, file_name)
}

#[cfg(test)]
fn parse_runtime_log_config_from_toml(
    content: &str,
    file_name: &str,
) -> Result<RuntimeLogConfig, String> {
    let parsed: PixyTomlLogFile =
        toml::from_str(content).map_err(|error| format!("parse log config failed: {error}"))?;
    Ok(build_runtime_log_config(
        &parsed.log,
        &parsed.env,
        file_name,
    ))
}

fn build_runtime_log_config(
    log: &PixyTomlLog,
    env_map: &HashMap<String, String>,
    file_name: &str,
) -> RuntimeLogConfig {
    let base_path = log
        .path
        .as_deref()
        .and_then(|value| resolve_config_value(value, env_map))
        .map(|value| expand_home_path(value.trim()))
        .unwrap_or_else(default_log_dir);
    let file_path = base_path.join(file_name);
    let level = log
        .level
        .as_deref()
        .and_then(|value| resolve_config_value(value, env_map))
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_LOG_LEVEL.to_string());
    let rotate_size_mb = log
        .rotate_size_mb
        .unwrap_or(DEFAULT_LOG_ROTATE_SIZE_MB)
        .max(1);

    RuntimeLogConfig {
        file_path,
        level,
        rotate_size_bytes: rotate_size_mb * 1024 * 1024,
        stdout: log.stdout.unwrap_or(DEFAULT_LOG_STDOUT),
    }
}

fn expand_home_path(path: &str) -> PathBuf {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return default_log_dir();
    }
    if trimmed == "~" {
        return home_dir();
    }
    if let Some(suffix) = trimmed.strip_prefix("~/") {
        return home_dir().join(suffix);
    }
    PathBuf::from(trimmed)
}

#[derive(Debug)]
struct SizeRotatingFileWriter {
    file_path: PathBuf,
    rotated_path: PathBuf,
    max_size_bytes: u64,
}

impl SizeRotatingFileWriter {
    fn new(file_path: PathBuf, max_size_bytes: u64) -> Result<Self, String> {
        let parent = file_path
            .parent()
            .ok_or_else(|| format!("invalid log file path {}", file_path.display()))?;
        std::fs::create_dir_all(parent).map_err(|error| {
            format!("create log directory {} failed: {error}", parent.display())
        })?;

        let mut rotated_name = file_path
            .file_name()
            .map(|value| value.to_os_string())
            .unwrap_or_else(|| std::ffi::OsString::from("pixy.log"));
        rotated_name.push(".1");
        let rotated_path = file_path.with_file_name(rotated_name);

        Ok(Self {
            file_path,
            rotated_path,
            max_size_bytes,
        })
    }

    fn maybe_rotate(&self, incoming_len: usize) -> io::Result<()> {
        if self.max_size_bytes == 0 {
            return Ok(());
        }
        let current_size = std::fs::metadata(&self.file_path)
            .map(|metadata| metadata.len())
            .unwrap_or(0);
        if current_size.saturating_add(incoming_len as u64) <= self.max_size_bytes {
            return Ok(());
        }
        if self.rotated_path.exists() {
            let _ = std::fs::remove_file(&self.rotated_path);
        }
        if self.file_path.exists() {
            std::fs::rename(&self.file_path, &self.rotated_path)?;
        }
        Ok(())
    }
}

impl Write for SizeRotatingFileWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.maybe_rotate(buf.len())?;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.file_path)?;
        file.write_all(buf)?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
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
    let session_dir = args
        .session_dir
        .as_ref()
        .map(|path| resolve_path(&cwd, path))
        .unwrap_or_else(|| default_agent_dir().join("sessions"));

    let session_factory = CliSessionFactory::new(current_pixy_home_dir());
    let session_request = CliSessionRequest {
        session_file: args.session_file.clone(),
        include_default_skills: !args.no_skills,
        skill_paths: args.skills.clone(),
        runtime_overrides: RuntimeOverrides {
            api: args.api.clone(),
            provider: args.provider.clone(),
            model: args.model.clone(),
            base_url: args.base_url.clone(),
            context_window: args.context_window,
            max_tokens: args.max_tokens,
            ..RuntimeOverrides::default()
        },
        custom_system_prompt: args.system_prompt.clone(),
        no_tools: args.no_tools,
    };
    let mut session =
        session_factory.create_session(&session_request, &cwd, &agent_dir, &session_dir)?;
    let runtime = session.runtime().clone();
    pixy_ai::set_transport_retry_count(runtime.transport_retry_count);
    for diagnostic in &runtime.skill_diagnostics {
        eprintln!(
            "warning: skill {}: {}",
            diagnostic.path.display(),
            diagnostic.message
        );
    }
    let runtime_model = runtime.model.clone();
    let discovered_skills = runtime.skills.clone();
    let use_tui = args.prompt.is_none() && !args.no_tui;

    if let Some(prompt) = args.prompt.as_deref() {
        let active_session = session.ensure_session()?;
        println!(
            "session: {}",
            active_session
                .session_file()
                .ok_or_else(|| "session file unavailable".to_string())?
                .display()
        );
        println!("cwd: {}", cwd.display());
        println!(
            "model: {}/{}/{}",
            runtime_model.api, runtime_model.provider, runtime_model.id
        );
        run_prompt_streaming_cli(active_session, prompt, !args.hide_tool_results).await?;
        return Ok(());
    }

    if use_tui {
        let theme_name = resolve_tui_theme_name(args.theme.as_deref(), runtime.theme.as_deref())?;
        let theme = TuiTheme::from_name(theme_name.as_str())
            .ok_or_else(|| format!("unsupported theme '{theme_name}', expected dark or light"))?;
        let status_top = build_status_top_line(&cwd);
        let status_left = format!(
            "0.0%/{} (auto)",
            format_token_window(runtime_model.context_window)
        );
        let status_right = format!("{} • medium", runtime_model.id);
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

    if let Some(session_file) = session.session_file() {
        println!("session: {}", session_file.display());
    }
    println!("cwd: {}", cwd.display());
    println!(
        "model: {}/{}/{}",
        runtime_model.api, runtime_model.provider, runtime_model.id
    );

    if args.continue_first {
        let active_session = session.ensure_session()?;
        run_continue_streaming_cli(active_session, !args.hide_tool_results).await?;
    }

    println!("commands: /new, /continue, /resume [session], /session, /help, /exit");
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

#[cfg(test)]
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

async fn repl_loop(session: &mut CliSession, show_tool_results: bool) -> Result<(), String> {
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

        let Some(command) = ReplCommandParser::parse(&line) else {
            continue;
        };

        match command {
            ReplCommand::NewSession => match session.start_new_session() {
                Ok(path) => println!("new session: {}", path.display()),
                Err(error) => eprintln!("new session failed: {error}"),
            },
            ReplCommand::Resume { target } => {
                let target = if let Some(target) = target {
                    Some(target)
                } else {
                    match prompt_resume_target_selection(
                        session.recent_resumable_sessions(RESUME_PICKER_LIMIT)?,
                    ) {
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
            }
            ReplCommand::Exit => return Ok(()),
            ReplCommand::Help => {
                println!("commands:");
                println!("  /new       start a new session and reset current context");
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
            ReplCommand::Session => {
                if let Some(path) = session.session_file() {
                    println!("{}", path.display());
                } else {
                    println!("(no session file)");
                }
            }
            ReplCommand::Continue => {
                let active_session = match session.ensure_session() {
                    Ok(active) => active,
                    Err(error) => {
                        eprintln!("continue failed: {error}");
                        continue;
                    }
                };
                if let Err(error) =
                    run_continue_streaming_cli(active_session, show_tool_results).await
                {
                    eprintln!("continue failed: {error}");
                }
            }
            ReplCommand::Prompt { text } => {
                let active_session = match session.ensure_session() {
                    Ok(active) => active,
                    Err(error) => {
                        eprintln!("prompt failed: {error}");
                        continue;
                    }
                };
                if let Err(error) =
                    run_prompt_streaming_cli(active_session, text.as_str(), show_tool_results).await
                {
                    eprintln!("prompt failed: {error}");
                }
            }
        }
    }
}

fn prompt_resume_target_selection(candidates: Vec<PathBuf>) -> Result<Option<PathBuf>, String> {
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
    markdown_renderer: MarkdownCodeBlockRenderer,
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
            markdown_renderer: MarkdownCodeBlockRenderer::default(),
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
                    self.write_assistant_chunk("\n")?;
                    self.thinking_line_open = false;
                }
                self.write_assistant_chunk(delta.as_str())?;
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
                    self.write_assistant_chunk("\n")?;
                    self.assistant_delta_open = false;
                    self.thinking_line_open = false;
                    self.thinking_visual_lines = 0;
                }
                if !line.is_empty() {
                    self.write_assistant_chunk(line.as_str())?;
                    self.write_assistant_chunk("\n")?;
                }
            }
            AgentSessionStreamUpdate::ToolLine(line) => {
                if !self.show_tool_results {
                    return Ok(());
                }
                if self.assistant_delta_open || self.thinking_line_open {
                    self.write_assistant_chunk("\n")?;
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
            self.write_assistant_chunk("\n")?;
            self.assistant_delta_open = false;
            self.thinking_line_open = false;
            self.thinking_visual_lines = 0;
        }
        self.markdown_renderer
            .finish(&mut self.writer)
            .and_then(|_| self.writer.flush())
            .map_err(|error| format!("stdout write failed: {error}"))?;
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

    fn write_assistant_chunk(&mut self, chunk: &str) -> Result<(), String> {
        self.markdown_renderer
            .write_chunk(&mut self.writer, chunk)
            .and_then(|_| self.writer.flush())
            .map_err(|error| format!("stdout write failed: {error}"))
    }

    #[cfg(test)]
    fn into_inner(self) -> W {
        self.writer
    }
}

#[derive(Default)]
struct MarkdownCodeBlockRenderer {
    in_code_block: bool,
    current_language: Option<String>,
    pending_line: String,
}

impl MarkdownCodeBlockRenderer {
    fn write_chunk<W: Write>(&mut self, writer: &mut W, chunk: &str) -> io::Result<()> {
        let mut start = 0usize;
        for (idx, ch) in chunk.char_indices() {
            if ch != '\n' {
                continue;
            }
            self.pending_line.push_str(&chunk[start..idx]);
            self.flush_complete_line(writer, true)?;
            start = idx + ch.len_utf8();
        }
        if start < chunk.len() {
            self.pending_line.push_str(&chunk[start..]);
            self.flush_partial_line(writer)?;
        }
        Ok(())
    }

    fn finish<W: Write>(&mut self, writer: &mut W) -> io::Result<()> {
        if self.pending_line.is_empty() {
            return Ok(());
        }
        self.flush_pending_without_newline(writer)
    }

    fn flush_complete_line<W: Write>(
        &mut self,
        writer: &mut W,
        append_newline: bool,
    ) -> io::Result<()> {
        let line = std::mem::take(&mut self.pending_line);
        if let Some(language) = parse_markdown_fence(&line) {
            if self.in_code_block {
                self.in_code_block = false;
                self.current_language = None;
            } else {
                self.in_code_block = true;
                self.current_language = language;
            }
            return Ok(());
        }
        self.write_line(writer, line.as_str(), append_newline)
    }

    fn flush_pending_without_newline<W: Write>(&mut self, writer: &mut W) -> io::Result<()> {
        let line = std::mem::take(&mut self.pending_line);
        if let Some(language) = parse_markdown_fence(&line) {
            if self.in_code_block {
                self.in_code_block = false;
                self.current_language = None;
            } else {
                self.in_code_block = true;
                self.current_language = language;
            }
            return Ok(());
        }
        self.write_line(writer, line.as_str(), false)
    }

    fn flush_partial_line<W: Write>(&mut self, writer: &mut W) -> io::Result<()> {
        if self.pending_line.is_empty() || line_might_be_markdown_fence(self.pending_line.as_str())
        {
            return Ok(());
        }
        let partial = std::mem::take(&mut self.pending_line);
        self.write_line(writer, partial.as_str(), false)
    }

    fn write_line<W: Write>(
        &self,
        writer: &mut W,
        line: &str,
        append_newline: bool,
    ) -> io::Result<()> {
        if self.in_code_block {
            write_highlighted_code_line(writer, line, self.current_language.as_deref())?;
        } else {
            write!(writer, "{line}")?;
        }
        if append_newline {
            write!(writer, "\n")?;
        }
        Ok(())
    }
}

fn parse_markdown_fence(line: &str) -> Option<Option<String>> {
    let trimmed = line.trim();
    let rest = trimmed.strip_prefix("```")?;
    let language = rest
        .split_whitespace()
        .next()
        .filter(|token| !token.is_empty())
        .map(|token| token.to_ascii_lowercase());
    Some(language)
}

fn line_might_be_markdown_fence(line: &str) -> bool {
    let trimmed = line.trim_start();
    (trimmed.starts_with('`') && "```".starts_with(trimmed)) || trimmed.starts_with("```")
}

fn write_highlighted_code_line<W: Write>(
    writer: &mut W,
    line: &str,
    language: Option<&str>,
) -> io::Result<()> {
    let chars = line.chars().collect::<Vec<_>>();
    if chars.is_empty() {
        write!(writer, "{MARKDOWN_CODE_BASE_STYLE}{ANSI_STYLE_RESET}")?;
        return Ok(());
    }

    let mut idx = 0usize;
    while idx < chars.len() {
        if idx + 1 < chars.len() && chars[idx] == '/' && chars[idx + 1] == '/' {
            let fragment = chars[idx..].iter().collect::<String>();
            write!(writer, "{MARKDOWN_CODE_COMMENT_STYLE}{fragment}")?;
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
            write!(writer, "{MARKDOWN_CODE_STRING_STYLE}{fragment}")?;
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
            write!(writer, "{MARKDOWN_CODE_NUMBER_STYLE}{fragment}")?;
            continue;
        }

        if is_identifier_start(chars[idx]) {
            let start = idx;
            idx += 1;
            while idx < chars.len() && is_identifier_continue(chars[idx]) {
                idx += 1;
            }
            let token = chars[start..idx].iter().collect::<String>();
            if is_code_keyword(token.as_str(), language) {
                write!(writer, "{MARKDOWN_CODE_KEYWORD_STYLE}{token}")?;
            } else {
                write!(writer, "{MARKDOWN_CODE_BASE_STYLE}{token}")?;
            }
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
        let fragment = chars[start..idx].iter().collect::<String>();
        write!(writer, "{MARKDOWN_CODE_BASE_STYLE}{fragment}")?;
    }

    write!(writer, "{ANSI_STYLE_RESET}")?;
    Ok(())
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
    let mut stdout = io::stdout();
    if let Err(error) = render_messages_to_writer(messages, show_tool_results, &mut stdout) {
        eprintln!("[cli_error] failed to render messages: {error}");
    }
}

fn render_messages_to_writer<W: Write>(
    messages: &[Message],
    show_tool_results: bool,
    writer: &mut W,
) -> io::Result<()> {
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
                                writeln!(writer, "{text}")?;
                            }
                        }
                        AssistantContentBlock::Thinking { thinking, .. } => {
                            if !thinking.trim().is_empty() {
                                writeln!(writer, "[thinking] {thinking}")?;
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
                writeln!(writer, "[tool:{tool_name}:{status}]")?;
                if should_render_tool_result_content(tool_name) {
                    for block in content {
                        match block {
                            ToolResultContentBlock::Text { text, .. } => {
                                writeln!(writer, "{text}")?
                            }
                            ToolResultContentBlock::Image { .. } => {
                                writeln!(writer, "(image tool result omitted)")?
                            }
                        }
                    }
                }
            }
            Message::User { .. } => {}
        }
    }
    Ok(())
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

fn resolve_path(cwd: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    }
}

fn init_conf_dir(conf_dir: Option<&Path>) {
    let resolved = resolve_pixy_home_dir(conf_dir);
    let _ = CONF_DIR.set(resolved);
}

fn resolve_pixy_home_dir(conf_dir: Option<&Path>) -> PathBuf {
    conf_dir
        .map(resolve_pixy_home_arg)
        .unwrap_or_else(default_pixy_home_dir)
}

fn resolve_pixy_home_arg(path: &Path) -> PathBuf {
    let expanded = expand_path_with_home(path);
    if expanded.is_absolute() {
        expanded
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(expanded)
    }
}

fn expand_path_with_home(path: &Path) -> PathBuf {
    let raw = path.to_string_lossy();
    if raw == "~" {
        return home_dir();
    }
    if let Some(suffix) = raw.strip_prefix("~/") {
        return home_dir().join(suffix);
    }
    path.to_path_buf()
}

fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn default_pixy_home_dir() -> PathBuf {
    home_dir().join(DEFAULT_PIXY_HOME_DIR_NAME)
}

fn current_pixy_home_dir() -> PathBuf {
    CONF_DIR
        .get()
        .cloned()
        .unwrap_or_else(|| resolve_pixy_home_dir(None))
}

fn default_log_dir() -> PathBuf {
    current_pixy_home_dir().join("logs")
}

fn default_pixy_config_path() -> PathBuf {
    current_pixy_home_dir().join("pixy.toml")
}

fn default_agent_dir() -> PathBuf {
    current_pixy_home_dir().join("agent")
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

#[cfg(test)]
#[derive(Debug, Clone, Default)]
struct AgentSettingsFile {
    default_provider: Option<String>,
    theme: Option<String>,
    transport_retry_count: Option<usize>,
    skills: Vec<String>,
    env: HashMap<String, String>,
}

#[cfg(test)]
#[derive(Debug, Clone, Default)]
struct ModelsFile {
    providers: HashMap<String, ProviderConfig>,
}

#[cfg(test)]
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

#[cfg(test)]
fn default_provider_weight() -> u8 {
    1
}

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
#[derive(Debug, Clone, Default, Deserialize)]
struct PixyTomlLlm {
    #[serde(default)]
    default_provider: Option<String>,
    #[serde(default)]
    providers: Vec<PixyTomlProvider>,
}

#[cfg(test)]
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

#[cfg(test)]
#[derive(Debug, Clone, Default)]
struct AgentLocalConfig {
    settings: AgentSettingsFile,
    models: ModelsFile,
}

#[cfg(test)]
#[derive(Debug, Clone)]
struct ResolvedRuntimeConfig {
    model: Model,
    model_catalog: Vec<Model>,
    api_key: Option<String>,
}

#[cfg(test)]
#[derive(Debug, Clone)]
pub struct LLMRouter {
    slots: Vec<Slot>,
}

#[cfg(test)]
#[derive(Debug, Clone)]
struct Slot {
    provider: String,
    weight: u8,
}

#[cfg(test)]
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

#[cfg(test)]
fn load_agent_local_config(_agent_dir: &Path) -> Result<AgentLocalConfig, String> {
    let config_path = default_pixy_config_path();
    let config = read_toml_if_exists::<PixyTomlFile>(&config_path)?;
    Ok(config
        .map(convert_pixy_toml_to_local_config)
        .unwrap_or_default())
}

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
fn resolve_runtime_config_for_tests(
    args: &ChatArgs,
    local: &AgentLocalConfig,
) -> Result<ResolvedRuntimeConfig, String> {
    resolve_runtime_config_for_tests_with_seed(args, local, runtime_router_seed())
}

#[cfg(test)]
fn runtime_router_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
fn resolve_runtime_config_for_tests_with_seed(
    args: &ChatArgs,
    local: &AgentLocalConfig,
    router_seed: u64,
) -> Result<ResolvedRuntimeConfig, String> {
    CliRuntimeConfigResolver::new(args, local, router_seed).resolve_runtime_config_with_seed()
}

#[cfg(test)]
struct CliRuntimeConfigResolver<'a> {
    args: &'a ChatArgs,
    local: &'a AgentLocalConfig,
    router_seed: u64,
}

#[cfg(test)]
impl<'a> CliRuntimeConfigResolver<'a> {
    fn new(args: &'a ChatArgs, local: &'a AgentLocalConfig, router_seed: u64) -> Self {
        Self {
            args,
            local,
            router_seed,
        }
    }

    fn resolve_runtime_config_with_seed(&self) -> Result<ResolvedRuntimeConfig, String> {
        self.resolve_runtime_config_with_seed_impl()
    }

    fn resolve_runtime_config_with_seed_impl(&self) -> Result<ResolvedRuntimeConfig, String> {
        let cli_model_parts = split_provider_model(self.args.model.as_deref());
        let explicit_provider =
            first_non_empty([self.args.provider.clone(), cli_model_parts.0.clone()]);
        let routed_provider = if explicit_provider.is_none() {
            resolve_weighted_provider_selection(self.local, self.router_seed)?
        } else {
            None
        };
        let settings_provider = self
            .local
            .settings
            .default_provider
            .clone()
            .filter(|value| value.trim() != "*");

        let provider = first_non_empty([
            explicit_provider,
            routed_provider,
            settings_provider,
            infer_single_provider(self.local),
            Some("openai".to_string()),
        ])
        .ok_or_else(|| "Unable to resolve provider".to_string())?;

        let provider_config = self.local.models.providers.get(&provider);
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
            self.args.api.clone(),
            selected_model_cfg.and_then(|model| model.api.clone()),
            provider_config.and_then(|provider| provider.api.clone()),
            infer_api_for_provider(&provider_name),
        ])
        .ok_or_else(|| format!("Unable to resolve API for provider '{provider}'"))?;

        let cli_base_url = self
            .args
            .base_url
            .as_ref()
            .and_then(|value| resolve_config_value(value, &self.local.settings.env));
        let model_base_url = selected_model_cfg
            .and_then(|model| model.base_url.as_ref())
            .and_then(|value| resolve_config_value(value, &self.local.settings.env));
        let provider_base_url = provider_config
            .and_then(|provider| provider.base_url.as_ref())
            .and_then(|value| resolve_config_value(value, &self.local.settings.env));
        let base_url = first_non_empty([
            cli_base_url,
            model_base_url,
            provider_base_url,
            default_base_url_for_api(&api),
        ])
        .ok_or_else(|| format!("Unable to resolve base URL for api '{api}'"))?;

        let context_window = self
            .args
            .context_window
            .or_else(|| selected_model_cfg.and_then(|model| model.context_window))
            .unwrap_or(200_000);
        let max_tokens = self
            .args
            .max_tokens
            .or_else(|| selected_model_cfg.and_then(|model| model.max_tokens))
            .unwrap_or(8_192);

        let api_key = provider_config
            .and_then(|provider_cfg| provider_cfg.api_key.as_ref())
            .and_then(|value| resolve_config_value(value, &self.local.settings.env))
            .or_else(|| infer_api_key_from_settings(&provider_name, &self.local.settings.env))
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
}

#[cfg(test)]
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
        "default_provider='*' requires at least one chat provider with non-zero weight in <conf_dir>/pixy.toml".to_string()
    })?;
    Ok(Some(provider))
}

#[cfg(test)]
fn provider_config_default_model(provider_config: Option<&ProviderConfig>) -> Option<String> {
    provider_config.and_then(|config| {
        first_non_empty([
            config.default_model.clone(),
            config.models.first().map(|entry| entry.id.clone()),
        ])
    })
}

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
fn is_chat_provider(config: &ProviderConfig) -> bool {
    config
        .kind
        .as_deref()
        .map(str::trim)
        .map_or(true, |kind| kind.eq_ignore_ascii_case("chat"))
}

#[cfg(test)]
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

#[cfg(test)]
fn default_reasoning_enabled_for_api(api: &str) -> bool {
    matches!(
        api,
        "openai-responses"
            | "openai-completions"
            | "openai-codex-responses"
            | "azure-openai-responses"
    )
}

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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
    fn cli_accepts_explicit_cli_subcommand() {
        let parsed = Cli::try_parse_from(["pixy", "cli", "--no-tui"]);
        assert!(
            parsed.is_ok(),
            "pixy cli should be accepted as explicit CLI entrypoint"
        );
    }

    #[test]
    fn cli_accepts_gateway_start_daemon_subcommand() {
        let parsed = Cli::try_parse_from(["pixy", "gateway", "start", "--daemon"]);
        assert!(
            parsed.is_ok(),
            "pixy gateway start --daemon should be accepted"
        );
    }

    #[test]
    fn cli_accepts_conf_dir_global_flag() {
        let parsed =
            Cli::try_parse_from(["pixy", "--conf-dir", "/tmp/pixy-conf", "gateway", "start"]);
        assert!(
            parsed.is_ok(),
            "pixy should accept --conf-dir as global flag"
        );
    }

    #[test]
    fn gateway_command_tokens_include_daemon_flag_when_requested() {
        let tokens = gateway_command_tokens(
            &GatewaySubcommand::Start(GatewayStartArgs { daemon: true }),
            None,
        );
        assert_eq!(tokens, vec!["start".to_string(), "--daemon".to_string()]);
    }

    #[test]
    fn gateway_command_tokens_encode_restart_subcommand() {
        let tokens = gateway_command_tokens(&GatewaySubcommand::Restart, None);
        assert_eq!(tokens, vec!["restart".to_string()]);
    }

    #[test]
    fn gateway_command_tokens_include_conf_dir_when_provided() {
        let conf_dir = Path::new("/tmp/pixy-conf");
        let tokens = gateway_command_tokens(&GatewaySubcommand::Stop, Some(conf_dir));
        assert_eq!(
            tokens,
            vec![
                "--conf-dir".to_string(),
                "/tmp/pixy-conf".to_string(),
                "stop".to_string(),
            ]
        );
    }

    #[test]
    fn new_repl_command_matches_exact_new_token() {
        use crate::cli_app::{ReplCommand, ReplCommandParser};

        assert_eq!(
            ReplCommandParser::parse("/new"),
            Some(ReplCommand::NewSession)
        );
        assert_eq!(
            ReplCommandParser::parse(" /new "),
            Some(ReplCommand::NewSession)
        );
        assert_eq!(
            ReplCommandParser::parse("/new task"),
            Some(ReplCommand::Prompt {
                text: "/new task".to_string()
            })
        );
        assert_eq!(
            ReplCommandParser::parse("new"),
            Some(ReplCommand::Prompt {
                text: "new".to_string()
            })
        );
    }

    #[test]
    fn repl_command_parser_recognizes_control_commands() {
        use crate::cli_app::{ReplCommand, ReplCommandParser};

        assert_eq!(
            ReplCommandParser::parse("/new"),
            Some(ReplCommand::NewSession)
        );
        assert_eq!(
            ReplCommandParser::parse("/resume"),
            Some(ReplCommand::Resume { target: None })
        );
        assert_eq!(
            ReplCommandParser::parse("/resume  abc.jsonl "),
            Some(ReplCommand::Resume {
                target: Some("abc.jsonl".to_string())
            })
        );
        assert_eq!(
            ReplCommandParser::parse("/continue"),
            Some(ReplCommand::Continue)
        );
        assert_eq!(
            ReplCommandParser::parse("/session"),
            Some(ReplCommand::Session)
        );
        assert_eq!(ReplCommandParser::parse("/help"), Some(ReplCommand::Help));
        assert_eq!(ReplCommandParser::parse("/quit"), Some(ReplCommand::Exit));
        assert_eq!(ReplCommandParser::parse("/exit"), Some(ReplCommand::Exit));
    }

    #[test]
    fn repl_command_parser_treats_other_text_as_prompt() {
        use crate::cli_app::{ReplCommand, ReplCommandParser};

        assert_eq!(
            ReplCommandParser::parse(" summarize this file "),
            Some(ReplCommand::Prompt {
                text: "summarize this file".to_string()
            })
        );
        assert_eq!(ReplCommandParser::parse(""), None);
        assert_eq!(ReplCommandParser::parse("   "), None);
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
            resolve_runtime_config_for_tests(&args, &local).expect("runtime config should resolve");
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

        let weighted_anthropic = resolve_runtime_config_for_tests_with_seed(&args, &local, 10)
            .expect("runtime config should resolve");
        assert_eq!(weighted_anthropic.model.provider, "anthropic");

        let weighted_openai = resolve_runtime_config_for_tests_with_seed(&args, &local, 90)
            .expect("runtime config should resolve");
        assert_eq!(weighted_openai.model.provider, "openai");
    }

    #[test]
    fn resolve_runtime_config_matches_runtime_module_with_same_seed() {
        let dir = tempdir().expect("tempdir");
        let pixy_dir = dir.path().join(".pixy");
        std::fs::create_dir_all(&pixy_dir).expect("create .pixy");
        std::fs::write(
            pixy_dir.join("pixy.toml"),
            r#"
[env]
PIXY_TEST_OPENAI_KEY = "sk-from-settings"

[llm]
default_provider = "*"

[[llm.providers]]
name = "openai"
kind = "chat"
provider = "openai"
api = "openai-responses"
base_url = "https://api.openai.com/v1"
api_key = "$PIXY_TEST_OPENAI_KEY"
model = "gpt-5.3-codex"
weight = 90

[[llm.providers]]
name = "anthropic"
kind = "chat"
provider = "anthropic"
api = "anthropic-messages"
base_url = "https://api.anthropic.com/v1"
api_key = "sk-anthropic"
model = "claude-3-5-sonnet-latest"
weight = 10
"#,
        )
        .expect("write pixy.toml");

        unsafe {
            std::env::set_var("HOME", dir.path());
        }

        let args = test_chat_args();
        let local = load_agent_local_config(Path::new(".")).expect("load local config");
        let cli_resolved = resolve_runtime_config_for_tests_with_seed(&args, &local, 42)
            .expect("cli runtime config should resolve");

        let runtime_options = RuntimeLoadOptions {
            conf_dir: Some(pixy_dir.clone()),
            agent_dir: Some(dir.path().join("agent")),
            load_skills: false,
            include_default_skills: false,
            skill_paths: vec![],
            overrides: RuntimeOverrides {
                api: args.api.clone(),
                provider: args.provider.clone(),
                model: args.model.clone(),
                base_url: args.base_url.clone(),
                context_window: args.context_window,
                max_tokens: args.max_tokens,
                ..RuntimeOverrides::default()
            },
        };
        let runtime_resolved = runtime_options
            .resolve_runtime_with_seed(Path::new("."), 42)
            .expect("runtime module should resolve");

        assert_eq!(cli_resolved.model.provider, runtime_resolved.model.provider);
        assert_eq!(cli_resolved.model.id, runtime_resolved.model.id);
        assert_eq!(cli_resolved.model.api, runtime_resolved.model.api);
        assert_eq!(cli_resolved.model.base_url, runtime_resolved.model.base_url);
        assert_eq!(
            cli_resolved.model.reasoning,
            runtime_resolved.model.reasoning
        );
        assert_eq!(
            cli_resolved.model.reasoning_effort,
            runtime_resolved.model.reasoning_effort
        );
        assert_eq!(cli_resolved.api_key, runtime_resolved.api_key);
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

        let error = resolve_runtime_config_for_tests_with_seed(&args, &local, 0)
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

        let error = resolve_runtime_config_for_tests_with_seed(&args, &local, 0)
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

        let resolved = resolve_runtime_config_for_tests_with_seed(&args, &local, 0)
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

        let resolved = resolve_runtime_config_for_tests_with_seed(&args, &local, 0)
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
            resolve_runtime_config_for_tests(&args, &local).expect("runtime config should resolve");
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
            resolve_runtime_config_for_tests(&args, &local).expect("runtime config should resolve");
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
            resolve_runtime_config_for_tests(&args, &local).expect("runtime config should resolve");
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
            resolve_runtime_config_for_tests(&args, &local).expect("runtime config should resolve");
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
            resolve_runtime_config_for_tests(&args, &local).expect("runtime config should resolve");
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
    fn render_messages_omits_read_tool_text_blocks() {
        let messages = vec![Message::ToolResult {
            tool_call_id: "call-read-1".to_string(),
            tool_name: "read".to_string(),
            content: vec![ToolResultContentBlock::Text {
                text: "secret file body".to_string(),
                text_signature: None,
            }],
            details: None,
            is_error: false,
            timestamp: 1_700_000_000_010,
        }];

        let mut output = Vec::<u8>::new();
        render_messages_to_writer(&messages, true, &mut output).expect("render succeeds");
        assert_eq!(
            String::from_utf8(output).expect("utf-8"),
            "[tool:read:ok]\n",
            "read tool output body should be hidden in CLI message rendering"
        );
    }

    #[test]
    fn render_messages_keeps_non_read_tool_text_blocks() {
        let messages = vec![Message::ToolResult {
            tool_call_id: "call-write-1".to_string(),
            tool_name: "write".to_string(),
            content: vec![ToolResultContentBlock::Text {
                text: "write diff stat".to_string(),
                text_signature: None,
            }],
            details: None,
            is_error: false,
            timestamp: 1_700_000_000_020,
        }];

        let mut output = Vec::<u8>::new();
        render_messages_to_writer(&messages, true, &mut output).expect("render succeeds");
        assert_eq!(
            String::from_utf8(output).expect("utf-8"),
            "[tool:write:ok]\nwrite diff stat\n"
        );
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
    fn cli_stream_renderer_renders_fenced_code_blocks_with_markdown_style() {
        let mut renderer = CliStreamRenderer::new(Vec::<u8>::new(), true);

        renderer
            .on_update(AgentSessionStreamUpdate::AssistantTextDelta(
                "Before\n```rust\nfn main() {\n".to_string(),
            ))
            .expect("delta write succeeds");
        renderer
            .on_update(AgentSessionStreamUpdate::AssistantTextDelta(
                "    println!(\"hi\");\n}\n```\nAfter".to_string(),
            ))
            .expect("delta write succeeds");
        renderer.finish().expect("finish succeeds");

        let output = String::from_utf8(renderer.into_inner()).expect("utf-8 output");
        assert!(output.contains("Before\n"));
        assert!(output.contains("\u{1b}[48;5;236m\u{1b}[38;5;111mfn"));
        assert!(
            output.contains("\u{1b}[48;5;236m\u{1b}[38;5;150m\"hi\""),
            "string token should use markdown code string style"
        );
        assert!(
            output.contains("println"),
            "code block content should still be present"
        );
        assert!(
            !output.contains("```"),
            "fence markers should not be printed"
        );
        assert!(output.ends_with("After\n"));
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
            resolve_runtime_config_for_tests(&args, &local).expect("runtime config should resolve");
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
            resolve_runtime_config_for_tests(&args, &local).expect("runtime config should resolve");
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
    fn parse_runtime_log_config_from_toml_reads_log_section() {
        let config = parse_runtime_log_config_from_toml(
            r#"
[env]
LOG_LEVEL = "debug"

[log]
path = "/tmp/pixy-logs"
level = "$LOG_LEVEL"
rotate_size_mb = 12
stdout = true
"#,
            "pixy.log",
        )
        .expect("log config should parse");

        assert_eq!(config.file_path, PathBuf::from("/tmp/pixy-logs/pixy.log"));
        assert_eq!(config.level, "debug");
        assert_eq!(config.rotate_size_bytes, 12 * 1024 * 1024);
        assert!(config.stdout);
    }

    #[test]
    fn parse_runtime_log_config_from_toml_uses_defaults_when_missing() {
        let config = parse_runtime_log_config_from_toml(
            r#"
[llm]
default_provider = "openai"
"#,
            "pixy.log",
        )
        .expect("config without log section should still parse");

        assert!(config.file_path.ends_with(Path::new(".pixy/logs/pixy.log")));
        assert_eq!(config.level, "info");
        assert_eq!(config.rotate_size_bytes, 100 * 1024 * 1024);
        assert!(!config.stdout);
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

        let resolved = resolve_runtime_config_for_tests_with_seed(&args, &local, 0)
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
            resolve_runtime_config_for_tests(&args, &local).expect("runtime config should resolve");
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
                source: SkillSource::Project,
                disable_model_invocation: false,
            },
            Skill {
                name: "path-skill".to_string(),
                description: "path".to_string(),
                file_path: cwd.join("custom/path-skill/SKILL.md"),
                base_dir: cwd.join("custom/path-skill"),
                source: SkillSource::Path,
                disable_model_invocation: false,
            },
            Skill {
                name: "user-skill".to_string(),
                description: "user".to_string(),
                file_path: agent_dir.join("skills/user/SKILL.md"),
                base_dir: agent_dir.join("skills/user"),
                source: SkillSource::User,
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
