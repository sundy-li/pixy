use std::collections::HashMap;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use crate::cli_app::{
    CliSession, CliSessionFactory, CliSessionRequest, ReplCommand, ReplCommandParser,
};
use crate::{AgentMode, AgentSession, AgentSessionStreamUpdate, RuntimeOverrides, Skill};
use clap::{Args, Parser, Subcommand};
use pixy_ai::{AssistantContentBlock, Message, StopReason, ToolResultContentBlock};
use pixy_tui::{parse_key_id, KeyBinding, TuiKeyBindings, TuiOptions, TuiTheme};
use serde::Deserialize;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

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
        let status_left = AgentMode::default().label().to_string();
        let status_right = format_status_model_label(
            runtime_model.provider.as_str(),
            runtime_model.id.as_str(),
            runtime_model.reasoning,
        );
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
            show_tool_results: false,
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
    cwd.display().to_string()
}

fn format_status_model_label(provider_name: &str, model: &str, _reasoning: bool) -> String {
    let provider_name = provider_name.trim();
    let model = model.trim();
    if provider_name.is_empty() && model.is_empty() {
        return "unknown".to_string();
    }
    if provider_name.is_empty() {
        return model.to_string();
    }
    if model.is_empty() {
        return provider_name.to_string();
    }
    format!("{provider_name}:{model}")
}

fn build_startup_resource_lines(cwd: &Path, agent_dir: &Path, skills: &[Skill]) -> Vec<String> {
    let _ = cwd;
    let _ = agent_dir;
    let _ = skills;
    vec![]
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
    current_pixy_home_dir().join("agents")
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

    if changed {
        Some(keybindings)
    } else {
        None
    }
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
#[path = "../tests/unit/cli_unit.rs"]
mod tests;
