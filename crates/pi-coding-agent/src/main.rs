use std::collections::HashMap;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use clap::{Args, Parser};
use pi_ai::{
    AssistantContentBlock, Cost, Message, Model, SimpleStreamOptions, StopReason,
    ToolResultContentBlock,
};
use pi_coding_agent::{
    AgentSession, AgentSessionConfig, AgentSessionStreamUpdate, SessionManager, create_coding_tools,
};
use pi_tui::{KeyBinding, TuiKeyBindings, TuiOptions, parse_key_id};
use serde::Deserialize;

#[derive(Parser, Debug)]
#[command(name = "pi", version, about = "pi-rs interactive CLI")]
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
    #[arg(
        long,
        default_value = "You are pi, a pragmatic coding assistant. Keep responses concise and actionable."
    )]
    system_prompt: String,
    #[arg(long)]
    prompt: Option<String>,
    #[arg(long, default_value_t = false)]
    continue_first: bool,
    #[arg(long, default_value_t = false)]
    no_tools: bool,
    #[arg(long, default_value_t = false)]
    hide_tool_results: bool,
    #[arg(long, default_value_t = false)]
    no_tui: bool,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    if let Err(error) = run(cli.chat).await {
        eprintln!("error: {error}");
        std::process::exit(1);
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
    let local_config = load_agent_local_config(&agent_dir)?;
    let runtime = resolve_runtime_config(&args, &local_config)?;

    let session_dir = args
        .session_dir
        .as_ref()
        .map(|path| resolve_path(&cwd, path))
        .unwrap_or_else(|| default_agent_dir().join("sessions"));
    let mut session = create_session(&args, &runtime, &cwd, &session_dir)?;
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
        let status_top = build_status_top_line(&cwd);
        let status_left = format!(
            "0.0%/{} (auto)",
            format_token_window(runtime.model.context_window)
        );
        let status_right = format!("{} • medium", runtime.model.id);
        let mut tui_options = TuiOptions {
            app_name: "pi".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            show_tool_results: !args.hide_tool_results,
            status_top,
            status_left,
            status_right,
            ..TuiOptions::default()
        };
        if let Some(keybindings) = load_tui_keybindings(&agent_dir) {
            tui_options.keybindings = keybindings;
        }
        return pi_tui::run_tui(&mut session, tui_options).await;
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

fn create_session(
    args: &ChatArgs,
    runtime: &ResolvedRuntimeConfig,
    cwd: &Path,
    session_dir: &Path,
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
        move |model: Model, context: pi_ai::Context, options: Option<SimpleStreamOptions>| {
            let mut resolved_options = options.unwrap_or_default();
            if resolved_options.stream.api_key.is_none() {
                resolved_options.stream.api_key = runtime_api_key.clone();
            }
            pi_ai::stream_simple(model, context, Some(resolved_options))
        },
    );

    let config = AgentSessionConfig {
        model: runtime.model.clone(),
        system_prompt: args.system_prompt.clone(),
        stream_fn,
        tools,
    };
    let mut session = AgentSession::new(session_manager, config);
    session.set_model_catalog(runtime.model_catalog.clone());
    Ok(session)
}

async fn repl_loop(session: &mut AgentSession, show_tool_results: bool) -> Result<(), String> {
    loop {
        print!("pi> ");
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
            let target = input
                .strip_prefix("/resume")
                .map(str::trim)
                .filter(|value| !value.is_empty());
            match session.resume(target) {
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
                    "  /resume [session]  resume from latest history session or a specific session file"
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
}

impl<W: Write> CliStreamRenderer<W> {
    fn new(writer: W, show_tool_results: bool) -> Self {
        Self {
            writer,
            show_tool_results,
            saw_updates: false,
            assistant_delta_open: false,
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
                write!(self.writer, "{delta}")
                    .and_then(|_| self.writer.flush())
                    .map_err(|error| format!("stdout write failed: {error}"))?;
                self.assistant_delta_open = true;
            }
            AgentSessionStreamUpdate::AssistantLine(line) => {
                if self.assistant_delta_open {
                    writeln!(self.writer)
                        .map_err(|error| format!("stdout write failed: {error}"))?;
                    self.assistant_delta_open = false;
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
                if self.assistant_delta_open {
                    writeln!(self.writer)
                        .map_err(|error| format!("stdout write failed: {error}"))?;
                    self.assistant_delta_open = false;
                }
                writeln!(self.writer, "{line}")
                    .map_err(|error| format!("stdout write failed: {error}"))?;
            }
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<(), String> {
        if self.assistant_delta_open {
            writeln!(self.writer).map_err(|error| format!("stdout write failed: {error}"))?;
            self.assistant_delta_open = false;
        }
        Ok(())
    }

    #[cfg(test)]
    fn into_inner(self) -> W {
        self.writer
    }
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

fn default_agent_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".pi/agent")
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

#[derive(Debug, Clone, Default, Deserialize)]
struct AgentSettingsFile {
    #[serde(rename = "defaultProvider")]
    default_provider: Option<String>,
    #[serde(rename = "defaultModel")]
    default_model: Option<String>,
    #[serde(default)]
    env: HashMap<String, String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct ModelsFile {
    #[serde(default)]
    providers: HashMap<String, ProviderConfig>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct ProviderConfig {
    api: Option<String>,
    #[serde(rename = "baseUrl")]
    base_url: Option<String>,
    #[serde(rename = "apiKey")]
    api_key: Option<String>,
    #[serde(default)]
    models: Vec<ProviderModelConfig>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct ProviderModelConfig {
    id: String,
    api: Option<String>,
    #[serde(rename = "baseUrl")]
    base_url: Option<String>,
    #[serde(rename = "contextWindow")]
    context_window: Option<u32>,
    #[serde(rename = "maxTokens")]
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

fn load_agent_local_config(agent_dir: &Path) -> Result<AgentLocalConfig, String> {
    let settings_path = agent_dir.join("settings.json");
    let models_path = agent_dir.join("models.json");

    let settings = read_json_if_exists::<AgentSettingsFile>(&settings_path)?;
    let models = read_json_if_exists::<ModelsFile>(&models_path)?;

    Ok(AgentLocalConfig {
        settings: settings.unwrap_or_default(),
        models: models.unwrap_or_default(),
    })
}

fn read_json_if_exists<T>(path: &Path) -> Result<Option<T>, String>
where
    T: for<'de> Deserialize<'de>,
{
    if !path.exists() {
        return Ok(None);
    }

    let content = std::fs::read_to_string(path)
        .map_err(|error| format!("read {} failed: {error}", path.display()))?;
    let parsed = serde_json::from_str::<T>(&content)
        .map_err(|error| format!("parse {} failed: {error}", path.display()))?;
    Ok(Some(parsed))
}

fn resolve_runtime_config(
    args: &ChatArgs,
    local: &AgentLocalConfig,
) -> Result<ResolvedRuntimeConfig, String> {
    let cli_model_parts = split_provider_model(args.model.as_deref());
    let settings_model_parts = split_provider_model(local.settings.default_model.as_deref());

    let provider = first_non_empty([
        args.provider.clone(),
        cli_model_parts.0.clone(),
        local.settings.default_provider.clone(),
        settings_model_parts.0.clone(),
        infer_single_provider(local),
        Some("openai".to_string()),
    ])
    .ok_or_else(|| "Unable to resolve provider".to_string())?;

    let model_id = first_non_empty([
        cli_model_parts.1.clone(),
        settings_model_parts.1.clone(),
        default_model_for_provider(&provider),
    ])
    .ok_or_else(|| format!("Unable to resolve model for provider '{provider}'"))?;

    let provider_config = local.models.providers.get(&provider);
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
        infer_api_for_provider(&provider),
    ])
    .ok_or_else(|| format!("Unable to resolve API for provider '{provider}'"))?;

    let base_url = first_non_empty([
        args.base_url.clone(),
        selected_model_cfg.and_then(|model| model.base_url.clone()),
        provider_config.and_then(|provider| provider.base_url.clone()),
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
        .or_else(|| infer_api_key_from_settings(&provider, &local.settings.env))
        .or_else(|| std::env::var(primary_env_key_for_provider(&provider)).ok());

    let model = Model {
        id: model_id.clone(),
        name: model_id,
        api,
        provider,
        base_url,
        reasoning: false,
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
                        &model.provider,
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

fn infer_single_provider(local: &AgentLocalConfig) -> Option<String> {
    if local.models.providers.len() == 1 {
        local.models.providers.keys().next().cloned()
    } else {
        None
    }
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

    Model {
        id: config.id.clone(),
        name: config.id.clone(),
        api,
        provider: provider.to_string(),
        base_url,
        reasoning: false,
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
        "openai" | "openai-completions" => Some("openai-completions".to_string()),
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
        "openai" | "openai-completions" => Some("gpt-4.1-mini".to_string()),
        "openai-responses" | "azure-openai" | "azure-openai-responses" => {
            Some("gpt-4.1-mini".to_string())
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
    use super::*;
    use tempfile::tempdir;

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
            system_prompt: "test".to_string(),
            prompt: None,
            continue_first: false,
            no_tools: false,
            hide_tool_results: false,
            no_tui: false,
        };

        let local = AgentLocalConfig {
            settings: AgentSettingsFile {
                default_provider: Some("anthropic".to_string()),
                default_model: Some("claude-opus-4-6".to_string()),
                env: HashMap::from([(
                    "ANTHROPIC_AUTH_TOKEN".to_string(),
                    "token-from-settings".to_string(),
                )]),
            },
            models: ModelsFile {
                providers: HashMap::from([(
                    "anthropic".to_string(),
                    ProviderConfig {
                        api: None,
                        base_url: Some("https://custom.anthropic.local".to_string()),
                        api_key: Some("model-json-token".to_string()),
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
            system_prompt: "test".to_string(),
            prompt: None,
            continue_first: false,
            no_tools: false,
            hide_tool_results: false,
            no_tui: false,
        };

        let local = AgentLocalConfig {
            settings: AgentSettingsFile {
                default_provider: Some("anthropic".to_string()),
                default_model: Some("claude-opus-4-6".to_string()),
                env: HashMap::new(),
            },
            models: ModelsFile::default(),
        };

        let resolved =
            resolve_runtime_config(&args, &local).expect("runtime config should resolve");
        assert_eq!(resolved.model.provider, "openai");
        assert_eq!(resolved.model.api, "openai-completions");
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
            system_prompt: "test".to_string(),
            prompt: None,
            continue_first: false,
            no_tools: false,
            hide_tool_results: false,
            no_tui: false,
        };

        let local = AgentLocalConfig {
            settings: AgentSettingsFile {
                default_provider: None,
                default_model: None,
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
    fn resolve_runtime_config_builds_model_catalog_with_selected_model_first() {
        let args = ChatArgs {
            api: None,
            provider: Some("openai".to_string()),
            model: Some("gpt-4.1-mini".to_string()),
            base_url: None,
            context_window: None,
            max_tokens: None,
            agent_dir: None,
            cwd: None,
            session_dir: None,
            session_file: None,
            system_prompt: "test".to_string(),
            prompt: None,
            continue_first: false,
            no_tools: false,
            hide_tool_results: false,
            no_tui: false,
        };

        let local = AgentLocalConfig {
            settings: AgentSettingsFile::default(),
            models: ModelsFile {
                providers: HashMap::from([(
                    "openai".to_string(),
                    ProviderConfig {
                        api: Some("openai-completions".to_string()),
                        base_url: Some("https://api.openai.com/v1".to_string()),
                        api_key: None,
                        models: vec![
                            ProviderModelConfig {
                                id: "gpt-4.1".to_string(),
                                api: None,
                                base_url: None,
                                context_window: None,
                                max_tokens: None,
                            },
                            ProviderModelConfig {
                                id: "gpt-4.1-mini".to_string(),
                                api: None,
                                base_url: None,
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
        assert_eq!(resolved.model_catalog[0].id, "gpt-4.1-mini");
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
  "followUp": "alt+enter"
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
    }

    #[test]
    fn load_tui_keybindings_ignores_invalid_json() {
        let dir = tempdir().expect("tempdir");
        let config_path = dir.path().join("keybindings.json");
        std::fs::write(&config_path, "{").expect("write invalid keybindings");

        assert!(load_tui_keybindings(dir.path()).is_none());
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
            system_prompt: "test".to_string(),
            prompt: None,
            continue_first: false,
            no_tools: false,
            hide_tool_results: false,
            no_tui: false,
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
}
