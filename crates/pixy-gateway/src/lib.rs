use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::Deserialize;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

pub mod channels;
pub mod config;
pub mod runtime;

const GATEWAY_RUNTIME_DIR_ENV: &str = "PIXY_GATEWAY_DIR";
const STOP_WAIT_TIMEOUT: Duration = Duration::from_secs(2);
const STOP_POLL_INTERVAL: Duration = Duration::from_millis(50);
const DEFAULT_LOG_LEVEL: &str = "info";
const DEFAULT_LOG_ROTATE_SIZE_MB: u64 = 100;
const DEFAULT_LOG_STDOUT: bool = false;
pub const DEFAULT_PROMPT_INTRO: &str = "You are pixy, an expert coding assistant and coding agent harness running via pixy gateway service. You must help users from {channel}, you can help users by reading files, executing commands, editing code, and writing new files.";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GatewayCommand {
    Start(GatewayStartOptions),
    Stop,
    Restart,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GatewayStartOptions {
    pub daemon: bool,
}

pub fn init_conf_dir(conf_dir: Option<PathBuf>) {
    config::init_conf_dir(conf_dir);
}

#[derive(Debug, Clone)]
struct GatewayRuntimePaths {
    runtime_dir: PathBuf,
    pid_file: PathBuf,
    state_file: PathBuf,
}

impl GatewayRuntimePaths {
    fn resolve() -> Self {
        let runtime_dir = std::env::var_os(GATEWAY_RUNTIME_DIR_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(default_runtime_dir);
        Self {
            pid_file: runtime_dir.join("gateway.pid"),
            state_file: runtime_dir.join("gateway.state.json"),
            runtime_dir,
        }
    }

    fn ensure_runtime_dir(&self) -> Result<(), String> {
        fs::create_dir_all(&self.runtime_dir).map_err(|error| {
            format!(
                "create gateway runtime dir {} failed: {error}",
                self.runtime_dir.display()
            )
        })
    }

    fn cleanup_runtime_files(&self) -> Result<(), String> {
        remove_if_exists(&self.pid_file)?;
        remove_if_exists(&self.state_file)?;
        Ok(())
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
        fs::create_dir_all(parent).map_err(|error| {
            format!("create log directory {} failed: {error}", parent.display())
        })?;

        let mut rotated_name = file_path
            .file_name()
            .map(|value| value.to_os_string())
            .unwrap_or_else(|| std::ffi::OsString::from("gateway.log"));
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
        let current_size = fs::metadata(&self.file_path)
            .map(|metadata| metadata.len())
            .unwrap_or(0);
        if current_size.saturating_add(incoming_len as u64) <= self.max_size_bytes {
            return Ok(());
        }
        if self.rotated_path.exists() {
            let _ = fs::remove_file(&self.rotated_path);
        }
        if self.file_path.exists() {
            fs::rename(&self.file_path, &self.rotated_path)?;
        }
        Ok(())
    }
}

impl Write for SizeRotatingFileWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.maybe_rotate(buf.len())?;
        let mut file = fs::OpenOptions::new()
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

pub fn init_tracing() {
    static TRACE_GUARD: OnceLock<WorkerGuard> = OnceLock::new();

    let config = load_runtime_log_config("gateway.log");
    let file_writer =
        match SizeRotatingFileWriter::new(config.file_path.clone(), config.rotate_size_bytes) {
            Ok(writer) => writer,
            Err(error) => {
                eprintln!("warning: failed to initialize gateway tracing writer: {error}");
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
    let stdout_layer = tracing_subscriber::fmt::layer().with_ansi(false);
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
            "warning: failed to initialize gateway tracing subscriber for {}: {error}",
            config.file_path.display()
        );
    }
}

fn load_runtime_log_config(file_name: &str) -> RuntimeLogConfig {
    let path = config::default_pixy_config_path();
    let parsed = if path.exists() {
        fs::read_to_string(&path)
            .ok()
            .and_then(|content| toml::from_str::<PixyTomlLogFile>(&content).ok())
            .unwrap_or_default()
    } else {
        PixyTomlLogFile::default()
    };
    build_runtime_log_config(&parsed.log, &parsed.env, file_name)
}

fn build_runtime_log_config(
    log: &PixyTomlLog,
    env_map: &HashMap<String, String>,
    file_name: &str,
) -> RuntimeLogConfig {
    let path = log
        .path
        .as_deref()
        .and_then(|value| resolve_config_value(value, env_map))
        .map(|value| expand_home_path(value.trim()))
        .unwrap_or_else(default_log_dir);
    let file_path = path.join(file_name);
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

fn expand_home_path(path: &str) -> PathBuf {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return default_log_dir();
    }
    if trimmed == "~" {
        return std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));
    }
    if let Some(suffix) = trimmed.strip_prefix("~/") {
        return std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."))
            .join(suffix);
    }
    PathBuf::from(trimmed)
}

fn default_log_dir() -> PathBuf {
    config::current_pixy_home_dir().join("logs")
}

pub async fn run_gateway_command(command: GatewayCommand) -> Result<(), String> {
    match command {
        GatewayCommand::Start(start) => {
            if start.daemon {
                start_daemon().await
            } else {
                run_gateway_serve().await
            }
        }
        GatewayCommand::Stop => stop_daemon().await,
        GatewayCommand::Restart => {
            stop_daemon().await?;
            start_daemon().await
        }
    }
}

pub async fn run_gateway_serve() -> Result<(), String> {
    let config_path = config::default_pixy_config_path();
    let config = config::load_gateway_config(&config_path)?;
    if let Some(retry_count) = config.transport_retry_count {
        pixy_ai::set_transport_retry_count(retry_count);
    }
    runtime::serve_gateway(config).await
}

async fn start_daemon() -> Result<(), String> {
    let paths = GatewayRuntimePaths::resolve();
    paths.ensure_runtime_dir()?;
    if let Some(existing_pid) = read_pid_file(&paths.pid_file)? {
        if is_process_alive(existing_pid) {
            return Err(format!(
                "gateway daemon is already running with pid {existing_pid}"
            ));
        }
        paths.cleanup_runtime_files()?;
    }

    let current_exe = std::env::current_exe()
        .map_err(|error| format!("resolve current executable failed: {error}"))?;
    let child = Command::new(current_exe)
        .arg("--conf-dir")
        .arg(config::current_pixy_home_dir())
        .arg("serve")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| format!("spawn gateway daemon failed: {error}"))?;

    let pid = child.id();
    fs::write(&paths.pid_file, format!("{pid}\n")).map_err(|error| {
        format!(
            "write pid file {} failed: {error}",
            paths.pid_file.display()
        )
    })?;
    let started_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let state_json =
        format!("{{\"pid\":{pid},\"mode\":\"daemon\",\"started_at_unix\":{started_at}}}\n");
    fs::write(&paths.state_file, state_json).map_err(|error| {
        format!(
            "write state file {} failed: {error}",
            paths.state_file.display()
        )
    })?;
    println!(
        "[gateway] daemon started pid={} runtime_dir={}",
        pid,
        paths.runtime_dir.display()
    );
    Ok(())
}

async fn stop_daemon() -> Result<(), String> {
    let paths = GatewayRuntimePaths::resolve();
    let Some(pid) = read_pid_file(&paths.pid_file)? else {
        // Also clean stale state when pid file is missing.
        paths.cleanup_runtime_files()?;
        return Ok(());
    };

    if is_process_alive(pid) {
        let _ = send_signal(pid, "TERM");
        wait_for_process_exit(pid, STOP_WAIT_TIMEOUT);
        if is_process_alive(pid) {
            let _ = send_signal(pid, "KILL");
            wait_for_process_exit(pid, STOP_WAIT_TIMEOUT);
        }
    }

    if is_process_alive(pid) {
        return Err(format!("failed to stop gateway daemon pid {pid}"));
    }

    paths.cleanup_runtime_files()?;
    Ok(())
}

fn default_runtime_dir() -> PathBuf {
    config::current_pixy_home_dir().join("gateway")
}

fn read_pid_file(path: &Path) -> Result<Option<u32>, String> {
    if !path.exists() {
        return Ok(None);
    }
    let content = fs::read_to_string(path)
        .map_err(|error| format!("read pid file {} failed: {error}", path.display()))?;
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    trimmed
        .parse::<u32>()
        .map(Some)
        .map_err(|error| format!("parse pid file {} failed: {error}", path.display()))
}

fn send_signal(pid: u32, signal: &str) -> Result<(), String> {
    let status = Command::new("kill")
        .arg(format!("-{signal}"))
        .arg(pid.to_string())
        .status()
        .map_err(|error| format!("run kill -{signal} {pid} failed: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("kill -{signal} {pid} exited with status {status}"))
    }
}

fn is_process_alive(pid: u32) -> bool {
    let exists = Command::new("kill")
        .arg("-0")
        .arg(pid.to_string())
        .status()
        .map(|status| status.success())
        .unwrap_or(false);
    exists && !is_process_zombie(pid)
}

fn is_process_zombie(pid: u32) -> bool {
    #[cfg(target_os = "linux")]
    {
        let stat_path = format!("/proc/{pid}/stat");
        let Ok(stat) = fs::read_to_string(stat_path) else {
            return false;
        };
        let Some(paren_end) = stat.rfind(')') else {
            return false;
        };
        let state = stat
            .get(paren_end + 2..)
            .and_then(|rest| rest.chars().next())
            .unwrap_or('\0');
        state == 'Z'
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = pid;
        false
    }
}

fn wait_for_process_exit(pid: u32, timeout: Duration) {
    let deadline = SystemTime::now() + timeout;
    while is_process_alive(pid) {
        if SystemTime::now() >= deadline {
            break;
        }
        thread::sleep(STOP_POLL_INTERVAL);
    }
}

fn remove_if_exists(path: &Path) -> Result<(), String> {
    if path.exists() {
        fs::remove_file(path)
            .map_err(|error| format!("remove file {} failed: {error}", path.display()))?;
    }
    Ok(())
}

#[cfg(unix)]
async fn wait_for_shutdown_signal() -> Result<(), String> {
    use tokio::signal::unix::{signal, SignalKind};
    let mut sigterm = signal(SignalKind::terminate())
        .map_err(|error| format!("register SIGTERM handler failed: {error}"))?;
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        _ = sigterm.recv() => {}
    }
    Ok(())
}

#[cfg(not(unix))]
async fn wait_for_shutdown_signal() -> Result<(), String> {
    tokio::signal::ctrl_c()
        .await
        .map_err(|error| format!("wait for ctrl+c failed: {error}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::sync::{Mutex, OnceLock};

    use super::*;
    use tempfile::tempdir;

    struct EnvVarGuard {
        key: &'static str,
        previous: Option<String>,
    }

    impl EnvVarGuard {
        fn set_path(key: &'static str, value: &Path) -> Self {
            let previous = std::env::var(key).ok();
            // SAFETY: tests serialize env-var writes via TEST_LOCK to avoid concurrent mutation.
            unsafe { std::env::set_var(key, value) };
            Self { key, previous }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            if let Some(previous) = self.previous.take() {
                // SAFETY: tests serialize env-var writes via TEST_LOCK to avoid concurrent mutation.
                unsafe { std::env::set_var(self.key, previous) };
            } else {
                // SAFETY: tests serialize env-var writes via TEST_LOCK to avoid concurrent mutation.
                unsafe { std::env::remove_var(self.key) };
            }
        }
    }

    fn test_lock() -> &'static Mutex<()> {
        static TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        TEST_LOCK.get_or_init(|| Mutex::new(()))
    }

    fn runtime_paths(runtime_dir: &Path) -> (PathBuf, PathBuf) {
        (
            runtime_dir.join("gateway.pid"),
            runtime_dir.join("gateway.state.json"),
        )
    }

    #[tokio::test]
    async fn start_daemon_writes_runtime_files_and_stop_cleans_them_up() {
        let _guard = test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let dir = tempdir().expect("tempdir should be created");
        let runtime_dir = dir.path().join("gateway-runtime");
        let _env = EnvVarGuard::set_path("PIXY_GATEWAY_DIR", &runtime_dir);

        run_gateway_command(GatewayCommand::Start(GatewayStartOptions { daemon: true }))
            .await
            .expect("daemon start should succeed");

        let (pid_file, state_file) = runtime_paths(&runtime_dir);
        assert!(pid_file.exists(), "pid file should be created");
        assert!(state_file.exists(), "state file should be created");

        run_gateway_command(GatewayCommand::Stop)
            .await
            .expect("daemon stop should succeed");
        assert!(
            !pid_file.exists(),
            "pid file should be cleaned up after stop"
        );
        assert!(
            !state_file.exists(),
            "state file should be cleaned up after stop"
        );
    }

    #[tokio::test]
    async fn restart_without_running_daemon_starts_new_daemon_process() {
        let _guard = test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let dir = tempdir().expect("tempdir should be created");
        let runtime_dir = dir.path().join("gateway-runtime");
        let _env = EnvVarGuard::set_path("PIXY_GATEWAY_DIR", &runtime_dir);

        run_gateway_command(GatewayCommand::Restart)
            .await
            .expect("restart should start daemon when not running");

        let (pid_file, state_file) = runtime_paths(&runtime_dir);
        assert!(pid_file.exists(), "pid file should be created by restart");
        assert!(
            state_file.exists(),
            "state file should be created by restart"
        );

        run_gateway_command(GatewayCommand::Stop)
            .await
            .expect("daemon stop should succeed");
    }

    #[test]
    fn default_prompt_intro_mentions_gateway_role() {
        assert!(
            DEFAULT_PROMPT_INTRO.contains("gateway"),
            "gateway prompt intro should describe the gateway role"
        );
    }

    #[test]
    fn build_runtime_log_config_reads_log_settings() {
        let log = PixyTomlLog {
            path: Some("/tmp/pixy-logs".to_string()),
            level: Some("$LOG_LEVEL".to_string()),
            rotate_size_mb: Some(8),
            stdout: Some(true),
        };
        let env_map = HashMap::from([("LOG_LEVEL".to_string(), "debug".to_string())]);
        let resolved = build_runtime_log_config(&log, &env_map, "gateway.log");

        assert_eq!(
            resolved.file_path,
            PathBuf::from("/tmp/pixy-logs/gateway.log")
        );
        assert_eq!(resolved.level, "debug");
        assert_eq!(resolved.rotate_size_bytes, 8 * 1024 * 1024);
        assert!(resolved.stdout);
    }

    #[test]
    fn build_runtime_log_config_uses_defaults_when_unset() {
        let resolved =
            build_runtime_log_config(&PixyTomlLog::default(), &HashMap::new(), "gateway.log");
        assert!(resolved
            .file_path
            .ends_with(Path::new(".pixy/logs/gateway.log")));
        assert_eq!(resolved.level, "info");
        assert_eq!(resolved.rotate_size_bytes, 100 * 1024 * 1024);
        assert!(!resolved.stdout);
    }
}
