use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use pixy_agent_core::{AgentTool, AgentToolExecuteFn, AgentToolExecutor, AgentToolResult};
use pixy_ai::{PiAiError, PiAiErrorCode};
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PermissionMode {
    #[serde(alias = "off", alias = "auto-off")]
    AutoOff,
    #[serde(alias = "full", alias = "auto-full")]
    AutoFull,
}

impl Default for PermissionMode {
    fn default() -> Self {
        Self::AutoOff
    }
}

impl PermissionMode {
    pub fn cycle(self) -> Self {
        match self {
            Self::AutoOff => Self::AutoFull,
            Self::AutoFull => Self::AutoOff,
        }
    }

    pub fn status_label(self) -> &'static str {
        match self {
            Self::AutoOff => "Auto (off)",
            Self::AutoFull => "Auto (full)",
        }
    }

    pub fn status_line(self) -> &'static str {
        match self {
            Self::AutoOff => {
                "Auto (off) - workspace edits/commands, internet & outside edits need approval"
            }
            Self::AutoFull => {
                "Auto (full) - internet and outside-workspace edits are allowed without approval"
            }
        }
    }
}

pub type SharedPermissionMode = Arc<Mutex<PermissionMode>>;

pub fn new_shared_permission_mode(mode: PermissionMode) -> SharedPermissionMode {
    Arc::new(Mutex::new(mode))
}

pub fn current_permission_mode_state(mode: &SharedPermissionMode) -> PermissionMode {
    *mode.lock().unwrap_or_else(|poison| poison.into_inner())
}

pub fn cycle_permission_mode_state(mode: &SharedPermissionMode) -> PermissionMode {
    let mut guard = mode.lock().unwrap_or_else(|poison| poison.into_inner());
    let next = guard.cycle();
    *guard = next;
    next
}

pub fn apply_permission_mode_to_tools(
    tools: &mut [AgentTool],
    workspace_root: &Path,
    mode: SharedPermissionMode,
) {
    let workspace_root = workspace_root.to_path_buf();
    for tool in tools.iter_mut() {
        match tool.name.as_str() {
            "bash" | "write" | "edit" => {
                let inner = tool.execute.clone();
                tool.execute = Arc::new(PermissionGuardedExecutor {
                    tool_name: tool.name.clone(),
                    workspace_root: workspace_root.clone(),
                    mode: mode.clone(),
                    inner,
                });
            }
            _ => {}
        }
    }
}

struct PermissionGuardedExecutor {
    tool_name: String,
    workspace_root: PathBuf,
    mode: SharedPermissionMode,
    inner: AgentToolExecuteFn,
}

#[async_trait]
impl AgentToolExecutor for PermissionGuardedExecutor {
    async fn execute(
        &self,
        tool_call_id: String,
        args: Value,
    ) -> Result<AgentToolResult, PiAiError> {
        let mode = current_permission_mode_state(&self.mode);
        if mode == PermissionMode::AutoOff {
            if let Some(error) =
                self.approval_required_message_for_args(args.as_object(), self.tool_name.as_str())
            {
                return Err(PiAiError::new(PiAiErrorCode::ToolExecutionFailed, error));
            }
        }
        self.inner.execute(tool_call_id, args).await
    }
}

impl PermissionGuardedExecutor {
    fn approval_required_message_for_args(
        &self,
        args: Option<&serde_json::Map<String, Value>>,
        tool_name: &str,
    ) -> Option<String> {
        let args = args?;
        match tool_name {
            "bash" => {
                let command = args.get("command").and_then(Value::as_str)?;
                if command_requests_internet(command) {
                    Some(
                        "Approval required in Auto (off): internet access is blocked. Press Ctrl+L to switch to Auto (full), then retry."
                            .to_string(),
                    )
                } else {
                    None
                }
            }
            "write" | "edit" => {
                let path = args.get("path").and_then(Value::as_str)?;
                let resolved = resolve_path(&self.workspace_root, path);
                if path_is_within_workspace(&self.workspace_root, &resolved) {
                    None
                } else {
                    Some(format!(
                        "Approval required in Auto (off): editing files outside the current workspace is blocked (target: {}). Press Ctrl+L to switch to Auto (full), then retry.",
                        resolved.display()
                    ))
                }
            }
            _ => None,
        }
    }
}

fn command_requests_internet(command: &str) -> bool {
    let lowered = command.to_ascii_lowercase();
    if lowered.contains("http://")
        || lowered.contains("https://")
        || lowered.contains("ftp://")
        || lowered.contains("ssh://")
        || lowered.contains("git@")
    {
        return true;
    }

    let padded = format!(
        " {} ",
        lowered
            .replace('\n', " ")
            .replace('\r', " ")
            .replace('\t', " ")
    );
    const NETWORK_PATTERNS: &[&str] = &[
        " curl ",
        " wget ",
        " ssh ",
        " scp ",
        " sftp ",
        " rsync ",
        " nc ",
        " ncat ",
        " netcat ",
        " telnet ",
        " dig ",
        " nslookup ",
        " traceroute ",
        " tracert ",
        " ping ",
        " ftp ",
        " git clone ",
        " git fetch ",
        " git pull ",
        " git push ",
        " docker pull ",
        " docker push ",
    ];
    NETWORK_PATTERNS
        .iter()
        .any(|pattern| padded.contains(pattern))
}

fn resolve_path(workspace_root: &Path, raw_path: &str) -> PathBuf {
    let normalized = if let Some(stripped) = raw_path.strip_prefix('@') {
        stripped
    } else {
        raw_path
    };
    let expanded = expand_home(normalized);
    let path = PathBuf::from(expanded);
    if path.is_absolute() {
        path
    } else {
        workspace_root.join(path)
    }
}

fn expand_home(path: &str) -> String {
    if path == "~" {
        return std::env::var("HOME").unwrap_or_else(|_| path.to_string());
    }
    if let Some(rest) = path.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return format!("{home}/{rest}");
        }
    }
    path.to_string()
}

fn path_is_within_workspace(workspace_root: &Path, candidate: &Path) -> bool {
    let normalized_workspace = normalize_path_for_scope(workspace_root);
    let normalized_candidate = normalize_path_for_scope(candidate);
    normalized_candidate.starts_with(normalized_workspace)
}

fn normalize_path_for_scope(path: &Path) -> PathBuf {
    if let Ok(canonical) = std::fs::canonicalize(path) {
        return canonical;
    }

    if let Some(parent) = path.parent() {
        if let Ok(canonical_parent) = std::fs::canonicalize(parent) {
            if let Some(name) = path.file_name() {
                return canonical_parent.join(name);
            }
            return canonical_parent;
        }
    }

    path.to_path_buf()
}
