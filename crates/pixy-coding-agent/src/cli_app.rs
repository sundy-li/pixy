use std::path::{Path, PathBuf};

use pixy_ai::Message;

use crate::{
    agent_session::{build_session_resume_candidate, SessionResumeCandidate},
    create_session_from_runtime, AgentSession, ResolvedRuntime, RuntimeLoadOptions,
    RuntimeOverrides, SessionManager,
};

#[derive(Debug, Clone)]
pub(crate) struct CliSessionRequest {
    pub(crate) session_file: Option<PathBuf>,
    pub(crate) include_default_skills: bool,
    pub(crate) skill_paths: Vec<String>,
    pub(crate) runtime_overrides: RuntimeOverrides,
    pub(crate) custom_system_prompt: Option<String>,
    pub(crate) no_tools: bool,
}

pub(crate) struct CliSessionFactory {
    pixy_home_dir: PathBuf,
}

impl CliSessionFactory {
    pub(crate) fn new(pixy_home_dir: PathBuf) -> Self {
        Self { pixy_home_dir }
    }

    fn runtime_options(&self, request: &CliSessionRequest, agent_dir: &Path) -> RuntimeLoadOptions {
        RuntimeLoadOptions {
            conf_dir: Some(self.pixy_home_dir.clone()),
            agent_dir: Some(agent_dir.to_path_buf()),
            load_skills: true,
            include_default_skills: request.include_default_skills,
            skill_paths: request.skill_paths.clone(),
            overrides: request.runtime_overrides.clone(),
        }
    }

    pub(crate) fn resolve_runtime(
        &self,
        request: &CliSessionRequest,
        cwd: &Path,
        agent_dir: &Path,
    ) -> Result<ResolvedRuntime, String> {
        self.runtime_options(request, agent_dir)
            .resolve_runtime(cwd)
    }

    pub(crate) fn create_session(
        &self,
        request: &CliSessionRequest,
        cwd: &Path,
        agent_dir: &Path,
        session_dir: &Path,
    ) -> Result<CliSession, String> {
        let runtime = self.resolve_runtime(request, cwd, agent_dir)?;
        let resolved_session_file = request
            .session_file
            .as_ref()
            .map(|path| resolve_path(cwd, path));
        Ok(CliSession::new(
            cwd.to_path_buf(),
            session_dir.to_path_buf(),
            runtime,
            request.custom_system_prompt.clone(),
            request.no_tools,
            resolved_session_file,
        ))
    }
}

pub(crate) struct CliSession {
    cwd: PathBuf,
    session_dir: PathBuf,
    runtime: ResolvedRuntime,
    custom_system_prompt: Option<String>,
    no_tools: bool,
    resolved_session_file: Option<PathBuf>,
    session: Option<AgentSession>,
}

impl CliSession {
    fn new(
        cwd: PathBuf,
        session_dir: PathBuf,
        runtime: ResolvedRuntime,
        custom_system_prompt: Option<String>,
        no_tools: bool,
        resolved_session_file: Option<PathBuf>,
    ) -> Self {
        Self {
            cwd,
            session_dir,
            runtime,
            custom_system_prompt,
            no_tools,
            resolved_session_file,
            session: None,
        }
    }

    pub(crate) fn runtime(&self) -> &ResolvedRuntime {
        &self.runtime
    }

    pub(crate) fn session_file(&self) -> Option<PathBuf> {
        self.session
            .as_ref()
            .and_then(|session| session.session_file().cloned())
            .or_else(|| self.resolved_session_file.clone())
    }

    pub(crate) fn session_messages(&self) -> Option<Vec<Message>> {
        self.session
            .as_ref()
            .map(|session| session.build_session_context().messages)
    }

    pub(crate) fn ensure_session(&mut self) -> Result<&mut AgentSession, String> {
        if self.session.is_none() {
            let manager = if let Some(session_file) = self.resolved_session_file.take() {
                SessionManager::load(&session_file)?
            } else {
                self.create_new_session_manager()?
            };
            self.install_session(manager);
        }
        self.session
            .as_mut()
            .ok_or_else(|| "session initialization failed".to_string())
    }

    pub(crate) fn start_new_session(&mut self) -> Result<PathBuf, String> {
        if self.session.is_none() {
            if self.resolved_session_file.is_none() {
                let manager = self.create_new_session_manager()?;
                let path = manager.session_file().cloned().ok_or_else(|| {
                    "session manager did not return session file path".to_string()
                })?;
                self.install_session(manager);
                return Ok(path);
            }
            self.ensure_session()?;
        }
        self.session
            .as_mut()
            .ok_or_else(|| "session initialization failed".to_string())?
            .start_new_session()
    }

    pub(crate) fn resume(&mut self, target: Option<&str>) -> Result<PathBuf, String> {
        if let Some(session) = self.session.as_mut() {
            return session.resume(target);
        }

        let target_path =
            resolve_resume_target_without_active_session(target, &self.cwd, &self.session_dir)?;
        let manager = SessionManager::load(&target_path)?;
        self.install_session(manager);
        Ok(target_path)
    }

    pub(crate) fn recent_resumable_sessions(&self, limit: usize) -> Result<Vec<PathBuf>, String> {
        if limit == 0 {
            return Ok(vec![]);
        }

        if let Some(session) = &self.session {
            return session.recent_resumable_sessions(limit);
        }

        let mut files = list_session_files(&self.session_dir)?;
        if let Some(current_session_file) = self.resolved_session_file.as_deref() {
            files.retain(|path| !paths_equal(path, current_session_file));
        }
        files.sort_by(|left, right| right.file_name().cmp(&left.file_name()));
        files.truncate(limit);
        Ok(files)
    }

    pub(crate) fn recent_resumable_session_candidates(
        &self,
        limit: usize,
    ) -> Result<Vec<SessionResumeCandidate>, String> {
        self.recent_resumable_sessions(limit)?
            .into_iter()
            .map(build_session_resume_candidate)
            .collect()
    }

    fn install_session(&mut self, session_manager: SessionManager) {
        let session = create_session_from_runtime(
            &self.cwd,
            session_manager,
            &self.runtime,
            self.custom_system_prompt.as_deref(),
            self.no_tools,
        );
        self.session = Some(session);
        self.resolved_session_file = None;
    }

    fn create_new_session_manager(&self) -> Result<SessionManager, String> {
        let cwd_text = self
            .cwd
            .to_str()
            .ok_or_else(|| format!("cwd is not valid UTF-8: {}", self.cwd.display()))?;
        SessionManager::create(cwd_text, &self.session_dir)
    }
}

fn resolve_resume_target_without_active_session(
    target: Option<&str>,
    cwd: &Path,
    session_dir: &Path,
) -> Result<PathBuf, String> {
    let Some(target_value) = target.map(str::trim).filter(|value| !value.is_empty()) else {
        return latest_session_in_dir(session_dir, None);
    };

    if target_value.eq_ignore_ascii_case("latest") {
        return latest_session_in_dir(session_dir, None);
    }

    resolve_explicit_session_target(target_value, cwd, session_dir)
}

fn resolve_explicit_session_target(
    target: &str,
    cwd: &Path,
    session_dir: &Path,
) -> Result<PathBuf, String> {
    let raw = PathBuf::from(target);
    if raw.is_absolute() {
        if raw.is_file() {
            return Ok(raw);
        }
        return Err(format!("Session file not found: {}", raw.display()));
    }

    let mut candidates = vec![cwd.join(&raw), session_dir.join(&raw)];
    if raw.extension().is_none() {
        let mut with_ext = raw.clone();
        with_ext.set_extension("jsonl");
        candidates.push(cwd.join(&with_ext));
        candidates.push(session_dir.join(with_ext));
    }

    if let Some(found) = candidates.into_iter().find(|path| path.is_file()) {
        return Ok(found);
    }

    if !target.contains('/') {
        let mut fuzzy_matches = list_session_files(session_dir)?
            .into_iter()
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .map(|name| name.contains(target))
                    .unwrap_or(false)
            })
            .collect::<Vec<_>>();
        fuzzy_matches.sort_by(|left, right| left.file_name().cmp(&right.file_name()));
        if let Some(found) = fuzzy_matches.pop() {
            return Ok(found);
        }
    }

    Err(format!(
        "Session not found for '{target}'. Use absolute path or a file under {}",
        session_dir.display()
    ))
}

fn latest_session_in_dir(session_dir: &Path, exclude: Option<&Path>) -> Result<PathBuf, String> {
    let mut files = list_session_files(session_dir)?;
    if let Some(excluded) = exclude {
        files.retain(|path| !paths_equal(path, excluded));
    }

    files.sort_by(|left, right| left.file_name().cmp(&right.file_name()));
    files.pop().ok_or_else(|| {
        format!(
            "No resumable sessions found under {}",
            session_dir.display()
        )
    })
}

fn list_session_files(session_dir: &Path) -> Result<Vec<PathBuf>, String> {
    if !session_dir.exists() {
        return Ok(vec![]);
    }

    let entries = std::fs::read_dir(session_dir).map_err(|error| {
        format!(
            "Read session directory failed ({}): {error}",
            session_dir.display()
        )
    })?;

    let mut files = vec![];
    for entry in entries {
        let entry = entry.map_err(|error| format!("Read session dir entry failed: {error}"))?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("jsonl"))
            .unwrap_or(false)
        {
            files.push(path);
        }
    }
    Ok(files)
}

fn paths_equal(left: &Path, right: &Path) -> bool {
    if left == right {
        return true;
    }
    match (std::fs::canonicalize(left), std::fs::canonicalize(right)) {
        (Ok(left), Ok(right)) => left == right,
        _ => false,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ReplCommand {
    NewSession,
    Resume { target: Option<String> },
    Continue,
    Session,
    Help,
    Exit,
    Prompt { text: String },
}

pub(crate) struct ReplCommandParser;

impl ReplCommandParser {
    pub(crate) fn parse(input: &str) -> Option<ReplCommand> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return None;
        }

        if trimmed == "/new" {
            return Some(ReplCommand::NewSession);
        }

        if let Some(rest) = trimmed.strip_prefix("/resume") {
            let target = rest.trim();
            return Some(ReplCommand::Resume {
                target: if target.is_empty() {
                    None
                } else {
                    Some(target.to_string())
                },
            });
        }

        match trimmed {
            "/exit" | "/quit" => Some(ReplCommand::Exit),
            "/help" | "?" => Some(ReplCommand::Help),
            "/session" => Some(ReplCommand::Session),
            "/continue" => Some(ReplCommand::Continue),
            _ => Some(ReplCommand::Prompt {
                text: trimmed.to_string(),
            }),
        }
    }
}

fn resolve_path(cwd: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    }
}
