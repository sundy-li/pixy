use std::fmt;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use pixy_coding_agent::RuntimeLoadOptions;

use crate::pixy_home::resolve_pixy_home_dir;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CheckStatus {
    Pass,
    Warn,
    Fail,
}

impl fmt::Display for CheckStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CheckStatus::Pass => write!(f, "PASS"),
            CheckStatus::Warn => write!(f, "WARN"),
            CheckStatus::Fail => write!(f, "FAIL"),
        }
    }
}

#[derive(Debug, Clone)]
struct CheckResult {
    name: &'static str,
    status: CheckStatus,
    detail: String,
}

#[derive(Debug, Clone, Default)]
struct DoctorReport {
    checks: Vec<CheckResult>,
}

impl DoctorReport {
    fn push(&mut self, name: &'static str, status: CheckStatus, detail: impl Into<String>) {
        self.checks.push(CheckResult {
            name,
            status,
            detail: detail.into(),
        });
    }

    fn fail_count(&self) -> usize {
        self.checks
            .iter()
            .filter(|check| check.status == CheckStatus::Fail)
            .count()
    }

    fn warn_count(&self) -> usize {
        self.checks
            .iter()
            .filter(|check| check.status == CheckStatus::Warn)
            .count()
    }
}

pub fn run_doctor(conf_dir: Option<PathBuf>) -> Result<(), String> {
    let conf_dir = resolve_pixy_home_dir(conf_dir.as_deref());
    let report = collect_doctor_report(&conf_dir);

    println!("pixy doctor report");
    println!("conf_dir: {}", conf_dir.display());
    for check in &report.checks {
        println!("[{}] {:<20} {}", check.status, check.name, check.detail);
    }

    let fails = report.fail_count();
    let warns = report.warn_count();
    println!(
        "summary: {} checks, {} fail, {} warn",
        report.checks.len(),
        fails,
        warns
    );

    if fails > 0 {
        Err(format!("doctor found {fails} failing checks"))
    } else {
        Ok(())
    }
}

fn collect_doctor_report(conf_dir: &Path) -> DoctorReport {
    let mut report = DoctorReport::default();

    let conf_dir_result = ensure_dir_writable(conf_dir);
    match conf_dir_result {
        Ok(()) => report.push("conf dir", CheckStatus::Pass, "readable and writable"),
        Err(error) => report.push("conf dir", CheckStatus::Fail, error),
    }

    let config_path = conf_dir.join("pixy.toml");
    if config_path.exists() {
        report.push(
            "config file",
            CheckStatus::Pass,
            format!("found {}", config_path.display()),
        );
    } else {
        report.push(
            "config file",
            CheckStatus::Warn,
            format!(
                "missing {} (copy from pixy.toml.sample to initialize)",
                config_path.display()
            ),
        );
    }

    let cwd = std::env::current_dir().unwrap_or_else(|_| conf_dir.to_path_buf());
    let options = RuntimeLoadOptions {
        conf_dir: Some(conf_dir.to_path_buf()),
        load_skills: false,
        include_default_skills: false,
        ..RuntimeLoadOptions::default()
    };

    match options.resolve_runtime(&cwd) {
        Ok(runtime) => {
            report.push(
                "runtime config",
                CheckStatus::Pass,
                format!("model={} api={}", runtime.model.id, runtime.model.api),
            );
            if runtime.api_key.is_some() {
                report.push(
                    "provider key",
                    CheckStatus::Pass,
                    "resolved api_key for selected provider",
                );
            } else {
                report.push(
                    "provider key",
                    CheckStatus::Warn,
                    "selected provider api_key unresolved",
                );
            }

            if runtime.model_catalog.is_empty() {
                report.push("model catalog", CheckStatus::Warn, "model catalog is empty");
            } else {
                report.push(
                    "model catalog",
                    CheckStatus::Pass,
                    format!("{} model(s) available", runtime.model_catalog.len()),
                );
            }
        }
        Err(error) => {
            let status = if config_path.exists() {
                CheckStatus::Fail
            } else {
                CheckStatus::Warn
            };
            report.push("runtime config", status, error);
        }
    }

    for (name, path) in [
        ("log dir", conf_dir.join("logs")),
        ("gateway dir", conf_dir.join("gateway")),
        ("sessions dir", conf_dir.join("agent").join("sessions")),
    ] {
        match ensure_dir_writable(&path) {
            Ok(()) => report.push(name, CheckStatus::Pass, format!("{}", path.display())),
            Err(error) => report.push(name, CheckStatus::Fail, error),
        }
    }

    report
}

fn ensure_dir_writable(path: &Path) -> Result<(), String> {
    fs::create_dir_all(path)
        .map_err(|error| format!("create {} failed: {error}", path.display()))?;

    let probe = path.join(".pixy-doctor-probe");
    let mut file = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&probe)
        .map_err(|error| format!("write test in {} failed: {error}", path.display()))?;
    file.write_all(b"ok")
        .map_err(|error| format!("write test in {} failed: {error}", path.display()))?;
    fs::remove_file(&probe)
        .map_err(|error| format!("cleanup test in {} failed: {error}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doctor_warns_when_config_file_missing() {
        let temp = tempfile::tempdir().expect("create tempdir");
        let report = collect_doctor_report(temp.path());
        let config_check = report
            .checks
            .iter()
            .find(|check| check.name == "config file")
            .expect("config file check should exist");
        assert_eq!(config_check.status, CheckStatus::Warn);
    }

    #[test]
    fn doctor_passes_runtime_with_minimal_valid_config() {
        let temp = tempfile::tempdir().expect("create tempdir");
        let config = r#"
[llm]
default_provider = "openai"

[[llm.providers]]
name = "openai"
kind = "chat"
provider = "openai"
api = "openai-completions"
base_url = "https://api.openai.com/v1"
api_key = "test-key"
model = "gpt-4o-mini"
weight = 1
"#;
        fs::write(temp.path().join("pixy.toml"), config).expect("write config");

        let report = collect_doctor_report(temp.path());
        let runtime_check = report
            .checks
            .iter()
            .find(|check| check.name == "runtime config")
            .expect("runtime config check should exist");
        assert_eq!(runtime_check.status, CheckStatus::Pass);
        assert_eq!(report.fail_count(), 0);
    }
}
