use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_REPO: &str = "sundy-li/pixy";
const DEFAULT_VERSION: &str = "latest";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct UpdateCommandArgs {
    pub(crate) version: Option<String>,
    pub(crate) repo: Option<String>,
}

pub(crate) fn run_update(args: UpdateCommandArgs) -> Result<(), String> {
    let repo = normalize_repo(args.repo.as_deref());
    let version = normalize_version(args.version.as_deref());
    let current_executable = std::env::current_exe()
        .map_err(|error| format!("resolve current executable failed: {error}"))?;
    let install_dir = current_executable
        .parent()
        .ok_or_else(|| {
            format!(
                "cannot resolve install directory from {}",
                current_executable.display()
            )
        })?
        .to_path_buf();

    let script_url = installer_script_url(repo.as_str());
    let script_path = temporary_script_path(script_extension());
    download_installer_script(script_url.as_str(), script_path.as_path())?;

    println!(
        "updating pixy from {repo} ({version}) into {}",
        install_dir.display()
    );
    let result = run_installer_script(
        script_path.as_path(),
        install_dir.as_path(),
        repo.as_str(),
        version.as_str(),
    );
    let _ = fs::remove_file(&script_path);
    result?;

    println!("pixy update finished");
    Ok(())
}

fn normalize_repo(input: Option<&str>) -> String {
    input
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(DEFAULT_REPO)
        .to_string()
}

fn normalize_version(input: Option<&str>) -> String {
    input
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(DEFAULT_VERSION)
        .to_string()
}

#[cfg(windows)]
fn installer_script_url(repo: &str) -> String {
    format!("https://raw.githubusercontent.com/{repo}/main/scripts/install.ps1")
}

#[cfg(not(windows))]
fn installer_script_url(repo: &str) -> String {
    format!("https://raw.githubusercontent.com/{repo}/main/scripts/install.sh")
}

#[cfg(windows)]
fn script_extension() -> &'static str {
    "ps1"
}

#[cfg(not(windows))]
fn script_extension() -> &'static str {
    "sh"
}

fn temporary_script_path(extension: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!(
        "pixy-update-{}-{nanos}.{extension}",
        std::process::id()
    ))
}

fn download_installer_script(script_url: &str, path: &Path) -> Result<(), String> {
    let status = Command::new("curl")
        .arg("-fsSL")
        .arg(script_url)
        .arg("-o")
        .arg(path)
        .status()
        .map_err(|error| format!("download installer script failed: {error}"))?;
    if !status.success() {
        return Err(format!(
            "download installer script failed with status {status}"
        ));
    }
    Ok(())
}

fn run_installer_script(
    script_path: &Path,
    install_dir: &Path,
    repo: &str,
    version: &str,
) -> Result<(), String> {
    let (program, arguments) = build_installer_invocation(script_path, install_dir, repo, version);
    let status = Command::new(&program)
        .args(&arguments)
        .status()
        .map_err(|error| format!("run update installer failed: {error}"))?;
    if !status.success() {
        return Err(format!("update installer failed with status {status}"));
    }
    Ok(())
}

#[cfg(windows)]
fn build_installer_invocation(
    script_path: &Path,
    install_dir: &Path,
    repo: &str,
    version: &str,
) -> (String, Vec<String>) {
    let arguments = vec![
        "-NoProfile".to_string(),
        "-ExecutionPolicy".to_string(),
        "Bypass".to_string(),
        "-File".to_string(),
        script_path.display().to_string(),
        "-InstallDir".to_string(),
        install_dir.display().to_string(),
        "-Repo".to_string(),
        repo.to_string(),
        "-Version".to_string(),
        version.to_string(),
    ];
    ("powershell".to_string(), arguments)
}

#[cfg(not(windows))]
fn build_installer_invocation(
    script_path: &Path,
    install_dir: &Path,
    repo: &str,
    version: &str,
) -> (String, Vec<String>) {
    let arguments = vec![
        script_path.display().to_string(),
        "--install-dir".to_string(),
        install_dir.display().to_string(),
        "--repo".to_string(),
        repo.to_string(),
        "--version".to_string(),
        version.to_string(),
    ];
    ("bash".to_string(), arguments)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_repo_defaults_to_official_repository() {
        assert_eq!(normalize_repo(None), "sundy-li/pixy");
        assert_eq!(normalize_repo(Some("")), "sundy-li/pixy");
        assert_eq!(normalize_repo(Some("  ")), "sundy-li/pixy");
    }

    #[test]
    fn normalize_version_defaults_to_latest() {
        assert_eq!(normalize_version(None), "latest");
        assert_eq!(normalize_version(Some("")), "latest");
        assert_eq!(normalize_version(Some("   ")), "latest");
    }

    #[test]
    fn normalize_inputs_preserve_non_empty_values() {
        assert_eq!(normalize_repo(Some("acme/pixy-fork")), "acme/pixy-fork");
        assert_eq!(normalize_version(Some("v1.2.3")), "v1.2.3");
        assert_eq!(normalize_version(Some("1.2.3")), "1.2.3");
    }

    #[cfg(not(windows))]
    #[test]
    fn build_installer_invocation_uses_bash_with_expected_arguments() {
        let script = Path::new("/tmp/install.sh");
        let install_dir = Path::new("/usr/local/bin");
        let (program, args) =
            build_installer_invocation(script, install_dir, "owner/repo", "v1.0.0");
        assert_eq!(program, "bash");
        assert_eq!(
            args,
            vec![
                "/tmp/install.sh",
                "--install-dir",
                "/usr/local/bin",
                "--repo",
                "owner/repo",
                "--version",
                "v1.0.0",
            ]
        );
    }
}
