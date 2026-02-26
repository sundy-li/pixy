use std::path::{Path, PathBuf};

const DEFAULT_PIXY_HOME_DIR_NAME: &str = ".pixy";

pub fn resolve_pixy_home_dir(conf_dir: Option<&Path>) -> PathBuf {
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

fn default_pixy_home_dir() -> PathBuf {
    home_dir().join(DEFAULT_PIXY_HOME_DIR_NAME)
}

fn home_dir() -> PathBuf {
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home);
    }
    if let Some(profile) = std::env::var_os("USERPROFILE") {
        return PathBuf::from(profile);
    }
    PathBuf::from(".")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_pixy_home_dir_uses_absolute_override_directly() {
        let path = resolve_pixy_home_dir(Some(Path::new("/tmp/pixy-home")));
        assert_eq!(path, PathBuf::from("/tmp/pixy-home"));
    }

    #[test]
    fn resolve_pixy_home_dir_uses_default_suffix_without_override() {
        let path = resolve_pixy_home_dir(None);
        assert!(
            path.ends_with(".pixy"),
            "expected default pixy home directory to end with .pixy, got {}",
            path.display()
        );
    }
}
