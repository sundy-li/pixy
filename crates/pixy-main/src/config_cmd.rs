use std::fs;
use std::path::{Path, PathBuf};

use crate::pixy_home::resolve_pixy_home_dir;

const PIXY_TOML_SAMPLE: &str = include_str!("../../../pixy.toml.sample");

pub fn run_config_init(conf_dir: Option<PathBuf>) -> Result<(), String> {
    let pixy_home_dir = resolve_pixy_home_dir(conf_dir.as_deref());

    for path in init_directories(&pixy_home_dir) {
        fs::create_dir_all(&path)
            .map_err(|error| format!("create {} failed: {error}", path.display()))?;
        println!("created: {}", path.display());
    }

    let config_path = pixy_home_dir.join("pixy.toml");
    if config_path.exists() {
        println!("kept: {}", config_path.display());
    } else {
        fs::write(&config_path, PIXY_TOML_SAMPLE)
            .map_err(|error| format!("write {} failed: {error}", config_path.display()))?;
        println!("created: {}", config_path.display());
    }

    Ok(())
}

fn init_directories(pixy_home_dir: &Path) -> Vec<PathBuf> {
    vec![
        pixy_home_dir.to_path_buf(),
        pixy_home_dir.join("agent"),
        pixy_home_dir.join("agent").join("sessions"),
        pixy_home_dir.join("logs"),
        pixy_home_dir.join("workspace"),
        pixy_home_dir.join("workspace").join("tmp"),
        pixy_home_dir.join("gateway"),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_directories_include_expected_runtime_paths() {
        let root = PathBuf::from("/tmp/pixy-home");
        let dirs = init_directories(&root);
        assert!(dirs.contains(&root.join("agent")));
        assert!(dirs.contains(&root.join("agent").join("sessions")));
        assert!(dirs.contains(&root.join("logs")));
        assert!(dirs.contains(&root.join("workspace")));
        assert!(dirs.contains(&root.join("gateway")));
    }
}
