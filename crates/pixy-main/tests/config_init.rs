use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use tempfile::tempdir;

const PIXY_TOML_SAMPLE: &str = include_str!("../../../pixy.toml.sample");

#[test]
fn config_init_creates_pixy_home_tree_and_pixy_toml_from_sample() {
    let conf_dir = tempdir().expect("create temp conf dir");

    let output = run_pixy_config_init(conf_dir.path());
    assert_command_succeeded(&output);

    let pixy_toml = conf_dir.path().join("pixy.toml");
    assert!(pixy_toml.is_file(), "expected {}", pixy_toml.display());

    for dir in [
        conf_dir.path().join("agents"),
        conf_dir.path().join("agents").join("sessions"),
        conf_dir.path().join("logs"),
        conf_dir.path().join("workspace"),
        conf_dir.path().join("gateway"),
    ] {
        assert!(dir.is_dir(), "expected {}", dir.display());
    }

    let generated = fs::read_to_string(&pixy_toml).expect("read generated pixy.toml");
    assert_eq!(
        generated, PIXY_TOML_SAMPLE,
        "generated config should match sample"
    );
}

#[test]
fn config_init_keeps_existing_pixy_toml() {
    let conf_dir = tempdir().expect("create temp conf dir");
    let config_path = conf_dir.path().join("pixy.toml");
    let original = "[llm]\ndefault_provider = \"local\"\n";
    fs::write(&config_path, original).expect("seed pixy.toml");

    let output = run_pixy_config_init(conf_dir.path());
    assert_command_succeeded(&output);

    let preserved = fs::read_to_string(&config_path).expect("read preserved pixy.toml");
    assert_eq!(
        preserved, original,
        "existing pixy.toml must not be overwritten"
    );
}

fn run_pixy_config_init(conf_dir: &Path) -> Output {
    Command::new(pixy_binary_path())
        .arg("--conf-dir")
        .arg(conf_dir)
        .arg("config")
        .arg("init")
        .output()
        .expect("execute pixy binary")
}

fn pixy_binary_path() -> PathBuf {
    for key in ["CARGO_BIN_EXE_pixy", "NEXTEST_BIN_EXE_pixy"] {
        if let Some(path) = std::env::var_os(key) {
            return PathBuf::from(path);
        }
    }

    let fallback = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/debug/pixy");
    if fallback.is_file() {
        return fallback;
    }

    panic!(
        "unable to resolve pixy binary path from env (CARGO_BIN_EXE_pixy / NEXTEST_BIN_EXE_pixy) \
or fallback {}",
        fallback.display()
    );
}

fn assert_command_succeeded(output: &Output) {
    if output.status.success() {
        return;
    }
    panic!(
        "pixy exited with status {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
