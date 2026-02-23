use crossterm::event::{KeyCode, KeyModifiers};
use pixy_tui::backend::StreamUpdate;
use pixy_tui::keybindings::{TuiKeyBindings, parse_key_id};
use pixy_tui::options::TuiOptions;
use pixy_tui::theme::TuiTheme;
use serde_json::Value;
use std::fs;
use std::path::Path;

#[test]
fn modular_public_api_paths_are_available() {
    let parsed = parse_key_id("ctrl+c").expect("ctrl+c should parse");
    assert_eq!(parsed.code, KeyCode::Char('c'));
    assert_eq!(parsed.modifiers, KeyModifiers::CONTROL);

    let defaults = TuiKeyBindings::default();
    assert!(!defaults.submit.is_empty());

    let options = TuiOptions::default();
    assert_eq!(options.theme, TuiTheme::Dark);

    let update = StreamUpdate::AssistantLine("ok".to_string());
    match update {
        StreamUpdate::AssistantLine(line) => assert_eq!(line, "ok"),
        _ => panic!("expected assistant line"),
    }
}

#[test]
fn built_in_theme_files_exist_and_are_valid_json() {
    let theme_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("themes");
    for name in ["dark", "light"] {
        let path = theme_dir.join(format!("{name}.json"));
        let content = fs::read_to_string(&path).unwrap_or_else(|error| {
            panic!("missing built-in theme file {}: {error}", path.display())
        });
        let value: Value = serde_json::from_str(&content)
            .unwrap_or_else(|error| panic!("invalid json in {}: {error}", path.display()));
        assert_eq!(value.get("name").and_then(Value::as_str), Some(name));
    }
}
