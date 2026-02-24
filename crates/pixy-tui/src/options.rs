use std::path::PathBuf;

use crate::{TuiKeyBindings, TuiTheme};

#[derive(Clone, Debug)]
pub struct TuiOptions {
    pub app_name: String,
    pub version: String,
    pub show_tool_results: bool,
    pub keybindings: TuiKeyBindings,
    pub initial_help: bool,
    pub theme: TuiTheme,
    pub status_top: String,
    pub status_left: String,
    pub status_right: String,
    pub input_history_path: Option<PathBuf>,
    pub input_history_limit: usize,
    pub enable_mouse_capture: bool,
    pub startup_resource_lines: Vec<String>,
}

impl Default for TuiOptions {
    fn default() -> Self {
        Self {
            app_name: "pixy".to_string(),
            version: String::new(),
            show_tool_results: true,
            keybindings: TuiKeyBindings::default(),
            initial_help: false,
            theme: TuiTheme::default(),
            status_top: String::new(),
            status_left: String::new(),
            status_right: String::new(),
            input_history_path: None,
            input_history_limit: 256,
            enable_mouse_capture: false,
            startup_resource_lines: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::TuiOptions;

    #[test]
    fn default_app_name_is_pixy() {
        assert_eq!(TuiOptions::default().app_name, "pixy");
    }
}
