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
}

impl Default for TuiOptions {
    fn default() -> Self {
        Self {
            app_name: "pi".to_string(),
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
        }
    }
}
