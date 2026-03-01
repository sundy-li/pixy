pub(crate) const FORCE_EXIT_SIGNAL: &str = "__FORCE_EXIT__";
pub(crate) const FORCE_EXIT_STATUS: &str = "force exiting...";
pub(crate) const PASTED_TEXT_PREVIEW_LIMIT: usize = 100;
pub(crate) const RESUME_LIST_LIMIT: usize = 10;
pub(crate) const INPUT_RENDER_LEFT_PADDING: &str = " ";
pub(crate) const INPUT_PLACEHOLDER_HINTS: &[&str] =
    &["Try \"Search the documentation for this library\""];
pub(crate) const INPUT_AREA_FIXED_HEIGHT: u16 = 3;
pub(crate) const STATUS_HINT_LEFT: &str =
    "shift+tab to toggle thinking, ctrl+L to switch permission";
pub(crate) const STATUS_HINT_RIGHT: &str = "ctrl+N to cycle models";

pub(crate) fn primary_input_placeholder_hint() -> &'static str {
    INPUT_PLACEHOLDER_HINTS.first().copied().unwrap_or("")
}
