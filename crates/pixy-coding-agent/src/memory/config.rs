use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Configuration for memory module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Base directory for memory storage.
    pub memory_dir: PathBuf,

    /// Maximum number of days to keep memory files.
    pub retention_days: Option<u32>,

    /// Whether to enable automatic memory flushing.
    pub auto_flush: bool,

    /// Optional token threshold for auto flush.
    pub flush_threshold_tokens: Option<u64>,

    /// Number of messages to include in memory flush.
    pub flush_message_count: usize,

    /// Pattern for memory file names.
    pub file_pattern: String,

    /// Default maximum result count for search.
    pub search_max_results: usize,

    /// Default minimum score threshold for search.
    pub search_min_score: f32,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            memory_dir: PathBuf::from("./memory"),
            retention_days: Some(30),
            auto_flush: true,
            flush_threshold_tokens: None,
            flush_message_count: 10,
            file_pattern: String::from("%Y-%m-%d.md"),
            search_max_results: 10,
            search_min_score: 0.1,
        }
    }
}

impl MemoryConfig {
    /// Create a new memory config with custom directory.
    pub fn new(memory_dir: impl Into<PathBuf>) -> Self {
        Self {
            memory_dir: memory_dir.into(),
            ..Default::default()
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.flush_message_count == 0 {
            return Err("flush_message_count must be greater than 0".to_string());
        }
        if self.file_pattern.trim().is_empty() {
            return Err("file_pattern cannot be empty".to_string());
        }
        if self.search_max_results == 0 {
            return Err("search_max_results must be greater than 0".to_string());
        }
        if !(0.0..=1.0).contains(&self.search_min_score) {
            return Err("search_min_score must be between 0.0 and 1.0".to_string());
        }
        Ok(())
    }
}
