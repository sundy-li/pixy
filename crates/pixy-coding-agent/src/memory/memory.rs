use super::config::MemoryConfig;
use super::file_store::{FileStore, FileStoreError};
use super::search::{MemorySearch, SearchResult};
use chrono::{Local, NaiveDate};
use std::path::PathBuf;
use thiserror::Error;

/// Errors that can occur in memory operations.
#[derive(Debug, Error)]
pub enum MemoryError {
    #[error("File store error: {0}")]
    FileStore(#[from] FileStoreError),

    #[error("Invalid configuration: {0}")]
    Config(String),

    #[error("Memory operation failed: {0}")]
    Operation(String),
}

/// Context for memory flush operation.
#[derive(Debug, Clone, Default)]
pub struct MemoryFlushContext {
    /// Session identifier.
    pub session_id: Option<String>,

    /// Agent identifier.
    pub agent_id: Option<String>,

    /// Token count in current session.
    pub token_count: usize,

    /// Compaction count.
    pub compaction_count: usize,

    /// Optional summary text.
    pub summary: Option<String>,

    /// Notes to record.
    pub notes: Vec<String>,

    /// Decisions made.
    pub decisions: Vec<String>,

    /// TODOs to track.
    pub todos: Vec<String>,

    /// Optional JSON metadata payload.
    pub metadata: Option<serde_json::Value>,
}

/// Memory manager that handles memory operations.
#[derive(Debug, Clone)]
pub struct MemoryManager {
    config: MemoryConfig,
    file_store: FileStore,
}

impl MemoryManager {
    /// Create a new memory manager with the given configuration.
    pub fn new(config: MemoryConfig) -> Result<Self, MemoryError> {
        config.validate().map_err(MemoryError::Config)?;
        let file_store = FileStore::new(&config.memory_dir, &config.file_pattern);
        file_store.init()?;
        Ok(Self { config, file_store })
    }

    /// Create a new memory manager with default configuration.
    pub fn new_default() -> Result<Self, MemoryError> {
        Self::new(MemoryConfig::default())
    }

    /// Return memory configuration.
    pub fn config(&self) -> &MemoryConfig {
        &self.config
    }

    /// Record a memory entry.
    pub fn record(&self, content: &str) -> Result<(), MemoryError> {
        self.file_store.append_today(content)?;
        Ok(())
    }

    /// Record a memory entry with metadata.
    pub fn record_with_metadata(
        &self,
        content: &str,
        metadata: serde_json::Value,
    ) -> Result<(), MemoryError> {
        let metadata_str = serde_json::to_string_pretty(&metadata)
            .map_err(|error| MemoryError::Operation(error.to_string()))?;
        let full_content = format!("{content}\n\n**Metadata:**\n```json\n{metadata_str}\n```");
        self.record(&full_content)
    }

    /// Record a session memory.
    pub fn record_session(
        &self,
        session_id: &str,
        messages: &[String],
        summary: Option<&str>,
    ) -> Result<(), MemoryError> {
        let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
        let mut content = String::new();
        content.push_str(&format!("## Session: {timestamp}\n\n"));
        content.push_str(&format!("- **Session ID**: {session_id}\n"));
        content.push_str(&format!("- **Recorded at**: {timestamp}\n"));
        if let Some(summary_text) = summary {
            content.push_str(&format!("- **Summary**: {summary_text}\n"));
        }

        if !messages.is_empty() {
            content.push_str("\n### Conversation\n");
            for (index, message) in messages.iter().enumerate() {
                content.push_str(&format!("{}. {message}\n", index + 1));
            }
        }

        self.record(&content)
    }

    /// Read today's memories.
    pub fn read_today(&self) -> Result<String, MemoryError> {
        Ok(self.file_store.read_today()?)
    }

    /// Read memories for a specific date.
    pub fn read_date(&self, date: &NaiveDate) -> Result<String, MemoryError> {
        let path = self.file_store.get_file_path(date);
        Ok(self.file_store.read_file(&path)?)
    }

    /// Read memories for a `YYYY-MM-DD` date string.
    pub fn read_date_string(&self, date: &str) -> Result<String, MemoryError> {
        let parsed = NaiveDate::parse_from_str(date.trim(), "%Y-%m-%d")
            .map_err(|error| MemoryError::Operation(format!("invalid date '{date}': {error}")))?;
        self.read_date(&parsed)
    }

    /// List all memory files.
    pub fn list_files(&self) -> Result<Vec<PathBuf>, MemoryError> {
        Ok(self.file_store.list_files()?)
    }

    /// Search memories and return path + snippet.
    pub fn search(&self, query: &str) -> Result<Vec<(PathBuf, String)>, MemoryError> {
        let scored = self.search_scored(
            query,
            self.config.search_max_results,
            self.config.search_min_score,
        )?;
        Ok(scored
            .into_iter()
            .map(|result| (result.path, result.snippet))
            .collect())
    }

    /// Search memories and return scored results.
    pub fn search_scored(
        &self,
        query: &str,
        max_results: usize,
        min_score: f32,
    ) -> Result<Vec<SearchResult>, MemoryError> {
        let threshold = min_score.clamp(0.0, 1.0);
        let search = MemorySearch::new(self.file_store.clone());
        Ok(search.search_text_with_options(query, max_results, threshold))
    }

    /// Perform memory flush.
    pub fn flush(&self, context: &MemoryFlushContext) -> Result<(), MemoryError> {
        let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
        let mut content = String::new();
        content.push_str(&format!("## Memory Flush: {timestamp}\n\n"));

        if let Some(session_id) = &context.session_id {
            content.push_str(&format!("- **Session**: {session_id}\n"));
        }
        if let Some(agent_id) = &context.agent_id {
            content.push_str(&format!("- **Agent**: {agent_id}\n"));
        }
        content.push_str(&format!("- **Tokens**: {}\n", context.token_count));
        content.push_str(&format!(
            "- **Compaction Count**: {}\n",
            context.compaction_count
        ));

        if let Some(summary) = &context.summary {
            content.push_str("\n### Summary\n");
            content.push_str(summary);
            content.push('\n');
        }

        if !context.notes.is_empty() {
            content.push_str("\n### Notes\n");
            for note in &context.notes {
                content.push_str(&format!("- {note}\n"));
            }
        }

        if !context.decisions.is_empty() {
            content.push_str("\n### Decisions\n");
            for decision in &context.decisions {
                content.push_str(&format!("- {decision}\n"));
            }
        }

        if !context.todos.is_empty() {
            content.push_str("\n### TODOs\n");
            for todo in &context.todos {
                content.push_str(&format!("- [ ] {todo}\n"));
            }
        }

        if let Some(metadata) = &context.metadata {
            let metadata_str = serde_json::to_string_pretty(metadata)
                .map_err(|error| MemoryError::Operation(error.to_string()))?;
            content.push_str("\n### Metadata\n");
            content.push_str("```json\n");
            content.push_str(&metadata_str);
            content.push_str("\n```\n");
        }

        self.record(&content)
    }

    /// Clean up old memory files based on retention policy.
    pub fn cleanup(&self) -> Result<usize, MemoryError> {
        if let Some(retention_days) = self.config.retention_days {
            Ok(self.file_store.cleanup_old_files(retention_days)?)
        } else {
            Ok(0)
        }
    }
}
