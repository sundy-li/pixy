use chrono::{Local, NaiveDate};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;
use walkdir::WalkDir;

/// Errors that can occur in file store operations.
#[derive(Debug, Error)]
pub enum FileStoreError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("Memory directory not found: {0}")]
    DirectoryNotFound(PathBuf),
}

/// A single memory entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryEntry {
    /// Date of the memory entry.
    pub date: NaiveDate,

    /// File path.
    pub path: PathBuf,

    /// Content of the memory entry.
    pub content: String,

    /// Metadata (optional).
    pub metadata: Option<serde_json::Value>,
}

/// File-based storage for memories.
#[derive(Debug, Clone)]
pub struct FileStore {
    memory_dir: PathBuf,
    file_pattern: String,
}

impl FileStore {
    /// Create a new file store.
    pub fn new(memory_dir: impl Into<PathBuf>, file_pattern: impl Into<String>) -> Self {
        Self {
            memory_dir: memory_dir.into(),
            file_pattern: file_pattern.into(),
        }
    }

    /// Initialize the memory directory.
    pub fn init(&self) -> Result<(), FileStoreError> {
        fs::create_dir_all(&self.memory_dir)?;
        Ok(())
    }

    /// Return memory root directory path.
    pub fn memory_dir(&self) -> &Path {
        &self.memory_dir
    }

    /// Get the file path for a specific date.
    pub fn get_file_path(&self, date: &NaiveDate) -> PathBuf {
        let filename = date.format(&self.file_pattern).to_string();
        self.memory_dir.join(filename)
    }

    /// Get today's file path.
    pub fn get_today_path(&self) -> PathBuf {
        let today = Local::now().date_naive();
        self.get_file_path(&today)
    }

    /// Append content to today's memory file.
    pub fn append_today(&self, content: &str) -> Result<(), FileStoreError> {
        let path = self.get_today_path();
        self.append_to_file(&path, content)
    }

    /// Append content to a specific file.
    pub fn append_to_file(&self, path: &Path, content: &str) -> Result<(), FileStoreError> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        let is_new_file = file.metadata()?.len() == 0;
        if is_new_file {
            let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
            writeln!(file, "# {timestamp}")?;
            writeln!(file)?;
        }

        writeln!(file, "{content}")?;
        writeln!(file)?;
        Ok(())
    }

    /// Read content from a specific file.
    pub fn read_file(&self, path: &Path) -> Result<String, FileStoreError> {
        Ok(fs::read_to_string(path)?)
    }

    /// Read today's memory file.
    pub fn read_today(&self) -> Result<String, FileStoreError> {
        let path = self.get_today_path();
        if path.exists() {
            self.read_file(&path)
        } else {
            Ok(String::new())
        }
    }

    /// List all memory markdown files in memory root directory.
    pub fn list_files(&self) -> Result<Vec<PathBuf>, FileStoreError> {
        if !self.memory_dir.exists() {
            return Err(FileStoreError::DirectoryNotFound(self.memory_dir.clone()));
        }

        let mut files = Vec::new();
        for entry in WalkDir::new(&self.memory_dir)
            .max_depth(1)
            .into_iter()
            .filter_map(Result::ok)
        {
            let path = entry.path();
            if path.is_file() && path.extension().is_some_and(|ext| ext == "md") {
                files.push(path.to_path_buf());
            }
        }
        files.sort();
        Ok(files)
    }

    /// Parse date from default file name `YYYY-MM-DD.md`.
    pub fn parse_date_from_filename(&self, filename: &str) -> Option<NaiveDate> {
        filename
            .strip_suffix(".md")
            .and_then(|date| NaiveDate::parse_from_str(date, "%Y-%m-%d").ok())
    }

    /// Get memory entries for a date range.
    pub fn get_entries_in_range(
        &self,
        start_date: &NaiveDate,
        end_date: &NaiveDate,
    ) -> Result<Vec<MemoryEntry>, FileStoreError> {
        let mut entries = Vec::new();
        for file_path in self.list_files()? {
            let Some(filename) = file_path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            let Some(date) = self.parse_date_from_filename(filename) else {
                continue;
            };
            if date < *start_date || date > *end_date {
                continue;
            }
            let content = self.read_file(&file_path)?;
            entries.push(MemoryEntry {
                date,
                path: file_path,
                content,
                metadata: None,
            });
        }
        Ok(entries)
    }

    /// Clean up old memory files based on retention policy.
    pub fn cleanup_old_files(&self, retention_days: u32) -> Result<usize, FileStoreError> {
        let cutoff_date = Local::now()
            .date_naive()
            .checked_sub_days(chrono::Days::new(retention_days as u64))
            .unwrap_or(NaiveDate::MIN);
        let mut deleted_count = 0usize;

        for file_path in self.list_files()? {
            let Some(filename) = file_path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            let Some(date) = self.parse_date_from_filename(filename) else {
                continue;
            };
            if date < cutoff_date {
                fs::remove_file(&file_path)?;
                deleted_count += 1;
            }
        }

        Ok(deleted_count)
    }
}
