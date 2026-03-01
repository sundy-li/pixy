use super::file_store::{FileStore, MemoryEntry};
use chrono::NaiveDate;
use std::path::PathBuf;

/// Search result.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    /// File path.
    pub path: PathBuf,

    /// Date of the memory entry.
    pub date: NaiveDate,

    /// Snippet containing the match.
    pub snippet: String,

    /// Relevance score (0.0 to 1.0).
    pub score: f32,

    /// Line numbers where matches were found.
    pub line_numbers: Vec<usize>,
}

/// Memory search functionality.
pub struct MemorySearch {
    file_store: FileStore,
}

impl MemorySearch {
    /// Create a new memory search instance.
    pub fn new(file_store: FileStore) -> Self {
        Self { file_store }
    }

    /// Simple text search across all memory files.
    pub fn search_text(&self, query: &str) -> Vec<SearchResult> {
        self.search_text_with_options(query, usize::MAX, 0.0)
    }

    /// Text search with result cap and score threshold.
    pub fn search_text_with_options(
        &self,
        query: &str,
        max_results: usize,
        min_score: f32,
    ) -> Vec<SearchResult> {
        if query.trim().is_empty() || max_results == 0 {
            return Vec::new();
        }

        let files = match self.file_store.list_files() {
            Ok(files) => files,
            Err(_) => return Vec::new(),
        };

        let mut results = Vec::new();
        for file_path in files {
            let Ok(content) = self.file_store.read_file(&file_path) else {
                continue;
            };
            let Some(filename) = file_path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            let Some(date) = self.file_store.parse_date_from_filename(filename) else {
                continue;
            };

            let (score, line_numbers) = Self::calculate_relevance(&content, query);
            if score < min_score || score <= 0.0 {
                continue;
            }

            results.push(SearchResult {
                path: file_path,
                date,
                snippet: Self::extract_best_snippet(&content, query, 180),
                score,
                line_numbers,
            });
        }

        results.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| right.date.cmp(&left.date))
                .then_with(|| left.path.cmp(&right.path))
        });
        results.truncate(max_results);
        results
    }

    /// Search within a date range.
    pub fn search_in_range(
        &self,
        query: &str,
        start_date: &NaiveDate,
        end_date: &NaiveDate,
    ) -> Vec<SearchResult> {
        if query.trim().is_empty() {
            return Vec::new();
        }
        let entries = match self.file_store.get_entries_in_range(start_date, end_date) {
            Ok(entries) => entries,
            Err(_) => return Vec::new(),
        };
        self.search_entries(entries, query, usize::MAX, 0.0)
    }

    /// Search by date.
    pub fn search_by_date(&self, date: &NaiveDate) -> Option<MemoryEntry> {
        let path = self.file_store.get_file_path(date);
        if !path.exists() {
            return None;
        }
        self.file_store
            .read_file(&path)
            .ok()
            .map(|content| MemoryEntry {
                date: *date,
                path,
                content,
                metadata: None,
            })
    }

    /// Get recent memories (last N days).
    pub fn get_recent(&self, days: u32) -> Vec<MemoryEntry> {
        let end_date = chrono::Local::now().date_naive();
        let start_date = end_date
            .checked_sub_days(chrono::Days::new(days as u64))
            .unwrap_or(NaiveDate::MIN);
        self.file_store
            .get_entries_in_range(&start_date, &end_date)
            .unwrap_or_default()
    }

    fn search_entries(
        &self,
        entries: Vec<MemoryEntry>,
        query: &str,
        max_results: usize,
        min_score: f32,
    ) -> Vec<SearchResult> {
        let mut results = Vec::new();
        for entry in entries {
            let (score, line_numbers) = Self::calculate_relevance(&entry.content, query);
            if score < min_score || score <= 0.0 {
                continue;
            }
            results.push(SearchResult {
                path: entry.path,
                date: entry.date,
                snippet: Self::extract_best_snippet(&entry.content, query, 180),
                score,
                line_numbers,
            });
        }
        results.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| right.date.cmp(&left.date))
                .then_with(|| left.path.cmp(&right.path))
        });
        results.truncate(max_results);
        results
    }

    /// Calculate relevance score for a query in content.
    fn calculate_relevance(content: &str, query: &str) -> (f32, Vec<usize>) {
        let query_lower = query.to_lowercase();
        let query_words = query_lower
            .split_whitespace()
            .filter(|word| !word.is_empty())
            .collect::<Vec<_>>();
        if query_words.is_empty() {
            return (0.0, Vec::new());
        }

        let mut score = 0.0_f32;
        let mut line_numbers = Vec::new();
        let line_count = content.lines().count().max(1) as f32;

        for (index, line) in content.lines().enumerate() {
            let line_lower = line.to_lowercase();
            let line_no = index + 1;
            let mut matched = false;

            if line_lower.contains(&query_lower) {
                score += 1.0;
                matched = true;
            }

            let line_words = line_lower.split_whitespace().collect::<Vec<_>>();
            for query_word in &query_words {
                if line_words
                    .iter()
                    .any(|line_word| line_word.contains(query_word))
                {
                    score += 0.1;
                    matched = true;
                }
            }

            if matched {
                line_numbers.push(line_no);
            }
        }

        if line_numbers.is_empty() {
            return (0.0, Vec::new());
        }

        let normalized = (score / (line_count * 1.1)).clamp(0.0, 1.0);
        (normalized, line_numbers)
    }

    /// Extract the best snippet containing the query.
    fn extract_best_snippet(content: &str, query: &str, max_length: usize) -> String {
        if max_length == 0 {
            return String::new();
        }

        let lines = content.lines().collect::<Vec<_>>();
        let query_lower = query.to_lowercase();
        let mut snippet = if let Some(index) = lines
            .iter()
            .position(|line| line.to_lowercase().contains(&query_lower))
        {
            let start = index.saturating_sub(1);
            let end = (index + 2).min(lines.len());
            lines[start..end].join("\n")
        } else if content.len() <= max_length {
            content.to_string()
        } else {
            content.chars().take(max_length).collect::<String>()
        };

        if snippet.chars().count() > max_length {
            snippet = snippet.chars().take(max_length).collect::<String>();
            snippet.push_str("...");
        }
        snippet
    }
}
