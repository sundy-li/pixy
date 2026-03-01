use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Default)]
pub struct ChildSessionStore {
    parent_session_id: String,
    by_task_id: HashMap<String, PathBuf>,
}

impl ChildSessionStore {
    pub fn new(parent_session_id: impl Into<String>) -> Self {
        Self {
            parent_session_id: parent_session_id.into(),
            by_task_id: HashMap::new(),
        }
    }

    pub fn parent_session_id(&self) -> &str {
        &self.parent_session_id
    }

    pub fn resolve(&self, task_id: &str) -> Option<PathBuf> {
        self.by_task_id.get(task_id).cloned()
    }

    pub fn insert(&mut self, task_id: &str, session_file: impl AsRef<Path>) -> Result<(), String> {
        let task_id = task_id.trim();
        if task_id.is_empty() {
            return Err("task_id cannot be empty".to_string());
        }

        let session_file = session_file.as_ref();
        if session_file.as_os_str().is_empty() {
            return Err("session_file cannot be empty".to_string());
        }

        let session_file = session_file.to_path_buf();
        if let Some(previous) = self
            .by_task_id
            .insert(task_id.to_string(), session_file.clone())
        {
            if previous != session_file {
                tracing::warn!(
                    task_id,
                    old_session_file = %previous.display(),
                    new_session_file = %session_file.display(),
                    "task_id remapped to a different child session file"
                );
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn task_id_store_tracks_session_file_per_task() {
        let mut store = ChildSessionStore::new("parent-1");
        store
            .insert("task-1", PathBuf::from("/tmp/child-1.jsonl"))
            .expect("insert task-1");

        assert_eq!(
            store.resolve("task-1"),
            Some(PathBuf::from("/tmp/child-1.jsonl"))
        );
    }

    #[test]
    fn task_id_store_rejects_empty_task_id() {
        let mut store = ChildSessionStore::new("parent-1");
        let error = store
            .insert("", PathBuf::from("/tmp/child-1.jsonl"))
            .expect_err("empty task id should be rejected");

        assert!(error.contains("task_id"));
    }

    #[test]
    fn task_id_store_is_scoped_by_parent_session_id() {
        let left = ChildSessionStore::new("parent-left");
        let right = ChildSessionStore::new("parent-right");

        assert_ne!(left.parent_session_id(), right.parent_session_id());
    }

    #[test]
    fn task_id_store_allows_overwrite_for_existing_task_id() {
        let mut store = ChildSessionStore::new("parent-1");
        store
            .insert("task-1", PathBuf::from("/tmp/child-1.jsonl"))
            .expect("insert task-1 first path");
        store
            .insert("task-1", PathBuf::from("/tmp/child-2.jsonl"))
            .expect("insert task-1 second path");

        assert_eq!(
            store.resolve("task-1"),
            Some(PathBuf::from("/tmp/child-2.jsonl"))
        );
    }
}
