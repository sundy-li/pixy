use chrono::{Days, Local, NaiveDate};
use pixy_coding_agent::memory::prelude::*;
use tempfile::tempdir;

#[test]
fn memory_config_validation_rejects_invalid_search_settings() {
    let mut config = MemoryConfig::default();
    config.search_max_results = 0;
    assert!(config.validate().is_err());

    let mut config = MemoryConfig::default();
    config.search_min_score = 1.5;
    assert!(config.validate().is_err());
}

#[test]
fn file_store_can_list_and_cleanup_date_files() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempdir()?;
    let store = FileStore::new(temp_dir.path(), "%Y-%m-%d.md");
    store.init()?;

    let today = Local::now().date_naive();
    let old = today
        .checked_sub_days(Days::new(31))
        .unwrap_or(NaiveDate::MIN);

    let old_path = store.get_file_path(&old);
    let today_path = store.get_file_path(&today);
    store.append_to_file(&old_path, "old memory")?;
    store.append_to_file(&today_path, "new memory")?;

    let files = store.list_files()?;
    assert_eq!(files.len(), 2);

    let deleted = store.cleanup_old_files(30)?;
    assert_eq!(deleted, 1);
    assert!(!old_path.exists());
    assert!(today_path.exists());
    Ok(())
}

#[test]
fn memory_manager_records_and_searches_with_score() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempdir()?;
    let mut config = MemoryConfig::new(temp_dir.path());
    config.search_max_results = 5;
    config.search_min_score = 0.0;
    let manager = MemoryManager::new(config)?;

    manager.record("Working on memory module integration for Pixy.")?;
    manager.record("Compaction summary should be flushed into memory.")?;

    let results = manager.search_scored("memory", 5, 0.0)?;
    assert!(!results.is_empty());
    assert!(results.iter().any(|entry| entry.snippet.contains("memory")));

    let plain = manager.search("compaction")?;
    assert!(!plain.is_empty());
    Ok(())
}

#[test]
fn memory_flush_persists_summary_and_metadata() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempdir()?;
    let config = MemoryConfig::new(temp_dir.path());
    let manager = MemoryManager::new(config)?;

    let context = MemoryFlushContext {
        session_id: Some("session-1".to_string()),
        agent_id: Some("pixy-agent".to_string()),
        token_count: 1234,
        compaction_count: 2,
        summary: Some("Compacted earlier conversation state.".to_string()),
        notes: vec!["keep latest branch context".to_string()],
        decisions: vec!["persist summary before compaction".to_string()],
        todos: vec!["re-run failing tests".to_string()],
        metadata: Some(serde_json::json!({"source":"unit-test"})),
    };
    manager.flush(&context)?;

    let content = manager.read_today()?;
    assert!(content.contains("Memory Flush"));
    assert!(content.contains("session-1"));
    assert!(content.contains("Compacted earlier conversation state."));
    assert!(content.contains("\"source\": \"unit-test\""));
    Ok(())
}
