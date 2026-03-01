use std::sync::{Arc, Mutex};

use pixy_ai::ToolResultContentBlock;
use pixy_coding_agent::{
    create_memory_tool,
    memory::{MemoryConfig, MemoryManager},
};
use serde_json::json;
use tempfile::tempdir;

fn first_text(content: &[ToolResultContentBlock]) -> &str {
    match content.first() {
        Some(ToolResultContentBlock::Text { text, .. }) => text.as_str(),
        _ => "",
    }
}

#[tokio::test]
async fn memory_tool_record_search_and_get_actions_work() {
    let dir = tempdir().expect("tempdir");
    let manager = MemoryManager::new(MemoryConfig::new(dir.path()))
        .expect("memory manager should initialize");
    let tool = create_memory_tool(Arc::new(Mutex::new(manager)), 10, 0.0);

    tool.execute
        .execute(
            "call-1".to_string(),
            json!({
                "action": "record",
                "content": "Implemented memory tool integration."
            }),
        )
        .await
        .expect("record action should succeed");

    let search = tool
        .execute
        .execute(
            "call-2".to_string(),
            json!({
                "action": "search",
                "query": "memory tool"
            }),
        )
        .await
        .expect("search action should succeed");
    assert!(
        first_text(&search.content).contains("memory"),
        "search output should include memory snippet"
    );
    assert!(
        search.details["count"].as_u64().unwrap_or(0) >= 1,
        "search result count should be >= 1"
    );

    let get = tool
        .execute
        .execute(
            "call-3".to_string(),
            json!({
                "action": "get"
            }),
        )
        .await
        .expect("get action should succeed");
    assert!(
        first_text(&get.content).contains("Implemented memory tool integration."),
        "get output should include recorded memory"
    );
}

#[tokio::test]
async fn memory_tool_flush_and_cleanup_actions_work() {
    let dir = tempdir().expect("tempdir");
    let manager = MemoryManager::new(MemoryConfig::new(dir.path()))
        .expect("memory manager should initialize");
    let tool = create_memory_tool(Arc::new(Mutex::new(manager)), 10, 0.0);

    let flush = tool
        .execute
        .execute(
            "call-1".to_string(),
            json!({
                "action": "flush",
                "session_id": "session-xyz",
                "agent_id": "pixy",
                "token_count": 4096,
                "compaction_count": 1,
                "summary": "Compaction executed",
                "notes": ["keep latest state"],
                "metadata": {"source": "memory_tool_test"}
            }),
        )
        .await
        .expect("flush action should succeed");
    assert!(
        first_text(&flush.content).contains("Memory flush recorded"),
        "flush should return success text"
    );

    let get = tool
        .execute
        .execute(
            "call-2".to_string(),
            json!({
                "action": "get"
            }),
        )
        .await
        .expect("get action should succeed");
    assert!(first_text(&get.content).contains("Memory Flush"));
    assert!(first_text(&get.content).contains("session-xyz"));

    let cleanup = tool
        .execute
        .execute(
            "call-3".to_string(),
            json!({
                "action": "cleanup"
            }),
        )
        .await
        .expect("cleanup action should succeed");
    assert_eq!(
        cleanup.details["action"].as_str(),
        Some("cleanup"),
        "cleanup action details should be present"
    );
}

#[tokio::test]
async fn memory_tool_rejects_unknown_action() {
    let dir = tempdir().expect("tempdir");
    let manager = MemoryManager::new(MemoryConfig::new(dir.path()))
        .expect("memory manager should initialize");
    let tool = create_memory_tool(Arc::new(Mutex::new(manager)), 10, 0.0);

    let error = tool
        .execute
        .execute(
            "call-1".to_string(),
            json!({
                "action": "unknown"
            }),
        )
        .await
        .expect_err("unknown action should fail");
    assert_eq!(error.code, pixy_ai::PiAiErrorCode::ToolArgumentsInvalid);
}
