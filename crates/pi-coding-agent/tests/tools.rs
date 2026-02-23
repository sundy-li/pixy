use std::fs;

use pi_ai::ToolResultContentBlock;
use pi_coding_agent::{
    create_bash_tool, create_coding_tools, create_edit_tool, create_read_tool, create_write_tool,
};
use serde_json::json;
use tempfile::tempdir;

fn first_text(blocks: &[ToolResultContentBlock]) -> String {
    blocks
        .iter()
        .find_map(|block| match block {
            ToolResultContentBlock::Text { text, .. } => Some(text.clone()),
            _ => None,
        })
        .unwrap_or_default()
}

#[tokio::test]
async fn write_tool_creates_parent_dirs_and_read_tool_supports_offset_limit() {
    let dir = tempdir().expect("tempdir");
    let write_tool = create_write_tool(dir.path());
    let read_tool = create_read_tool(dir.path());

    let content = "line-1\nline-2\nline-3\nline-4";
    write_tool
        .execute
        .execute(
            "call-write".to_string(),
            json!({
                "path": "nested/file.txt",
                "content": content
            }),
        )
        .await
        .expect("write should succeed");

    let result = read_tool
        .execute
        .execute(
            "call-read".to_string(),
            json!({
                "path": "nested/file.txt",
                "offset": 2,
                "limit": 2
            }),
        )
        .await
        .expect("read should succeed");

    let text = first_text(&result.content);
    assert!(text.starts_with("line-2\nline-3"));
}

#[tokio::test]
async fn edit_tool_replaces_unique_match() {
    let dir = tempdir().expect("tempdir");
    fs::write(dir.path().join("edit.txt"), "before OLD after").expect("seed file");
    let edit_tool = create_edit_tool(dir.path());

    edit_tool
        .execute
        .execute(
            "call-edit".to_string(),
            json!({
                "path": "edit.txt",
                "oldText": "OLD",
                "newText": "NEW"
            }),
        )
        .await
        .expect("edit should succeed");

    let updated = fs::read_to_string(dir.path().join("edit.txt")).expect("read edited file");
    assert_eq!(updated, "before NEW after");
}

#[tokio::test]
async fn edit_tool_rejects_non_unique_match() {
    let dir = tempdir().expect("tempdir");
    fs::write(dir.path().join("edit.txt"), "dup dup").expect("seed file");
    let edit_tool = create_edit_tool(dir.path());

    let error = edit_tool
        .execute
        .execute(
            "call-edit".to_string(),
            json!({
                "path": "edit.txt",
                "oldText": "dup",
                "newText": "x"
            }),
        )
        .await
        .expect_err("edit should fail for non-unique oldText");

    assert!(error.contains("must be unique"));
}

#[tokio::test]
async fn bash_tool_returns_output_and_exit_code_errors() {
    let dir = tempdir().expect("tempdir");
    let bash_tool = create_bash_tool(dir.path());

    let ok = bash_tool
        .execute
        .execute(
            "call-bash-ok".to_string(),
            json!({
                "command": "printf \"hello\""
            }),
        )
        .await
        .expect("bash should succeed");
    assert_eq!(first_text(&ok.content), "hello");

    let error = bash_tool
        .execute
        .execute(
            "call-bash-err".to_string(),
            json!({
                "command": "echo fail >&2; exit 7"
            }),
        )
        .await
        .expect_err("bash should return error for non-zero exit code");
    assert!(error.contains("Command exited with code 7"));
    assert!(error.contains("fail"));
}

#[test]
fn create_coding_tools_returns_expected_order() {
    let dir = tempdir().expect("tempdir");
    let tools = create_coding_tools(dir.path());
    let names = tools.into_iter().map(|tool| tool.name).collect::<Vec<_>>();
    assert_eq!(names, vec!["read", "bash", "edit", "write"]);
}
