use std::fs;

use pixy_ai::{PiAiErrorCode, ToolResultContentBlock};
use pixy_coding_agent::{
    create_bash_tool, create_coding_tools, create_edit_tool, create_list_directory_tool,
    create_read_tool, create_write_tool,
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
    let write_status = write_tool
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
    let write_text = first_text(&write_status.content);
    assert!(write_text.contains("nested/file.txt"));
    assert!(write_text.contains('|'));
    assert!(write_text.contains('+'));

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

    let edit_result = edit_tool
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
    let edit_text = first_text(&edit_result.content);
    assert!(edit_text.contains("edit.txt"));
    assert!(edit_text.contains('|'));
    assert!(edit_text.contains('+'));
    assert!(edit_text.contains('-'));

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

    assert_eq!(error.code, PiAiErrorCode::ToolExecutionFailed);
    assert!(error.message.contains("must be unique"));
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
    assert_eq!(error.code, PiAiErrorCode::ToolExecutionFailed);
    assert!(error.message.contains("Command exited with code 7"));
    assert!(error.message.contains("fail"));
}

#[tokio::test]
async fn bash_tool_unwraps_nested_bash_lc_commands_before_execution() {
    let dir = tempdir().expect("tempdir");
    let bash_tool = create_bash_tool(dir.path());

    let result = bash_tool
        .execute
        .execute(
            "call-bash-nested".to_string(),
            json!({
                "command": "bash -lc 'ps -p $PPID -o comm='"
            }),
        )
        .await
        .expect("bash should succeed");

    let parent_process = first_text(&result.content).trim().to_string();
    assert_ne!(
        parent_process, "bash",
        "nested wrapping should be removed before execution"
    );
}

#[tokio::test]
async fn list_directory_tool_lists_entries_and_allows_absolute_paths_outside_workspace() {
    let dir = tempdir().expect("tempdir");
    let outside = tempdir().expect("outside tempdir");
    fs::create_dir_all(dir.path().join("nested")).expect("create nested dir");
    fs::write(dir.path().join("hello.txt"), "hello").expect("seed file");
    fs::write(outside.path().join("outside.txt"), "outside").expect("seed outside file");
    let list_directory_tool = create_list_directory_tool(dir.path());

    let listed = list_directory_tool
        .execute
        .execute("call-list".to_string(), json!({ "path": "" }))
        .await
        .expect("list should succeed");
    let listed_text = first_text(&listed.content);
    assert!(listed_text.contains("nested/"));
    assert!(listed_text.contains("hello.txt  (5 bytes)"));

    let outside_list = list_directory_tool
        .execute
        .execute(
            "call-list-outside".to_string(),
            json!({ "path": outside.path().display().to_string() }),
        )
        .await
        .expect("listing outside workspace should succeed");
    let outside_text = first_text(&outside_list.content);
    assert!(outside_text.contains("outside.txt  (7 bytes)"));
}

#[test]
fn create_coding_tools_returns_expected_order() {
    let dir = tempdir().expect("tempdir");
    let tools = create_coding_tools(dir.path());
    let names = tools.into_iter().map(|tool| tool.name).collect::<Vec<_>>();
    assert_eq!(
        names,
        vec!["list_directory", "read", "bash", "edit", "write"]
    );
}
