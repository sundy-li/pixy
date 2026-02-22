use pi_ai::{Message, UserContent, UserContentBlock};
use pi_coding_agent::{
    BRANCH_SUMMARY_PREFIX, BRANCH_SUMMARY_SUFFIX, BashExecutionMessage, BranchSummaryMessage,
    COMPACTION_SUMMARY_PREFIX, COMPACTION_SUMMARY_SUFFIX, CodingMessage, CompactionSummaryMessage,
    CustomMessage, convert_to_llm,
};

fn user_message(text: &str, ts: i64) -> Message {
    Message::User {
        content: UserContent::Text(text.to_string()),
        timestamp: ts,
    }
}

#[test]
fn convert_to_llm_maps_custom_messages_and_filters_excluded_bash() {
    let input = vec![
        CodingMessage::Agent(user_message("hello", 1)),
        CodingMessage::BashExecution(BashExecutionMessage {
            role: "bashExecution".to_string(),
            command: "ls".to_string(),
            output: "a.txt".to_string(),
            exit_code: Some(0),
            cancelled: false,
            truncated: false,
            full_output_path: None,
            timestamp: 2,
            exclude_from_context: Some(true),
        }),
        CodingMessage::Custom(CustomMessage {
            role: "custom".to_string(),
            custom_type: "note".to_string(),
            content: UserContent::Text("custom text".to_string()),
            display: true,
            details: None,
            timestamp: 3,
        }),
        CodingMessage::BranchSummary(BranchSummaryMessage {
            role: "branchSummary".to_string(),
            summary: "branch summary".to_string(),
            from_id: "id-1".to_string(),
            timestamp: 4,
        }),
        CodingMessage::CompactionSummary(CompactionSummaryMessage {
            role: "compactionSummary".to_string(),
            summary: "compact summary".to_string(),
            tokens_before: 100,
            timestamp: 5,
        }),
    ];

    let output = convert_to_llm(&input);
    assert_eq!(
        output.len(),
        4,
        "excluded bash message must be filtered out"
    );

    match &output[1] {
        Message::User { content, .. } => match content {
            UserContent::Text(text) => assert_eq!(text, "custom text"),
            _ => panic!("custom message should map to user text"),
        },
        _ => panic!("expected user message"),
    }

    match &output[2] {
        Message::User { content, .. } => match content {
            UserContent::Blocks(blocks) => {
                let text = blocks
                    .iter()
                    .filter_map(|block| {
                        if let UserContentBlock::Text { text, .. } = block {
                            Some(text.clone())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("");
                assert_eq!(
                    text,
                    format!("{BRANCH_SUMMARY_PREFIX}branch summary{BRANCH_SUMMARY_SUFFIX}")
                );
            }
            _ => panic!("branch summary should map to user block content"),
        },
        _ => panic!("expected user message"),
    }

    match &output[3] {
        Message::User { content, .. } => match content {
            UserContent::Blocks(blocks) => {
                let text = blocks
                    .iter()
                    .filter_map(|block| {
                        if let UserContentBlock::Text { text, .. } = block {
                            Some(text.clone())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("");
                assert_eq!(
                    text,
                    format!(
                        "{COMPACTION_SUMMARY_PREFIX}compact summary{COMPACTION_SUMMARY_SUFFIX}"
                    )
                );
            }
            _ => panic!("compaction summary should map to user block content"),
        },
        _ => panic!("expected user message"),
    }
}
