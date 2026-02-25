use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use pixy_ai::{Message, StopReason, UserContent, UserContentBlock};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::messages::{
    BRANCH_SUMMARY_PREFIX, BRANCH_SUMMARY_SUFFIX, COMPACTION_SUMMARY_PREFIX,
    COMPACTION_SUMMARY_SUFFIX,
};

pub const CURRENT_SESSION_VERSION: u32 = 3;

fn default_session_version() -> u32 {
    1
}

fn default_custom_message_display() -> bool {
    true
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SessionHeader {
    #[serde(rename = "type")]
    pub type_field: String,
    #[serde(default = "default_session_version")]
    pub version: u32,
    pub id: String,
    pub timestamp: String,
    pub cwd: String,
    #[serde(rename = "parentSession", skip_serializing_if = "Option::is_none")]
    pub parent_session: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SessionEntry {
    #[serde(rename = "message")]
    Message {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: String,
        message: Message,
    },
    #[serde(rename = "thinking_level_change")]
    ThinkingLevelChange {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: String,
        #[serde(rename = "thinkingLevel")]
        thinking_level: String,
    },
    #[serde(rename = "model_change")]
    ModelChange {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: String,
        provider: String,
        #[serde(rename = "modelId")]
        model_id: String,
    },
    #[serde(rename = "branch_summary")]
    BranchSummary {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: String,
        #[serde(rename = "fromId")]
        from_id: String,
        summary: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<Value>,
        #[serde(rename = "fromHook", skip_serializing_if = "Option::is_none")]
        from_hook: Option<bool>,
    },
    #[serde(rename = "compaction")]
    Compaction {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: String,
        summary: String,
        #[serde(rename = "firstKeptEntryId", skip_serializing_if = "Option::is_none")]
        first_kept_entry_id: Option<String>,
        #[serde(rename = "tokensBefore")]
        tokens_before: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<Value>,
        #[serde(rename = "fromHook", skip_serializing_if = "Option::is_none")]
        from_hook: Option<bool>,
    },
    #[serde(rename = "custom")]
    Custom {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: String,
        #[serde(rename = "customType")]
        custom_type: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        data: Option<Value>,
    },
    #[serde(rename = "custom_message")]
    CustomMessage {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: String,
        #[serde(rename = "customType")]
        custom_type: String,
        content: UserContent,
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<Value>,
        #[serde(default = "default_custom_message_display")]
        display: bool,
    },
    #[serde(rename = "label")]
    Label {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: String,
        #[serde(rename = "targetId")]
        target_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        label: Option<String>,
    },
    #[serde(rename = "session_info")]
    SessionInfo {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
}

impl SessionEntry {
    fn id(&self) -> &str {
        match self {
            SessionEntry::Message { id, .. } => id,
            SessionEntry::ThinkingLevelChange { id, .. } => id,
            SessionEntry::ModelChange { id, .. } => id,
            SessionEntry::BranchSummary { id, .. } => id,
            SessionEntry::Compaction { id, .. } => id,
            SessionEntry::Custom { id, .. } => id,
            SessionEntry::CustomMessage { id, .. } => id,
            SessionEntry::Label { id, .. } => id,
            SessionEntry::SessionInfo { id, .. } => id,
        }
    }

    fn parent_id(&self) -> Option<&str> {
        match self {
            SessionEntry::Message { parent_id, .. } => parent_id.as_deref(),
            SessionEntry::ThinkingLevelChange { parent_id, .. } => parent_id.as_deref(),
            SessionEntry::ModelChange { parent_id, .. } => parent_id.as_deref(),
            SessionEntry::BranchSummary { parent_id, .. } => parent_id.as_deref(),
            SessionEntry::Compaction { parent_id, .. } => parent_id.as_deref(),
            SessionEntry::Custom { parent_id, .. } => parent_id.as_deref(),
            SessionEntry::CustomMessage { parent_id, .. } => parent_id.as_deref(),
            SessionEntry::Label { parent_id, .. } => parent_id.as_deref(),
            SessionEntry::SessionInfo { parent_id, .. } => parent_id.as_deref(),
        }
    }

    fn to_context_message(&self) -> Option<Message> {
        match self {
            SessionEntry::Message { message, .. } => Some(message.clone()),
            SessionEntry::CustomMessage {
                content, timestamp, ..
            } => Some(Message::User {
                content: content.clone(),
                timestamp: parse_timestamp_millis(timestamp),
            }),
            SessionEntry::BranchSummary {
                summary, timestamp, ..
            } => Some(Message::User {
                content: UserContent::Blocks(vec![UserContentBlock::Text {
                    text: format!("{BRANCH_SUMMARY_PREFIX}{summary}{BRANCH_SUMMARY_SUFFIX}"),
                    text_signature: None,
                }]),
                timestamp: parse_timestamp_millis(timestamp),
            }),
            SessionEntry::ThinkingLevelChange { .. }
            | SessionEntry::ModelChange { .. }
            | SessionEntry::Compaction { .. }
            | SessionEntry::Custom { .. }
            | SessionEntry::Label { .. }
            | SessionEntry::SessionInfo { .. } => None,
        }
    }

    fn to_compaction_summary_message(&self) -> Option<Message> {
        match self {
            SessionEntry::Compaction {
                summary, timestamp, ..
            } => Some(Message::User {
                content: UserContent::Blocks(vec![UserContentBlock::Text {
                    text: format!(
                        "{COMPACTION_SUMMARY_PREFIX}{summary}{COMPACTION_SUMMARY_SUFFIX}"
                    ),
                    text_signature: None,
                }]),
                timestamp: parse_timestamp_millis(timestamp),
            }),
            _ => None,
        }
    }

    fn is_assistant_error_message(&self) -> bool {
        matches!(
            self,
            SessionEntry::Message {
                message: Message::Assistant {
                    stop_reason: StopReason::Error,
                    ..
                },
                ..
            }
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SessionContext {
    pub messages: Vec<Message>,
}

pub struct SessionManager {
    session_file: PathBuf,
    header: SessionHeader,
    entries: Vec<SessionEntry>,
    by_id: HashMap<String, usize>,
    leaf_id: Option<String>,
    next_id: u64,
}

impl SessionManager {
    pub fn create(cwd: &str, session_dir: impl AsRef<Path>) -> Result<Self, String> {
        let session_dir = session_dir.as_ref();
        fs::create_dir_all(session_dir)
            .map_err(|error| format!("create session dir failed: {error}"))?;

        let timestamp = now_millis().to_string();
        let session_id = format!("session-{timestamp}");
        let session_file = session_dir.join(format!("{session_id}.jsonl"));
        let header = SessionHeader {
            type_field: "session".to_string(),
            version: CURRENT_SESSION_VERSION,
            id: session_id,
            timestamp: timestamp.clone(),
            cwd: cwd.to_string(),
            parent_session: None,
        };

        let manager = Self {
            session_file,
            header,
            entries: vec![],
            by_id: HashMap::new(),
            leaf_id: None,
            next_id: 1,
        };
        manager.persist_header()?;
        Ok(manager)
    }

    pub fn load(session_file: impl AsRef<Path>) -> Result<Self, String> {
        let session_file = session_file.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .read(true)
            .open(&session_file)
            .map_err(|error| format!("open session file failed: {error}"))?;
        let mut lines = BufReader::new(file).lines();

        let header_line = lines
            .next()
            .ok_or_else(|| "session file is empty".to_string())?
            .map_err(|error| format!("read session header failed: {error}"))?;
        let header: SessionHeader = serde_json::from_str(&header_line)
            .map_err(|error| format!("parse session header failed: {error}"))?;

        let mut entries = Vec::new();
        let mut by_id = HashMap::new();
        let mut leaf_id = None;
        let mut max_numeric_id = 0_u64;

        for line_result in lines {
            let line =
                line_result.map_err(|error| format!("read session entry failed: {error}"))?;
            if line.trim().is_empty() {
                continue;
            }
            let entry: SessionEntry = serde_json::from_str(&line)
                .map_err(|error| format!("parse session entry failed: {error}"))?;

            if let Ok(id_value) = u64::from_str_radix(entry.id(), 16) {
                max_numeric_id = max_numeric_id.max(id_value);
            }

            by_id.insert(entry.id().to_string(), entries.len());
            leaf_id = Some(entry.id().to_string());
            entries.push(entry);
        }

        let fallback_next_id = entries.len() as u64 + 1;
        let next_id = max_numeric_id.max(fallback_next_id - 1) + 1;

        Ok(Self {
            session_file,
            header,
            entries,
            by_id,
            leaf_id,
            next_id,
        })
    }

    pub fn append_message(&mut self, message: Message) -> Result<String, String> {
        let id = format!("{:08x}", self.next_id);
        self.next_id += 1;
        let entry = SessionEntry::Message {
            id: id.clone(),
            parent_id: self.leaf_id.clone(),
            timestamp: now_millis().to_string(),
            message,
        };
        self.by_id.insert(id.clone(), self.entries.len());
        self.entries.push(entry.clone());
        self.leaf_id = Some(id.clone());
        self.append_entry(&entry)?;
        Ok(id)
    }

    pub fn branch(&mut self, branch_from_id: &str) -> Result<(), String> {
        if !self.by_id.contains_key(branch_from_id) {
            return Err(format!("branch id not found: {branch_from_id}"));
        }
        self.leaf_id = Some(branch_from_id.to_string());
        Ok(())
    }

    pub fn branch_with_summary(
        &mut self,
        branch_from_id: Option<&str>,
        summary: &str,
    ) -> Result<String, String> {
        match branch_from_id {
            Some(id) => self.branch(id)?,
            None => {
                self.leaf_id = None;
            }
        }

        let parent_id = self.leaf_id.clone();
        let id = format!("{:08x}", self.next_id);
        self.next_id += 1;
        let entry = SessionEntry::BranchSummary {
            id: id.clone(),
            parent_id,
            timestamp: now_millis().to_string(),
            from_id: branch_from_id.unwrap_or("root").to_string(),
            summary: summary.to_string(),
            details: None,
            from_hook: None,
        };

        self.by_id.insert(id.clone(), self.entries.len());
        self.entries.push(entry.clone());
        self.leaf_id = Some(id.clone());
        self.append_entry(&entry)?;
        Ok(id)
    }

    pub fn append_compaction(
        &mut self,
        summary: &str,
        first_kept_entry_id: Option<&str>,
        tokens_before: u64,
    ) -> Result<String, String> {
        if let Some(first_kept_id) = first_kept_entry_id {
            if !self.by_id.contains_key(first_kept_id) {
                return Err(format!("first kept entry id not found: {first_kept_id}"));
            }
        }

        let id = format!("{:08x}", self.next_id);
        self.next_id += 1;
        let entry = SessionEntry::Compaction {
            id: id.clone(),
            parent_id: self.leaf_id.clone(),
            timestamp: now_millis().to_string(),
            summary: summary.to_string(),
            first_kept_entry_id: first_kept_entry_id.map(ToOwned::to_owned),
            tokens_before,
            details: None,
            from_hook: None,
        };

        self.by_id.insert(id.clone(), self.entries.len());
        self.entries.push(entry.clone());
        self.leaf_id = Some(id.clone());
        self.append_entry(&entry)?;
        Ok(id)
    }

    pub fn append_thinking_level_change(&mut self, thinking_level: &str) -> Result<String, String> {
        let id = format!("{:08x}", self.next_id);
        self.next_id += 1;
        let entry = SessionEntry::ThinkingLevelChange {
            id: id.clone(),
            parent_id: self.leaf_id.clone(),
            timestamp: now_millis().to_string(),
            thinking_level: thinking_level.to_string(),
        };

        self.by_id.insert(id.clone(), self.entries.len());
        self.entries.push(entry.clone());
        self.leaf_id = Some(id.clone());
        self.append_entry(&entry)?;
        Ok(id)
    }

    pub fn append_model_change(
        &mut self,
        provider: &str,
        model_id: &str,
    ) -> Result<String, String> {
        let id = format!("{:08x}", self.next_id);
        self.next_id += 1;
        let entry = SessionEntry::ModelChange {
            id: id.clone(),
            parent_id: self.leaf_id.clone(),
            timestamp: now_millis().to_string(),
            provider: provider.to_string(),
            model_id: model_id.to_string(),
        };

        self.by_id.insert(id.clone(), self.entries.len());
        self.entries.push(entry.clone());
        self.leaf_id = Some(id.clone());
        self.append_entry(&entry)?;
        Ok(id)
    }

    pub fn append_custom_entry(
        &mut self,
        custom_type: &str,
        data: Option<Value>,
    ) -> Result<String, String> {
        let id = format!("{:08x}", self.next_id);
        self.next_id += 1;
        let entry = SessionEntry::Custom {
            id: id.clone(),
            parent_id: self.leaf_id.clone(),
            timestamp: now_millis().to_string(),
            custom_type: custom_type.to_string(),
            data,
        };

        self.by_id.insert(id.clone(), self.entries.len());
        self.entries.push(entry.clone());
        self.leaf_id = Some(id.clone());
        self.append_entry(&entry)?;
        Ok(id)
    }

    pub fn append_custom_message_entry(
        &mut self,
        custom_type: &str,
        content: UserContent,
        display: bool,
        details: Option<Value>,
    ) -> Result<String, String> {
        let id = format!("{:08x}", self.next_id);
        self.next_id += 1;
        let entry = SessionEntry::CustomMessage {
            id: id.clone(),
            parent_id: self.leaf_id.clone(),
            timestamp: now_millis().to_string(),
            custom_type: custom_type.to_string(),
            content,
            details,
            display,
        };

        self.by_id.insert(id.clone(), self.entries.len());
        self.entries.push(entry.clone());
        self.leaf_id = Some(id.clone());
        self.append_entry(&entry)?;
        Ok(id)
    }

    pub fn append_label(&mut self, target_id: &str, label: Option<&str>) -> Result<String, String> {
        let id = format!("{:08x}", self.next_id);
        self.next_id += 1;
        let entry = SessionEntry::Label {
            id: id.clone(),
            parent_id: self.leaf_id.clone(),
            timestamp: now_millis().to_string(),
            target_id: target_id.to_string(),
            label: label.map(ToOwned::to_owned),
        };

        self.by_id.insert(id.clone(), self.entries.len());
        self.entries.push(entry.clone());
        self.leaf_id = Some(id.clone());
        self.append_entry(&entry)?;
        Ok(id)
    }

    pub fn append_session_info(&mut self, name: Option<&str>) -> Result<String, String> {
        let id = format!("{:08x}", self.next_id);
        self.next_id += 1;
        let entry = SessionEntry::SessionInfo {
            id: id.clone(),
            parent_id: self.leaf_id.clone(),
            timestamp: now_millis().to_string(),
            name: name.map(ToOwned::to_owned),
        };

        self.by_id.insert(id.clone(), self.entries.len());
        self.entries.push(entry.clone());
        self.leaf_id = Some(id.clone());
        self.append_entry(&entry)?;
        Ok(id)
    }

    pub fn rewind_leaf_if_last_assistant_error(&mut self) -> bool {
        let Some(leaf_id) = self.leaf_id.clone() else {
            return false;
        };

        let Some(index) = self.by_id.get(&leaf_id).copied() else {
            return false;
        };

        let parent = match &self.entries[index] {
            entry if entry.is_assistant_error_message() => entry.parent_id().map(ToOwned::to_owned),
            _ => return false,
        };

        self.leaf_id = parent;
        true
    }

    pub fn session_file(&self) -> Option<&PathBuf> {
        Some(&self.session_file)
    }

    pub fn cwd(&self) -> &str {
        &self.header.cwd
    }

    pub fn latest_model_change(&self) -> Option<(String, String)> {
        self.current_path_entries()
            .iter()
            .rev()
            .find_map(|entry| match entry {
                SessionEntry::ModelChange {
                    provider, model_id, ..
                } => Some((provider.clone(), model_id.clone())),
                _ => None,
            })
    }

    pub fn first_kept_entry_id_for_recent_messages(
        &self,
        keep_recent_messages: usize,
    ) -> Option<String> {
        let context_entries = self
            .current_path_entries()
            .into_iter()
            .filter(|entry| entry.to_context_message().is_some())
            .collect::<Vec<_>>();

        if context_entries.is_empty() {
            return None;
        }

        if keep_recent_messages == 0 {
            return Some(context_entries[0].id().to_string());
        }

        if context_entries.len() <= keep_recent_messages {
            return None;
        }

        let first_kept_idx = context_entries.len() - keep_recent_messages;
        Some(context_entries[first_kept_idx].id().to_string())
    }

    pub fn build_session_context(&self) -> SessionContext {
        let path_entries = self.current_path_entries();

        let mut messages = Vec::new();
        let latest_compaction_idx = path_entries
            .iter()
            .enumerate()
            .rfind(|(_, entry)| matches!(entry, SessionEntry::Compaction { .. }))
            .map(|(idx, _)| idx);

        if let Some(compaction_idx) = latest_compaction_idx {
            let compaction_entry = path_entries[compaction_idx];
            if let Some(message) = compaction_entry.to_compaction_summary_message() {
                messages.push(message);
            }

            let kept_start_idx = match compaction_entry {
                SessionEntry::Compaction {
                    first_kept_entry_id: Some(first_kept_id),
                    ..
                } => path_entries
                    .iter()
                    .position(|entry| entry.id() == first_kept_id)
                    .unwrap_or(compaction_idx + 1),
                _ => compaction_idx + 1,
            };

            for entry in &path_entries[kept_start_idx..compaction_idx] {
                if let Some(message) = entry.to_context_message() {
                    messages.push(message);
                }
            }

            for entry in &path_entries[(compaction_idx + 1)..] {
                if let Some(message) = entry.to_context_message() {
                    messages.push(message);
                }
            }
        } else {
            for entry in path_entries {
                if let Some(message) = entry.to_context_message() {
                    messages.push(message);
                }
            }
        }

        SessionContext { messages }
    }

    fn current_path_entry_indices(&self) -> Vec<usize> {
        let mut path_indices = Vec::new();
        let mut current_id = self.leaf_id.clone();

        while let Some(id) = current_id {
            let Some(index) = self.by_id.get(&id).copied() else {
                break;
            };
            path_indices.push(index);
            let entry = &self.entries[index];
            current_id = entry.parent_id().map(ToOwned::to_owned);
        }

        path_indices.reverse();
        path_indices
    }

    fn current_path_entries(&self) -> Vec<&SessionEntry> {
        self.current_path_entry_indices()
            .iter()
            .map(|index| &self.entries[*index])
            .collect::<Vec<_>>()
    }

    fn persist_header(&self) -> Result<(), String> {
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.session_file)
            .map_err(|error| format!("open session file failed: {error}"))?;
        let line = serde_json::to_string(&self.header)
            .map_err(|error| format!("serialize header failed: {error}"))?;
        file.write_all(line.as_bytes())
            .map_err(|error| format!("write header failed: {error}"))?;
        file.write_all(b"\n")
            .map_err(|error| format!("write header newline failed: {error}"))?;
        Ok(())
    }

    fn append_entry(&self, entry: &SessionEntry) -> Result<(), String> {
        let mut file = OpenOptions::new()
            .append(true)
            .open(&self.session_file)
            .map_err(|error| format!("open session file for append failed: {error}"))?;
        let line = serde_json::to_string(entry)
            .map_err(|error| format!("serialize session entry failed: {error}"))?;
        file.write_all(line.as_bytes())
            .map_err(|error| format!("append session entry failed: {error}"))?;
        file.write_all(b"\n")
            .map_err(|error| format!("append session newline failed: {error}"))?;
        Ok(())
    }
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

fn now_millis_i64() -> i64 {
    now_millis().min(i64::MAX as u128) as i64
}

fn parse_timestamp_millis(timestamp: &str) -> i64 {
    if let Ok(parsed) = timestamp.parse::<i64>() {
        return parsed;
    }

    if let Ok(parsed) = chrono::DateTime::parse_from_rfc3339(timestamp) {
        return parsed.timestamp_millis();
    }

    now_millis_i64()
}
