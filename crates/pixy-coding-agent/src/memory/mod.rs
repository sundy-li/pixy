//! Memory module for Pixy agent with date-based file storage.
//!
//! This module provides memory functionality similar to OpenClaw's memory system,
//! with date-based file organization and search capabilities.

pub mod config;
pub mod file_store;
pub mod memory;
pub mod search;

pub use config::MemoryConfig;
pub use file_store::{FileStore, FileStoreError, MemoryEntry};
pub use memory::{MemoryError, MemoryFlushContext, MemoryManager};
pub use search::{MemorySearch, SearchResult};

/// Re-export common memory types.
pub mod prelude {
    pub use super::config::MemoryConfig;
    pub use super::file_store::{FileStore, FileStoreError, MemoryEntry};
    pub use super::memory::{MemoryError, MemoryFlushContext, MemoryManager};
    pub use super::search::{MemorySearch, SearchResult};
}
