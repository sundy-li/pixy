mod bash;
mod common;
mod edit;
mod read;
mod write;

use std::path::Path;

use pixy_agent_core::AgentTool;

pub use bash::create_bash_tool;
pub use edit::create_edit_tool;
pub use read::create_read_tool;
pub use write::create_write_tool;

pub fn create_coding_tools(cwd: impl AsRef<Path>) -> Vec<AgentTool> {
    let cwd = cwd.as_ref().to_path_buf();
    vec![
        create_read_tool(&cwd),
        create_bash_tool(&cwd),
        create_edit_tool(&cwd),
        create_write_tool(&cwd),
    ]
}
