# Multi-Agent Hooks Example

This example shows **routing + hooks fully in TOML** (including hooks inside plugin manifest).

## Files

- `pixy.toml`: enables `multi_agent` and loads plugin manifests
- `basic-plugin.toml`: declares subagents + policy routing rules + declarative hooks
- (optional) Rust hooks can still be added with `MultiAgentHook`, but this example does not require Rust code.

## When does subagent routing happen?

1. Parent model emits a `task` tool call.
2. `TaskDispatcher` runs declarative `before_task_dispatch` hooks (from plugin manifests and/or `pixy.toml`).
3. Policy routing runs (`fallback_subagent`, allow/deny rules).
4. Dispatcher resolves target subagent and executes/reuses child session by `task_id`.
5. Declarative `after_task_result` hooks can rewrite final summary/details (including running `bash`).

## Quick start

1. Copy plugin manifest:

   ```bash
   mkdir -p ~/.pixy/plugins
   cp examples/multi-agent-hooks/basic-plugin.toml ~/.pixy/plugins/basic-plugin.toml
   ```

2. Merge `examples/multi-agent-hooks/pixy.toml` into `~/.pixy/pixy.toml`.
3. Start Pixy. Routing + hook behavior will be driven by TOML only.
