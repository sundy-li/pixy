# Multi-Agent Plugin Example

This folder demonstrates a minimal declarative plugin setup for Pixy multi-agent orchestration.

## Files

- `pixy.toml`: runtime config enabling multi-agent + `task` tool
- `basic-plugin.toml`: plugin manifest defining one extra subagent and policy

## Quick start

1. Copy the plugin manifest into your Pixy config directory:

   ```bash
   mkdir -p ~/.pixy/plugins
   cp examples/multi-agent/basic-plugin.toml ~/.pixy/plugins/basic-plugin.toml
   ```

2. Merge the `examples/multi-agent/pixy.toml` sections into `~/.pixy/pixy.toml`.

3. Start Pixy and delegate through `task` when the model needs a subagent.

For manifest field reference, see `docs/multi-agent-plugin-manifest.md`.
