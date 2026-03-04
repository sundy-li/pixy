# Multi-Agent Mission Plugin Example

This example provides a **pluginized mission flow** inspired by `docs/droid_arch.md`:

- default agent: orchestrator (task analysis and decomposition only)
- `code` subagent: implementation worker
- `review` subagent: validation worker

Target flow:

`default (orchestrator) -> code -> review -> (if fail) code -> review ... until pass`

## Files

- `pixy.toml`: enables `multi_agent` and loads plugin manifest
- `mission-plugin.toml`: defines policy and declarative hooks for mission protocol
- `agents/code.toml`: code worker subagent profile
- `agents/review.toml`: review worker subagent profile

`agents/*.toml` supports `provider` (maps to `llm.providers[].name`) and `model` overrides per subagent.

## What the plugin enforces

- Extends `task` tool description with mission contract so the default agent behaves as orchestrator.
- Adds role contracts for `code` and `review` at `before_task_dispatch`.
- Enforces review status output format:
  - `REVIEW_STATUS: PASS`
  - `REVIEW_STATUS: FAIL`
- Adds post-review decision tags in task summary:
  - `[mission-decision] pass`
  - `[mission-decision] retry-code`

## Quick start

1. Copy plugin manifest and sidecar agents into your Pixy config directory:

   ```bash
   mkdir -p ~/.pixy/plugins
   cp examples/multi-agent-mission-plugin/mission-plugin.toml ~/.pixy/plugins/mission-plugin.toml
   mkdir -p ~/.pixy/plugins/agents
   cp examples/multi-agent-mission-plugin/agents/*.toml ~/.pixy/plugins/agents/
   ```

2. Merge `examples/multi-agent-mission-plugin/pixy.toml` into `~/.pixy/pixy.toml`.

   If you launch with a custom config directory (for example `./target/debug/pixy cli --conf-dir ~/.pixy2`),
   copy everything into that directory instead:

   ```bash
   mkdir -p ~/.pixy2/plugins
   cp examples/multi-agent-mission-plugin/mission-plugin.toml ~/.pixy2/plugins/mission-plugin.toml
   mkdir -p ~/.pixy2/plugins/agents
   cp examples/multi-agent-mission-plugin/agents/*.toml ~/.pixy2/plugins/agents/
   # and ensure ~/.pixy2/pixy.toml contains:
   # [multi_agent]
   # enabled = true
   # [[multi_agent.plugins]]
   # path = "plugins/mission-plugin.toml"
   ```

3. Start Pixy and ask for a coding task. The orchestrator is expected to run:

   1. `task(subagent_type="code", task_id="mission-code", ...)`
   2. `task(subagent_type="review", task_id="mission-review", ...)`
   3. If review returns `REVIEW_STATUS: FAIL`, send review feedback to `code` and repeat.

## Notes

- Current hook model cannot auto-trigger a new `task` call by itself.
- The loop is enforced through plugin-injected mission protocol and structured review output.
- Keep separate stable `task_id` values for `code` and `review` workers to avoid cross-role session reuse.
