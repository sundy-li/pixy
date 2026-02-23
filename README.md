# pixy

`pixy` is a Rust coding-agent workspace inspired by [`pi-mono`](https://github.com/badlogic/pi-mono).

The goal is pragmatic behavior compatibility with `pi-mono`, with Rust-native architecture, tests, and iteration speed.

## What Works Today

- Provider-agnostic LLM layer (`pixy-ai`) with event streaming and tool-call support.
- Built-in providers:
- `openai-responses` (with automatic 404 fallback to `openai-completions`)
- `openai-completions`
- `openai-codex-responses`
- `azure-openai-responses`
- `anthropic-messages`
- `google-generative-ai`
- `google-gemini-cli`
- `google-vertex`
- `bedrock-converse-stream`
- Agent loop core (`pixy-agent-core`) with retries, fallback, metrics, and abort.
- Coding session orchestration (`pixy-coding-agent`) with persistent sessions, branch, compaction, and resume.
- Terminal UI (`pixy-tui`) with streaming output, interrupt, model switching, and theme/keybinding customization.
- Skills discovery and prompt injection compatible with the current project conventions.

## Workspace Layout

- `crates/pixy-ai`: provider abstraction, streaming protocol, built-in provider adapters.
- `crates/pixy-agent-core`: agent loop, tool execution, retries/fallback, metrics events.
- `crates/pixy-coding-agent`: CLI binary (`pixy`), session manager, tools, skills integration.
- `crates/pixy-tui`: TUI rendering, keybindings, themes, transcript behavior.
- `docs/plan.md`: phased architecture and migration plan.
- `docs/task.md`: execution checklist and recent completion log.

## Quick Start

### Prerequisites

- Rust stable toolchain.
- At least one provider credential (for example `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`).

### Run Interactive TUI

```bash
cargo run -p pixy-coding-agent --bin pixy
```

### Run One Prompt (Non-interactive)

```bash
cargo run -p pixy-coding-agent --bin pixy -- --prompt "Summarize this repository structure."
```

### Run in REPL Mode (No TUI)

```bash
cargo run -p pixy-coding-agent --bin pixy -- --no-tui
```

### Show CLI Help

```bash
cargo run -p pixy-coding-agent --bin pixy -- --help
```

## CLI Options (Most Used)

- `--provider <name>`: provider alias (`openai`, `anthropic`, `google`, `bedrock`, ...).
- `--model <id>`: model id, or `provider/model` shorthand.
- `--api <api-name>`: force exact API adapter.
- `--base-url <url>`: override provider base URL.
- `--prompt <text>`: run single-shot prompt and exit.
- `--no-tui`: force plain terminal mode.
- `--session-file <path>`: load/continue a specific session file.
- `--continue-first`: immediately run a continue turn at startup.
- `--no-tools`: disable built-in coding tools.
- `--hide-tool-results`: hide tool output lines in CLI/TUI transcript.
- `--skill <path>`: add explicit skill path (repeatable).
- `--no-skills`: disable default skill auto-discovery.
- `--theme <dark|light>`: select TUI theme.

## Session Commands (REPL)

- `/continue`: continue from current context without adding a user message.
- `/resume [session]`: resume latest or specific historical session.
- `/session`: print current session file path.
- `/help`: show command help.
- `/exit` or `/quit`: quit.

## Runtime Files and Directories

Default runtime data lives under `~/.pixy`:

```text
~/.pixy/
  pixy.log
  agent/
    settings.json
    models.json
    keybindings.json
    input_history.jsonl
    sessions/
```

- Default agent dir: `~/.pixy/agent`
- Default sessions dir: `~/.pixy/agent/sessions`
- Tracing log file: `~/.pixy/pixy.log`

## Configuration

### `settings.json`

`~/.pixy/agent/settings.json` supports:

- `defaultProvider`
- `defaultModel`
- `theme`
- `skills` (extra skill paths)
- `env` (key-value map for model/provider config indirection)

Example:

```json
{
  "defaultProvider": "openai",
  "defaultModel": "openai/gpt-4.1-mini",
  "theme": "dark",
  "skills": ["./skills", "~/.agents/skills"],
  "env": {
    "OPENAI_KEY": "sk-...",
    "ANTHROPIC_TOKEN": "..."
  }
}
```

### `models.json`

`~/.pixy/agent/models.json` supports provider-level defaults and model catalog entries.

Example:

```json
{
  "providers": {
    "openai": {
      "api": "openai-completions",
      "baseUrl": "https://api.openai.com/v1",
      "apiKey": "$OPENAI_KEY",
      "models": [
        {
          "id": "gpt-4.1",
          "contextWindow": 200000,
          "maxTokens": 8192,
          "reasoning": true,
          "reasoningEffort": "high"
        },
        {
          "id": "gpt-4.1-mini"
        }
      ]
    },
    "anthropic": {
      "api": "anthropic-messages",
      "baseUrl": "https://api.anthropic.com/v1",
      "apiKey": "$ANTHROPIC_TOKEN",
      "models": [
        {
          "id": "claude-3-5-sonnet-latest"
        }
      ]
    }
  }
}
```

Notes:

- `apiKey` supports `$ENV_KEY` indirection through `settings.env` and process env.
- `reasoning` and `reasoningEffort` are model-level runtime config fields.
- For OpenAI completions payloads, `reasoningEffort` is emitted as `reasoning_effort` when enabled.

## Provider Alias Resolution

Common aliases resolved by CLI runtime:

- `openai` -> `openai-responses`
- `openai-completions` -> `openai-completions`
- `openai-responses` -> `openai-responses`
- `codex` / `openai-codex-responses` -> `openai-codex-responses`
- `azure-openai` / `azure-openai-responses` -> `azure-openai-responses`
- `anthropic` -> `anthropic-messages`
- `google` / `google-generative-ai` -> `google-generative-ai`
- `google-gemini-cli` -> `google-gemini-cli`
- `google-vertex` -> `google-vertex`
- `bedrock` / `amazon-bedrock` -> `bedrock-converse-stream`

## System Prompt

`pixy` builds system prompt text dynamically:

- Default prompt uses structured sections (`<identity>`, `<runtime_contract>`, `<tools_contract>`).
- Runtime context appends current datetime and working directory.
- `--system-prompt` accepts literal text or file path (file content is used if found).
- Skills metadata is injected as `<available_skills>...</available_skills>` when `read` tool is available.

## Skills

`pixy` supports `SKILL.md` discovery and prompt exposure.

Default discovery locations:

- `~/.pixy/agent/skills`
- `~/.agents/skills`
- `<cwd>/.pixy/skills`
- `.agents/skills` from `<cwd>` up to git root

Extra skill sources:

- `settings.json` -> `skills: ["..."]`
- CLI -> repeatable `--skill <path>`

Skill frontmatter keys:

- `name`
- `description` (required)
- `disable-model-invocation` (optional, default `false`)

Example `SKILL.md`:

```md
---
name: repo-inspector
description: Analyze repository structure and architecture files.
disable-model-invocation: false
---

Use ripgrep for discovery, then summarize architecture-relevant files.
```

Notes:

- Invalid skill files are skipped with diagnostics.
- Name collisions keep the first loaded skill and emit warnings.

## TUI Themes and Keybindings

Built-in themes:

- `dark`
- `light`

Theme files:

- `crates/pixy-tui/themes/dark.json`
- `crates/pixy-tui/themes/light.json`

Keybinding overrides are loaded from `~/.pixy/agent/keybindings.json`.

Common defaults:

- `Enter`: submit
- `Shift+Enter` / `Ctrl+J`: newline
- `Alt+Enter`: follow-up queue
- `Esc`: interrupt streaming/tool execution
- `Ctrl+D`: force exit
- `Ctrl+P` / `Ctrl+Shift+P`: cycle model
- `Ctrl+L`: select model

## Development

### Format

```bash
cargo fmt
```

### Test (single crate)

```bash
cargo test -p pixy-coding-agent
```

### Test (workspace)

```bash
cargo test
```

### Optional `just` shortcuts

```bash
just dev-cli
just cli
```

## Project Status

Current implementation status and near-term tasks:

- `docs/plan.md`
- `docs/task.md`
