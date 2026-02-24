# pixy

`pixy` is a Rust coding-agent workspace inspired by [`pi-mono`](https://github.com/badlogic/pi-mono).

## Workspace Layout

- `crates/pixy-ai`: provider abstraction, streaming protocol, built-in provider adapters.
- `crates/pixy-agent-core`: agent loop, tool execution, retries/fallback, metrics events.
- `crates/pixy-coding-agent`: CLI binary (`pixy`), session manager, tools, skills integration.
- `crates/pixy-tui`: TUI rendering, keybindings, themes, transcript behavior.

## Quick Start

### Prerequisites

- Rust stable toolchain.
- At least one provider credential (for example `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`).
- Initialize the config: `mkdir -p  ~/.pixy && cp pixy.toml.sample ~/.pixy/pixy.toml`

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

## Built-in Coding Tools

- `list_directory`: list directory entries under workspace path (directories end with `/`).
- `read`: read UTF-8 file content with optional `offset` and `limit`.
- `bash`: execute shell commands in workspace.
- `edit`: replace a unique text fragment in an existing file.
- `write`: create or overwrite files.

## Session Commands (REPL)

- `/continue`: continue from current context without adding a user message.
- `/resume [session]`: show recent sessions and choose one to resume (TUI opens a picker with task summary + updated time, Enter to confirm; or pass a specific session file).
- `/session`: print current session file path.
- `/help`: show command help.
- `/exit` or `/quit`: quit.

## Runtime Files and Directories

Default runtime data lives under `~/.pixy`:

```text
~/.pixy/
  pixy.log
  pixy.toml
  agent/
    keybindings.json
    input_history.jsonl
    sessions/
```

- Default agent dir: `~/.pixy/agent`
- Default sessions dir: `~/.pixy/agent/sessions`
- Tracing log file: `~/.pixy/pixy.log`

## Configuration

### `pixy.toml`

Example [pixy.toml.sample](./pixy.toml.sample)

Notes:
- `api_key` supports `$ENV_KEY`: it resolves from `[env]` in `pixy.toml` first, then falls back to process environment variables.
- `default_provider = "*"` routes by provider `weight`.
- `weight` is provider-level and must be `< 100`. `0` means that provider is never selected by weighted routing.
- Only providers with `kind = "chat"` participate in session routing and requests; `embedding` providers are ignored.
- `reasoning` / `reasoning_effort` / `context_window` / `max_tokens` are runtime model parameters.

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
