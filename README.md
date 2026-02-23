# pixy

`pixy` is a Rust-based coding agent workspace inspired by [`pi-mono`](https://github.com/badlogic/pi-mono).

This repository focuses on rebuilding key `pi-mono` capabilities in Rust while keeping behavior aligned where practical.

## Inspiration

This project is directly inspired by `pi-mono`:

- architecture and feature planning track `pi-mono` capabilities
- protocol and session behavior aim for semantic compatibility
- implementation targets Rust-native reliability, testing, and iteration speed

## Workspace Layout

- `crates/pixy-ai`: model/provider abstraction, streaming APIs, provider adapters
- `crates/pixy-agent-core`: agent loop, tool execution, retries/fallback, metrics
- `crates/pixy-coding-agent`: coding session orchestration and CLI binary (`pixy`)
- `crates/pixy-tui`: terminal UI components used by the CLI
- `docs/plan.md`: migration and architecture plan
- `docs/task.md`: execution checklist and recent completion log

## Quick Start

### Prerequisites

- Rust toolchain (stable)
- API credentials for your target provider (for example `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`)

### Run Interactive TUI

```bash
cargo run -p pixy-coding-agent --bin pixy
```

### Run Single Prompt (non-interactive)

```bash
cargo run -p pixy-coding-agent --bin pixy -- --prompt "Summarize this repository structure."
```

### CLI Help

```bash
cargo run -p pixy-coding-agent --bin pixy -- --help
```

## Configuration and Data Paths

Default runtime data now lives under `~/.pixy`:

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

Notes:

- default agent directory: `~/.pixy/agent`
- default session directory: `~/.pixy/agent/sessions`
- tracing log file: `~/.pixy/pixy.log`

## System Prompt Behavior

`pixy` builds the system prompt dynamically at runtime:

- includes available tool descriptions
- includes execution guidelines
- appends current date/time
- appends current working directory

`--system-prompt` supports either:

- a literal prompt string, or
- a file path (if the file exists, its content is used as the prompt body)

## Development

### Format

```bash
cargo fmt
```

### Test (affected crate)

```bash
cargo test -p pixy-coding-agent
```

### Test (full workspace)

```bash
cargo test
```

## Current Status

The current implementation focus and completed milestones are tracked in:

- `docs/plan.md`
- `docs/task.md`
