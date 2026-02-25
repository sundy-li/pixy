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

## Gateway (Telegram + Feishu)

Run gateway in foreground:

```bash
cargo run -p pixy-coding-agent --bin pixy -- gateway start
```

Run gateway daemon / lifecycle:

```bash
cargo run -p pixy-coding-agent --bin pixy -- gateway start --daemon
cargo run -p pixy-coding-agent --bin pixy -- gateway restart
cargo run -p pixy-coding-agent --bin pixy -- gateway stop
```

Gateway runtime files:

```text
~/.pixy/gateway/
  gateway.pid
  gateway.state.json
```

Gateway channels are configured in the same `~/.pixy/pixy.toml` file:

```toml
[gateway]
enabled = true
bind = "0.0.0.0:8080"
request_timeout_ms = 20000

[[gateway.channels]]
name = "tg-main"
kind = "telegram"
mode = "polling"
bot_token = "$TELEGRAM_BOT_TOKEN"
# proxy_url = "socks5://127.0.0.1:7891"
poll_interval_ms = 1500
update_limit = 50
allowed_user_ids = ["123456789"]

[[gateway.channels]]
name = "feishu-main"
kind = "feishu"
mode = "webhook"
app_id = "$FEISHU_APP_ID"
app_secret = "$FEISHU_APP_SECRET"
verification_token = "$FEISHU_VERIFICATION_TOKEN"
# proxy_url = "http://127.0.0.1:7890"
poll_interval_ms = 100
allowed_user_ids = ["ou_xxx"]
```

Notes:
- Telegram uses polling (`getUpdates`) and supports `typing...` status while processing.
- Feishu uses webhook subscription; configure callback URL as `http://<gateway-bind>/webhook/feishu/{channel_name}`.
- `allowed_user_ids` is required and currently only private chat is processed.
- Send `/new` in channel chat to reset that user's routed session context.
- Gateway session files are partitioned by month: `~/.pixy/agent/sessions/<YYYY>/<MM>/...`.

## Built-in Coding Tools

- `list_directory`: list directory entries under workspace path (directories end with `/`).
- `read`: read UTF-8 file content with optional `offset` and `limit`.
- `bash`: execute shell commands in workspace.
- `edit`: replace a unique text fragment in an existing file.
- `write`: create or overwrite files.

## Session Commands (REPL)

- `/continue`: continue from current context without adding a user message.
- `/new`: force start a new session and clear current conversation context.
- `/resume [session]`: show recent sessions and choose one to resume (TUI opens a picker with task summary + updated time, Enter to confirm; or pass a specific session file).
- `/session`: print current session file path.
- `/help`: show command help.
- `/exit` or `/quit`: quit.

## Runtime Files and Directories

Default runtime data lives under `~/.pixy`:

```text
~/.pixy/
  pixy.toml
  agent/
    keybindings.json
    input_history.jsonl
    sessions/
```

- Default conf dir: `~/.pixy` (override with `--conf-dir <path>`)
- Default agent dir: `<conf_dir>/agent`
- Default sessions dir: `<conf_dir>/agent/sessions`
- Default log files (configurable via `[log].path`): `<conf_dir>/logs/pixy.log`, `<conf_dir>/logs/gateway.log`

## Configuration

### `pixy.toml`

Example [pixy.toml.sample](./pixy.toml.sample)

Notes:
- `pixy` and `pixy-gateway` support `--conf-dir <path>`; default is `~/.pixy`.
- `api_key` supports `$ENV_KEY`: it resolves from `[env]` in `pixy.toml` first, then falls back to process environment variables.
- `default_provider = "*"` routes by provider `weight`.
- `weight` is provider-level and must be `< 100`. `0` means that provider is never selected by weighted routing.
- Only providers with `kind = "chat"` participate in session routing and requests; `embedding` providers are ignored.
- `reasoning` / `reasoning_effort` / `context_window` / `max_tokens` are runtime model parameters.

### `[log]` section

`pixy.toml` supports a top-level `[log]` section for both `pixy` and `pixy-gateway`:

```toml
[log]
path = "~/.pixy/logs/"
level = "info"
rotate_size_mb = 100
stdout = false
```

- `path`: log directory (default `<conf_dir>/logs`, where `conf_dir` defaults to `~/.pixy`)
- `level`: default filter when `RUST_LOG` is unset
- `rotate_size_mb`: size-based rotation threshold per log file
- `stdout`: whether logs are also emitted to stdout

## Development

### Format

```bash
cargo fmt
```

### Test (single crate)

```bash
cargo nextest run -p pixy-coding-agent
```

### Test (workspace)

```bash
cargo nextest run
```

### Optional `just` shortcuts

```bash
just dev-cli
just cli
```
