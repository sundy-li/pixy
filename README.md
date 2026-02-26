# pixy

`pixy` is a Rust coding-agent runtime inspired by [`pi-mono`](https://github.com/badlogic/pi-mono).

## Quick Install (Linux / macOS / Windows)

### Option A (recommended): one-liner installer (Linux / macOS)

```bash
curl -fsSL https://raw.githubusercontent.com/sundy-li/pixy/main/scripts/install.sh | bash
```

> Security note: for production or restricted environments, review `scripts/install.sh` before running the one-liner.

### Option B (recommended on Windows PowerShell): one-liner installer

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "iwr https://raw.githubusercontent.com/sundy-li/pixy/main/scripts/install.ps1 -UseBasicParsing | iex"
```

> Windows one-liner currently installs `x86_64` release assets only.

### Option C: clone and run local installer

```bash
git clone https://github.com/sundy-li/pixy.git
cd pixy
./scripts/install.sh
```

```powershell
git clone https://github.com/sundy-li/pixy.git
cd pixy
powershell -NoProfile -ExecutionPolicy Bypass -File .\\scripts\\install.ps1
```

Installer defaults:
- installs latest release from `sundy-li/pixy`
- installs binary to `~/.local/bin/pixy`
- verifies release checksum via `SHA256SUMS`

Common overrides:

```bash
PIXY_VERSION=v0.1.0 ./scripts/install.sh
PIXY_INSTALL_DIR=/usr/local/bin ./scripts/install.sh
PIXY_REPO=sundy-li/pixy ./scripts/install.sh
```

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\\scripts\\install.ps1 -Version v0.1.0
powershell -NoProfile -ExecutionPolicy Bypass -File .\\scripts\\install.ps1 -InstallDir "$env:USERPROFILE\\bin"
powershell -NoProfile -ExecutionPolicy Bypass -File .\\scripts\\install.ps1 -Repo sundy-li/pixy
```

## Bootstrap + Onboard

`bootstrap.sh` can install pixy, initialize config, and optionally onboard provider settings in one flow.

```bash
./scripts/bootstrap.sh --onboard --provider openai --api-key "sk-..."
```

```bash
./scripts/bootstrap.sh --version v0.1.0 --conf-dir ~/.pixy --onboard --provider anthropic --api-key "sk-ant-..."
```

Manual onboarding only:

```bash
./scripts/onboard.sh --provider openai --api-key "sk-..." --force
```

## Quick Start (30 seconds)

1. Initialize pixy home and default config:

```bash
pixy config init
```

2. Edit `~/.pixy/pixy.toml` and set your API key(s).

3. Verify installation:

```bash
pixy --help
pixy doctor
```

4. Start interactive chat:

```bash
pixy
```

Run one prompt:

```bash
pixy --prompt "Summarize this repository structure."
```

Run REPL without TUI:

```bash
pixy --no-tui
```

## Minimal `pixy.toml`

Use this as the smallest chat setup:

```toml
[env]
OPENAI_API_KEY = "sk-..."

[llm]
default_provider = "openai"

[[llm.providers]]
name = "openai"
kind = "chat"
provider = "openai"
api = "openai-responses"
base_url = "https://api.openai.com/v1"
api_key = "$OPENAI_API_KEY"
model = "gpt-5.3-codex"
weight = 1
```

Full sample: [`pixy.toml.sample`](./pixy.toml.sample)

## Gateway Quick Setup (Telegram / Feishu)

Start gateway in foreground:

```bash
pixy gateway start
```

Daemon lifecycle:

```bash
pixy gateway start --daemon
pixy gateway restart
pixy gateway stop
```

Gateway runtime files:

```text
~/.pixy/gateway/
  gateway.pid
  gateway.state.json
```

`gateway.channels` are configured in `~/.pixy/pixy.toml`.
- Telegram uses polling (`getUpdates`)
- Feishu uses webhook route: `/webhook/feishu/{channel_name}`
- `/new` in chat resets routed session context

## Upgrade / Uninstall

Upgrade to latest:

```bash
pixy update
```

Upgrade to a fixed version:

```bash
pixy update --version v0.1.0
```

Use a custom release repository:

```bash
pixy update --repo owner/repo
```

Uninstall:

```bash
rm -f ~/.local/bin/pixy
```

```powershell
Remove-Item "$env:USERPROFILE\\.local\\bin\\pixy.exe" -Force
```

## Troubleshooting

- `pixy: command not found`
  - Ensure install dir is in PATH (default `~/.local/bin`).
- checksum error
  - Re-run installer and verify network/proxy settings.
- cannot connect provider
  - Check `api_key`, `base_url`, and network reachability.
- runtime/config issues
  - Run `pixy doctor` for a local environment report.

## Development

### Workspace layout

- `crates/pixy-main`: unified `pixy` command dispatcher (`pixy cli`, `pixy gateway`)
- `crates/pixy-coding-agent`: coding agent runtime and `pixy-cli` binary
- `crates/pixy-gateway`: gateway runtime
- `crates/pixy-ai`: provider abstraction and streaming protocol
- `crates/pixy-agent-core`: agent loop, tools, retries/fallback
- `crates/pixy-tui`: terminal UI

### Run from source

```bash
cargo run -p pixy-main --bin pixy
cargo run -p pixy-main --bin pixy -- --help
cargo run -p pixy-main --bin pixy -- gateway start
```

### Tests

```bash
cargo c
cargo nextest run -p pixy-coding-agent -p pixy-main -p pixy-gateway
cargo nextest run
```

### Package manager manifests

Generate Homebrew/Scoop manifests from a release tag:

```bash
./scripts/generate-package-manifests.sh --version v0.1.0
```

Generated files:
- `packaging/homebrew/pixy.rb`
- `packaging/scoop/pixy.json`
