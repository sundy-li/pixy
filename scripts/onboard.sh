#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${PIXY_CONFIG_FILE:-$HOME/.pixy/pixy.toml}"
PROVIDER="${PIXY_PROVIDER:-openai}"
API_KEY="${PIXY_API_KEY:-}"
MODEL="${PIXY_MODEL:-}"
FORCE=0

usage() {
  cat <<USAGE
pixy onboarding helper

Usage:
  onboard.sh [options]

Options:
  --config <path>        Config path (default: ~/.pixy/pixy.toml)
  --provider <name>      Provider: openai | anthropic (default: openai)
  --api-key <key>        API key value (if omitted, prompt in interactive shell)
  --model <id>           Model id override
  --force                Overwrite existing config file
  -h, --help             Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="${2:-}"
      shift 2
      ;;
    --provider)
      PROVIDER="${2:-}"
      shift 2
      ;;
    --api-key)
      API_KEY="${2:-}"
      shift 2
      ;;
    --model)
      MODEL="${2:-}"
      shift 2
      ;;
    --force)
      FORCE=1
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$CONFIG_PATH" ]]; then
  echo "error: config path cannot be empty" >&2
  exit 1
fi

case "$PROVIDER" in
  openai|anthropic) ;;
  *)
    echo "error: unsupported provider '$PROVIDER' (expected: openai|anthropic)" >&2
    exit 1
    ;;
esac

if [[ -f "$CONFIG_PATH" && "$FORCE" -ne 1 ]]; then
  echo "config exists: $CONFIG_PATH"
  echo "use --force to overwrite"
  exit 0
fi

if [[ -z "$API_KEY" ]]; then
  if [[ -t 0 ]]; then
    read -r -s -p "Enter API key for $PROVIDER: " API_KEY
    echo
  else
    echo "error: --api-key is required in non-interactive mode" >&2
    exit 1
  fi
fi

if [[ -z "$API_KEY" ]]; then
  echo "error: API key cannot be empty" >&2
  exit 1
fi

if [[ -z "$MODEL" ]]; then
  if [[ "$PROVIDER" == "openai" ]]; then
    MODEL="gpt-5.3-codex"
  else
    MODEL="claude-3-5-sonnet-latest"
  fi
fi

toml_escape() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  printf '%s' "$value"
}

API_KEY_ESCAPED="$(toml_escape "$API_KEY")"
MODEL_ESCAPED="$(toml_escape "$MODEL")"
CONFIG_DIR="$(dirname "$CONFIG_PATH")"
mkdir -p "$CONFIG_DIR"

if [[ "$PROVIDER" == "openai" ]]; then
  cat > "$CONFIG_PATH" <<TOML
[env]
OPENAI_API_KEY = "$API_KEY_ESCAPED"

[llm]
default_provider = "openai"

[[llm.providers]]
name = "openai"
kind = "chat"
provider = "openai"
api = "openai-responses"
base_url = "https://api.openai.com/v1"
api_key = "\$OPENAI_API_KEY"
model = "$MODEL_ESCAPED"
weight = 1

[log]
path = "~/.pixy/logs/"
level = "info"
rotate_size_mb = 100
stdout = false
TOML
else
  cat > "$CONFIG_PATH" <<TOML
[env]
ANTHROPIC_API_KEY = "$API_KEY_ESCAPED"

[llm]
default_provider = "anthropic"

[[llm.providers]]
name = "anthropic"
kind = "chat"
provider = "anthropic"
api = "anthropic-messages"
base_url = "https://api.anthropic.com/v1"
api_key = "\$ANTHROPIC_API_KEY"
model = "$MODEL_ESCAPED"
weight = 1

[log]
path = "~/.pixy/logs/"
level = "info"
rotate_size_mb = 100
stdout = false
TOML
fi

chmod 600 "$CONFIG_PATH" || true

echo "onboard complete: $CONFIG_PATH"
echo "next: pixy --help"
echo "next: pixy doctor"
