#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION="${PIXY_VERSION:-latest}"
REPO="${PIXY_REPO:-sundy-li/pixy}"
INSTALL_DIR="${PIXY_INSTALL_DIR:-$HOME/.local/bin}"
CONF_DIR="${PIXY_CONF_DIR:-$HOME/.pixy}"
DO_ONBOARD=0
PROVIDER="${PIXY_PROVIDER:-openai}"
API_KEY="${PIXY_API_KEY:-}"
MODEL="${PIXY_MODEL:-}"
FORCE_ONBOARD=0
SKIP_INSTALL=0

usage() {
  cat <<USAGE
pixy bootstrap

Usage:
  bootstrap.sh [options]

Options:
  --version <tag>          Install version (default: latest)
  --repo <owner/repo>      Release repo (default: sundy-li/pixy)
  --install-dir <path>     Install directory (default: ~/.local/bin)
  --conf-dir <path>        Config directory (default: ~/.pixy)
  --skip-install           Do not run installer
  --onboard                Run onboarding after install/init
  --provider <name>        Onboard provider: openai|anthropic
  --api-key <value>        Onboard API key
  --model <id>             Onboard model override
  --force-onboard          Overwrite existing config in onboard
  -h, --help               Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)
      VERSION="${2:-}"
      shift 2
      ;;
    --repo)
      REPO="${2:-}"
      shift 2
      ;;
    --install-dir)
      INSTALL_DIR="${2:-}"
      shift 2
      ;;
    --conf-dir)
      CONF_DIR="${2:-}"
      shift 2
      ;;
    --skip-install)
      SKIP_INSTALL=1
      shift 1
      ;;
    --onboard)
      DO_ONBOARD=1
      shift 1
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
    --force-onboard)
      FORCE_ONBOARD=1
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

if [[ "$SKIP_INSTALL" -ne 1 ]]; then
  if command -v pixy >/dev/null 2>&1; then
    echo "pixy already installed: $(command -v pixy)"
  else
    "$SCRIPT_DIR/install.sh" --version "$VERSION" --repo "$REPO" --install-dir "$INSTALL_DIR"
  fi
fi

mkdir -p "$CONF_DIR"
CONFIG_PATH="$CONF_DIR/pixy.toml"
if [[ ! -f "$CONFIG_PATH" ]]; then
  if [[ -f "$SCRIPT_DIR/../pixy.toml.sample" ]]; then
    cp "$SCRIPT_DIR/../pixy.toml.sample" "$CONFIG_PATH"
    echo "initialized config: $CONFIG_PATH"
  fi
fi

if [[ "$DO_ONBOARD" -eq 1 ]]; then
  ONBOARD_ARGS=(--config "$CONFIG_PATH" --provider "$PROVIDER")
  if [[ -n "$API_KEY" ]]; then
    ONBOARD_ARGS+=(--api-key "$API_KEY")
  fi
  if [[ -n "$MODEL" ]]; then
    ONBOARD_ARGS+=(--model "$MODEL")
  fi
  if [[ "$FORCE_ONBOARD" -eq 1 ]]; then
    ONBOARD_ARGS+=(--force)
  fi
  "$SCRIPT_DIR/onboard.sh" "${ONBOARD_ARGS[@]}"
fi

if command -v pixy >/dev/null 2>&1; then
  pixy --help >/dev/null
elif [[ -x "$INSTALL_DIR/pixy" ]]; then
  "$INSTALL_DIR/pixy" --help >/dev/null
fi

echo "bootstrap complete"
echo "next: pixy doctor"
echo "next: pixy"
