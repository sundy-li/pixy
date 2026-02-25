#!/usr/bin/env bash
set -euo pipefail

REPO="${PIXY_REPO:-sundy-li/pixy}"
VERSION="${PIXY_VERSION:-latest}"
INSTALL_DIR="${PIXY_INSTALL_DIR:-$HOME/.local/bin}"

usage() {
  cat <<USAGE
pixy installer

Usage:
  install.sh [options]

Options:
  -v, --version <tag>      Install a specific version (for example: v0.1.0)
  -d, --install-dir <dir>  Install directory (default: ~/.local/bin)
  -r, --repo <owner/repo>  GitHub repository (default: sundy-li/pixy)
  -h, --help               Show this help message

Environment variables:
  PIXY_VERSION             Same as --version
  PIXY_INSTALL_DIR         Same as --install-dir
  PIXY_REPO                Same as --repo
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -v|--version)
      VERSION="${2:-}"
      shift 2
      ;;
    -d|--install-dir)
      INSTALL_DIR="${2:-}"
      shift 2
      ;;
    -r|--repo)
      REPO="${2:-}"
      shift 2
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

if [[ -z "$VERSION" || -z "$INSTALL_DIR" || -z "$REPO" ]]; then
  echo "error: version/install-dir/repo cannot be empty" >&2
  exit 1
fi

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "error: required command not found: $1" >&2
    exit 1
  fi
}

require_cmd curl
require_cmd tar

if ! command -v sha256sum >/dev/null 2>&1 && ! command -v shasum >/dev/null 2>&1; then
  echo "error: missing checksum tool (sha256sum or shasum)" >&2
  exit 1
fi

OS="$(uname -s)"
ARCH="$(uname -m)"

case "$ARCH" in
  x86_64|amd64) ARCH="x86_64" ;;
  aarch64|arm64) ARCH="aarch64" ;;
  *)
    echo "error: unsupported architecture: $ARCH" >&2
    exit 1
    ;;
esac

case "$OS" in
  Linux) TARGET_SUFFIX="unknown-linux-gnu" ;;
  Darwin) TARGET_SUFFIX="apple-darwin" ;;
  *)
    echo "error: unsupported operating system: $OS (use release assets manually)" >&2
    exit 1
    ;;
esac

TARGET="${ARCH}-${TARGET_SUFFIX}"
ASSET="pixy-${TARGET}.tar.gz"

if [[ "$VERSION" == "latest" ]]; then
  BASE_URL="https://github.com/${REPO}/releases/latest/download"
  DISPLAY_VERSION="latest"
else
  if [[ "$VERSION" != v* ]]; then
    VERSION="v${VERSION}"
  fi
  BASE_URL="https://github.com/${REPO}/releases/download/${VERSION}"
  DISPLAY_VERSION="$VERSION"
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

ASSET_URL="${BASE_URL}/${ASSET}"
SUMS_URL="${BASE_URL}/SHA256SUMS"

echo "==> Downloading ${ASSET} (${DISPLAY_VERSION})"
curl -fL --retry 3 --connect-timeout 10 -o "${TMP_DIR}/${ASSET}" "$ASSET_URL"

echo "==> Downloading SHA256SUMS"
curl -fL --retry 3 --connect-timeout 10 -o "${TMP_DIR}/SHA256SUMS" "$SUMS_URL"

EXPECTED_SUM="$(awk -v name="$ASSET" '{
  file=$2
  sub(/^\.\//, "", file)
  if (file == name) {
    print $1
  }
}' "${TMP_DIR}/SHA256SUMS")"
if [[ -z "$EXPECTED_SUM" ]]; then
  echo "error: checksum entry not found for ${ASSET}" >&2
  exit 1
fi

if command -v sha256sum >/dev/null 2>&1; then
  ACTUAL_SUM="$(sha256sum "${TMP_DIR}/${ASSET}" | awk '{print $1}')"
else
  ACTUAL_SUM="$(shasum -a 256 "${TMP_DIR}/${ASSET}" | awk '{print $1}')"
fi

if [[ "$EXPECTED_SUM" != "$ACTUAL_SUM" ]]; then
  echo "error: checksum mismatch for ${ASSET}" >&2
  echo "expected: $EXPECTED_SUM" >&2
  echo "actual:   $ACTUAL_SUM" >&2
  exit 1
fi

echo "==> Checksum verified"

tar -xzf "${TMP_DIR}/${ASSET}" -C "$TMP_DIR"

if [[ ! -f "${TMP_DIR}/pixy" ]]; then
  echo "error: archive does not contain pixy binary" >&2
  exit 1
fi

mkdir -p "$INSTALL_DIR"
install -m 0755 "${TMP_DIR}/pixy" "${INSTALL_DIR}/pixy"

echo "==> pixy installed to ${INSTALL_DIR}/pixy"
if [[ ":$PATH:" != *":${INSTALL_DIR}:"* ]]; then
  echo "==> Add to PATH:"
  echo "    export PATH=\"${INSTALL_DIR}:\$PATH\""
fi

echo "==> Verify: pixy --help"
