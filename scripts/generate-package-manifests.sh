#!/usr/bin/env bash
set -euo pipefail

REPO="${PIXY_REPO:-sundy-li/pixy}"
VERSION="${PIXY_VERSION:-}"
OUT_DIR="${PIXY_PACKAGING_DIR:-packaging}"

usage() {
  cat <<USAGE
Generate package manager manifests for a released pixy version.

Usage:
  generate-package-manifests.sh --version <tag> [options]

Options:
  -v, --version <tag>      Release version/tag (for example: v0.1.0)
  -r, --repo <owner/repo>  GitHub repo (default: sundy-li/pixy)
  -o, --out-dir <path>     Output directory (default: packaging)
  -h, --help               Show this help message
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -v|--version)
      VERSION="${2:-}"
      shift 2
      ;;
    -r|--repo)
      REPO="${2:-}"
      shift 2
      ;;
    -o|--out-dir)
      OUT_DIR="${2:-}"
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

if [[ -z "$VERSION" ]]; then
  echo "error: --version is required" >&2
  usage >&2
  exit 1
fi
if [[ "$VERSION" != v* ]]; then
  VERSION="v$VERSION"
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "error: curl is required" >&2
  exit 1
fi

BASE_URL="https://github.com/${REPO}/releases/download/${VERSION}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

SUMS_FILE="$TMP_DIR/SHA256SUMS"
curl -fsSL "$BASE_URL/SHA256SUMS" -o "$SUMS_FILE"

sum_for() {
  local asset="$1"
  awk -v name="$asset" '{
    file=$2
    sub(/^\.\//, "", file)
    if (file == name) {
      print $1
      exit
    }
  }' "$SUMS_FILE"
}

MAC_ARM_ASSET="pixy-aarch64-apple-darwin.tar.gz"
MAC_X86_ASSET="pixy-x86_64-apple-darwin.tar.gz"
LINUX_ARM_ASSET="pixy-aarch64-unknown-linux-gnu.tar.gz"
LINUX_X86_ASSET="pixy-x86_64-unknown-linux-gnu.tar.gz"
WIN_X86_ASSET="pixy-x86_64-pc-windows-msvc.zip"

MAC_ARM_SHA="$(sum_for "$MAC_ARM_ASSET")"
MAC_X86_SHA="$(sum_for "$MAC_X86_ASSET")"
LINUX_ARM_SHA="$(sum_for "$LINUX_ARM_ASSET")"
LINUX_X86_SHA="$(sum_for "$LINUX_X86_ASSET")"
WIN_X86_SHA="$(sum_for "$WIN_X86_ASSET")"

for value in "$MAC_ARM_SHA" "$MAC_X86_SHA" "$LINUX_ARM_SHA" "$LINUX_X86_SHA" "$WIN_X86_SHA"; do
  if [[ -z "$value" ]]; then
    echo "error: required asset checksum missing in SHA256SUMS" >&2
    exit 1
  fi
done

VERSION_NO_V="${VERSION#v}"
HOMEBREW_DIR="$OUT_DIR/homebrew"
SCOOP_DIR="$OUT_DIR/scoop"
mkdir -p "$HOMEBREW_DIR" "$SCOOP_DIR"

cat > "$HOMEBREW_DIR/pixy.rb" <<FORMULA
class Pixy < Formula
  desc "Rust coding-agent runtime"
  homepage "https://github.com/${REPO}"
  version "${VERSION_NO_V}"
  license "MIT"

  on_macos do
    on_arm do
      url "${BASE_URL}/${MAC_ARM_ASSET}"
      sha256 "${MAC_ARM_SHA}"
    end
    on_intel do
      url "${BASE_URL}/${MAC_X86_ASSET}"
      sha256 "${MAC_X86_SHA}"
    end
  end

  on_linux do
    on_arm do
      url "${BASE_URL}/${LINUX_ARM_ASSET}"
      sha256 "${LINUX_ARM_SHA}"
    end
    on_intel do
      url "${BASE_URL}/${LINUX_X86_ASSET}"
      sha256 "${LINUX_X86_SHA}"
    end
  end

  def install
    bin.install "pixy"
  end

  test do
    assert_match "pixy", shell_output("#{bin}/pixy --help")
  end
end
FORMULA

cat > "$SCOOP_DIR/pixy.json" <<SCOOP
{
  "version": "${VERSION_NO_V}",
  "description": "Rust coding-agent runtime",
  "homepage": "https://github.com/${REPO}",
  "license": "MIT",
  "architecture": {
    "64bit": {
      "url": "${BASE_URL}/${WIN_X86_ASSET}",
      "hash": "${WIN_X86_SHA}"
    }
  },
  "bin": "pixy.exe"
}
SCOOP

echo "Generated:"
echo "  $HOMEBREW_DIR/pixy.rb"
echo "  $SCOOP_DIR/pixy.json"
