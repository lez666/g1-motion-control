#!/usr/bin/env bash
set -euo pipefail

# G1 Motion Control - Environment Bootstrap Script

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "📦 [1/3] Updating submodules..."
git submodule update --init --recursive

echo "pip [2/3] Installing controller dependencies (pynput, loguru, termcolor)..."
# These are required for the custom keyboard controller
pip install pynput loguru termcolor

echo "✅ [3/3] Bootstrap complete!"
echo ""
echo "Next steps:"
echo "1. Setup your simulator in third_party/holosoma/scripts/"
echo "2. Check README.md for training and simulation commands."
