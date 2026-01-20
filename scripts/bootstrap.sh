#!/usr/bin/env bash
set -euo pipefail

# Bootstrap entry for g1-motion-control
# - pulls submodules
# - delegates env setup to holosoma scripts

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

git submodule update --init --recursive

echo "[OK] submodules synced"
echo "Next: see third_party/holosoma/scripts for simulator-specific setup."
