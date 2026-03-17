#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$(command -v python3 || command -v python)"

if [[ -z "${PYTHON_BIN}" ]]; then
    echo "python3/python not found" >&2
    exit 1
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/build_daily_memory.py" \
    --timezone "Asia/Shanghai" \
    --memory-file "${HOME}/.codex/memories/important-memory.md"
