#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG_FILE="${SCRIPT_DIR}/../config/robotwin_client.yaml"

load_config() {
    local config_file="$1"
    eval "$(
        python - "$config_file" <<'PY'
import shlex
import sys
import yaml

config_file = sys.argv[1]
defaults = {
    "gpu_id": 0,
    "port": 29056,
}
with open(config_file, "r", encoding="utf-8") as f:
    loaded = yaml.safe_load(f) or {}
if not isinstance(loaded, dict):
    raise SystemExit(f"Config must be a mapping: {config_file}")
defaults.update(loaded)
for key, value in defaults.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
    )"
}

if [[ "${1:-}" == "--config" ]]; then
    if [[ -z "${2:-}" ]]; then
        echo "missing config path after --config" >&2
        exit 1
    fi
    load_config "$2"
else
    load_config "$DEFAULT_CONFIG_FILE"
fi

gpu_id=${gpu_id:-0}
START_PORT=${START_PORT:-${port:-29056}}

save_root='visualization/'
mkdir -p "$save_root"

echo "server config:"
echo "  gpu_id=${gpu_id}"
echo "  port=${START_PORT}"
echo "  save_root=${save_root}"

CUDA_VISIBLE_DEVICES=${gpu_id} python wan_va/wan_va_server.py \
    --config-name robotwin \
    --port "$START_PORT" \
    --save_root "$save_root"
