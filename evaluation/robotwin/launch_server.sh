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
echo "  video_guidance_scale=${video_guidance_scale:-5}"
echo "  action_guidance_scale=${action_guidance_scale:-1}"
echo "  online_scheduler_mode=${online_scheduler_mode:-fixed}"
echo "  fixed_video_steps=${fixed_video_steps:-25}"
echo "  enable_hazard_scheduler_runtime=${enable_hazard_scheduler_runtime:-false}"
echo "  enable_offload=${enable_offload:-false}"
if [[ -n "${hazard_checkpoint_path:-}" ]]; then
    echo "  hazard_checkpoint_path=${hazard_checkpoint_path}"
fi

hazard_enabled_normalized="$(echo "${enable_hazard_scheduler_runtime:-false}" | tr '[:upper:]' '[:lower:]')"
hazard_return_metadata_normalized="$(echo "${hazard_return_metadata:-true}" | tr '[:upper:]' '[:lower:]')"

CUDA_VISIBLE_DEVICES=${gpu_id} python wan_va/wan_va_server.py \
    --config-name robotwin \
    --port "$START_PORT" \
    --save_root "$save_root" \
    --guidance-scale "${video_guidance_scale:-5}" \
    --action-guidance-scale "${action_guidance_scale:-1}" \
    --online-scheduler-mode "${online_scheduler_mode:-fixed}" \
    --fixed-video-steps "${fixed_video_steps:-25}" \
    --enable-hazard-scheduler-runtime "$([[ "${hazard_enabled_normalized}" == "true" ]] && echo 1 || echo 0)" \
    --hazard-checkpoint "${hazard_checkpoint_path:-}" \
    --hazard-eta "${hazard_eta:-0.5}" \
    --hazard-k-min "${hazard_k_min:-3}" \
    --hazard-k-max "${hazard_k_max:-25}" \
    --hazard-feature-source "${hazard_feature_source:-cond}" \
    --hazard-return-metadata "$([[ "${hazard_return_metadata_normalized}" == "true" ]] && echo 1 || echo 0)" \
    --save-debug-artifacts "$([[ "$(echo "${save_debug_artifacts:-true}" | tr '[:upper:]' '[:lower:]')" == "true" ]] && echo 1 || echo 0)" \
    --enable-offload "$([[ "$(echo "${enable_offload:-false}" | tr '[:upper:]' '[:lower:]')" == "true" ]] && echo 1 || echo 0)"
