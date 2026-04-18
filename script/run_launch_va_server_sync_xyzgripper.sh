#!/usr/bin/env bash

set -euo pipefail
set -x

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

umask 007

NGPU="${NGPU:-1}"
MASTER_PORT="${MASTER_PORT:-29501}"
PORT="${PORT:-29536}"
LOG_RANK="${LOG_RANK:-0}"
TORCHFT_LIGHTHOUSE="${TORCHFT_LIGHTHOUSE:-http://localhost:29510}"
CONFIG_NAME="${CONFIG_NAME:-ur10_follower_safe_xyzgripper}"
GPU_ID="${GPU_ID:-0}"

default_model_path="${repo_root}/train_out/ur10_gamepad_drawer_block_xyzgripper_max512_4gpu_20260412/infer_model_step_1500_torch"
export LINGBOT_VA_MODEL_PATH="${LINGBOT_VA_MODEL_PATH:-${default_model_path}}"
export LINGBOT_VA_SAVE_ROOT="${LINGBOT_VA_SAVE_ROOT:-${repo_root}/visualization/ur10_real_step1500_xyzgripper}"

mkdir -p "${LINGBOT_VA_SAVE_ROOT}"

overrides=""
if [[ $# -ne 0 ]]; then
    overrides="$*"
fi

echo "[repo_root] ${repo_root}"
echo "[model_path] ${LINGBOT_VA_MODEL_PATH}"
echo "[save_root] ${LINGBOT_VA_SAVE_ROOT}"
echo "[port] ${PORT}"
echo "[ngpu] ${NGPU}"
echo "[gpu_id] ${GPU_ID}"
echo "[attn_mode] $(python - <<'PY'
import json
import os
from pathlib import Path

config_path = Path(os.environ["LINGBOT_VA_MODEL_PATH"]) / "transformer" / "config.json"
if config_path.is_file():
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    print(cfg.get("attn_mode", "unknown"))
else:
    print("missing")
PY
)"
echo "[camera_keys] ${LINGBOT_VA_UR10_CAMERA_KEYS:-observation.images.third,observation.images.wrist}"

export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}" \
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
TORCHFT_LIGHTHOUSE="${TORCHFT_LIGHTHOUSE}" \
python -m torch.distributed.run \
    --nproc_per_node="${NGPU}" \
    --local-ranks-filter="${LOG_RANK}" \
    --master_port "${MASTER_PORT}" \
    --tee 3 \
    -m wan_va.wan_va_server_ur10_follower_safe_xyzgripper \
    --config-name "${CONFIG_NAME}" \
    --port "${PORT}" \
    --save_root "${LINGBOT_VA_SAVE_ROOT}" \
    ${overrides}
