#!/usr/bin/env bash

set -euo pipefail

run_name="${1:?run_name required}"
profile_path="${2:?profile_path required}"

source /home/syr/anaconda3/etc/profile.d/conda.sh
conda activate lingbot-va

cd /home/syr/code/lingbot-va
export PYTHONUNBUFFERED=1
export LINGBOT_VA_FORCE_ATTN_MODE=torch
export LINGBOT_HAZARD_RUN_NAME="${run_name}"
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_CUMEM_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[launcher] run_name=${run_name}"
echo "[launcher] profile=${profile_path}"
echo "[launcher] python=$(which python)"
echo "[launcher] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"

exec python -m torch.distributed.run \
  --nproc_per_node=2 \
  --master_port=29535 \
  -m wan_va.train_hazard \
  --config "${profile_path}"
