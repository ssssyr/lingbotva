#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

run_name="${RUN_NAME:-ur10_gamepad_drawer_block_xyzgripper_max512_4gpu_20260412}"
step="${STEP:-3500}"
poll_seconds="${POLL_SECONDS:-60}"
remote_save_root="${REMOTE_SAVE_ROOT:-/cpfs01/projects-HDD/cfff-4a2485d4a88d_HDD/ct_24210860031/812/SYR/code/lingbot-va/train_out/${run_name}}"
remote_checkpoint="${remote_save_root}/checkpoints/checkpoint_step_${step}/transformer"
local_save_root="${LOCAL_SAVE_ROOT:-${repo_root}/train_out/${run_name}}"
local_checkpoint_root="${local_save_root}/checkpoints/checkpoint_step_${step}"
local_transformer_dir="${local_checkpoint_root}/transformer"
local_infer_dir="${local_save_root}/infer_model_step_${step}_torch"
attn_mode="${ATTN_MODE:-torch}"

timestamp() {
    date '+%F %T'
}

echo "[$(timestamp)] monitor start run=${run_name} step=${step} poll=${poll_seconds}s"
echo "[$(timestamp)] remote_checkpoint=${remote_checkpoint}"
echo "[$(timestamp)] local_infer_dir=${local_infer_dir}"

while true; do
    if CFFF_REMOTE_ROOT='/home/ct_24210860031/.cache/cfff-code/lingbot-va' \
        CFFF_CONDA_ENV='' \
        "${script_dir}/cfff-run.sh" \
        "test -s \"${remote_checkpoint}/config.json\" && test -s \"${remote_checkpoint}/diffusion_pytorch_model.safetensors\""; then
        echo "[$(timestamp)] remote checkpoint_step_${step} looks complete, fetching"
        rm -rf "${local_checkpoint_root}" "${local_infer_dir}"
        STEP="${step}" ATTN_MODE="${attn_mode}" "${script_dir}/fetch_prepare_ur10_xyzgripper_latest_infer_model.sh"
        echo "[$(timestamp)] fetch+assemble done infer_model=${local_infer_dir}"
        exit 0
    fi

    echo "[$(timestamp)] remote checkpoint_step_${step} not ready yet"
    sleep "${poll_seconds}"
done
