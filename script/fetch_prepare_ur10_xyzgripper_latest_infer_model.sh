#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

run_name="${RUN_NAME:-ur10_gamepad_drawer_block_xyzgripper_max512_4gpu_20260412}"
remote_save_root="${REMOTE_SAVE_ROOT:-/cpfs01/projects-HDD/cfff-4a2485d4a88d_HDD/ct_24210860031/812/SYR/code/lingbot-va/train_out/${run_name}}"
local_save_root="${LOCAL_SAVE_ROOT:-${repo_root}/train_out/${run_name}}"
base_model="${BASE_MODEL:-/home/syr/code/models/lingbot-va-posttrain-robotwin}"
attn_mode="${ATTN_MODE:-torch}"
step_arg="${STEP:-latest}"

if [[ "${step_arg}" == "latest" ]]; then
    latest_checkpoint="$(
        CFFF_REMOTE_ROOT='/home/ct_24210860031/.cache/cfff-code/lingbot-va' \
        CFFF_CONDA_ENV='' \
        "${script_dir}/cfff-run.sh" \
        "ls -dt \"${remote_save_root}/checkpoints\"/checkpoint_step_* 2>/dev/null | head -n 1"
    )"
    latest_checkpoint="$(echo "${latest_checkpoint}" | tr -d '\r' | tail -n 1)"
    if [[ -z "${latest_checkpoint}" ]]; then
        echo "failed to resolve latest checkpoint under ${remote_save_root}/checkpoints" >&2
        exit 1
    fi
    checkpoint_step="$(basename "${latest_checkpoint}")"
else
    checkpoint_step="checkpoint_step_${step_arg}"
    latest_checkpoint="${remote_save_root}/checkpoints/${checkpoint_step}"
fi

transformer_remote="${latest_checkpoint}/transformer/"
transformer_local="${local_save_root}/checkpoints/${checkpoint_step}/transformer/"
infer_model_local="${local_save_root}/infer_model_step_${checkpoint_step#checkpoint_step_}_${attn_mode}"

echo "[run_name] ${run_name}"
echo "[remote_checkpoint] ${latest_checkpoint}"
echo "[local_transformer] ${transformer_local}"
echo "[infer_model] ${infer_model_local}"
echo "[base_model] ${base_model}"
echo "[attn_mode] ${attn_mode}"

mkdir -p "${local_save_root}/checkpoints/${checkpoint_step}"

"${script_dir}/cfff-fetch.sh" "${transformer_remote}" "${transformer_local}"

"${script_dir}/assemble_robotwin_infer_model.sh" \
    --base-model "${base_model}" \
    --transformer-dir "${transformer_local%/}" \
    --output-dir "${infer_model_local}" \
    --attn-mode "${attn_mode}"

echo
echo "Prepared UR10 inference model:"
echo "  ${infer_model_local}"
echo
echo "Next:"
echo "  export LINGBOT_VA_MODEL_PATH=${infer_model_local}"
echo "  bash script/run_launch_va_server_sync_xyzgripper.sh"
