#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="/home/ct_24210860031/812/SYR/code/lingbot-va-home"
CONDA_SH="/cpfs01/projects-HDD/cfff-4a2485d4a88d_HDD/ct_24210860031/miniconda3/etc/profile.d/conda.sh"
DATASET_PATH="/home/ct_24210860031/812/SYR/datasets/robotwin-clean-and-aug-lerobot"
MODEL_PATH="/home/ct_24210860031/models/lingbot-va-base"
JOB_NAME="robotwin-4gpu-noflex-cu118-auto"

cd "${REPO_ROOT}"
source "${CONDA_SH}"
conda activate lingbot-va-cu118

if pgrep -af "wan_va.train --config-name robotwin_train" >/dev/null; then
    echo "training_already_running=1"
    exit 0
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
log_dir="logs/cfff-jobs/${JOB_NAME}"
log_file="${log_dir}/${timestamp}.log"
mkdir -p "${log_dir}"

if [[ ! -f "${MODEL_PATH}/transformer/config.json" ]]; then
    echo "missing_model_config=${MODEL_PATH}/transformer/config.json"
    exit 1
fi

nohup env \
    LINGBOT_VA_FORCE_ATTN_MODE=torch \
    LINGBOT_VA_SAVE_INTERVAL=10000 \
    LINGBOT_VA_ENABLE_WANDB=0 \
    LINGBOT_VA_DATASET_PATH="${DATASET_PATH}" \
    LINGBOT_VA_TRAIN_MODEL_PATH="${MODEL_PATH}" \
    bash script/run_va_posttrain_profile.sh train_profiles/robotwin_author_4gpu.env \
    > "${log_file}" 2>&1 < /dev/null &

echo "pid=$!"
echo "log=${log_file}"
