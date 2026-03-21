#!/usr/bin/bash

set -x

umask 007
 
NGPU=${NGPU:-"8"}
MASTER_PORT=${MASTER_PORT:-"29501"}
PORT=${PORT:-"1106"}
LOG_RANK=${LOG_RANK:-"0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"robotwin_train"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

set_optional_env() {
    local name="$1"
    local value="${!name:-}"
    if [[ -n "${value}" ]]; then
        export "${name}=${value}"
    else
        unset "${name}"
    fi
}

set_optional_env "WANDB_API_KEY"
set_optional_env "WANDB_BASE_URL"
set_optional_env "WANDB_TEAM_NAME"
export WANDB_PROJECT="${WANDB_PROJECT:-va_robotwin}"

## node setting
num_gpu=${NGPU}
master_port=${MASTER_PORT}
log_rank=${LOG_RANK}
torchft_lighthouse=${TORCHFT_LIGHTHOUSE}
config_name=${CONFIG_NAME}

## cmd setting
export TOKENIZERS_PARALLELISM=false
export LINGBOT_VA_DISABLE_FLEX_COMPILE="${LINGBOT_VA_DISABLE_FLEX_COMPILE:-0}"
export LINGBOT_VA_DIST_STRATEGY="${LINGBOT_VA_DIST_STRATEGY:-ddp}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
PYTORCH_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${torchft_lighthouse} \
python -m torch.distributed.run \
    --nproc_per_node=${num_gpu} \
    --local-ranks-filter=${log_rank} \
    --master_port ${master_port} \
    --tee 3 \
    -m wan_va.train --config-name ${config_name} $overrides
