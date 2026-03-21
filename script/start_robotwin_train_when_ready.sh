#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPECTED_PTH_COUNT=82414
CHECK_INTERVAL_SECONDS=300
REMOTE_DATASET="/home/ct_24210860031/812/SYR/datasets/robotwin-clean-and-aug-lerobot"
REMOTE_MODEL_CONFIG="/home/ct_24210860031/models/lingbot-va-base/transformer/config.json"
CONDA_SH="/cpfs01/projects-HDD/cfff-4a2485d4a88d_HDD/ct_24210860031/miniconda3/etc/profile.d/conda.sh"
REMOTE_START_SCRIPT="${REPO_ROOT}/script/start_robotwin_train_remote.sh"

cd "${REPO_ROOT}"

log() {
    printf '[auto-start] %s %s\n' "$(date -Iseconds)" "$*"
}

remote_count_command() {
    ./script/cfff-run.sh "dataset='${REMOTE_DATASET}'; count=\$(find \"\$dataset\" -type f -name '*.pth' | wc -l); if [ -f \"\$dataset/empty_emb.pt\" ]; then echo \"\$count ready\"; else echo \"\$count missing_empty_emb\"; fi"
}

remote_env_check() {
    CFFF_CONDA_SH="${CONDA_SH}" CFFF_CONDA_ENV='' ./script/cfff-run.sh \
        "source '${CONDA_SH}' && conda activate lingbot-va-cu118 && python -c 'import torch, wan_va.train; print(torch.__version__)'"
}

log "waiting_for_dataset remote=${REMOTE_DATASET} expected_pth=${EXPECTED_PTH_COUNT}"
while true; do
    dataset_status="$(remote_count_command | tail -n 1 | tr -d '\r')"
    count="${dataset_status%% *}"
    state="${dataset_status#* }"
    log "dataset_status count=${count} state=${state}"
    if [[ "${count}" == "${EXPECTED_PTH_COUNT}" && "${state}" == "ready" ]]; then
        break
    fi
    sleep "${CHECK_INTERVAL_SECONDS}"
done

log "dataset_ready"

while true; do
    if ./script/cfff-run.sh "test -f '${REMOTE_MODEL_CONFIG}'"; then
        log "model_ready path=${REMOTE_MODEL_CONFIG}"
        break
    fi
    log "model_not_ready retry_after=${CHECK_INTERVAL_SECONDS}s"
    sleep "${CHECK_INTERVAL_SECONDS}"
done

while true; do
    if remote_env_check >/tmp/robotwin_train_env_check.log 2>&1; then
        log "env_ready"
        break
    fi
    log "env_not_ready retry_after=${CHECK_INTERVAL_SECONDS}s"
    sleep "${CHECK_INTERVAL_SECONDS}"
done

log "starting_remote_training"
./script/cfff-run.sh "bash -s" < "${REMOTE_START_SCRIPT}"
log "remote_training_start_requested"
