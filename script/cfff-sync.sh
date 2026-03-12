#!/usr/bin/env bash

set -euo pipefail

if git_root="$(git rev-parse --show-toplevel 2>/dev/null)"; then
    local_root="$git_root"
else
    local_root="$(pwd)"
fi

repo_name="$(basename "$local_root")"
remote_host="${CFFF_HOST:-cfff}"
remote_base="${CFFF_REMOTE_BASE:-/cpfs01/projects-HDD/cfff-4a2485d4a88d_HDD/ct_24210860031/812/SYR/code}"
remote_root="${CFFF_REMOTE_ROOT:-$remote_base/$repo_name}"

ssh "$remote_host" "mkdir -p $remote_root"

rsync \
    -az \
    --info=progress2 \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude '__pycache__/' \
    --exclude '.pytest_cache/' \
    --exclude '.mypy_cache/' \
    --exclude '.ruff_cache/' \
    --exclude '.vscode/' \
    --exclude '.idea/' \
    --exclude 'logs/' \
    --exclude 'pids.txt' \
    --exclude 'train_out/' \
    --exclude 'wandb/' \
    --exclude 'visualization/' \
    --exclude 'results/' \
    --exclude 'results_*/' \
    --exclude 'eval_result/' \
    --exclude 'RoboTwin/' \
    "$@" \
    "$local_root/" \
    "$remote_host:$remote_root/"
