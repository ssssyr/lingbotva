#!/usr/bin/env bash

set -euo pipefail

REPO_ID="${REPO_ID:-Sylvest/libero_plus_lerobot}"
TARGET="${TARGET:-/mnt/sda/syr/datasets/libero_plus_lerobot}"
MAX_WORKERS="${MAX_WORKERS:-4}"

export HF_TOKEN="${HF_TOKEN:-}"
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:7890}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:7890}"
export http_proxy="${http_proxy:-$HTTP_PROXY}"
export https_proxy="${https_proxy:-$HTTPS_PROXY}"
unset HF_ENDPOINT || true
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

mkdir -p "$TARGET"

printf '[start] %s\n' "$(date -Iseconds)"
printf '[repo] %s\n' "$REPO_ID"
printf '[target] %s\n' "$TARGET"
printf '[http_proxy] %s\n' "$HTTP_PROXY"
printf '[max_workers] %s\n' "$MAX_WORKERS"

hf download \
    --repo-type dataset \
    --local-dir "$TARGET" \
    --max-workers "$MAX_WORKERS" \
    "$REPO_ID"

printf '[done] %s\n' "$(date -Iseconds)"
