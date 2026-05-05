#!/usr/bin/env bash

set -euo pipefail

REPO_ID="${REPO_ID:-Sylvest/libero_plus_lerobot}"
TARGET="${TARGET:-/mnt/sda/syr/datasets/libero_plus_lerobot}"
THREADS="${THREADS:-1}"
JOBS="${JOBS:-2}"

export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:7890}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:7890}"
export http_proxy="${http_proxy:-$HTTP_PROXY}"
export https_proxy="${https_proxy:-$HTTPS_PROXY}"
unset HF_ENDPOINT || true

mkdir -p "$TARGET"

printf '[start] %s\n' "$(date -Iseconds)"
printf '[repo] %s\n' "$REPO_ID"
printf '[target] %s\n' "$TARGET"
printf '[http_proxy] %s\n' "$HTTP_PROXY"
printf '[threads] %s\n' "$THREADS"
printf '[jobs] %s\n' "$JOBS"

set +e
/mnt/sda/syr/tools/hfd.sh "$REPO_ID" \
    --dataset \
    --local-dir "$TARGET" \
    --tool aria2c \
    -x "$THREADS" \
    -j "$JOBS"
status=$?
set -e

printf '[exit_code] %s\n' "$status"
printf '[done] %s\n' "$(date -Iseconds)"
exit "$status"
