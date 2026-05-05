#!/usr/bin/env bash

set -euo pipefail

REPO_ID="${REPO_ID:-Sylvest/libero_plus_lerobot}"
TARGET="${TARGET:-/mnt/sda/syr/datasets/libero_plus_lerobot}"
MAX_WORKERS="${MAX_WORKERS:-4}"
CHUNK_START="${CHUNK_START:-0}"
CHUNK_END="${CHUNK_END:-14}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

mkdir -p "$TARGET"

download() {
    local pattern="$1"
    echo "[download] pattern=$pattern"
    hf download \
        --repo-type dataset \
        --local-dir "$TARGET" \
        --max-workers "$MAX_WORKERS" \
        --include "$pattern" \
        "$REPO_ID"
}

printf '[start] %s\n' "$(date -Iseconds)"
printf '[repo] %s\n' "$REPO_ID"
printf '[target] %s\n' "$TARGET"
printf '[endpoint] %s\n' "$HF_ENDPOINT"
printf '[max_workers] %s\n' "$MAX_WORKERS"
printf '[chunk_range] %s-%s\n' "$CHUNK_START" "$CHUNK_END"

download ".gitattributes"
download "README.md"
download "meta/**"

for idx in $(seq "$CHUNK_START" "$CHUNK_END"); do
    chunk=$(printf "%03d" "$idx")
    download "data/chunk-$chunk/*"
done

for idx in $(seq "$CHUNK_START" "$CHUNK_END"); do
    chunk=$(printf "%03d" "$idx")
    download "videos/chunk-$chunk/**"
done

printf '[done] %s\n' "$(date -Iseconds)"
