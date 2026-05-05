#!/usr/bin/env bash

set -euo pipefail

REPO_ID="${REPO_ID:-Sylvest/libero_plus_lerobot}"
ENDPOINT="${ENDPOINT:-https://hf-mirror.com}"
TARGET="${TARGET:-/mnt/sda/syr/datasets/libero_plus_lerobot}"
REVISION="${REVISION:-main}"
JOBS="${JOBS:-4}"
RETRY="${RETRY:-20}"
STATE_DIR="${STATE_DIR:-$TARGET/.curl-mirror}"

worker() {
    local rel="$1"
    local out="$TARGET/$rel"
    local part="$out.part"
    local url="$ENDPOINT/datasets/$REPO_ID/resolve/$REVISION/$rel"

    if [ -s "$out" ]; then
        printf '[skip] %s\n' "$rel"
        return 0
    fi

    mkdir -p "$(dirname "$out")"

    curl \
        -fL \
        -C - \
        --retry "$RETRY" \
        --retry-delay 3 \
        --retry-all-errors \
        --connect-timeout 15 \
        --speed-time 60 \
        --speed-limit 1024 \
        -o "$part" \
        "$url"

    mv "$part" "$out"
    printf '[ok] %s\n' "$rel"
}

if [ "${1:-}" = "__worker__" ]; then
    worker "$2"
    exit 0
fi

mkdir -p "$TARGET" "$STATE_DIR"

META_JSON="$STATE_DIR/repo_metadata.json"
FILELIST="$STATE_DIR/filelist.txt"

printf '[start] %s\n' "$(date -Iseconds)"
printf '[repo] %s\n' "$REPO_ID"
printf '[endpoint] %s\n' "$ENDPOINT"
printf '[target] %s\n' "$TARGET"
printf '[jobs] %s\n' "$JOBS"
printf '[retry] %s\n' "$RETRY"

curl -fLsS "$ENDPOINT/api/datasets/$REPO_ID" -o "$META_JSON.tmp"
mv "$META_JSON.tmp" "$META_JSON"

jq -r '.siblings[].rfilename' "$META_JSON" > "$FILELIST"
printf '[file_count] %s\n' "$(wc -l < "$FILELIST" | tr -d ' ')"

xargs -d '\n' -P "$JOBS" -n 1 "$0" __worker__ < "$FILELIST"

printf '[done] %s\n' "$(date -Iseconds)"
