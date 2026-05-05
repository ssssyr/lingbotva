#!/usr/bin/env bash

set -uo pipefail

LOCAL_DATASET_ROOT="${LOCAL_DATASET_ROOT:-/mnt/sda/syr/datasets/libero_plus_lerobot}"
REMOTE_HOST="${REMOTE_HOST:-cfff_2}"
REMOTE_DATASET_ROOT="${REMOTE_DATASET_ROOT:-/home/ct_24210860031/812/SYR/data/libero_plus_lerobot}"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-15}"
CONNECT_TIMEOUT_SECONDS="${CONNECT_TIMEOUT_SECONDS:-20}"
IO_TIMEOUT_SECONDS="${IO_TIMEOUT_SECONDS:-600}"
SSH_SERVER_ALIVE_INTERVAL="${SSH_SERVER_ALIVE_INTERVAL:-30}"
SSH_SERVER_ALIVE_COUNT_MAX="${SSH_SERVER_ALIVE_COUNT_MAX:-12}"
RSYNC_SSH_BIN="${RSYNC_SSH_BIN:-ssh}"

ssh_rsh=(
    "${RSYNC_SSH_BIN}"
    -T
    -o "ConnectTimeout=${CONNECT_TIMEOUT_SECONDS}"
    -o "ServerAliveInterval=${SSH_SERVER_ALIVE_INTERVAL}"
    -o "ServerAliveCountMax=${SSH_SERVER_ALIVE_COUNT_MAX}"
    -o "TCPKeepAlive=yes"
    -o "IPQoS=throughput"
)

shutdown_requested=0
on_shutdown() {
    shutdown_requested=1
    printf '[stop] signal received at %s\n' "$(date -Iseconds)"
}
trap on_shutdown INT TERM

ensure_remote_root() {
    "${ssh_rsh[@]}" "${REMOTE_HOST}" "mkdir -p ${REMOTE_DATASET_ROOT@Q}"
}

rsync_once() {
    rsync \
        -az \
        --info=progress2 \
        --human-readable \
        --partial \
        --append-verify \
        --timeout="${IO_TIMEOUT_SECONDS}" \
        -e "$(printf '%q ' "${ssh_rsh[@]}")" \
        --exclude '.cache/' \
        --exclude '.curl-mirror/' \
        --exclude '.hfd/' \
        "${LOCAL_DATASET_ROOT}/" \
        "${REMOTE_HOST}:${REMOTE_DATASET_ROOT}/"
}

printf '[start] %s\n' "$(date -Iseconds)"
printf '[local] %s\n' "${LOCAL_DATASET_ROOT}"
printf '[remote_host] %s\n' "${REMOTE_HOST}"
printf '[remote] %s\n' "${REMOTE_DATASET_ROOT}"
printf '[retry_sleep] %ss\n' "${RETRY_SLEEP_SECONDS}"
printf '[connect_timeout] %ss\n' "${CONNECT_TIMEOUT_SECONDS}"
printf '[io_timeout] %ss\n' "${IO_TIMEOUT_SECONDS}"
printf '[ssh_server_alive] interval=%ss count_max=%s\n' "${SSH_SERVER_ALIVE_INTERVAL}" "${SSH_SERVER_ALIVE_COUNT_MAX}"

ensure_remote_root

attempt=1
while true; do
    if [[ "${shutdown_requested}" -ne 0 ]]; then
        printf '[stopped] %s\n' "$(date -Iseconds)"
        exit 130
    fi

    started_at="$(date +%s)"
    printf '[attempt] %s at %s\n' "${attempt}" "$(date -Iseconds)"
    rsync_once
    status=$?
    ended_at="$(date +%s)"
    duration=$((ended_at - started_at))
    printf '[attempt_end] %s status=%s duration=%ss at %s\n' "${attempt}" "${status}" "${duration}" "$(date -Iseconds)"

    if [[ "${status}" -eq 0 ]]; then
        printf '[done] %s\n' "$(date -Iseconds)"
        exit 0
    fi

    printf '[retry] status=%s sleep=%ss at %s\n' "${status}" "${RETRY_SLEEP_SECONDS}" "$(date -Iseconds)"
    sleep "${RETRY_SLEEP_SECONDS}"
    attempt=$((attempt + 1))
done
