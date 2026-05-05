#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

watch_log_dir="${repo_root}/logs/train_watch"
watch_log_file="${WATCH_LOG_FILE:-${watch_log_dir}/libero_plus_latents_watch.log}"
watch_pid_file="${WATCH_PID_FILE:-${watch_log_dir}/libero_plus_latents_watch.pid}"
watch_tmux_session="${WATCH_TMUX_SESSION:-libero-plus-latents-watch}"
python_bin="${WATCH_PYTHON:-python}"

mkdir -p "${watch_log_dir}"

export WATCH_TITLE="${WATCH_TITLE:-LIBERO+ VAE Watch}"

watch_cmd=(
    "${python_bin}" "${script_dir}/watch_local_libero_plus_latents.py"
    --dataset-root "${DATASET_ROOT:-/mnt/sda/syr/datasets/libero_plus_lerobot}"
    --log-file "${LATENT_LOG_FILE:-/home/syr/code/lingbot-va/logs/libero_plus_extract/libero_plus_lerobot_latents_tmux_overwrite_20260419_130226.log}"
    --process-pattern "${LATENT_PROCESS_PATTERN:-extract_libero_plus_lerobot_latents.py --dataset-root /mnt/sda/syr/datasets/libero_plus_lerobot}"
    --expected-files "${LATENT_EXPECTED_FILES:-28694}"
    --poll-seconds "${POLL_SECONDS:-60}"
    --notify-cooldown "${NOTIFY_COOLDOWN:-3600}"
    "$@"
)

if [[ "${WATCH_FOREGROUND:-0}" == "1" ]]; then
    exec "${watch_cmd[@]}"
fi

if command -v tmux >/dev/null 2>&1; then
    tmux kill-session -t "${watch_tmux_session}" 2>/dev/null || true
    tmux new-session -d -s "${watch_tmux_session}" "cd ${repo_root@Q} && ${watch_cmd[*]@Q} > ${watch_log_file@Q} 2>&1"
    tmux_pid="$(tmux list-panes -t "${watch_tmux_session}" -F '#{pane_pid}' | head -n 1)"
    echo "${tmux_pid}" > "${watch_pid_file}"
    printf 'TMUX_SESSION=%s
PID=%s
LOG=%s
PIDFILE=%s
' "${watch_tmux_session}" "${tmux_pid}" "${watch_log_file}" "${watch_pid_file}"
    exit 0
fi

nohup "${watch_cmd[@]}" > "${watch_log_file}" 2>&1 < /dev/null &
watch_pid=$!
echo "${watch_pid}" > "${watch_pid_file}"
printf 'PID=%s
LOG=%s
PIDFILE=%s
' "${watch_pid}" "${watch_log_file}" "${watch_pid_file}"
