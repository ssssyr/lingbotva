#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

job_name="${JOB_4GPU:-ur10-gamepad-xyzgripper-max512-4gpu-r1}"
watch_log_dir="${repo_root}/logs/train_watch"
watch_log_file="${WATCH_LOG_FILE:-${watch_log_dir}/${job_name}_watch.log}"
watch_pid_file="${WATCH_PID_FILE:-${watch_log_dir}/${job_name}_watch.pid}"
watch_tmux_session="${WATCH_TMUX_SESSION:-${job_name}-watch}"
python_bin="${WATCH_PYTHON:-python}"

mkdir -p "${watch_log_dir}"

watch_cmd=(
    "${python_bin}" "${script_dir}/watch_cfff_train_jobs.py"
    --poll-seconds "${POLL_SECONDS:-60}"
    --notify-cooldown "${NOTIFY_COOLDOWN:-3600}"
    --remote-root "${CFFF_REMOTE_ROOT:-/home/ct_24210860031/.cache/cfff-code/lingbot-va}"
    --job "4gpu=${job_name}"
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
    printf 'TMUX_SESSION=%s\nPID=%s\nLOG=%s\nPIDFILE=%s\n' "${watch_tmux_session}" "${tmux_pid}" "${watch_log_file}" "${watch_pid_file}"
    exit 0
fi

nohup "${watch_cmd[@]}" > "${watch_log_file}" 2>&1 < /dev/null &
watch_pid=$!
echo "${watch_pid}" > "${watch_pid_file}"
printf 'PID=%s\nLOG=%s\nPIDFILE=%s\n' "${watch_pid}" "${watch_log_file}" "${watch_pid_file}"
