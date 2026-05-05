#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

remote_host="${CFFF_HOST:-cfff_2}"
remote_root="${CFFF_REMOTE_ROOT:-/home/ct_24210860031/812/SYR/code/lingbot-va-libero}"
job_name="${LIBERO_PLUS_EVAL_WATCH_JOB_NAME:-libero-plus-eval-libero10-4gpu}"
watch_log_dir="${repo_root}/logs/train_watch"
watch_log_file="${WATCH_LOG_FILE:-${watch_log_dir}/${job_name}_watch.log}"
watch_pid_file="${WATCH_PID_FILE:-${watch_log_dir}/${job_name}_watch.pid}"
watch_tmux_session="${WATCH_TMUX_SESSION:-${job_name}-watch}"
python_bin="${WATCH_PYTHON:-python}"

mkdir -p "${watch_log_dir}"

watch_cmd=(
    env "CFFF_HOST=${remote_host}"
    "${python_bin}" "${script_dir}/watch_libero_plus_eval_progress.py"
    --poll-seconds "${POLL_SECONDS:-60}"
    --notify-cooldown "${NOTIFY_COOLDOWN:-3600}"
    --title "${WATCH_TITLE:-LIBERO+ Eval Watch}"
    --remote-root "${remote_root}"
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
'         "${watch_tmux_session}" "${tmux_pid}" "${watch_log_file}" "${watch_pid_file}"
    exit 0
fi

nohup "${watch_cmd[@]}" > "${watch_log_file}" 2>&1 < /dev/null &
watch_pid=$!
echo "${watch_pid}" > "${watch_pid_file}"
printf 'PID=%s
LOG=%s
PIDFILE=%s
' "${watch_pid}" "${watch_log_file}" "${watch_pid_file}"
