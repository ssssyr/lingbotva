#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

log_dir="${UPLOAD_LOG_DIR:-${repo_root}/logs/data_transfer}"
timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="${UPLOAD_LOG_FILE:-${log_dir}/libero_plus_upload_cfff2_${timestamp}.log}"
pid_file="${UPLOAD_PID_FILE:-${log_dir}/libero_plus_upload_cfff2.pid}"
base_session="${UPLOAD_TMUX_SESSION:-libero-plus-upload-cfff2}"

mkdir -p "${log_dir}"

upload_cmd=(
    "${script_dir}/run_libero_plus_upload_cfff2.sh"
    "$@"
)

if [[ "${UPLOAD_FOREGROUND:-0}" == "1" ]]; then
    exec "${upload_cmd[@]}"
fi

session_name="${base_session}"
if command -v tmux >/dev/null 2>&1 && tmux has-session -t "${session_name}" 2>/dev/null; then
    session_name="${base_session}-${timestamp}"
fi

if command -v tmux >/dev/null 2>&1; then
    tmux new-session -d -s "${session_name}" "cd ${repo_root@Q} && ${upload_cmd[*]@Q} > ${log_file@Q} 2>&1"
    tmux_pid="$(tmux list-panes -t "${session_name}" -F '#{pane_pid}' | head -n 1)"
    echo "${tmux_pid}" > "${pid_file}"
    printf 'TMUX_SESSION=%s\nPID=%s\nLOG=%s\nPIDFILE=%s\n' "${session_name}" "${tmux_pid}" "${log_file}" "${pid_file}"
    exit 0
fi

nohup "${upload_cmd[@]}" > "${log_file}" 2>&1 < /dev/null &
upload_pid=$!
echo "${upload_pid}" > "${pid_file}"
printf 'PID=%s\nLOG=%s\nPIDFILE=%s\n' "${upload_pid}" "${log_file}" "${pid_file}"
