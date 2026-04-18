#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

remote_host="${CFFF_HOST:-cfff_2}"
remote_root="${CFFF_REMOTE_ROOT:-/home/ct_24210860031/812/SYR/code/lingbot-va-libero}"
remote_log_glob="${LIBERO_REMOTE_LOG_GLOB:-/home/ct_24210860031/812/SYR/logs/libero/train-base-eqbs16-*.log}"
job_name="${LIBERO_WATCH_JOB_NAME:-libero-base-eqbs16-4gpu}"
watch_log_dir="${repo_root}/logs/train_watch"
watch_log_file="${WATCH_LOG_FILE:-${watch_log_dir}/${job_name}_watch.log}"
watch_pid_file="${WATCH_PID_FILE:-${watch_log_dir}/${job_name}_watch.pid}"
watch_tmux_session="${WATCH_TMUX_SESSION:-${job_name}-watch}"
python_bin="${WATCH_PYTHON:-python}"

mkdir -p "${watch_log_dir}"

remote_cmd=$(cat <<EOF
set -euo pipefail
mkdir -p logs/cfff-jobs/${job_name}
latest_log=\$(ls -1t ${remote_log_glob} 2>/dev/null | head -n 1)
if [ -z "\${latest_log}" ]; then
    echo "No LIBERO log matched: ${remote_log_glob}" >&2
    exit 1
fi
link_dir=logs/cfff-jobs/${job_name}
link_path="\${link_dir}/\$(basename "\${latest_log}")"
find "\${link_dir}" -maxdepth 1 -type f -name '*.log' ! -samefile "\${latest_log}" -delete 2>/dev/null || true
rm -f "\${link_path}"
ln "\${latest_log}" "\${link_path}"
echo "linked=\${link_path}"
EOF
)

CFFF_HOST="${remote_host}" CFFF_REMOTE_ROOT="${remote_root}" "${script_dir}/cfff-run.sh" "${remote_cmd}"

watch_cmd=(
    env "CFFF_HOST=${remote_host}"
    "${python_bin}" "${script_dir}/watch_cfff_train_jobs.py"
    --poll-seconds "${POLL_SECONDS:-60}"
    --notify-cooldown "${NOTIFY_COOLDOWN:-1800}"
    --title "${WATCH_TITLE:-LIBERO Train Watch}"
    --remote-root "${remote_root}"
    --job "libero=${job_name}"
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
