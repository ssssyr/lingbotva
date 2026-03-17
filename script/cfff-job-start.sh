#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <job-name> \"<remote command>\"" >&2
    exit 1
fi

job_name="$1"
shift
job_command="$*"
remote_conda_sh="${CFFF_CONDA_SH-}"
remote_conda_env="${CFFF_CONDA_ENV-base}"

if ! [[ "$job_name" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "job-name must match [A-Za-z0-9._-]+" >&2
    exit 1
fi

./script/cfff-run.sh "
job_name=$(printf '%q' "$job_name")
job_command=$(printf '%q' "$job_command")
remote_conda_sh=$(printf '%q' "$remote_conda_sh")
remote_conda_env=$(printf '%q' "$remote_conda_env")
timestamp=\$(date +%Y%m%d-%H%M%S)
log_dir=\"logs/cfff-jobs/\$job_name\"
log_file=\"\$log_dir/\$timestamp.log\"
runner_file=\"\$log_dir/\$timestamp.runner.sh\"
session_name=\"\${job_name}-\${timestamp}\"
workdir=\$(pwd)

mkdir -p \"\$log_dir\"

if ! command -v tmux >/dev/null 2>&1; then
    echo 'tmux is not installed on the remote host' >&2
    exit 1
fi

printf -v job_name_shell '%q' \"\$job_name\"
printf -v job_command_shell '%q' \"\$job_command\"
printf -v workdir_shell '%q' \"\$workdir\"

{
    printf '%s\n' '#!/usr/bin/env bash'
    printf '%s\n' 'set -u'
    printf 'job_name=%q\n' \"\$job_name\"
    printf 'job_command=%q\n' \"\$job_command\"
    printf 'preferred_conda_sh=%q\n' \"\$remote_conda_sh\"
    printf 'conda_env=%q\n' \"\$remote_conda_env\"
    printf 'workdir=%q\n' \"\$workdir\"
    printf '%s\n' 'cd \"\$workdir\"'
    printf '%s\n' 'cfff_source_conda() {'
    printf '%s\n' '    if [ -n \"\$preferred_conda_sh\" ] && [ -f \"\$preferred_conda_sh\" ]; then'
    printf '%s\n' '        . \"\$preferred_conda_sh\"'
    printf '%s\n' '        return 0'
    printf '%s\n' '    fi'
    printf '%s\n' ''
    printf '%s\n' '    for candidate in \"\${HOME}/miniconda3/etc/profile.d/conda.sh\" \"\${HOME}/anaconda3/etc/profile.d/conda.sh\" \"/opt/conda/etc/profile.d/conda.sh\"; do'
    printf '%s\n' '        if [ -f \"\$candidate\" ]; then'
    printf '%s\n' '            . \"\$candidate\"'
    printf '%s\n' '            return 0'
    printf '%s\n' '        fi'
    printf '%s\n' '    done'
    printf '%s\n' ''
    printf '%s\n' '    return 1'
    printf '%s\n' '}'
    printf '%s\n' 'cfff_source_conda >/dev/null 2>&1 || true'
    printf '%s\n' 'if [ -n \"\$conda_env\" ] && command -v conda >/dev/null 2>&1; then'
    printf '%s\n' '    conda activate \"\$conda_env\"'
    printf '%s\n' 'fi'
    printf '%s\n' 'set -o pipefail'
    printf '%s\n' 'echo [cfff-job] started_at=\$(date -Iseconds) job=\$job_name'
    printf '%s\n' 'echo [cfff-job] workdir=\$(pwd)'
    printf '%s\n' 'if [ -n \"\$conda_env\" ]; then echo [cfff-job] conda_env=\$conda_env; fi'
    printf '%s\n' 'printf '\''[cfff-job] command=%s\n'\'' \"\$job_command\"'
    printf '%s\n' 'set +e'
    printf '%s\n' 'bash -c \"\$job_command\"'
    printf '%s\n' 'job_status=\$?'
    printf '%s\n' 'set -e'
    printf '%s\n' 'printf '\''[cfff-job] finished_at=%s exit_code=%s\n'\'' \"\$(date -Iseconds)\" \"\$job_status\"'
    printf '%s\n' 'exit \"\$job_status\"'
} > \"\$runner_file\"

chmod +x \"\$runner_file\"

printf -v runner_file_quoted '%q' \"\$runner_file\"
printf -v log_file_quoted '%q' \"\$log_file\"

tmux new-session -d -s \"\$session_name\" \"bash \$runner_file_quoted > \$log_file_quoted 2>&1\"

printf 'session=%s\nlog=%s\n' \"\$session_name\" \"\$log_file\"
"
