#!/usr/bin/env bash

set -euo pipefail

if git_root="$(git rev-parse --show-toplevel 2>/dev/null)"; then
    local_root="$git_root"
else
    local_root="$(pwd)"
fi

repo_name="$(basename "$local_root")"
remote_host="${CFFF_HOST:-cfff}"
remote_base="${CFFF_REMOTE_BASE:-/cpfs01/projects-HDD/cfff-4a2485d4a88d_HDD/ct_24210860031/812/SYR/code}"
remote_root="${CFFF_REMOTE_ROOT:-$remote_base/$repo_name}"
job_name="${1-}"

if [ -n "$job_name" ]; then
    remote_script="
cd $remote_root
log_file=\$(find logs/cfff-jobs/$(printf '%q' "$job_name") -maxdepth 1 -type f -name '*.log' 2>/dev/null | sort | tail -n 1)
if [ -z \"\$log_file\" ]; then
    echo \"No log found for job: $(printf '%q' "$job_name")\" >&2
    exit 1
fi
echo \"==> \$log_file <==\"
exec tail -n 50 -f \"\$log_file\"
"
else
    remote_script="
cd $remote_root
log_file=\$(find logs/cfff-jobs -type f -name '*.log' 2>/dev/null | sort | tail -n 1)
if [ -z \"\$log_file\" ]; then
    echo \"No logs found under logs/cfff-jobs\" >&2
    exit 1
fi
echo \"==> \$log_file <==\"
exec tail -n 50 -f \"\$log_file\"
"
fi

printf -v remote_script_quoted "%q" "$remote_script"

exec ssh -t "$remote_host" "bash -lc $remote_script_quoted"
