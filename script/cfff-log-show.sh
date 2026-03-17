#!/usr/bin/env bash

set -euo pipefail

job_name="${1-}"
lines="${2-200}"

if ! [[ "$lines" =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 [job-name] [lines]" >&2
    exit 1
fi

if [ -n "$job_name" ]; then
    ./script/cfff-run.sh "
log_file=\$(find logs/cfff-jobs/$(printf '%q' "$job_name") -maxdepth 1 -type f -name '*.log' 2>/dev/null | sort | tail -n 1)
if [ -z \"\$log_file\" ]; then
    echo \"No log found for job: $(printf '%q' "$job_name")\" >&2
    exit 1
fi
echo \"==> \$log_file <==\"
tail -n $(printf '%q' "$lines") \"\$log_file\"
"
    exit 0
fi

./script/cfff-run.sh "
log_file=\$(find logs/cfff-jobs -type f -name '*.log' 2>/dev/null | sort | tail -n 1)
if [ -z \"\$log_file\" ]; then
    echo 'No logs found under logs/cfff-jobs' >&2
    exit 1
fi
echo \"==> \$log_file <==\"
tail -n $(printf '%q' "$lines") \"\$log_file\"
"
