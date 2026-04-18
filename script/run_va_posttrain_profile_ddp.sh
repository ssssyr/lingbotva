#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LINGBOT_VA_DISABLE_FSDP=1
bash "${script_dir}/run_va_posttrain_profile.sh" "$@"
