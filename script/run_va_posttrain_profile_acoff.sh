#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LINGBOT_VA_ENABLE_AC=0
bash "${script_dir}/run_va_posttrain_profile.sh" "$@"
