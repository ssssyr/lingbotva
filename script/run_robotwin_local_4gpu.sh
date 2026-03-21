#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
profile_path="${repo_root}/train_profiles/robotwin_local_4gpu.yaml"
conda_sh="${repo_root}/.local/miniconda3/etc/profile.d/conda.sh"

if [[ ! -f "${conda_sh}" ]]; then
    echo "error: conda activation script not found: ${conda_sh}" >&2
    exit 1
fi

# shellcheck disable=SC1090
. "${conda_sh}"
conda activate lingbot-va

python "${script_dir}/check_robotwin_train_ready.py"

exec bash "${script_dir}/run_va_posttrain_profile.sh" "${profile_path}" "$@"
