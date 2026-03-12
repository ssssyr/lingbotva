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
remote_conda_sh="${CFFF_CONDA_SH-}"
remote_conda_env="${CFFF_CONDA_ENV-base}"

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 \"<remote command>\"" >&2
    exit 1
fi

remote_command="$*"
printf -v remote_conda_sh_quoted "%q" "$remote_conda_sh"
printf -v remote_conda_env_quoted "%q" "$remote_conda_env"

remote_script=$(cat <<EOF
preferred_conda_sh=$remote_conda_sh_quoted
conda_env=$remote_conda_env_quoted

mkdir -p $remote_root
cd $remote_root

cfff_source_conda() {
    if [ -n "\$preferred_conda_sh" ] && [ -f "\$preferred_conda_sh" ]; then
        . "\$preferred_conda_sh"
        return 0
    fi

    for candidate in "\${HOME}/miniconda3/etc/profile.d/conda.sh" "\${HOME}/anaconda3/etc/profile.d/conda.sh" "/opt/conda/etc/profile.d/conda.sh"; do
        if [ -f "\$candidate" ]; then
            . "\$candidate"
            return 0
        fi
    done

    return 1
}

cfff_source_conda >/dev/null 2>&1 || true

if [ -n "\$conda_env" ] && command -v conda >/dev/null 2>&1; then
    conda activate "\$conda_env"
fi

$remote_command
EOF
)

printf -v remote_script_quoted "%q" "$remote_script"

exec ssh "$remote_host" "bash -lc $remote_script_quoted"
