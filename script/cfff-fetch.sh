#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  script/cfff-fetch.sh <remote-path> <local-dest>

Description:
  Copy a file or directory from the remote CFFF repo checkout to the local machine.

Arguments:
  remote-path  Remote source path. If relative, it is resolved under the remote repo root.
  local-dest   Local destination path. Parent directories are created automatically.

Examples:
  script/cfff-fetch.sh \
    train_out/robotwin_author_1gpu_wandb_s10k/checkpoints/checkpoint_step_50000/transformer/ \
    train_out/robotwin_author_1gpu_wandb_s10k/checkpoints/checkpoint_step_50000/transformer/

  script/cfff-fetch.sh \
    /cpfs01/projects-HDD/.../lingbot-va/train_out/foo/bar \
    /tmp/bar
EOF
}

if [[ $# -ne 2 ]]; then
    usage >&2
    exit 1
fi

if git_root="$(git rev-parse --show-toplevel 2>/dev/null)"; then
    local_root="$git_root"
else
    local_root="$(pwd)"
fi

repo_name="$(basename "$local_root")"
remote_host="${CFFF_HOST:-cfff}"
remote_base="${CFFF_REMOTE_BASE:-/cpfs01/projects-HDD/cfff-4a2485d4a88d_HDD/ct_24210860031/812/SYR/code}"
remote_root="${CFFF_REMOTE_ROOT:-$remote_base/$repo_name}"

remote_path="$1"
local_dest="$2"

if [[ "$remote_path" != /* ]]; then
    remote_path="${remote_root}/${remote_path}"
fi

mkdir -p "$(dirname "$local_dest")"

rsync -az --info=progress2 "${remote_host}:${remote_path}" "${local_dest}"
