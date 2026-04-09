#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

profile_path="${1:-${TRAIN_PROFILE:-${repo_root}/train_profiles/robotwin_hazard_local_8gpu.yaml}}"
if [[ "${profile_path}" != /* ]]; then
    profile_path="${repo_root}/${profile_path}"
fi

if [[ ! -f "${profile_path}" ]]; then
    echo "error: profile not found: ${profile_path}" >&2
    exit 1
fi

eval "$(
python3 - "$profile_path" <<'PY'
import os
import shlex
import sys
import yaml

profile_path = sys.argv[1]
with open(profile_path, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle) or {}

launcher = cfg.get("launcher", {}) or {}
logging_cfg = cfg.get("logging", {}) or {}

values = {
    "NGPU": launcher.get("ngpu", 8),
    "MASTER_PORT": launcher.get("master_port", 29528),
    "RUN_NAME_PREFIX": logging_cfg.get("run_name_prefix", os.path.splitext(os.path.basename(profile_path))[0]),
}

for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"

conda_sh="${CONDA_SH:-${repo_root}/.local/miniconda3/etc/profile.d/conda.sh}"
conda_env_path="${CONDA_ENV_PATH:-${repo_root}/.local/miniconda3/envs/lingbot-va}"
background="${HAZARD_BACKGROUND:-1}"

if [[ ! -f "${conda_sh}" ]]; then
    echo "error: conda.sh not found: ${conda_sh}" >&2
    exit 1
fi

run_name="${RUN_NAME:-${RUN_NAME_PREFIX}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${repo_root}/logs/local-jobs"
launcher_log="${repo_root}/logs/local-jobs/${run_name}.launcher.log"
launcher_pid="${repo_root}/logs/local-jobs/${run_name}.launcher.pid"

launch_cmd=$(
    cat <<EOF
source $(printf '%q' "${conda_sh}")
conda activate $(printf '%q' "${conda_env_path}")
cd $(printf '%q' "${repo_root}")
export PYTHONUNBUFFERED=1
export LINGBOT_VA_FORCE_ATTN_MODE=torch
export LINGBOT_HAZARD_RUN_NAME=$(printf '%q' "${run_name}")
exec python -m torch.distributed.run --nproc_per_node=$(printf '%q' "${NGPU}") --master_port=$(printf '%q' "${MASTER_PORT}") -m wan_va.train_hazard --config $(printf '%q' "${profile_path}")
EOF
)

if [[ "${background}" == "1" ]]; then
    nohup env \
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
        NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" \
        NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}" \
        NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}" \
        NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}" \
        PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
        /bin/bash -lc "${launch_cmd}" > "${launcher_log}" 2>&1 < /dev/null &
    pid=$!
    echo "${pid}" > "${launcher_pid}"
    printf 'RUN_NAME=%s\nPID=%s\nLOG=%s\nPIDFILE=%s\n' "${run_name}" "${pid}" "${launcher_log}" "${launcher_pid}"
else
    exec env \
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
        NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" \
        NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}" \
        NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}" \
        NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}" \
        PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
        /bin/bash -lc "${launch_cmd}"
fi
