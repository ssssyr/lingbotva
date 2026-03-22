#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

profile_path="${1:-${TRAIN_PROFILE:-${repo_root}/train_profiles/robotwin_author_4gpu.yaml}}"
if [[ "${profile_path}" != /* ]]; then
    profile_path="${repo_root}/${profile_path}"
fi

if [[ ! -f "${profile_path}" ]]; then
    echo "error: train profile not found: ${profile_path}" >&2
    exit 1
fi

if [[ $# -gt 0 ]]; then
    shift
fi

load_yaml_profile() {
    local config_file="$1"
    local python_bin
    local rendered

    if command -v python3 >/dev/null 2>&1; then
        python_bin="python3"
    elif command -v python >/dev/null 2>&1; then
        python_bin="python"
    else
        echo "error: python3/python not found; cannot load YAML profile" >&2
        exit 1
    fi

    rendered="$(
        "${python_bin}" - "$config_file" <<'PY'
import os
import shlex
import sys

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "PyYAML is required to load YAML training profiles. "
        "Install it with `pip install PyYAML`."
    ) from exc

config_file = sys.argv[1]
with open(config_file, "r", encoding="utf-8") as f:
    loaded = yaml.safe_load(f) or {}
if not isinstance(loaded, dict):
    raise SystemExit(f"Config must be a mapping: {config_file}")

def get_section(name):
    value = loaded.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise SystemExit(f"Section `{name}` must be a mapping: {config_file}")
    return value

def env_override(name, value):
    return os.environ.get(name, value)

def optional_env_override(name, value=None):
    env_value = os.environ.get(name)
    if env_value is not None:
        return env_value
    if value in (None, ""):
        return None
    return value

launcher = get_section("launcher")
paths = get_section("paths")
logging = get_section("logging")
training = get_section("training")
model = get_section("model")
trainable = get_section("trainable")

assignments = {
    "CONFIG_NAME": env_override("CONFIG_NAME", launcher.get("config_name", "robotwin_train")),
    "NGPU": env_override("NGPU", launcher.get("ngpu", 8)),
    "MASTER_PORT": env_override("MASTER_PORT", launcher.get("master_port", 29501)),
    "LOG_RANK": env_override("LOG_RANK", launcher.get("log_rank", 0)),
    "TORCHFT_LIGHTHOUSE": env_override("TORCHFT_LIGHTHOUSE", launcher.get("torchft_lighthouse", "http://localhost:29510")),
    "LINGBOT_VA_FORCE_ATTN_MODE": optional_env_override("LINGBOT_VA_FORCE_ATTN_MODE", launcher.get("force_attn_mode")),
    "LINGBOT_VA_TRAIN_MODEL_PATH": env_override("LINGBOT_VA_TRAIN_MODEL_PATH", paths.get("train_model_path", "")),
    "LINGBOT_VA_DATASET_PATH": env_override("LINGBOT_VA_DATASET_PATH", paths.get("dataset_path", "")),
    "LINGBOT_VA_SAVE_ROOT": env_override("LINGBOT_VA_SAVE_ROOT", paths.get("save_root", "./train_out")),
    "LINGBOT_VA_REFERENCE_MODEL_PATH": env_override("LINGBOT_VA_REFERENCE_MODEL_PATH", paths.get("reference_model_path", "")),
    "LINGBOT_VA_ENABLE_WANDB": env_override("LINGBOT_VA_ENABLE_WANDB", int(bool(logging.get("enable_wandb", False)))),
    "WANDB_PROJECT": env_override("WANDB_PROJECT", logging.get("wandb_project", "va_robotwin")),
    "WANDB_TEAM_NAME": optional_env_override("WANDB_TEAM_NAME", logging.get("wandb_team_name")),
    "WANDB_BASE_URL": optional_env_override("WANDB_BASE_URL", logging.get("wandb_base_url")),
    "WANDB_API_KEY": optional_env_override("WANDB_API_KEY", logging.get("wandb_api_key")),
    "LINGBOT_VA_LOAD_WORKERS": env_override("LINGBOT_VA_LOAD_WORKERS", training.get("load_workers", 16)),
    "LINGBOT_VA_INIT_WORKERS": env_override("LINGBOT_VA_INIT_WORKERS", training.get("init_workers", 8)),
    "LINGBOT_VA_SAVE_INTERVAL": env_override("LINGBOT_VA_SAVE_INTERVAL", training.get("save_interval", 1000)),
    "LINGBOT_VA_GC_INTERVAL": env_override("LINGBOT_VA_GC_INTERVAL", training.get("gc_interval", 50)),
    "LINGBOT_VA_CFG_PROB": env_override("LINGBOT_VA_CFG_PROB", training.get("cfg_prob", 0.1)),
    "LINGBOT_VA_LR": env_override("LINGBOT_VA_LR", training.get("learning_rate", "1e-5")),
    "LINGBOT_VA_WEIGHT_DECAY": env_override("LINGBOT_VA_WEIGHT_DECAY", training.get("weight_decay", 0.1)),
    "LINGBOT_VA_WARMUP_STEPS": env_override("LINGBOT_VA_WARMUP_STEPS", training.get("warmup_steps", 10)),
    "LINGBOT_VA_BATCH_SIZE": env_override("LINGBOT_VA_BATCH_SIZE", training.get("batch_size", 1)),
    "LINGBOT_VA_GRAD_ACCUM_STEPS": env_override("LINGBOT_VA_GRAD_ACCUM_STEPS", training.get("gradient_accumulation_steps", 1)),
    "LINGBOT_VA_NUM_STEPS": env_override("LINGBOT_VA_NUM_STEPS", training.get("num_steps", 50000)),
    "LINGBOT_VA_ENABLE_ACTION_ADAPTER": env_override(
        "LINGBOT_VA_ENABLE_ACTION_ADAPTER",
        int(bool(model.get("enable_action_residual_adapter", False))),
    ),
    "LINGBOT_VA_ACTION_ADAPTER_DIM": env_override(
        "LINGBOT_VA_ACTION_ADAPTER_DIM",
        model.get("action_adapter_dim", 256),
    ),
    "LINGBOT_VA_ACTION_ADAPTER_DROPOUT": env_override(
        "LINGBOT_VA_ACTION_ADAPTER_DROPOUT",
        model.get("action_adapter_dropout", 0.0),
    ),
    "LINGBOT_VA_FREEZE_BACKBONE": env_override(
        "LINGBOT_VA_FREEZE_BACKBONE",
        int(bool(trainable.get("freeze_backbone", False))),
    ),
    "LINGBOT_VA_FREEZE_EMBEDDINGS": env_override(
        "LINGBOT_VA_FREEZE_EMBEDDINGS",
        int(bool(trainable.get("freeze_embeddings", False))),
    ),
    "LINGBOT_VA_TRAIN_ACTION_ADAPTER": env_override(
        "LINGBOT_VA_TRAIN_ACTION_ADAPTER",
        int(bool(trainable.get("train_action_adapter", False))),
    ),
    "LINGBOT_VA_TRAIN_ACTION_HEAD": env_override(
        "LINGBOT_VA_TRAIN_ACTION_HEAD",
        int(bool(trainable.get("train_action_head", False))),
    ),
    "LINGBOT_VA_TRAIN_VIDEO_HEADS": env_override(
        "LINGBOT_VA_TRAIN_VIDEO_HEADS",
        int(bool(trainable.get("train_video_heads", False))),
    ),
    "LINGBOT_VA_TRAIN_TIME_EMBEDDER": env_override(
        "LINGBOT_VA_TRAIN_TIME_EMBEDDER",
        int(bool(trainable.get("train_time_embedder", False))),
    ),
}

for key, value in assignments.items():
    if value is None:
        print(f"unset {key}")
        continue
    print(f"{key}={shlex.quote(str(value))}")
PY
    )"
    set -a
    eval "${rendered}"
    set +a
}

case "${profile_path}" in
    *.yaml|*.yml)
        load_yaml_profile "${profile_path}"
        ;;
    *)
        set -a
        # shellcheck disable=SC1090
        source "${profile_path}"
        set +a
        ;;
esac

cd "${repo_root}"

echo "[profile] ${profile_path}"
echo "[config] ${CONFIG_NAME:-robotwin_train}"
echo "[ngpu] ${NGPU:-8}"
echo "[train_model] ${LINGBOT_VA_TRAIN_MODEL_PATH:-UNSET}"
echo "[dataset] ${LINGBOT_VA_DATASET_PATH:-UNSET}"
echo "[save_root] ${LINGBOT_VA_SAVE_ROOT:-./train_out}"
echo "[steps] ${LINGBOT_VA_NUM_STEPS:-50000}"
echo "[lr] ${LINGBOT_VA_LR:-1e-5}"
echo "[init_workers] ${LINGBOT_VA_INIT_WORKERS:-8}"

bash "${script_dir}/run_va_posttrain.sh" "$@"
