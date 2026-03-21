#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  script/assemble_robotwin_infer_model.sh \
    --base-model <complete-model-dir> \
    --transformer-dir <checkpoint-transformer-dir> \
    --output-dir <inference-model-dir> \
    [--attn-mode torch|flashattn]

Description:
  Build an inference-ready LingBot-VA model directory by copying a complete base model
  (vae/tokenizer/text_encoder + metadata) and then replacing its transformer/ folder
  with a trained checkpoint transformer.

Examples:
  script/assemble_robotwin_infer_model.sh \
    --base-model /mnt/sda/syr/models/lingbot-va-posttrain-robotwin \
    --transformer-dir train_out/robotwin_author_1gpu_wandb_s10k/checkpoints/checkpoint_step_50000/transformer \
    --output-dir train_out/robotwin_author_1gpu_wandb_s10k/infer_model_step_50000 \
    --attn-mode torch
EOF
}

base_model=""
transformer_dir=""
output_dir=""
attn_mode="torch"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-model)
            base_model="$2"
            shift 2
            ;;
        --transformer-dir)
            transformer_dir="$2"
            shift 2
            ;;
        --output-dir)
            output_dir="$2"
            shift 2
            ;;
        --attn-mode)
            attn_mode="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "$base_model" || -z "$transformer_dir" || -z "$output_dir" ]]; then
    usage >&2
    exit 1
fi

if [[ "$attn_mode" != "torch" && "$attn_mode" != "flashattn" ]]; then
    echo "--attn-mode must be 'torch' or 'flashattn'" >&2
    exit 1
fi

for path in "$base_model" "$transformer_dir"; do
    if [[ ! -d "$path" ]]; then
        echo "directory not found: $path" >&2
        exit 1
    fi
done

for req in vae tokenizer text_encoder transformer; do
    if [[ ! -e "$base_model/$req" ]]; then
        echo "base model is missing required path: $base_model/$req" >&2
        exit 1
    fi
done

for req in config.json diffusion_pytorch_model.safetensors; do
    if [[ ! -f "$transformer_dir/$req" ]]; then
        echo "checkpoint transformer is missing required file: $transformer_dir/$req" >&2
        exit 1
    fi
done

mkdir -p "$output_dir"

# Copy the complete skeleton first. Keep metadata files such as configuration.json and README.md.
rsync -a --delete \
    --exclude '._____temp/' \
    "$base_model/" \
    "$output_dir/"

# Then replace the skeleton transformer with the trained checkpoint transformer.
rsync -a --delete \
    "$transformer_dir/" \
    "$output_dir/transformer/"

python - "$output_dir/transformer/config.json" "$attn_mode" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
attn_mode = sys.argv[2]
config = json.loads(config_path.read_text(encoding="utf-8"))
config["attn_mode"] = attn_mode
config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"updated {config_path} attn_mode={attn_mode}")
PY

echo "Inference-ready model assembled at: $output_dir"
echo "Base skeleton: $base_model"
echo "Transformer source: $transformer_dir"
echo "Attn mode: $attn_mode"
