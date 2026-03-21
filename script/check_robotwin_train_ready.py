#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the local LingBot-VA RobotWin fine-tuning setup."
    )
    parser.add_argument(
        "--model-root",
        default="/home/syr/code/lingbotva/artifacts/models/lingbot-va-base",
    )
    parser.add_argument(
        "--dataset-root",
        default="/data/syr/datasets/robotwin-clean-and-aug-lerobot",
    )
    parser.add_argument(
        "--expected-attn-mode",
        default="flex",
    )
    parser.add_argument(
        "--expected-dataset-count",
        type=int,
        default=100,
    )
    return parser.parse_args()


def check_model(model_root: Path, expected_attn_mode: str) -> list[str]:
    errors: list[str] = []
    required_paths = [
        model_root / "transformer" / "config.json",
        model_root / "transformer" / "diffusion_pytorch_model-00001-of-00003.safetensors",
        model_root / "transformer" / "diffusion_pytorch_model-00002-of-00003.safetensors",
        model_root / "transformer" / "diffusion_pytorch_model-00003-of-00003.safetensors",
        model_root / "text_encoder" / "model-00001-of-00003.safetensors",
        model_root / "text_encoder" / "model-00002-of-00003.safetensors",
        model_root / "text_encoder" / "model-00003-of-00003.safetensors",
        model_root / "tokenizer" / "tokenizer.json",
        model_root / "vae" / "diffusion_pytorch_model.safetensors",
    ]
    for path in required_paths:
        if not path.exists():
            errors.append(f"missing model path: {path}")

    config_path = model_root / "transformer" / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        attn_mode = config.get("attn_mode")
        if attn_mode != expected_attn_mode:
            errors.append(
                f"unexpected attn_mode in {config_path}: {attn_mode!r} != {expected_attn_mode!r}"
            )
    return errors


def check_dataset(dataset_root: Path, expected_dataset_count: int) -> tuple[list[str], dict[str, int]]:
    errors: list[str] = []
    stats: dict[str, int] = {}

    empty_emb = dataset_root / "empty_emb.pt"
    manifest_path = dataset_root / ".cache" / "hf_mirror_download" / "manifest.txt"
    state_path = dataset_root / ".cache" / "hf_mirror_download" / "state.json"

    if not empty_emb.exists():
        errors.append(f"missing dataset file: {empty_emb}")
    if not manifest_path.exists():
        errors.append(f"missing manifest file: {manifest_path}")
        return errors, stats
    if not state_path.exists():
        errors.append(f"missing state file: {state_path}")
        return errors, stats

    manifest_files = [line.strip() for line in manifest_path.read_text().splitlines() if line.strip()]
    state = json.loads(state_path.read_text(encoding="utf-8"))

    missing_files = [rel for rel in manifest_files if not (dataset_root / rel).exists()]
    info_json_count = sum(1 for _ in dataset_root.rglob("meta/info.json"))

    stats["manifest_files"] = len(manifest_files)
    stats["missing_files"] = len(missing_files)
    stats["info_json_count"] = info_json_count
    stats["listing_complete"] = int(bool(state.get("listing_complete", False)))

    if not state.get("listing_complete", False):
        errors.append("dataset manifest listing is not complete yet")
    if missing_files:
        errors.append(f"dataset download still missing {len(missing_files)} files")
    if info_json_count != expected_dataset_count:
        errors.append(
            f"dataset count mismatch: found {info_json_count} meta/info.json files, expected {expected_dataset_count}"
        )

    return errors, stats


def main() -> int:
    args = parse_args()
    model_root = Path(args.model_root)
    dataset_root = Path(args.dataset_root)

    errors: list[str] = []
    errors.extend(check_model(model_root, args.expected_attn_mode))
    dataset_errors, stats = check_dataset(dataset_root, args.expected_dataset_count)
    errors.extend(dataset_errors)

    print(f"model_root={model_root}")
    print(f"dataset_root={dataset_root}")
    for key in sorted(stats):
        print(f"{key}={stats[key]}")

    if errors:
        print("ready=no")
        for item in errors:
            print(f"error={item}")
        return 1

    print("ready=yes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
