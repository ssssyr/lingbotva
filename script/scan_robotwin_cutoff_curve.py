#!/usr/bin/env python3
"""Offline cutoff_idx scan for video-to-action handoff quality."""

import argparse
import csv
import json
import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_va.configs import VA_CONFIGS
from wan_va.dataset.lerobot_latent_dataset import MultiLatentLeRobotDataset
from wan_va.modules.utils import load_transformer
from wan_va.utils.cutoff_scan import CutoffScanner, scan_cutoff_curve, summarize_cutoff_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Scan discrete video cutoff_idx values offline and measure action loss curves."
        )
    )
    parser.add_argument("--config", default="robotwin_train", help="Config name in wan_va.configs.VA_CONFIGS")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Path to checkpoint root or transformer directory. "
            "If omitted, uses config.wan22_pretrained_model_name_or_path/transformer."
        ),
    )
    parser.add_argument("--dataset-path", default=None, help="Override config dataset path")
    parser.add_argument(
        "--dataset-substring",
        default='',
        help='Comma-separated repo path substrings used to keep only matching LeRobot subdatasets.',
    )
    parser.add_argument(
        "--cutoff-indices",
        default="1,4,8,12,16,20,25",
        help="Comma-separated discrete cutoff_idx list, e.g. 1,4,8,12,16,20,25",
    )
    parser.add_argument("--max-samples", type=int, default=32, help="Max number of samples to scan; <=0 means all")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for deterministic per-sample noise")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--init-workers", type=int, default=1, help="Dataset init workers")
    parser.add_argument(
        "--max-datasets",
        type=int,
        default=0,
        help="Limit how many LeRobot subdatasets are initialized. 0 means all.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: ./train_out/cutoff_scan/<timestamp>",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device string, e.g. cuda:0 or cpu. Default: cuda:0 when available else cpu",
    )
    return parser.parse_args()


def parse_cutoff_indices(raw, max_step):
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        idx = int(item)
        if idx < 1 or idx > max_step:
            raise ValueError("cutoff_idx out of range: {}, expected [1, {}]".format(idx, max_step))
        values.append(idx)
    if not values:
        raise ValueError("No valid cutoff_idx parsed from --cutoff-indices")
    return sorted(set(values))


def resolve_transformer_dir(checkpoint, config):
    if checkpoint is None:
        return Path(config.wan22_pretrained_model_name_or_path) / "transformer"

    path = Path(checkpoint).expanduser().resolve()
    if path.is_dir() and (path / "config.json").exists():
        return path
    if path.is_dir() and (path / "transformer" / "config.json").exists():
        return path / "transformer"
    raise FileNotFoundError(
        "Cannot resolve transformer directory from checkpoint path: {}".format(checkpoint)
    )


def default_output_dir():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("./train_out") / "cutoff_scan" / stamp


def ensure_dataset_config(config, dataset_path):
    if dataset_path is not None:
        config.dataset_path = dataset_path
    if not hasattr(config, "dataset_path"):
        raise ValueError("Selected config has no dataset_path. Use a *_train config.")

    config.empty_emb_path = os.path.join(config.dataset_path, "empty_emb.pt")
    config.cfg_prob = 0.0


def main():
    args = parse_args()

    if args.config not in VA_CONFIGS:
        raise KeyError("Unknown config: {}. Available: {}".format(args.config, sorted(VA_CONFIGS.keys())))

    config = deepcopy(VA_CONFIGS[args.config])
    ensure_dataset_config(config, args.dataset_path)
    dataset_repo_filters = [v.strip() for v in args.dataset_substring.split(',') if v.strip()]
    if dataset_repo_filters:
        config.dataset_repo_filters = dataset_repo_filters
    if int(args.max_datasets) > 0:
        config.max_dataset_repos = int(args.max_datasets)

    device_str = args.device
    if device_str is None:
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    if not Path(config.empty_emb_path).exists():
        raise FileNotFoundError("empty_emb not found: {}".format(config.empty_emb_path))

    transformer_dir = resolve_transformer_dir(args.checkpoint, config)
    if not transformer_dir.exists():
        raise FileNotFoundError("transformer dir not found: {}".format(transformer_dir))

    cutoff_indices = parse_cutoff_indices(args.cutoff_indices, int(config.num_inference_steps))

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("config={}".format(args.config))
    print("device={}".format(device))
    print("dataset_path={}".format(config.dataset_path))
    print("transformer_dir={}".format(transformer_dir))
    print("cutoff_indices={}".format(cutoff_indices))
    print("output_dir={}".format(output_dir))

    dataset = MultiLatentLeRobotDataset(config, num_init_worker=args.init_workers)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )

    transformer = load_transformer(
        str(transformer_dir),
        torch_dtype=config.param_dtype,
        torch_device=device,
    )

    scanner = CutoffScanner(
        transformer=transformer,
        config=config,
        device=device,
        dtype=config.param_dtype,
        cache_name="pos",
    )

    with torch.no_grad():
        result = scan_cutoff_curve(
            scanner,
            dataloader,
            cutoff_indices,
            max_samples=int(args.max_samples),
            seed=int(args.seed),
            show_progress=True,
        )

    summary_rows = summarize_cutoff_metrics(result["metric_table"])

    summary_path = output_dir / "summary.csv"
    per_state_path = output_dir / "per_state_loss.pt"
    args_path = output_dir / "scan_args.json"

    if summary_rows:
        fieldnames = ["cutoff_idx", "num_states"]
        metric_names = [
            "masked_action_mse",
            "first_pred_action_mse",
            "denorm_action_l1",
        ]
        for name in metric_names:
            fieldnames.extend(["{}_mean".format(name), "{}_std".format(name)])
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
    else:
        summary_path.write_text("cutoff_idx,num_states\n", encoding="utf-8")

    torch.save(result["per_state_records"], str(per_state_path))

    run_args = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "resolved_transformer_dir": str(transformer_dir),
        "dataset_path": str(config.dataset_path),
        "dataset_substring": args.dataset_substring,
        "cutoff_indices": cutoff_indices,
        "max_samples": int(args.max_samples),
        "seed": int(args.seed),
        "num_workers": int(args.num_workers),
        "init_workers": int(args.init_workers),
        "max_datasets": int(args.max_datasets),
        "device": str(device),
        "num_states_scanned": int(result["num_states"]),
    }
    args_path.write_text(json.dumps(run_args, ensure_ascii=False, indent=2), encoding="utf-8")

    print("num_states_scanned={}".format(result["num_states"]))
    print("summary_csv={}".format(summary_path))
    print("per_state_loss={}".format(per_state_path))
    print("scan_args={}".format(args_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
