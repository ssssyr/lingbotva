#!/usr/bin/env python3
"""Plot cutoff scan results from summary.csv and per_state_loss.pt."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_va.configs import VA_CONFIGS

DEFAULT_METRICS = [
    'masked_action_mse',
    'denorm_action_l1',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot cutoff scan mean curves and per-state trends.'
    )
    parser.add_argument(
        '--scan-dir',
        required=True,
        help='Directory containing summary.csv, per_state_loss.pt, and scan_args.json.',
    )
    parser.add_argument(
        '--metrics',
        default=','.join(DEFAULT_METRICS),
        help='Comma-separated metric names to plot.',
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Directory to save plots. Defaults to <scan-dir>/plots.',
    )
    parser.add_argument(
        '--title-prefix',
        default='',
        help='Optional title prefix for the generated figures.',
    )
    return parser.parse_args()


def load_total_video_steps(scan_args):
    config_name = scan_args.get('config')
    if config_name in VA_CONFIGS:
        return int(VA_CONFIGS[config_name].num_inference_steps)
    return None


def compute_per_state_table(per_state_records, metrics):
    rows = []
    for rec in per_state_records:
        state_uid = rec['state_uid']
        metrics_by_cutoff = rec['metrics_by_cutoff']
        for cutoff_key, metric_values in metrics_by_cutoff.items():
            row = {
                'state_uid': state_uid,
                'cutoff_idx': int(cutoff_key),
            }
            for metric in metrics:
                row[metric] = float(metric_values[metric])
            rows.append(row)
    return pd.DataFrame(rows)


def metric_label(metric_name):
    labels = {
        'masked_action_mse': 'Masked Action MSE',
        'denorm_action_l1': 'Denorm Action L1',
        'first_pred_action_mse': 'First Predicted Action MSE',
    }
    return labels.get(metric_name, metric_name)


def budget_tick_labels(cutoff_values, total_video_steps):
    if not total_video_steps:
        return [str(v) for v in cutoff_values]
    return ['{}\n({:.0f}%)'.format(v, 100.0 * float(v) / float(total_video_steps)) for v in cutoff_values]


def plot_mean_curves(summary_df, metrics, total_video_steps, output_path, title_prefix=''):
    cutoff_values = summary_df['cutoff_idx'].tolist()
    xticklabels = budget_tick_labels(cutoff_values, total_video_steps)
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4.8), squeeze=False)
    axes = axes[0]

    for ax, metric in zip(axes, metrics):
        mean_col = '{}_mean'.format(metric)
        std_col = '{}_std'.format(metric)
        mean_vals = summary_df[mean_col].to_numpy(dtype=float)
        std_vals = summary_df[std_col].to_numpy(dtype=float)
        num_states = summary_df['num_states'].to_numpy(dtype=float)
        sem_vals = std_vals / num_states.clip(min=1.0) ** 0.5

        ax.plot(cutoff_values, mean_vals, marker='o', linewidth=2.0, color='#1f77b4')
        ax.fill_between(
            cutoff_values,
            mean_vals - sem_vals,
            mean_vals + sem_vals,
            color='#1f77b4',
            alpha=0.20,
            label='mean ± SEM',
        )
        ax.set_xticks(cutoff_values)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel('cutoff_idx')
        ax.set_ylabel(metric_label(metric))
        ax.set_title('{}{}'.format(title_prefix + ' ' if title_prefix else '', metric_label(metric)))
        ax.grid(True, alpha=0.25)
        ax.legend(loc='best')

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_per_state_curves(per_state_df, metrics, total_video_steps, output_path, title_prefix=''):
    cutoff_values = sorted(per_state_df['cutoff_idx'].unique().tolist())
    xticklabels = budget_tick_labels(cutoff_values, total_video_steps)
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4.8), squeeze=False)
    axes = axes[0]

    for ax, metric in zip(axes, metrics):
        for state_uid, group in per_state_df.groupby('state_uid'):
            group = group.sort_values('cutoff_idx')
            ax.plot(
                group['cutoff_idx'].to_numpy(dtype=float),
                group[metric].to_numpy(dtype=float),
                color='#9aa0a6',
                alpha=0.45,
                linewidth=1.0,
            )

        grouped = per_state_df.groupby('cutoff_idx')[metric]
        mean_vals = grouped.mean().reindex(cutoff_values).to_numpy(dtype=float)
        sem_vals = grouped.sem().fillna(0.0).reindex(cutoff_values).to_numpy(dtype=float)
        ax.plot(cutoff_values, mean_vals, marker='o', linewidth=2.4, color='#d62728', label='mean')
        ax.fill_between(
            cutoff_values,
            mean_vals - sem_vals,
            mean_vals + sem_vals,
            color='#d62728',
            alpha=0.18,
            label='mean ± SEM',
        )
        ax.set_xticks(cutoff_values)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel('cutoff_idx')
        ax.set_ylabel(metric_label(metric))
        ax.set_title('{}{} per-state'.format(title_prefix + ' ' if title_prefix else '', metric_label(metric)))
        ax.grid(True, alpha=0.25)
        ax.legend(loc='best')

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    scan_dir = Path(args.scan_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else scan_dir / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = scan_dir / 'summary.csv'
    per_state_path = scan_dir / 'per_state_loss.pt'
    scan_args_path = scan_dir / 'scan_args.json'

    if not summary_path.exists():
        raise FileNotFoundError('summary.csv not found at {}'.format(summary_path))
    if not per_state_path.exists():
        raise FileNotFoundError('per_state_loss.pt not found at {}'.format(per_state_path))
    if not scan_args_path.exists():
        raise FileNotFoundError('scan_args.json not found at {}'.format(scan_args_path))

    metrics = [m.strip() for m in args.metrics.split(',') if m.strip()]
    summary_df = pd.read_csv(summary_path).sort_values('cutoff_idx').reset_index(drop=True)
    with scan_args_path.open('r', encoding='utf-8') as f:
        scan_args = json.load(f)
    per_state_records = torch.load(str(per_state_path), map_location='cpu')
    per_state_df = compute_per_state_table(per_state_records, metrics)
    total_video_steps = load_total_video_steps(scan_args)

    mean_plot_path = output_dir / 'cutoff_mean_curves.png'
    state_plot_path = output_dir / 'cutoff_per_state_curves.png'

    plot_mean_curves(summary_df, metrics, total_video_steps, str(mean_plot_path), title_prefix=args.title_prefix)
    plot_per_state_curves(per_state_df, metrics, total_video_steps, str(state_plot_path), title_prefix=args.title_prefix)

    print('summary_csv={}'.format(summary_path))
    print('per_state_loss={}'.format(per_state_path))
    print('mean_plot={}'.format(mean_plot_path))
    print('per_state_plot={}'.format(state_plot_path))
    print('metrics={}'.format(metrics))


if __name__ == '__main__':
    raise SystemExit(main())
