#!/usr/bin/env python3
import argparse
import bisect
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def quantiles(arr: np.ndarray):
    qs = [0.01, 0.1, 0.5, 0.9, 0.99]
    vals = np.quantile(arr, qs, axis=0)
    return {
        'q01': vals[0].tolist(),
        'q10': vals[1].tolist(),
        'q50': vals[2].tolist(),
        'q90': vals[3].tolist(),
        'q99': vals[4].tolist(),
    }


def stats_for_matrix(arr: np.ndarray):
    out = {
        'min': arr.min(axis=0).tolist(),
        'max': arr.max(axis=0).tolist(),
        'mean': arr.mean(axis=0).tolist(),
        'std': arr.std(axis=0).tolist(),
        'count': [int(arr.shape[0])],
    }
    out.update(quantiles(arr))
    return out


def read_list_column(parquet_path: Path, col: str):
    pf = pq.ParquetFile(parquet_path)
    tbl = pf.read_row_group(0, columns=[col])
    data = tbl.column(0).to_pylist()
    return np.asarray(data, dtype=np.float32)


def ffmpeg_downsample(src: Path, dst: Path, stride: int, out_fps: int):
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        'ffmpeg', '-y', '-i', str(src),
        '-vf', f"select='not(mod(n\\,{stride}))',setpts=N/({out_fps}*TB)",
        '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-r', str(out_fps),
        str(dst)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-root', required=True)
    ap.add_argument('--output-root', required=True)
    ap.add_argument('--stride', type=int, default=2)
    ap.add_argument('--output-fps', type=int, default=None)
    ap.add_argument('--copy-tasks', action='store_true', default=True)
    ap.add_argument('--episode-limit', type=int, default=None)
    args = ap.parse_args()

    in_root = Path(args.input_root)
    out_root = Path(args.output_root)
    assert in_root.exists(), in_root
    if out_root.exists():
        raise SystemExit(f'output already exists: {out_root}')
    out_root.mkdir(parents=True)
    (out_root / 'data' / 'chunk-000').mkdir(parents=True)
    (out_root / 'videos' / 'chunk-000' / 'observation.images.third').mkdir(parents=True)
    (out_root / 'videos' / 'chunk-000' / 'observation.images.wrist').mkdir(parents=True)
    (out_root / 'meta').mkdir(parents=True)

    info = json.loads((in_root / 'meta' / 'info.json').read_text())
    in_fps = int(info['fps'])
    out_fps = args.output_fps or max(1, in_fps // args.stride)

    episodes = []
    with (in_root / 'meta' / 'episodes.jsonl').open() as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))

    all_action = []
    all_state = []
    episode_stats_lines = []
    total_frames = 0

    if args.episode_limit is not None:
        episodes = episodes[:args.episode_limit]

    for ep in episodes:
        ep_idx = int(ep['episode_index'])
        src_parquet = in_root / 'data' / 'chunk-000' / f'episode_{ep_idx:06d}.parquet'
        action = read_list_column(src_parquet, 'action')
        state = read_list_column(src_parquet, 'observation.state')
        length = len(action)
        keep_idx = np.arange(0, length, args.stride, dtype=np.int64)
        new_len = int(len(keep_idx))

        action_ds = action[keep_idx]
        state_ds = state[keep_idx]
        timestamps = (keep_idx / in_fps).astype(np.float64)
        frame_index = np.arange(new_len, dtype=np.int64)
        episode_index = np.full(new_len, ep_idx, dtype=np.int64)
        index = np.arange(new_len, dtype=np.int64)
        task_index = np.zeros(new_len, dtype=np.int64)

        table = pa.table({
            'action': pa.array(action_ds.tolist(), type=pa.list_(pa.float32())),
            'observation.state': pa.array(state_ds.tolist(), type=pa.list_(pa.float32())),
            'timestamp': pa.array(timestamps),
            'frame_index': pa.array(frame_index),
            'episode_index': pa.array(episode_index),
            'index': pa.array(index),
            'task_index': pa.array(task_index),
        })
        pq.write_table(table, out_root / 'data' / 'chunk-000' / f'episode_{ep_idx:06d}.parquet')

        for cam in ['observation.images.third', 'observation.images.wrist']:
            src_video = in_root / 'videos' / 'chunk-000' / cam / f'episode_{ep_idx:06d}.mp4'
            dst_video = out_root / 'videos' / 'chunk-000' / cam / f'episode_{ep_idx:06d}.mp4'
            ffmpeg_downsample(src_video, dst_video, args.stride, out_fps)

        new_action_cfg = []
        for seg in ep.get('action_config', []) or []:
            s = int(seg['start_frame'])
            e = int(seg['end_frame'])
            new_s = bisect.bisect_left(keep_idx.tolist(), s)
            new_e = bisect.bisect_left(keep_idx.tolist(), e)
            seg2 = dict(seg)
            seg2['start_frame'] = new_s
            seg2['end_frame'] = new_e
            new_action_cfg.append(seg2)

        ep_out = dict(ep)
        ep_out['length'] = new_len
        ep_out['action_config'] = new_action_cfg
        total_frames += new_len

        action_stats = stats_for_matrix(action_ds)
        state_stats = stats_for_matrix(state_ds)
        episode_stats_lines.append({'episode_index': ep_idx, 'stats': {'action': action_stats, 'observation.state': state_stats}})
        all_action.append(action_ds)
        all_state.append(state_ds)
        ep['_out'] = ep_out

    with (out_root / 'meta' / 'episodes.jsonl').open('w') as f:
        for ep in episodes:
            f.write(json.dumps(ep['_out'], ensure_ascii=False) + '\n')
    with (out_root / 'meta' / 'episodes_stats.jsonl').open('w') as f:
        for row in episode_stats_lines:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    shutil.copy2(in_root / 'meta' / 'tasks.jsonl', out_root / 'meta' / 'tasks.jsonl')

    action_all = np.concatenate(all_action, axis=0)
    state_all = np.concatenate(all_state, axis=0)
    old_stats = json.loads((in_root / 'meta' / 'stats.json').read_text())
    old_stats['action'] = stats_for_matrix(action_all)
    old_stats['observation.state'] = stats_for_matrix(state_all)
    with (out_root / 'meta' / 'stats.json').open('w') as f:
        json.dump(old_stats, f, ensure_ascii=False, indent=2)

    info['fps'] = out_fps
    info['total_frames'] = total_frames
    info['total_episodes'] = len(episodes)
    for k in ['data_files_size_in_mb', 'video_files_size_in_mb']:
        info[k] = 0
    features = info.get('features', {})
    for cam in ['observation.images.third', 'observation.images.wrist']:
        if cam in features and 'info' in features[cam]:
            features[cam]['info']['video.fps'] = out_fps
            features[cam]['info']['video.codec'] = 'h264'
            features[cam]['info']['video.pix_fmt'] = 'yuv420p'
    with (out_root / 'meta' / 'info.json').open('w') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f'created {out_root}')
    print(f'episodes={len(episodes)} total_frames={total_frames} fps={out_fps}')

if __name__ == '__main__':
    main()
