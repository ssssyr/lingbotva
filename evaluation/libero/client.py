import numpy as np
import argparse
import hashlib
import json
import re
import time
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_va.utils.Simple_Remote_Infer.deploy.websocket_client_policy import WebsocketClientPolicy
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from tqdm import tqdm
import imageio
import cv2


def save_video(real_obs_list, save_path, fps=15, video_names=["observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"]):
    if not real_obs_list:
        print("❌ No real observation frames")
        return

    first_obs = real_obs_list[0]
    base_h, width_base = first_obs[video_names[0]].shape[:2]
    target_size = (width_base, base_h)
    
    print(f"Saving video: {len(real_obs_list)} frames...")

    final_frames = [
        np.hstack([cv2.resize(obs[name], target_size) for name in video_names]).astype(np.uint8)
        for obs in real_obs_list
    ]

    imageio.mimsave(save_path, final_frames, fps=fps)
    print(f"✅ Video saved to: {save_path}")


def write_json_file(payload, path):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sanitize_prompt_for_path(prompt, max_prefix=80):
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", prompt).strip("._-")
    cleaned = re.sub(r"_+", "_", cleaned)
    if not cleaned:
        cleaned = "task"
    prefix = cleaned[:max_prefix].rstrip("._-")
    return prefix or "task"


def _task_dir_name(task_idx, prompt):
    prompt_hash = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]
    prompt_prefix = _sanitize_prompt_for_path(prompt)
    return f"{task_idx}_{prompt_prefix}_{prompt_hash}"


def _legacy_task_dir_name(task_idx, prompt):
    return f"{task_idx}_{prompt.replace(' ', '_')}"


def _candidate_task_dirs(out_dir, libero_benchmark, task_idx, prompt):
    root = Path(out_dir) / libero_benchmark
    names = [_task_dir_name(task_idx, prompt)]
    legacy = _legacy_task_dir_name(task_idx, prompt)
    if legacy not in names:
        names.append(legacy)
    return [root / name for name in names]


def get_existing_episode_results(out_dir, libero_benchmark, task_idx, prompt, test_num):
    results = {}
    for task_dir in _candidate_task_dirs(out_dir, libero_benchmark, task_idx, prompt):
        try:
            exists = task_dir.exists()
        except OSError:
            continue
        if not exists:
            continue
        try:
            video_paths = list(task_dir.glob("*.mp4"))
        except OSError:
            continue
        for video_path in video_paths:
            parts = video_path.stem.split("_", 1)
            if len(parts) != 2:
                continue
            try:
                episode_idx = int(parts[0])
            except ValueError:
                continue
            if episode_idx >= test_num:
                continue
            results[episode_idx] = parts[1] == "True"
    return results


def construct_single_env(env_args):
    count = 0
    env = None
    env_creation = False
    while not env_creation and count < 5:
        try:
            env = OffScreenRenderEnv(**env_args)
            env_creation = True
        except Exception as e:
            print(f"Error!!!  construct env failed: {e}")
            time.sleep(5)
            count += 1
    if count >= 5:
        return None
    return env


def _extract_obs(obs):
    """
    Extract agentview and eye_in_hand images from raw env obs dict.

    Avoids torch round-trip: the env already returns uint8 numpy arrays [H, W, C].
    We just flip the vertical axis ([::-1]) and make a contiguous copy once.
    """
    agentview = np.ascontiguousarray(obs["agentview_image"][::-1])
    eye_in_hand = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
    return {"observation.images.agentview_rgb": agentview, "observation.images.eye_in_hand_rgb": eye_in_hand}


def init_single_env(env_in, init_state):
    env_in.reset()
    env_in.set_init_state(init_state)
    for _ in range(5):
        obs, _, _, _ = env_in.step([0.] * 7)
    return _extract_obs(obs)


def env_one_step(env_in, action):
    obs, _, done, _ = env_in.step(action)
    return _extract_obs(obs), done


def run_one(model, libero_benchmark, task_idx, out_dir, episode_idx):
    benchmark_dict = benchmark.get_benchmark_dict()
    benchmark_instance = benchmark_dict[libero_benchmark]()
    num_tasks = benchmark_instance.get_num_tasks()
    assert task_idx < num_tasks, f"Error: error id must smaller than {num_tasks}"
    prompt = benchmark_instance.get_task(task_idx).language
    env_args = {
                "bddl_file_name": benchmark_instance.get_task_bddl_file_path(task_idx),
                "camera_heights": 128,
                "camera_widths": 128,
            }
    init_states = benchmark_instance.get_task_init_states(task_idx)

    cur_env = construct_single_env(env_args)
    first_obs = init_single_env(cur_env, init_states[episode_idx % init_states.shape[0]])

    ret = model.infer(dict(reset=True, prompt=prompt))

    full_obs_list = []
    done = False
    first = True
    while cur_env.env.timestep < 800:
        ret = model.infer(dict(obs=first_obs, prompt=prompt))
        action = ret['action']

        key_frame_list = []
        assert action.shape[2] % 4 == 0
        action_per_frame = action.shape[2] // 4
        start_idx = 1 if first else 0
        for i in range(start_idx, action.shape[1]):
            for j in range(action.shape[2]):
                ee_action = action[:, i, j]
                observes, done = env_one_step(cur_env, ee_action)
                if done:
                    break
                if (j+1) % action_per_frame == 0:
                    full_obs_list.append(observes)
                    key_frame_list.append(observes)

            if done:
                break

        first = False

        if done:
            break
        else:
            model.infer(dict(obs=key_frame_list, compute_kv_cache=True, imagine=False, state=action))

    out_file = Path(out_dir) / libero_benchmark / _task_dir_name(task_idx, prompt) / f"{episode_idx}_{done}.mp4"
    out_file.parent.mkdir(exist_ok=True, parents=True)

    save_video(
        real_obs_list=full_obs_list,
        save_path=out_file,
        fps=60,
        video_names=["observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"]
    )

    cur_env.close()
    return done


def run(libero_benchmark, port, out_dir, test_num, task_range=None):
    '''
        task_range: [start, end) for splitting tasks
    '''
    if task_range is None:
        benchmark_dict = benchmark.get_benchmark_dict()
        benchmark_instance = benchmark_dict[libero_benchmark]()
        num_tasks = benchmark_instance.get_num_tasks()
        progress_bar = tqdm(range(num_tasks), total=num_tasks)
    else:
        assert len(task_range) == 2, f'task_range: [start, end) for splitting tasks, however, task_range: {task_range}'
        num_tasks = task_range[1] - task_range[0]
        progress_bar = tqdm(range(task_range[0], task_range[1]), total=num_tasks)

    print(f"#################### Use benchmark: {libero_benchmark}, num_tasks: {num_tasks} #############")
    model = WebsocketClientPolicy(port=port)

    for task_idx in progress_bar:
        benchmark_dict = benchmark.get_benchmark_dict()
        benchmark_instance = benchmark_dict[libero_benchmark]()
        prompt = benchmark_instance.get_task(task_idx).language
        existing_results = get_existing_episode_results(out_dir, libero_benchmark, task_idx, prompt, test_num)
        succ_num = float(sum(existing_results.values()))
        completed_num = len(existing_results)
        episode_list = [idx for idx in range(test_num) if idx not in existing_results]

        if not episode_list:
            out_file = Path(out_dir) / f"{libero_benchmark}_{task_idx}.json"
            write_json_file({
                "succ_num": succ_num,
                "total_num": float(completed_num),
                "succ_rate": succ_num / completed_num if completed_num else 0.0,
            }, out_file)
            continue

        for episode_idx in tqdm(episode_list, total=len(episode_list)):
            res_i = run_one(model, libero_benchmark, task_idx, out_dir, episode_idx)
            succ_num += res_i
            completed_num += 1
            succ_rate = succ_num / completed_num
            print(f"Success rate: {succ_rate}, success num: {succ_num}, total num: {completed_num}")
            out_file = Path(out_dir) / f"{libero_benchmark}_{task_idx}.json"
            write_json_file({
                "succ_num": succ_num,
                "total_num": float(completed_num),
                "succ_rate": succ_rate,
                }, out_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--libero-benchmark",
        type=str,
        default="libero_10",
        choices=["libero_10", "libero_goal", "libero_spatial", "libero_object"],
        help="Benchmark name",
    )
    parser.add_argument(
        "--task-range",
        type=int,
        nargs="+",
        default=[0, 10],
        help="Task range [start, end) for splitting tasks",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=23908,
        help="WebSocket port",
    )
    parser.add_argument(
        "--test-num",
        type=int,
        default=50,
        help="Number of test episodes",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/libero",
        help="Output directory for results",
    )
    args = parser.parse_args()
    run(**vars(args))
    print("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    main()
