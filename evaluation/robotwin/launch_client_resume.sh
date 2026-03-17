#!/bin/bash
set -euo pipefail

export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:${LD_LIBRARY_PATH:-}

save_root=${1:-"results/results_full_3.11"}
test_num=${2:-5}
seed=${3:-0}

policy_name=${POLICY_NAME:-ACT}
task_config=${TASK_CONFIG:-demo_clean}
train_config_name=${TRAIN_CONFIG_NAME:-0}
model_name=${MODEL_NAME:-0}
start_port=${START_PORT:-29556}
num_gpus=${NUM_GPUS:-8}
video_guidance_scale=${VIDEO_GUIDANCE_SCALE:-5}
action_guidance_scale=${ACTION_GUIDANCE_SCALE:-1}
dry_run=${DRY_RUN:-0}

all_tasks=(
  stack_bowls_three
  handover_block
  hanging_mug
  scan_object
  lift_pot
  put_object_cabinet
  stack_blocks_three
  place_shoe
  adjust_bottle
  place_mouse_pad
  dump_bin_bigbin
  move_pillbottle_pad
  pick_dual_bottles
  shake_bottle
  place_fan
  turn_switch
  shake_bottle_horizontally
  place_container_plate
  rotate_qrcode
  place_object_stand
  put_bottles_dustbin
  move_stapler_pad
  place_burger_fries
  place_bread_basket
  pick_diverse_bottles
  open_microwave
  beat_block_hammer
  press_stapler
  click_bell
  move_playingcard_away
  open_laptop
  move_can_pot
  stack_bowls_two
  place_a2b_right
  stamp_seal
  place_object_basket
  handover_mic
  place_bread_skillet
  stack_blocks_two
  place_cans_plasticbox
  click_alarmclock
  blocks_ranking_size
  place_phone_stand
  place_can_basket
  place_object_scale
  place_a2b_left
  grab_roller
  place_dual_shoes
  place_empty_cup
  blocks_ranking_rgb
)

seed_tag=$((10000 * (1 + seed)))
metrics_root="/home/syr/code/lingbot-va/RoboTwin/${save_root}/stseed-${seed_tag}/metrics"

mapfile -t missing_tasks < <(
  python - "$metrics_root" "$test_num" "${all_tasks[@]}" <<'PY'
import json
import sys
from pathlib import Path

metrics_root = Path(sys.argv[1])
target_total = float(sys.argv[2])
tasks = sys.argv[3:]

for task in tasks:
    res_path = metrics_root / task / "res.json"
    if not res_path.exists():
        print(task)
        continue

    try:
        data = json.loads(res_path.read_text())
    except Exception:
        print(task)
        continue

    if float(data.get("total_num", 0.0)) < target_total:
        print(task)
PY
)

echo "save_root=${save_root}"
echo "metrics_root=${metrics_root}"
echo "seed=${seed}"
echo "test_num=${test_num}"
echo "start_port=${start_port}"
echo "num_gpus=${num_gpus}"

if [[ ${#missing_tasks[@]} -eq 0 ]]; then
    echo "No missing or incomplete tasks found."
    exit 0
fi

echo "Missing/incomplete tasks (${#missing_tasks[@]}): ${missing_tasks[*]}"

if [[ "${dry_run}" == "1" ]]; then
    echo "DRY_RUN=1 set; not launching clients."
    exit 0
fi

log_dir="./logs"
mkdir -p "${log_dir}"

batch_time=$(date +%Y%m%d_%H%M%S)
pid_file="pids_resume_${batch_time}.txt"
> "${pid_file}"

for ((batch_start=0; batch_start<${#missing_tasks[@]}; batch_start+=num_gpus)); do
    batch_tasks=("${missing_tasks[@]:batch_start:num_gpus}")
    echo "Launching batch $((batch_start / num_gpus + 1)): ${batch_tasks[*]}"

    batch_pids=()
    for i in "${!batch_tasks[@]}"; do
        task_name="${batch_tasks[$i]}"
        gpu_id=$((i % num_gpus))
        port=$((start_port + i))
        log_file="${log_dir}/${task_name}_${batch_time}.log"

        echo "[Batch $((batch_start / num_gpus + 1))][Task ${i}] task=${task_name} gpu=${gpu_id} port=${port} log=${log_file}"

        CUDA_VISIBLE_DEVICES=${gpu_id} \
        PYTHONWARNINGS=ignore::UserWarning \
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
        python -m evaluation.robotwin.eval_polict_client_openpi --config policy/${policy_name}/deploy_policy.yml \
            --overrides \
            --task_name "${task_name}" \
            --task_config "${task_config}" \
            --train_config_name "${train_config_name}" \
            --model_name "${model_name}" \
            --ckpt_setting "${model_name}" \
            --seed "${seed}" \
            --policy_name "${policy_name}" \
            --save_root "${save_root}" \
            --video_guidance_scale "${video_guidance_scale}" \
            --action_guidance_scale "${action_guidance_scale}" \
            --test_num "${test_num}" \
            --port "${port}" > "${log_file}" 2>&1 &

        pid=$!
        batch_pids+=("${pid}")
        echo "${pid}" | tee -a "${pid_file}" >/dev/null
    done

    echo "Waiting for batch $((batch_start / num_gpus + 1)) to finish..."
    wait "${batch_pids[@]}"
done

echo "All missing tasks finished. PIDs recorded in ${pid_file}."
