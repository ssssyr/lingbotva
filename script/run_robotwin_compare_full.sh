#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${REPO_ROOT}/RoboTwin}"
SERVER_PY="${REPO_ROOT}/.local/miniconda3/envs/lingbot-va/bin/python"
CLIENT_PY="${REPO_ROOT}/.local/miniconda3/envs/robotwin/bin/python"
SERVER_PORT_BASE="${SERVER_PORT_BASE:-29556}"
SEED="${SEED:-0}"
TEST_NUM="${TEST_NUM:-5}"
VIDEO_GUIDANCE_SCALE="${VIDEO_GUIDANCE_SCALE:-5}"
ACTION_GUIDANCE_SCALE="${ACTION_GUIDANCE_SCALE:-1}"
HAZARD_CHECKPOINT="${HAZARD_CHECKPOINT:-/data/syr/train_out/robotwin_hazard_local_8gpu/runs/robotwin_hazard_local_8gpu_20260409_104400/checkpoints/hazard_scheduler_step_9000.pt}"
MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/artifacts/models/lingbot-va-posttrain-robotwin-online}"
LOG_DIR="${REPO_ROOT}/logs/local-jobs"
FIXED_SAVE_ROOT="${FIXED_SAVE_ROOT:-results/compare_fixed_k25}"
HAZARD_SAVE_ROOT="${HAZARD_SAVE_ROOT:-results/compare_hazard_step9000}"

TASKS=(
  adjust_bottle
  beat_block_hammer
  blocks_ranking_rgb
  blocks_ranking_size
  click_alarmclock
  click_bell
  dump_bin_bigbin
  grab_roller
  handover_block
  handover_mic
  hanging_mug
  lift_pot
  move_can_pot
  move_pillbottle_pad
  move_playingcard_away
  move_stapler_pad
  open_laptop
  open_microwave
  pick_diverse_bottles
  pick_dual_bottles
  place_a2b_left
  place_a2b_right
  place_bread_basket
  place_bread_skillet
  place_burger_fries
  place_can_basket
  place_cans_plasticbox
  place_container_plate
  place_dual_shoes
  place_empty_cup
  place_fan
  place_mouse_pad
  place_object_basket
  place_object_scale
  place_object_stand
  place_phone_stand
  place_shoe
  press_stapler
  put_bottles_dustbin
  put_object_cabinet
  rotate_qrcode
  scan_object
  shake_bottle
  shake_bottle_horizontally
  stack_blocks_three
  stack_blocks_two
  stack_bowls_three
  stack_bowls_two
  stamp_seal
  turn_switch
)

prepare_model_root() {
  mkdir -p "$(dirname "${MODEL_ROOT}")"
  mkdir -p "${MODEL_ROOT}"
  ln -sfn /data/syr/models/lingbot-va-posttrain-robotwin/transformer "${MODEL_ROOT}/transformer"
  ln -sfn /data/syr/train_out/robotwin_local_4gpu/infer_model_step_25000/tokenizer "${MODEL_ROOT}/tokenizer"
  ln -sfn /data/syr/train_out/robotwin_local_4gpu/infer_model_step_25000/text_encoder "${MODEL_ROOT}/text_encoder"
  ln -sfn /data/syr/train_out/robotwin_local_4gpu/infer_model_step_25000/vae "${MODEL_ROOT}/vae"
}

prepare_robotwin_assets() {
  (
    cd "${ROBOTWIN_ROOT}"
    printf 'n\n' | PATH="${REPO_ROOT}/.local/miniconda3/envs/robotwin/bin:${PATH}" \
      python script/update_embodiment_config_path.py >/dev/null 2>&1 || true
  )
}

wait_for_port() {
  local port="$1"
  for _ in $(seq 1 120); do
    if python3 - <<PY
import socket
s = socket.socket()
s.settimeout(0.5)
try:
    s.connect(("127.0.0.1", ${port}))
except Exception:
    raise SystemExit(1)
finally:
    s.close()
PY
    then
      return 0
    fi
    sleep 1
  done
  return 1
}

start_server_pool() {
  local mode="$1"
  local save_root="$2"
  local fixed_steps="$3"
  local enable_hazard="$4"
  local server_tag="$5"

  SERVER_PIDS=()
  mkdir -p "${LOG_DIR}"

  for gpu in $(seq 0 7); do
    local port=$((SERVER_PORT_BASE + gpu))
    local log_file="${LOG_DIR}/${server_tag}_server_gpu${gpu}.log"
    : > "${log_file}"

    env CUDA_VISIBLE_DEVICES="${gpu}" \
      LINGBOT_VA_MODEL_PATH="${MODEL_ROOT}" \
      LINGBOT_VA_FORCE_ATTN_MODE=torch \
      "${SERVER_PY}" "${REPO_ROOT}/wan_va/wan_va_server.py" \
      --config-name robotwin \
      --port "${port}" \
      --save_root "visualization/${server_tag}" \
      --guidance-scale "${VIDEO_GUIDANCE_SCALE}" \
      --action-guidance-scale "${ACTION_GUIDANCE_SCALE}" \
      --online-scheduler-mode "${mode}" \
      --fixed-video-steps "${fixed_steps}" \
      --enable-hazard-scheduler-runtime "${enable_hazard}" \
      --hazard-checkpoint "${HAZARD_CHECKPOINT}" \
      --hazard-eta 0.5 \
      --hazard-k-min 3 \
      --hazard-k-max 25 \
      --hazard-feature-source cond \
      --hazard-return-metadata 1 \
      --save-debug-artifacts 0 > "${log_file}" 2>&1 &

    SERVER_PIDS+=("$!")
  done

  for gpu in $(seq 0 7); do
    local port=$((SERVER_PORT_BASE + gpu))
    if ! wait_for_port "${port}"; then
      echo "server on port ${port} failed to become ready" >&2
      stop_server_pool || true
      return 1
    fi
  done
}

stop_server_pool() {
  for pid in "${SERVER_PIDS[@]:-}"; do
    kill "${pid}" >/dev/null 2>&1 || true
  done
  for pid in "${SERVER_PIDS[@]:-}"; do
    wait "${pid}" >/dev/null 2>&1 || true
  done
  SERVER_PIDS=()
}

run_task_batch() {
  local save_root="$1"
  local condition_tag="$2"
  shift 2
  local batch_tasks=("$@")
  local batch_pids=()
  local batch_stamp
  batch_stamp="$(date +%Y%m%d_%H%M%S)"

  for i in "${!batch_tasks[@]}"; do
    local task_name="${batch_tasks[$i]}"
    local gpu_id="${i}"
    local port=$((SERVER_PORT_BASE + i))
    local log_file="${LOG_DIR}/${condition_tag}_${task_name}_${batch_stamp}.log"
    : > "${log_file}"

    env ROBOTWIN_ROOT="${ROBOTWIN_ROOT}" \
      CUDA_VISIBLE_DEVICES="${gpu_id}" \
      PYTHONWARNINGS=ignore::UserWarning \
      XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
      "${CLIENT_PY}" -m evaluation.robotwin.eval_polict_client_openpi \
      --config policy/ACT/deploy_policy.yml \
      --overrides \
      --task_name "${task_name}" \
      --task_config demo_clean \
      --train_config_name 0 \
      --model_name 0 \
      --ckpt_setting "${condition_tag}" \
      --seed "${SEED}" \
      --policy_name ACT \
      --save_root "${save_root}" \
      --video_guidance_scale "${VIDEO_GUIDANCE_SCALE}" \
      --action_guidance_scale "${ACTION_GUIDANCE_SCALE}" \
      --test_num "${TEST_NUM}" \
      --port "${port}" > "${log_file}" 2>&1 &

    batch_pids+=("$!")
  done

  local failed=0
  for pid in "${batch_pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  return "${failed}"
}

run_condition() {
  local mode="$1"
  local save_root="$2"
  local fixed_steps="$3"
  local enable_hazard="$4"
  local condition_tag="$5"

  echo "=== running condition ${condition_tag} ==="
  echo "save_root=${save_root}"
  echo "mode=${mode}"
  echo "fixed_steps=${fixed_steps}"
  echo "enable_hazard=${enable_hazard}"

  start_server_pool "${mode}" "${save_root}" "${fixed_steps}" "${enable_hazard}" "${condition_tag}"
  trap 'stop_server_pool' EXIT

  local total_tasks="${#TASKS[@]}"
  local batch_start=0
  while (( batch_start < total_tasks )); do
    local batch_tasks=("${TASKS[@]:batch_start:8}")
    echo "running ${condition_tag} batch start=${batch_start} tasks=${batch_tasks[*]}"
    if ! run_task_batch "${save_root}" "${condition_tag}" "${batch_tasks[@]}"; then
      echo "client batch failed for ${condition_tag} (start=${batch_start})" >&2
      return 1
    fi
    batch_start=$((batch_start + 8))
  done

  stop_server_pool
  trap - EXIT
}

summarize_results() {
  "${REPO_ROOT}/.local/miniconda3/envs/lingbot-va/bin/python" \
    "${REPO_ROOT}/script/summarize_robotwin_compare.py" \
    --robotwin-root "${ROBOTWIN_ROOT}" \
    --fixed-save-root "${FIXED_SAVE_ROOT}" \
    --hazard-save-root "${HAZARD_SAVE_ROOT}" \
    --seed "${SEED}"
}

main() {
  prepare_model_root
  prepare_robotwin_assets
  mkdir -p "${LOG_DIR}"

  run_condition fixed "${FIXED_SAVE_ROOT}" 25 0 fixed_k25
  run_condition hazard "${HAZARD_SAVE_ROOT}" 25 1 hazard_step9000
  summarize_results
}

main "$@"
