#!/bin/bash
export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:$LD_LIBRARY_PATH

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG_FILE="${SCRIPT_DIR}/../config/robotwin_client.yaml"

load_config() {
    local config_file="$1"
    eval "$(
        python - "$config_file" <<'PY'
import shlex
import sys
import yaml

config_file = sys.argv[1]
defaults = {
    "save_root": "./results",
    "task_name": "adjust_bottle",
    "render_freq": 0,
    "test_num": 100,
    "policy_name": "ACT",
    "task_config": "demo_clean",
    "train_config_name": 0,
    "model_name": 0,
    "seed": 0,
    "port": 29056,
    "video_guidance_scale": 5,
    "action_guidance_scale": 1,
}
with open(config_file, "r", encoding="utf-8") as f:
    loaded = yaml.safe_load(f) or {}
if not isinstance(loaded, dict):
    raise SystemExit(f"Config must be a mapping: {config_file}")
defaults.update(loaded)
for key, value in defaults.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
    )"
}

if [[ "${1:-}" == "--config" ]]; then
    if [[ -z "${2:-}" ]]; then
        echo "missing config path after --config" >&2
        exit 1
    fi
    load_config "$2"
elif [[ $# -eq 0 ]]; then
    load_config "$DEFAULT_CONFIG_FILE"
else
    save_root=${1:-'./results'}
    task_name=${2:-"adjust_bottle"}
    render_freq=${3:-0}
    test_num=${4:-100}
    policy_name=ACT
    task_config=demo_clean
    train_config_name=0
    model_name=0
    seed=0
    port=29056
    video_guidance_scale=5
    action_guidance_scale=1
fi

echo "client config:"
echo "  task_name=${task_name}"
echo "  save_root=${save_root}"
echo "  test_num=${test_num}"
echo "  render_freq=${render_freq}"
echo "  task_config=${task_config}"
echo "  port=${port}"

ALL_TASKS=(
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

run_one_task() {
    local current_task="$1"
    echo "----------------------------------------"
    echo "running task=${current_task} test_num=${test_num}"
    PYTHONWARNINGS=ignore::UserWarning \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python -m evaluation.robotwin.eval_polict_client_openpi --config policy/$policy_name/deploy_policy.yml \
        --overrides \
        --task_name ${current_task} \
        --task_config ${task_config} \
        --train_config_name ${train_config_name} \
        --model_name ${model_name} \
        --ckpt_setting ${model_name} \
        --seed ${seed} \
        --policy_name ${policy_name} \
        --save_root ${save_root} \
        --video_guidance_scale ${video_guidance_scale} \
        --action_guidance_scale ${action_guidance_scale} \
        --test_num ${test_num} \
        --render_freq ${render_freq} \
        --port ${port}
}

if [[ "${task_name}" == "__all__" ]]; then
    echo "mode=all_tasks"
    for current_task in "${ALL_TASKS[@]}"; do
        run_one_task "${current_task}" || exit $?
    done
else
    run_one_task "${task_name}"
fi
