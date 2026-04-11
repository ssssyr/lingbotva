START_PORT=${START_PORT:-29556}
MASTER_PORT=${MASTER_PORT:-29661}
LOG_DIR='./logs'
mkdir -p $LOG_DIR

save_root='./visualization/'
mkdir -p $save_root

VIDEO_GUIDANCE_SCALE=${VIDEO_GUIDANCE_SCALE:-5}
ACTION_GUIDANCE_SCALE=${ACTION_GUIDANCE_SCALE:-1}
ONLINE_SCHEDULER_MODE=${ONLINE_SCHEDULER_MODE:-fixed}
FIXED_VIDEO_STEPS=${FIXED_VIDEO_STEPS:-25}
ENABLE_HAZARD_SCHEDULER_RUNTIME=${ENABLE_HAZARD_SCHEDULER_RUNTIME:-0}
HAZARD_CHECKPOINT_PATH=${HAZARD_CHECKPOINT_PATH:-}
HAZARD_ETA=${HAZARD_ETA:-0.5}
HAZARD_K_MIN=${HAZARD_K_MIN:-3}
HAZARD_K_MAX=${HAZARD_K_MAX:-25}
HAZARD_FEATURE_SOURCE=${HAZARD_FEATURE_SOURCE:-cond}
HAZARD_RETURN_METADATA=${HAZARD_RETURN_METADATA:-1}
SAVE_DEBUG_ARTIFACTS=${SAVE_DEBUG_ARTIFACTS:-1}

batch_time=$(date +%Y%m%d_%H%M%S)


for i in {0..7}; do  
    CURRENT_PORT=$((START_PORT + i))
    CURRENT_MASTER_PORT=$((MASTER_PORT + i))

    LOG_FILE="${LOG_DIR}/server_${i}_${batch_time}.log"
    echo "[Task ${j}] GPU: ${i} | PORT: ${CURRENT_PORT} | MASTER_PORT: ${CURRENT_MASTER_PORT} | Log: ${LOG_FILE}"

    CUDA_VISIBLE_DEVICES=$i  \
    nohup python -m torch.distributed.run \
        --nproc_per_node 1 \
        --master_port $CURRENT_MASTER_PORT \
        wan_va/wan_va_server.py \
        --config-name robotwin \
        --save_root $save_root \
        --port $CURRENT_PORT \
        --guidance-scale $VIDEO_GUIDANCE_SCALE \
        --action-guidance-scale $ACTION_GUIDANCE_SCALE \
        --online-scheduler-mode $ONLINE_SCHEDULER_MODE \
        --fixed-video-steps $FIXED_VIDEO_STEPS \
        --enable-hazard-scheduler-runtime $ENABLE_HAZARD_SCHEDULER_RUNTIME \
        --hazard-checkpoint "$HAZARD_CHECKPOINT_PATH" \
        --hazard-eta $HAZARD_ETA \
        --hazard-k-min $HAZARD_K_MIN \
        --hazard-k-max $HAZARD_K_MAX \
        --hazard-feature-source $HAZARD_FEATURE_SOURCE \
        --hazard-return-metadata $HAZARD_RETURN_METADATA \
        --save-debug-artifacts $SAVE_DEBUG_ARTIFACTS > $LOG_FILE 2>&1 &
    sleep 2;
done

echo "All 8 instances have been launched in the background."
wait
