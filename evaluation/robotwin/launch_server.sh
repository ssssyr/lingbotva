START_PORT=${START_PORT:-29056}

save_root='visualization/'
mkdir -p $save_root

python wan_va/wan_va_server.py \
    --config-name robotwin \
    --port $START_PORT \
    --save_root $save_root

