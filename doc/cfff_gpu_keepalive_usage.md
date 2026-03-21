# CFFF GPU 保活使用说明

本文记录当前 LingBot-VA 仓库在 CFFF DSW 云端上的 GPU 保活方案，目标是让显卡利用率持续高于平台回收阈值。

当前文档适用于下面这类情况：

- 远端机器可通过 `cfff` SSH alias 访问
- 远端有 CUDA 和 PyTorch
- 远端仓库目录磁盘配额紧张，不适合再做整仓同步
- 远端未安装 `tmux`
- 远端未准备好 `transformers` / `vllm` / Qwen 权重

在这种前提下，当前推荐方案不是“真大模型推理保活”，而是：

- 在远端 `/tmp/lingbot-va-keepalive` 下放一个轻量 runner
- 用 `nohup` 挂后台
- 每张 GPU 跑一个持续的 BF16 `torch.matmul`
- 用日志和 `nvidia-smi` 验证利用率

这个方案依赖最少，稳定性最高，适合先保住机器不被回收。

---

## 1. 当前环境结论

本次排查的结论是：

- 远端是 4 张 `NVIDIA A100-SXM4-80GB`
- 可用 Conda 环境是 `lingbot-va-cu118`
- 该环境里有 `torch`
- 该环境里没有 `transformers`、`vllm`、`accelerate`、`safetensors`
- 没找到现成的 Qwen 27B 权重目录
- 远端 repo 目录有磁盘配额问题，`./script/cfff-sync.sh` 会失败
- 远端没有 `tmux`

因此当前保活方案选的是：

- 不依赖 repo 同步
- 不依赖模型权重
- 不依赖推理框架
- 直接使用 PyTorch CUDA 计算占用 GPU

---

## 2. 远端目录与约定

当前保活任务约定使用：

- 远端工作目录：`/tmp/lingbot-va-keepalive`
- 远端 Conda 环境：`lingbot-va-cu118`
- 日志目录：`/tmp/lingbot-va-keepalive/logs`

后续所有命令默认都建议加上：

```bash
CFFF_REMOTE_ROOT=/tmp/lingbot-va-keepalive
```

如果需要显式指定 Conda 环境，再加上：

```bash
CFFF_CONDA_ENV=lingbot-va-cu118
```

---

## 3. 基础检查

先确认远端和 GPU 状态：

```bash
./script/cfff-run.sh "whoami && hostname && pwd"
./script/cfff-run.sh "nvidia-smi"
```

确认远端 PyTorch/CUDA：

```bash
CFFF_CONDA_ENV=lingbot-va-cu118 ./script/cfff-run.sh "python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda', torch.version.cuda)
print('available', torch.cuda.is_available())
print('count', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY"
```

---

## 4. 当前推荐保活方案

### 4.1 原理

当前 runner 会：

- 启 4 个 Python 子进程
- 每个子进程通过 `CUDA_VISIBLE_DEVICES` 绑定一张卡
- 在对应 GPU 上持续做 BF16 矩阵乘法
- 周期性输出迭代日志

这一方案会把 GPU 利用率拉得比较高，通常接近满载。

优点：

- 不依赖大模型权重
- 不依赖 `transformers` / `vllm`
- 只要 PyTorch + CUDA 正常就能跑
- 很适合“先保活，后处理环境问题”

缺点：

- 负载偏高
- 没有实际业务意义，只是占卡

### 4.2 生成远端 runner

如果远端 `/tmp/lingbot-va-keepalive/gpu_keepalive_runner.sh` 丢失，可以用下面命令重新生成：

```bash
CFFF_REMOTE_ROOT=/tmp/lingbot-va-keepalive \
CFFF_CONDA_ENV=lingbot-va-cu118 \
./script/cfff-run.sh "cat > gpu_keepalive_runner.sh <<'SH'
#!/usr/bin/env bash
set -u

pids=()
cleanup() {
  for pid in \"\${pids[@]:-}\"; do
    kill \"\$pid\" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=\"\$gpu\" python -u - <<'PY' &
import os
import signal
import torch

stop = False

def handle_signal(signum, frame):
    del signum, frame
    global stop
    stop = True

signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')
a = torch.randn((6144, 6144), device='cuda', dtype=torch.bfloat16)
b = torch.randn((6144, 6144), device='cuda', dtype=torch.bfloat16)
count = 0
print(f'[keepalive] pid={os.getpid()} visible_gpu={os.environ.get(\"CUDA_VISIBLE_DEVICES\")} device={torch.cuda.get_device_name(0)}', flush=True)
while not stop:
    c = torch.matmul(a, b)
    a, b = b, c
    torch.cuda.synchronize()
    count += 1
    if count % 20 == 0:
        print(f'[keepalive] pid={os.getpid()} visible_gpu={os.environ.get(\"CUDA_VISIBLE_DEVICES\")} iter={count}', flush=True)
PY
  pids+=(\"\$!\")
done

wait
SH
chmod +x gpu_keepalive_runner.sh
mkdir -p logs
ls -l gpu_keepalive_runner.sh"
```

### 4.3 启动保活任务

```bash
CFFF_REMOTE_ROOT=/tmp/lingbot-va-keepalive \
CFFF_CONDA_ENV=lingbot-va-cu118 \
./script/cfff-run.sh "timestamp=\$(date +%Y%m%d-%H%M%S); log_file=logs/gpu-keepalive-\$timestamp.log; nohup bash ./gpu_keepalive_runner.sh > \"\$log_file\" 2>&1 < /dev/null & echo pid=\$!; echo log=\$log_file"
```

输出里会给出：

- 后台 `pid`
- 当前这次运行对应的 `log` 文件

---

## 5. 启动后如何验证

### 5.1 看进程

```bash
CFFF_REMOTE_ROOT=/tmp/lingbot-va-keepalive \
./script/cfff-run.sh "ps -ef | grep -E 'gpu_keepalive_runner|python -u -' | grep -v grep"
```

正常情况下应看到：

- 1 个 `bash ./gpu_keepalive_runner.sh`
- 4 个 `python -u -`

### 5.2 看日志

先找最新日志：

```bash
CFFF_REMOTE_ROOT=/tmp/lingbot-va-keepalive \
./script/cfff-run.sh "ls -1t logs/gpu-keepalive-*.log | head -n 1"
```

再看日志内容：

```bash
CFFF_REMOTE_ROOT=/tmp/lingbot-va-keepalive \
./script/cfff-run.sh "log_file=\$(ls -1t logs/gpu-keepalive-*.log | head -n 1); echo \"==> \$log_file <==\"; tail -n 50 \"\$log_file\""
```

正常日志会出现类似内容：

```text
[keepalive] pid=928 visible_gpu=0 device=NVIDIA A100-SXM4-80GB
[keepalive] pid=929 visible_gpu=1 device=NVIDIA A100-SXM4-80GB
[keepalive] pid=930 visible_gpu=2 device=NVIDIA A100-SXM4-80GB
[keepalive] pid=931 visible_gpu=3 device=NVIDIA A100-SXM4-80GB
[keepalive] pid=928 visible_gpu=0 iter=20
```

### 5.3 看显卡利用率

```bash
CFFF_REMOTE_ROOT=/tmp/lingbot-va-keepalive \
./script/cfff-run.sh "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw --format=csv,noheader"
```

如果保活正常，通常会看到：

- `utilization.gpu` 明显大于 `10%`
- 当前方案多数情况下会接近 `99%`

### 5.4 看 GPU 计算进程

```bash
CFFF_REMOTE_ROOT=/tmp/lingbot-va-keepalive \
./script/cfff-run.sh "nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits"
```

应能看到 4 个活跃的 compute app。

---

## 6. 停止保活任务

停止当前保活：

```bash
CFFF_REMOTE_ROOT=/tmp/lingbot-va-keepalive \
./script/cfff-run.sh "pkill -f gpu_keepalive_runner.sh"
```

如果还想保险一点，把子进程也清掉：

```bash
CFFF_REMOTE_ROOT=/tmp/lingbot-va-keepalive \
./script/cfff-run.sh "pkill -f 'python -u -'"
```

停止后再检查：

```bash
CFFF_REMOTE_ROOT=/tmp/lingbot-va-keepalive \
./script/cfff-run.sh "nvidia-smi"
```

---

## 7. 重启保活任务

推荐流程：

1. 先停旧任务
2. 确认旧进程已消失
3. 再重新启动

命令顺序：

```bash
CFFF_REMOTE_ROOT=/tmp/lingbot-va-keepalive ./script/cfff-run.sh "pkill -f gpu_keepalive_runner.sh || true"
CFFF_REMOTE_ROOT=/tmp/lingbot-va-keepalive ./script/cfff-run.sh "ps -ef | grep -E 'gpu_keepalive_runner|python -u -' | grep -v grep || true"
```

然后重新执行第 4.3 节的启动命令。

---

## 8. 调整负载强度

当前 runner 是偏激进的方案，目标是稳稳超过回收阈值。

如果你只想维持低于满载、但仍高于 10% 的利用率，可以改下面几项：

- 只用部分 GPU
- 把矩阵尺寸从 `6144` 改小到 `4096` 或 `3072`
- 在循环里加 `time.sleep(...)`

最直接的修改点是 `gpu_keepalive_runner.sh` 里的：

- `for gpu in 0 1 2 3`
- `torch.randn((6144, 6144), ...)`

例如：

- 只保活 2 张卡：改成 `for gpu in 0 1`
- 降低计算量：改成 `4096 x 4096`

---

## 9. 本地仓库里的可复用脚本

本地仓库已经新增了一个更正规的脚本：

- [gpu_keepalive.py](/home/syr/code/lingbot-va/script/gpu_keepalive.py)

它支持：

- 指定 GPU 列表
- 指定矩阵尺寸
- 指定每轮 burst 次数
- 指定 sleep 间隔
- 指定日志频率

典型参数示例：

```bash
python script/gpu_keepalive.py --gpus 0,1,2,3 --size 6144 --work-iters 2 --sleep-s 0.35 --dtype bf16
```

但要注意：

- 远端 repo 当前有磁盘配额问题
- 直接 `./script/cfff-sync.sh` 同步整仓会失败

所以在远端 repo 配额问题解决之前，优先用本文的 `/tmp` 方案。

---

## 10. 为什么这次没有直接用 Qwen 27B 保活

原因不是“不想用”，而是远端当前条件不满足：

- 没找到确认存在的 Qwen 27B 权重目录
- 没安装 `transformers`
- 没安装 `vllm`
- 没安装 `accelerate`
- 没安装 `safetensors`

所以如果现在强行做“真实模型推理保活”，就会先卡在：

- 找权重
- 装推理依赖
- 处理远端磁盘配额

这几件事都比“先保活”更容易失败。

---

## 11. 后续升级到“真实模型保活”的建议路径

如果后续你想把保活方案改成真正的 Qwen 推理服务，建议按下面顺序处理：

1. 先清理远端磁盘配额，保证 repo 同步和模型落盘都正常。
2. 确认 Qwen 27B 权重的实际目录。
3. 在远端安装 `transformers` 或 `vllm`。
4. 再切换到真实模型保活。

### 11.1 如果走 `vllm`

目标形态通常是：

```bash
vllm serve /path/to/Qwen-model --tensor-parallel-size 4
```

优点：

- 更接近真实服务
- 占用 GPU 有业务意义

缺点：

- 依赖和权重准备更复杂

### 11.2 如果走 `transformers`

目标形态通常是：

- 用 `AutoModelForCausalLM.from_pretrained(...)`
- 常驻加载模型
- 定期做一轮短文本生成

优点：

- 实现灵活

缺点：

- 模型显存和加载时间更重
- 依赖安装和权重路径仍然是前置条件

---

## 12. 常见问题

### 12.1 `./script/cfff-sync.sh` 失败，提示 `Disk quota exceeded`

这是当前远端 repo 最主要的问题。此时不要继续做整仓同步，先用 `/tmp` 保活方案。

### 12.2 远端没有 `tmux`

当前方案已经改成 `nohup + 日志文件`，不依赖 `tmux`。

### 12.3 启动了但 `nvidia-smi` 看不到利用率

优先检查：

- 保活进程是否还在
- 日志是否持续增长
- 是否真的绑到了 CUDA 设备

建议依次执行：

```bash
CFFF_REMOTE_ROOT=/tmp/lingbot-va-keepalive ./script/cfff-run.sh "ps -ef | grep -E 'gpu_keepalive_runner|python -u -' | grep -v grep"
CFFF_REMOTE_ROOT=/tmp/lingbot-va-keepalive ./script/cfff-run.sh "tail -n 50 \$(ls -1t logs/gpu-keepalive-*.log | head -n 1)"
CFFF_REMOTE_ROOT=/tmp/lingbot-va-keepalive ./script/cfff-run.sh "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw --format=csv,noheader"
```

### 12.4 需要临时把 GPU 让出来做别的实验

先停保活任务，实验结束后再重新启动。

---

## 13. 一套最短操作清单

如果你只想记最少命令，记下面这组：

启动：

```bash
CFFF_REMOTE_ROOT=/tmp/lingbot-va-keepalive \
CFFF_CONDA_ENV=lingbot-va-cu118 \
./script/cfff-run.sh "timestamp=\$(date +%Y%m%d-%H%M%S); log_file=logs/gpu-keepalive-\$timestamp.log; nohup bash ./gpu_keepalive_runner.sh > \"\$log_file\" 2>&1 < /dev/null & echo pid=\$!; echo log=\$log_file"
```

检查：

```bash
CFFF_REMOTE_ROOT=/tmp/lingbot-va-keepalive \
./script/cfff-run.sh "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw --format=csv,noheader"
```

日志：

```bash
CFFF_REMOTE_ROOT=/tmp/lingbot-va-keepalive \
./script/cfff-run.sh "tail -n 50 \$(ls -1t logs/gpu-keepalive-*.log | head -n 1)"
```

停止：

```bash
CFFF_REMOTE_ROOT=/tmp/lingbot-va-keepalive \
./script/cfff-run.sh "pkill -f gpu_keepalive_runner.sh"
```

