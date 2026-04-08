# Hazard Scheduler 离线评估

## 概述

离线评估脚本用于在验证集上评估训练好的 Hazard scheduler，并与固定步数的 baseline 进行对比。

## 使用方法

### 基本用法

```bash
./script/run_hazard_eval.sh <checkpoint_path> [config_path] [num_samples]
```

### 参数说明

- `checkpoint_path`（必需）：训练好的 scheduler 检查点路径
  - 例如：`train_out/robotwin_hazard_4gpu/hazard_scheduler_step_1000.pt`
- `config_path`（可选）：训练配置文件路径
  - 默认：`train_profiles/robotwin_hazard_4gpu.yaml`
- `num_samples`（可选）：评估样本数量
  - 默认：100
  - 设置为空或不传则评估整个验证集

### 示例

```bash
# 评估 step 1000 的检查点，使用 100 个样本
./script/run_hazard_eval.sh train_out/robotwin_hazard_4gpu/hazard_scheduler_step_1000.pt

# 评估整个验证集
./script/run_hazard_eval.sh train_out/robotwin_hazard_4gpu/hazard_scheduler_final.pt train_profiles/robotwin_hazard_4gpu.yaml

# 评估 500 个样本
./script/run_hazard_eval.sh train_out/robotwin_hazard_4gpu/hazard_scheduler_step_5000.pt train_profiles/robotwin_hazard_4gpu.yaml 500
```

## 评估指标

脚本会对比以下方法：

1. **Hazard Scheduler**（自适应步数）
2. **Baseline K=5**（固定 5 步）
3. **Baseline K=10**（固定 10 步）
4. **Baseline K=15**（固定 15 步）
5. **Baseline K=20**（固定 20 步）

### 输出指标

对每个方法，输出以下指标：

- **Reward**：奖励值（越大越好）
  - `Reward = -L_act - λ * video_steps`
- **Video Steps**：视频去噪步数（越少越好）
- **Action Loss**：动作预测损失（越小越好）

### 输出格式

#### 终端输出示例

```
================================================================================
EVALUATION SUMMARY
================================================================================

Hazard Scheduler:
  Reward:       -0.1234 ± 0.0056
  Video Steps:  7.32 ± 2.14
  Action Loss:  0.1234 ± 0.0056

Baselines:
  K=5:
    Reward:       -0.1456 ± 0.0067
    Action Loss:  0.1456 ± 0.0067
  K=10:
    Reward:       -0.1345 ± 0.0061
    Action Loss:  0.1345 ± 0.0061
  K=15:
    Reward:       -0.1567 ± 0.0072
    Action Loss:  0.1567 ± 0.0072
  K=20:
    Reward:       -0.1789 ± 0.0083
    Action Loss:  0.1789 ± 0.0083

Improvement over best baseline (baseline_K10):
  Reward improvement: +8.25%
================================================================================
```

#### JSON 输出

结果会保存到 `<checkpoint_dir>/eval_results/eval_<timestamp>.json`：

```json
{
  "hazard": {
    "reward_mean": -0.1234,
    "reward_std": 0.0056,
    "video_steps_mean": 7.32,
    "video_steps_std": 2.14,
    "action_loss_mean": 0.1234,
    "action_loss_std": 0.0056
  },
  "baseline_K5": {
    "reward_mean": -0.1456,
    "reward_std": 0.0067,
    "video_steps_mean": 5.0,
    "video_steps_std": 0.0,
    "action_loss_mean": 0.1456,
    "action_loss_std": 0.0067
  },
  ...
}
```

## 评估流程

1. **加载模型**：加载 transformer 和 scheduler_head
2. **加载检查点**：恢复训练好的 scheduler 权重
3. **准备数据**：加载验证集
4. **执行 rollout**：
   - 对每个样本，执行 Hazard rollout（自适应步数）
   - 对每个样本，执行固定 K 的 baseline rollout
5. **计算指标**：计算 reward、video_steps、action_loss
6. **聚合结果**：计算均值和标准差
7. **保存结果**：输出到终端和 JSON 文件

## 注意事项

1. **单 GPU 评估**：脚本使用单 GPU（CUDA:0），不支持分布式评估
2. **批大小为 1**：每次评估一个样本，确保结果准确
3. **确定性评估**：使用固定随机种子确保可复现性
4. **内存占用**：评估时会缓存视频帧，注意显存使用

## 故障排除

### 问题：找不到验证集

**解决方案**：确保数据集路径正确，并且包含 `val` split：

```bash
ls /path/to/dataset/val/
```

### 问题：显存不足

**解决方案**：减少 `num_samples` 或使用更小的 `window_size`：

```bash
./script/run_hazard_eval.sh checkpoint.pt config.yaml 50
```

### 问题：评估速度慢

**解决方案**：增加 DataLoader 的 `num_workers`，修改 `hazard_eval.py` 中的：

```python
val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=8,  # 增加到 8
    pin_memory=True,
)
```

## 下一步

评估完成后，根据结果决定：

1. **如果 Hazard 表现好**：继续训练更多步数，或集成到推理流程（任务 #3）
2. **如果 Hazard 表现差**：调整超参数（lambda_cost、lambda_kl）或重新训练
3. **如果步数分布不合理**：检查 Hazard 曲线设计，调整 K_max
