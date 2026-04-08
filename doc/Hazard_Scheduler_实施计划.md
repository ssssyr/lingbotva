# Hazard Scheduler 实施计划

## 总览

本文档提供 Hazard Scheduler V1 的详细代码实施计划，按照 3 个阶段、9 个任务组织。

**总时间估算**：3-4 周
- 阶段 1（基础设施）：1 周
- 阶段 2（训练框架）：1-2 周
- 阶段 3（评估验证）：1 周

---

## 阶段 1：基础设施搭建（1 周）

### 任务 1：实现 HazardSchedulerHead 模块

**文件**：`wan_va/modules/hazard_scheduler.py`

**功能**：
- 输入：video hidden feature (pooled) + step_id + 可选上下文
- 输出：`delta_H_k`（非负 Hazard 增量）
- 保证单调性：使用 `softplus` 激活

**核心代码结构**：
```python
class HazardSchedulerHead(nn.Module):
    def __init__(self, hidden_dim, output_dim=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # +1 for step_id
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        # 初始化输出层接近 0，让初始 delta_H_k 较小
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, video_feature, step_id):
        """
        Args:
            video_feature: [B, D] pooled video hidden feature
            step_id: [B] or [B, 1], normalized to [0, 1]
        Returns:
            delta_H_k: [B, 1], non-negative Hazard increment
        """
        if step_id.dim() == 1:
            step_id = step_id.unsqueeze(-1)

        x = torch.cat([video_feature, step_id], dim=-1)
        logit = self.mlp(x)
        delta_H_k = F.softplus(logit)  # 保证非负
        return delta_H_k
```

**验收标准**：
- [ ] 输出 `delta_H_k` 始终非负
- [ ] 支持 batch 输入
- [ ] 初始化后输出接近 0（避免过早停止）

---

### 任务 2：修改 model.py 暴露中间特征

**文件**：`wan_va/modules/model.py`

**修改点**：
在 `WanTransformer3DModel.forward()` 中添加 `return_video_features` 参数

**核心代码**：
```python
def forward(
    self,
    noisy_latents,
    timesteps,
    text_emb,
    grid_id,
    cache_name=None,
    return_video_features=False,  # 新增参数
    **kwargs,
):
    # ... 现有代码 ...

    # 在 video branch 处理后，添加 pooling
    if return_video_features:
        # 假设 video tokens 在前半部分
        video_tokens = hidden_states[:, :num_video_tokens]
        video_pooled = video_tokens.mean(dim=1)  # [B, D]
        return model_output, video_pooled

    return model_output
```

**验收标准**：
- [ ] 不影响现有训练流程（默认 `return_video_features=False`）
- [ ] 能正确返回 pooled video feature
- [ ] 特征维度与 hidden_dim 一致

---

### 任务 3：实现 hazard_runtime.py

**文件**：`wan_va/utils/hazard_runtime.py`

**功能**：
封装完整的 rollout 流程，复用 `CutoffScanner` 的 handoff 语义

**核心类**：
```python
class HazardRolloutRunner:
    def __init__(self, transformer, scheduler_head, config, device, dtype):
        self.transformer = transformer
        self.scheduler_head = scheduler_head
        self.config = config
        self.device = device
        self.dtype = dtype

        # 复用 CutoffScanner 的基础设施
        self.video_scheduler = FlowMatchScheduler(...)
        self.action_scheduler = FlowMatchScheduler(...)

        self.K_max = config.num_inference_steps  # 25
        self.K_min = config.get('hazard_k_min', 3)  # 最小步数

    def rollout_single_sample(self, video_noise, action_noise, text_emb, gt_action):
        """
        执行单次 rollout，返回 trajectory 和 reward

        Returns:
            trajectory: List[Dict] 包含每步的 state, delta_H_k, h_k, log_prob
            final_action: 预测的动作
            reward: R = -L_act - lambda_cost * video_steps
        """
        trajectory = []
        latents = video_noise.clone()
        H_cumulative = 0.0

        for step_idx in range(self.K_max):
            # 1. 运行 video branch 一步
            t = self.video_timesteps[step_idx]
            model_output, video_pooled = self.transformer(
                latents, t, text_emb, ...,
                return_video_features=True
            )
            latents = self.video_scheduler.step(model_output, t, latents)

            # 2. 计算 Hazard
            step_id = torch.tensor([step_idx / self.K_max], device=self.device)
            delta_H_k = self.scheduler_head(video_pooled, step_id)
            H_cumulative += delta_H_k.item()

            # 3. 计算停止概率
            h_k = 1 - torch.exp(-delta_H_k)

            # 4. 采样 stop 决策
            stop = torch.bernoulli(h_k).bool().item()

            # 5. 记录 log_prob
            if stop:
                log_prob = torch.log(h_k + 1e-8)
            else:
                log_prob = torch.log(1 - h_k + 1e-8)

            # 6. 存储到 trajectory
            trajectory.append({
                'step_idx': step_idx,
                'delta_H_k': delta_H_k,
                'h_k': h_k,
                'log_prob': log_prob,
                'stop': stop,
            })

            # 7. 如果停止，handoff 到动作分支
            if stop and step_idx >= self.K_min:
                break

        # 8. 生成动作（复用 CutoffScanner 的 action decode 逻辑）
        final_action = self._decode_action(latents, action_noise, text_emb)

        # 9. 计算 reward
        L_act = F.mse_loss(final_action, gt_action)
        video_steps = len(trajectory)
        reward = -L_act.item() - self.config.lambda_cost * video_steps

        return trajectory, final_action, reward

    def _decode_action(self, video_latents, action_noise, text_emb):
        """复用 CutoffScanner 的动作生成逻辑"""
        # 刷新 video cache
        # 运行 action branch
        # 返回预测动作
        pass
```

**验收标准**：
- [ ] 能完整运行 video rollout -> stop -> handoff -> action decode
- [ ] trajectory 记录完整（包含 log_prob）
- [ ] reward 计算正确

---

### 任务 4：编写单元测试

**文件**：
- `tests/test_hazard_scheduler.py`
- `tests/test_hazard_runtime.py`

**测试内容**：

**test_hazard_scheduler.py**：
```python
def test_hazard_monotonicity():
    """测试 Hazard 单调性"""
    scheduler_head = HazardSchedulerHead(hidden_dim=768)
    video_feature = torch.randn(4, 768)

    H_cumulative = 0.0
    for step_idx in range(10):
        step_id = torch.tensor([step_idx / 10.0]).expand(4, 1)
        delta_H_k = scheduler_head(video_feature, step_id)

        # 检查非负
        assert (delta_H_k >= 0).all()

        # 检查累积单调
        H_cumulative += delta_H_k.mean().item()
        assert H_cumulative >= 0

def test_stop_probability_range():
    """测试停止概率在 [0, 1] 范围"""
    delta_H_k = torch.tensor([0.1, 0.5, 1.0, 2.0])
    h_k = 1 - torch.exp(-delta_H_k)

    assert (h_k >= 0).all() and (h_k <= 1).all()
```

**test_hazard_runtime.py**：
```python
def test_rollout_single_sample():
    """测试完整 rollout 流程"""
    runner = HazardRolloutRunner(...)

    video_noise = torch.randn(1, C, F, H, W)
    action_noise = torch.randn(1, A, F, N, D)
    text_emb = torch.randn(1, L, D)
    gt_action = torch.randn(1, A, F, N, D)

    trajectory, final_action, reward = runner.rollout_single_sample(
        video_noise, action_noise, text_emb, gt_action
    )

    # 检查 trajectory 完整性
    assert len(trajectory) > 0
    assert all('log_prob' in step for step in trajectory)

    # 检查 reward 合理性
    assert isinstance(reward, float)
```

**验收标准**：
- [ ] 所有测试通过
- [ ] 覆盖核心功能（单调性、概率范围、rollout 流程）

---

## 阶段 2：训练框架实现（1-2 周）

### 任务 5：实现 hazard_trainer.py

**文件**：`wan_va/hazard_trainer.py`

**功能**：
实现 REINFORCE 训练循环，包括 rollout、reward 计算、baseline、梯度更新

**核心代码结构**：
```python
class HazardTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(f"cuda:{config.local_rank}")
        self.dtype = config.param_dtype

        # 加载预训练模型
        self.transformer = load_transformer(config.model_path)

        # 创建 scheduler head
        self.scheduler_head = HazardSchedulerHead(
            hidden_dim=config.hidden_dim
        ).to(self.device)

        # 冻结 backbone，只训练 scheduler head
        for param in self.transformer.parameters():
            param.requires_grad = False

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.scheduler_head.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # Rollout runner
        self.runner = HazardRolloutRunner(
            self.transformer,
            self.scheduler_head,
            config,
            self.device,
            self.dtype
        )

        # Reference baseline (固定 K=10)
        self.reference_K = config.get('reference_K', 10)

        # Reward 统计
        self.reward_mean = 0.0
        self.reward_std = 1.0

    def train_step(self, batch):
        """
        单步训练

        Args:
            batch: Dict 包含 video_noise, action_noise, text_emb, gt_action

        Returns:
            metrics: Dict 包含 loss, reward, video_steps 等
        """
        video_noise = batch['video_noise'].to(self.device)
        action_noise = batch['action_noise'].to(self.device)
        text_emb = batch['text_emb'].to(self.device)
        gt_action = batch['gt_action'].to(self.device)

        batch_size = video_noise.shape[0]

        # 1. Rollout sampled policy
        trajectories = []
        rewards = []
        video_steps_list = []

        for i in range(batch_size):
            trajectory, final_action, reward = self.runner.rollout_single_sample(
                video_noise[i:i+1],
                action_noise[i:i+1],
                text_emb[i:i+1],
                gt_action[i:i+1],
            )
            trajectories.append(trajectory)
            rewards.append(reward)
            video_steps_list.append(len(trajectory))

        # 2. Rollout reference policy (固定 K=10)
        rewards_ref = []
        for i in range(batch_size):
            _, _, reward_ref = self.runner.rollout_fixed_K(
                video_noise[i:i+1],
                action_noise[i:i+1],
                text_emb[i:i+1],
                gt_action[i:i+1],
                K=self.reference_K,
            )
            rewards_ref.append(reward_ref)

        # 3. 计算 advantage
        rewards = torch.tensor(rewards, device=self.device)
        rewards_ref = torch.tensor(rewards_ref, device=self.device)
        advantages = rewards - rewards_ref

        # 4. Advantage normalization (可选)
        if self.config.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 5. 计算 REINFORCE loss
        loss = 0.0
        for i, trajectory in enumerate(trajectories):
            for step in trajectory:
                log_prob = step['log_prob']
                loss -= log_prob * advantages[i]  # REINFORCE

        loss = loss / batch_size

        # 6. 添加 KL 正则（可选）
        if self.config.beta_kl > 0:
            kl_loss = self._compute_kl_regularization(trajectories)
            loss += self.config.beta_kl * kl_loss

        # 7. 添加 entropy bonus（可选）
        if self.config.beta_entropy > 0:
            entropy = self._compute_entropy(trajectories)
            loss -= self.config.beta_entropy * entropy

        # 8. 反向传播
        self.optimizer.zero_grad()
        loss.backward()

        # 9. 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.scheduler_head.parameters(),
            max_norm=self.config.max_grad_norm
        )

        self.optimizer.step()

        # 10. 返回 metrics
        metrics = {
            'loss': loss.item(),
            'reward_mean': rewards.mean().item(),
            'reward_std': rewards.std().item(),
            'advantage_mean': advantages.mean().item(),
            'video_steps_mean': sum(video_steps_list) / batch_size,
        }

        return metrics

    def _compute_kl_regularization(self, trajectories):
        """计算 KL(pi_theta || pi_ref)"""
        # 使用固定 Hazard 启发式作为 reference
        # KL = sum_k [h_k * log(h_k / h_ref_k) + (1-h_k) * log((1-h_k) / (1-h_ref_k))]
        pass

    def _compute_entropy(self, trajectories):
        """计算策略熵 H(pi) = -sum_k [h_k * log(h_k) + (1-h_k) * log(1-h_k)]"""
        entropy = 0.0
        for trajectory in trajectories:
            for step in trajectory:
                h_k = step['h_k']
                entropy -= h_k * torch.log(h_k + 1e-8)
                entropy -= (1 - h_k) * torch.log(1 - h_k + 1e-8)
        return entropy / len(trajectories)
```

**验收标准**：
- [ ] 训练循环能稳定运行
- [ ] loss 能正常反向传播
- [ ] 只有 scheduler_head 参数更新

---

### 任务 6：创建训练配置文件

**文件**：`train_profiles/hazard_stop_reward_v1.yaml`

**内容**：
```yaml
# Hazard Scheduler V1 训练配置

# 模型配置
model_path: "/path/to/pretrained/model"
hidden_dim: 768

# Scheduler 配置
num_inference_steps: 25  # K_max
hazard_k_min: 3  # 最小停止步数

# 训练配置
lr: 1.0e-5  # 低学习率保证稳定
weight_decay: 0.01
batch_size: 16
num_epochs: 10
max_grad_norm: 1.0

# Reward 配置
lambda_cost: 0.1  # 步数惩罚系数

# Baseline 配置
reference_K: 10  # 固定步数 baseline
normalize_advantage: true

# 正则化配置
beta_kl: 0.01  # KL 正则系数
beta_entropy: 0.001  # Entropy bonus 系数

# 数据配置
dataset_path: "/path/to/robotwin/dataset"
num_workers: 4

# 分布式配置
world_size: 4
local_rank: 0

# 日志配置
log_interval: 10
save_interval: 500
eval_interval: 100

# WandB 配置
enable_wandb: true
wandb_project: "hazard_scheduler_v1"
wandb_run_name: "stop_only_reinforce"
```

**验收标准**：
- [ ] 配置文件格式正确
- [ ] 所有必需参数都有默认值

---

### 任务 7：实现训练启动脚本

**文件**：`script/run_hazard_stop_reward.sh`

**内容**：
```bash
#!/bin/bash

# Hazard Scheduler V1 训练启动脚本

set -e

# 配置文件路径
CONFIG_PATH="train_profiles/hazard_stop_reward_v1.yaml"

# 分布式配置
export WORLD_SIZE=4
export MASTER_ADDR="localhost"
export MASTER_PORT=29500

# 启动训练
torchrun \
    --nproc_per_node=4 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    wan_va/hazard_trainer.py \
    --config $CONFIG_PATH \
    "$@"
```

**验收标准**：
- [ ] 能成功启动分布式训练
- [ ] 日志输出正常

---

## 阶段 3：评估与验证（1 周）

### 任务 8：实现离线评估脚本

**文件**：`script/eval_hazard_offline.py`

**功能**：
对比不同策略的离线性能

**核心代码**：
```python
def evaluate_policy(runner, dataloader, policy_type='learned'):
    """
    评估策略性能

    Args:
        policy_type: 'learned' | 'fixed_K10' | 'fixed_K25'

    Returns:
        metrics: Dict 包含 avg_video_steps, avg_action_loss, stop_distribution
    """
    total_video_steps = 0
    total_action_loss = 0.0
    stop_distribution = [0] * 25
    num_samples = 0

    for batch in tqdm(dataloader):
        video_noise = batch['video_noise']
        action_noise = batch['action_noise']
        text_emb = batch['text_emb']
        gt_action = batch['gt_action']

        for i in range(video_noise.shape[0]):
            if policy_type == 'learned':
                trajectory, final_action, reward = runner.rollout_single_sample(...)
                video_steps = len(trajectory)
                stop_idx = trajectory[-1]['step_idx']
            elif policy_type.startswith('fixed_K'):
                K = int(policy_type.split('K')[1])
                trajectory, final_action, reward = runner.rollout_fixed_K(..., K=K)
                video_steps = K
                stop_idx = K - 1

            action_loss = F.mse_loss(final_action, gt_action[i:i+1])

            total_video_steps += video_steps
            total_action_loss += action_loss.item()
            stop_distribution[stop_idx] += 1
            num_samples += 1

    metrics = {
        'avg_video_steps': total_video_steps / num_samples,
        'avg_action_loss': total_action_loss / num_samples,
        'stop_distribution': stop_distribution,
    }

    return metrics

# 主评估流程
if __name__ == '__main__':
    # 加载模型
    runner = HazardRolloutRunner(...)

    # 加载验证集
    dataloader = ...

    # 评估不同策略
    print("Evaluating learned policy...")
    metrics_learned = evaluate_policy(runner, dataloader, 'learned')

    print("Evaluating fixed K=10...")
    metrics_k10 = evaluate_policy(runner, dataloader, 'fixed_K10')

    print("Evaluating fixed K=25...")
    metrics_k25 = evaluate_policy(runner, dataloader, 'fixed_K25')

    # 打印对比结果
    print("\n=== Evaluation Results ===")
    print(f"Learned Policy:")
    print(f"  Avg Video Steps: {metrics_learned['avg_video_steps']:.2f}")
    print(f"  Avg Action Loss: {metrics_learned['avg_action_loss']:.4f}")

    print(f"\nFixed K=10:")
    print(f"  Avg Video Steps: {metrics_k10['avg_video_steps']:.2f}")
    print(f"  Avg Action Loss: {metrics_k10['avg_action_loss']:.4f}")

    print(f"\nFixed K=25:")
    print(f"  Avg Video Steps: {metrics_k25['avg_video_steps']:.2f}")
    print(f"  Avg Action Loss: {metrics_k25['avg_action_loss']:.4f}")

    # 可视化 stop 分布
    import matplotlib.pyplot as plt
    plt.bar(range(25), metrics_learned['stop_distribution'])
    plt.xlabel('Stop Step')
    plt.ylabel('Count')
    plt.title('Stop Distribution (Learned Policy)')
    plt.savefig('stop_distribution.png')
```

**验收标准**：
- [ ] 能对比多种策略
- [ ] 输出清晰的对比结果
- [ ] 生成 stop 分布可视化

---

### 任务 9：集成到 wan_va_server.py（可选）

**文件**：`wan_va/wan_va_server.py`

**修改点**：
如果离线评估效果好，将 HazardScheduler 集成到在线推理流程

**核心修改**：
```python
# 在 WanVAServer 初始化时加载 scheduler_head
self.scheduler_head = HazardSchedulerHead(...)
self.scheduler_head.load_state_dict(torch.load('scheduler_head.pth'))
self.scheduler_head.eval()

# 在推理循环中使用 scheduler
def generate_action(self, obs, text):
    # ... 初始化 ...

    H_cumulative = 0.0
    for step_idx in range(self.K_max):
        # 运行 video branch
        model_output, video_pooled = self.transformer(
            latents, t, text_emb, ...,
            return_video_features=True
        )
        latents = self.video_scheduler.step(model_output, t, latents)

        # 计算 Hazard
        step_id = torch.tensor([step_idx / self.K_max])
        delta_H_k = self.scheduler_head(video_pooled, step_id)
        H_cumulative += delta_H_k.item()

        # 确定性停止（推理模式）
        h_k = 1 - torch.exp(-delta_H_k)
        if step_idx >= self.K_min and h_k > self.eta:
            break

    # 生成动作
    final_action = self._decode_action(latents, ...)
    return final_action
```

**验收标准**：
- [ ] 在线推理能正常运行
- [ ] 推理速度满足要求（< 100ms/step）
- [ ] 不影响现有固定步数模式（通过配置切换）

---

## 实施顺序建议

### Week 1：基础设施
- Day 1-2：任务 1（HazardSchedulerHead）
- Day 3：任务 2（model.py 修改）
- Day 4-5：任务 3（hazard_runtime.py）
- Day 6-7：任务 4（单元测试）

### Week 2：训练框架
- Day 1-3：任务 5（hazard_trainer.py）
- Day 4：任务 6（配置文件）
- Day 5：任务 7（启动脚本）
- Day 6-7：调试训练流程

### Week 3：评估验证
- Day 1-2：任务 8（离线评估）
- Day 3-4：分析结果，调整超参数
- Day 5-7：任务 9（在线集成，可选）

---

## 关键风险与缓解

### 风险 1：梯度消失/爆炸
**缓解**：
- 使用梯度裁剪（max_norm=1.0）
- 降低学习率（1e-5）
- 监控梯度范数

### 风险 2：训练不稳定
**缓解**：
- 使用 reference baseline
- Advantage normalization
- 增加 KL 正则

### 风险 3：策略过早停止或始终跑满
**缓解**：
- 调整 lambda_cost
- 设置 K_min
- 调整 scheduler_head 初始化

### 风险 4：工程实现困难
**缓解**：
- 先在小数据集上验证
- 逐步增加复杂度
- 充分复用现有代码（CutoffScanner）

---

## 验收标准总结

### 阶段 1 验收
- [ ] 所有单元测试通过
- [ ] 能对单个样本完成完整 rollout
- [ ] 输出的 Hazard 和概率在合理范围

### 阶段 2 验收
- [ ] 训练能稳定运行 100 步以上
- [ ] loss 正常下降
- [ ] 平均视频步数出现变化

### 阶段 3 验收
- [ ] 相比固定 K=25，平均视频步数下降
- [ ] 动作误差不明显劣化
- [ ] stop 分布合理（不是全部过早停或全部跑满）

---

## 下一步行动

1. **立即开始**：任务 1（HazardSchedulerHead）
2. **并行准备**：准备预训练模型和数据集路径
3. **代码审查**：每完成一个任务，进行代码审查
4. **持续监控**：训练开始后，密切监控 metrics

---

**文档版本**：v1.0
**创建日期**：2025-01-XX
**维护者**：LingBot-VA Team
