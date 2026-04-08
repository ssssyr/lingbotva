# Hazard Scheduler 启发式初始化方案

## 文档版本
- **版本**: v1.0
- **日期**: 2026-04-07
- **作者**: Claude & User

---

## 一、方案概述

### 1.1 核心思想

在 PPO 训练 Hazard Scheduler 时，不从随机初始化开始，而是给模型一个**简单但合理的初始策略**（启发式），然后让 PPO 在这个基础上优化。

**类比**：
- ❌ 不是让婴儿从零学走路（随机初始化）
- ✅ 而是先教他爬（启发式），再学走路（PPO 优化）

### 1.2 为什么需要启发式初始化？

#### 问题：随机初始化的困境

PPO 从随机初始化开始训练时：

```python
# 随机初始化的模型输出
alpha = 0.23   # 随机
beta = 1.87    # 随机
delta_H = 0.05 # 随机

# 对应的策略：
r ~ Beta(0.23, 1.87)  # 均值 ≈ 0.11，偏向小跳步
h = 1 - exp(-0.05) ≈ 0.049  # 5% 概率停止
```

**导致的问题**：
- 策略完全随机，探索效率极低
- 可能所有样本都在 step 1 停止（delta_H 太大）
- 或所有样本都跑到 K_max（delta_H 太小）
- Reward 信号弱，训练难以收敛

#### 解决方案：启发式初始化

给模型一个简单的初始策略：
- **Hazard 策略**：随时间指数增长（早期小，后期大）
- **跳步策略**：均匀跳步（r=0.5）

然后让 PPO 在这个基础上优化。

---

## 二、启发式策略设计

### 2.1 Hazard 增长策略

#### 目标

设计一个简单的规则，让 Hazard 随时间增长：
- 早期：delta_H 小（不该停）
- 后期：delta_H 大（应该停）

#### 设计公式

```python
delta_H = scale * exp(growth_rate * t)
```

其中：
- `t ∈ [0, 1]`：归一化时间
- `scale`：缩放因子（控制整体大小）
- `growth_rate`：增长速率（控制增长快慢）

#### 参数选择

**scale（缩放因子）**：

```python
scale = 0.5  # 推荐值

# 效果：
t=0.0: delta_H = 0.5 * exp(0) = 0.5
t=0.5: delta_H = 0.5 * exp(1.0) ≈ 1.36
t=1.0: delta_H = 0.5 * exp(2.0) ≈ 3.69
```

**growth_rate（增长速率）**：

```python
growth_rate = 2.0  # 推荐值

# 不同 growth_rate 的效果：
growth_rate=1.0:  缓慢增长
growth_rate=2.0:  中等增长（推荐）
growth_rate=3.0:  快速增长
```

#### 完整示例

假设 K_max=10，每步的归一化时间：

```
step  t      delta_H = 0.5 * exp(2.0 * t)    H_cumulative    F = 1-exp(-H)
0     0.0    0.5                              0.5             0.39
1     0.1    0.61                             1.11            0.67
2     0.2    0.74                             1.85            0.84
3     0.3    0.91                             2.76            0.94
4     0.4    1.11                             3.87            0.98  ← 触发停止
5     0.5    1.36                             5.23            0.995
...
```

**效果**：在 step 4 左右，F 超过 0.85，触发停止。

---

### 2.2 跳步策略

#### 目标

设计一个简单的跳步策略，作为初始化。

#### 最简单的选择：均匀跳步

```python
mean = 0.5  # 均匀跳步
concentration = 5.0  # 控制方差

alpha = mean * concentration = 2.5
beta = (1 - mean) * concentration = 2.5
```

#### Beta 分布的性质

**Beta(α, β) 的均值和方差**：

```
均值 = α / (α + β)
方差 = αβ / [(α+β)²(α+β+1)]
```

**concentration 的作用**：

```python
# concentration 小 → 方差大（分散）
alpha, beta = 1.0, 1.0  # Beta(1, 1) = 均匀分布
# r 可以是 [0, 1] 中的任何值，方差 = 1/12 ≈ 0.083

# concentration 中等 → 方差中等（推荐）
alpha, beta = 2.5, 2.5  # Beta(2.5, 2.5)
# r 集中在 0.5 附近，方差 ≈ 0.042

# concentration 大 → 方差小（集中）
alpha, beta = 10.0, 10.0  # Beta(10, 10)
# r 非常集中在 0.5 附近，方差 ≈ 0.012
```

**推荐配置**：

```python
mean = 0.5  # 均匀跳步
concentration = 5.0  # 中等方差

alpha = 2.5
beta = 2.5
```

**效果**：
- 大部分时候跳 40%-60%
- 偶尔跳 30%-70%
- 很少跳 <20% 或 >80%

---

## 三、启发式权重的衰减

### 3.1 核心思想

训练初期：完全依赖启发式（weight=1.0）
训练中期：逐渐过渡到网络输出（weight 1.0→0.0）
训练后期：完全依赖网络输出（weight=0.0）

### 3.2 混合公式

```python
# 最终输出 = 启发式输出 × weight + 网络输出 × (1-weight)

w = heuristic_weight

alpha_final = w * alpha_heuristic + (1-w) * alpha_network
beta_final = w * beta_heuristic + (1-w) * beta_network
delta_H_final = w * delta_H_heuristic + (1-w) * delta_H_network
```

### 3.3 衰减策略

#### 策略 A：线性衰减（推荐）

```python
def update_heuristic_weight(self, progress):
    """
    progress ∈ [0, 1]，表示训练进度
    """
    # 前 50% 训练：权重从 1.0 → 0.0
    # 后 50% 训练：权重保持 0.0
    self.heuristic_weight = max(0.0, 1.0 - 2.0 * progress)
```

**可视化**：

```
weight
 1.0 |█████████████╲
     |              ╲
 0.5 |               ╲
     |                ╲___________
 0.0 |____________________________
     0%   25%   50%   75%   100%
              progress
```

**示例**：

```
progress=0.0:  weight=1.0  (完全启发式)
progress=0.1:  weight=0.8
progress=0.25: weight=0.5  (一半一半)
progress=0.5:  weight=0.0  (完全网络)
progress=0.75: weight=0.0
progress=1.0:  weight=0.0
```

#### 策略 B：余弦衰减（更平滑）

```python
def update_heuristic_weight(self, progress):
    """
    余弦衰减，更平滑
    """
    if progress < 0.5:
        # 前 50%：余弦衰减
        self.heuristic_weight = 0.5 * (1 + np.cos(np.pi * progress / 0.5))
    else:
        self.heuristic_weight = 0.0
```

#### 策略 C：延长衰减（更保守）

```python
def update_heuristic_weight(self, progress):
    """
    前 70% 训练才衰减到 0
    """
    self.heuristic_weight = max(0.0, 1.0 - progress / 0.7)
```

**选择建议**：
- 如果 PPO 训练稳定：用策略 A（线性，50% 衰减）
- 如果 PPO 训练不稳定：用策略 C（延长，70% 衰减）

---

## 四、代码实现

### 4.1 HazardSchedulerHead 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HazardSchedulerHead(nn.Module):
    """
    带启发式初始化的 Hazard Scheduler Head
    """

    def __init__(
        self,
        hidden_dim: int = 3072,
        time_embed_dim: int = 256,
        condition_dim: int = 128,
        output_dim: int = 256,
        # 启发式配置
        heuristic_hazard_scale: float = 0.5,
        heuristic_hazard_growth: float = 2.0,
        heuristic_jump_mean: float = 0.5,
        heuristic_jump_concentration: float = 5.0,
    ):
        super().__init__()

        # 网络结构
        input_dim = hidden_dim + time_embed_dim + condition_dim

        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.SiLU(),
            nn.LayerNorm(output_dim),
            nn.Dropout(0.1),
        )

        self.jump_head = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 2),  # (alpha, beta)
        )

        self.hazard_head = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1),  # delta_H
        )

        # 启发式配置
        self.heuristic_weight = 1.0  # 初始完全依赖启发式
        self.heuristic_hazard_scale = heuristic_hazard_scale
        self.heuristic_hazard_growth = heuristic_hazard_growth
        self.heuristic_jump_mean = heuristic_jump_mean
        self.heuristic_jump_concentration = heuristic_jump_concentration

    def forward(
        self,
        hidden_feature: torch.Tensor,  # [B, hidden_dim]
        time_embedding: torch.Tensor,  # [B, time_embed_dim]
        condition: torch.Tensor = None,  # [B, condition_dim]
    ):
        batch_size = hidden_feature.shape[0]

        # 默认 condition
        if condition is None:
            condition = torch.zeros(
                batch_size, 128,
                device=hidden_feature.device,
                dtype=hidden_feature.dtype
            )

        # 拼接输入
        z = torch.cat([hidden_feature, time_embedding, condition], dim=-1)

        # 网络输出
        shared = self.shared_net(z)

        # Jump head
        jump_params = self.jump_head(shared)
        network_alpha = F.softplus(jump_params[:, 0]) + 1.0
        network_beta = F.softplus(jump_params[:, 1]) + 1.0

        # Hazard head
        hazard_logit = self.hazard_head(shared).squeeze(-1)
        network_delta_H = F.softplus(hazard_logit)

        # 启发式输出
        if self.training and self.heuristic_weight > 0:
            heuristic_delta_H = self._compute_heuristic_hazard(
                time_embedding, batch_size
            )
            heuristic_alpha, heuristic_beta = self._compute_heuristic_jump(
                batch_size
            )

            # 混合
            w = self.heuristic_weight
            alpha = w * heuristic_alpha + (1 - w) * network_alpha
            beta = w * heuristic_beta + (1 - w) * network_beta
            delta_H = w * heuristic_delta_H + (1 - w) * network_delta_H
        else:
            # 推理时或权重为 0 时，只用网络输出
            alpha = network_alpha
            beta = network_beta
            delta_H = network_delta_H

        return alpha, beta, delta_H

    def _compute_heuristic_hazard(
        self,
        time_embedding: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """
        计算启发式 Hazard 增量

        策略：指数增长
        delta_H = scale * exp(growth * t)
        """
        # 从 time_embedding 中提取归一化时间
        # 假设 time_embedding 的第一个维度编码了时间信息
        t_raw = time_embedding[:, 0]
        t = torch.sigmoid(t_raw)  # 归一化到 [0, 1]

        # 指数增长
        delta_H = self.heuristic_hazard_scale * torch.exp(
            self.heuristic_hazard_growth * t
        )

        return delta_H

    def _compute_heuristic_jump(
        self,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        计算启发式跳步参数

        策略：均匀跳步 r=0.5
        """
        device = next(self.parameters()).device

        # Beta(α, β)，均值 = α/(α+β)
        mean = self.heuristic_jump_mean
        c = self.heuristic_jump_concentration

        alpha = torch.full(
            (batch_size,),
            mean * c,
            device=device,
            dtype=torch.float32
        )
        beta = torch.full(
            (batch_size,),
            (1 - mean) * c,
            device=device,
            dtype=torch.float32
        )

        return alpha, beta

    def update_heuristic_weight(self, progress: float):
        """
        更新启发式权重

        Args:
            progress: 训练进度，∈ [0, 1]
        """
        # 线性衰减：前 50% 训练从 1.0 → 0.0
        self.heuristic_weight = max(0.0, 1.0 - 2.0 * progress)

    def get_heuristic_info(self) -> dict:
        """
        获取启发式信息，用于日志
        """
        return {
            'heuristic_weight': self.heuristic_weight,
            'hazard_scale': self.heuristic_hazard_scale,
            'hazard_growth': self.heuristic_hazard_growth,
            'jump_mean': self.heuristic_jump_mean,
            'jump_concentration': self.heuristic_jump_concentration,
        }
```

### 4.2 训练循环

```python
def train_with_heuristic(model, dataloader, config):
    """
    带启发式初始化的 PPO 训练
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate']
    )

    total_steps = config['num_epochs'] * len(dataloader)
    current_step = 0

    for epoch in range(config['num_epochs']):
        for batch in dataloader:
            # 1. 更新启发式权重
            progress = current_step / total_steps
            model.scheduler_head.update_heuristic_weight(progress)

            # 2. Rollout 路径
            paths = rollout_paths(model, batch, config)

            # 3. 计算 Reward
            rewards = compute_rewards(paths, batch, config)

            # 4. PPO 更新
            loss = compute_ppo_loss(model, paths, rewards, config)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )
            optimizer.step()

            # 5. 日志
            if current_step % 100 == 0:
                heuristic_info = model.scheduler_head.get_heuristic_info()
                log_training_info(
                    current_step,
                    total_steps,
                    heuristic_info,
                    paths,
                    rewards,
                )

            current_step += 1
```

---

## 五、超参数配置

### 5.1 推荐配置

```python
config = {
    # Scheduler 配置
    'K_min': 2,
    'K_max': 10,
    'eta': 0.85,

    # 启发式配置
    'heuristic_hazard_scale': 0.5,  # 控制 Hazard 增长速度
    'heuristic_hazard_growth': 2.0,  # 控制增长陡峭程度
    'heuristic_jump_mean': 0.5,  # 均匀跳步
    'heuristic_jump_concentration': 5.0,  # Beta 分布集中度

    # PPO 配置
    'learning_rate': 1e-4,
    'ppo_epochs': 4,
    'ppo_clip_ratio': 0.2,
    'ppo_kl_coef': 0.01,

    # Reward 配置
    'lambda_cost': 0.01,  # 计算代价权重

    # 训练配置
    'num_epochs': 20,
    'batch_size': 8,
    'gradient_accumulation_steps': 4,
}
```

### 5.2 关键超参数说明

#### heuristic_hazard_scale

**作用**：控制 Hazard 的整体大小

```python
scale=0.3:  停止较晚（平均 6-8 步）
scale=0.5:  中等（平均 4-6 步）← 推荐
scale=0.8:  停止较早（平均 2-4 步）
```

**如何选择**：
- 在验证集上测试不同 scale
- 选择平均停止步数在 4-6 步的 scale

#### heuristic_hazard_growth

**作用**：控制 Hazard 增长速度

```python
growth=1.0:  缓慢增长（更平滑）
growth=2.0:  中等增长 ← 推荐
growth=3.0:  快速增长（更陡峭）
```

#### heuristic_jump_mean

**作用**：控制平均跳步比例

```python
mean=0.3:  小跳步（每步跳 30%）
mean=0.5:  中等跳步 ← 推荐
mean=0.7:  大跳步（每步跳 70%）
```

#### heuristic_jump_concentration

**作用**：控制跳步的方差

```python
concentration=2.0:  方差大（跳步多样）
concentration=5.0:  方差中等 ← 推荐
concentration=10.0: 方差小（跳步集中）
```

---

## 六、调试与验证

### 6.1 可视化启发式输出

```python
# 测试启发式策略
model.eval()
model.scheduler_head.heuristic_weight = 1.0  # 纯启发式

with torch.no_grad():
    H_cumulative = 0.0

    for step in range(10):
        # 模拟 time_embedding
        t = step / 10.0
        time_emb = get_time_embedding(t)

        # 启发式输出
        alpha, beta, delta_H = model.scheduler_head(
            hidden=torch.randn(1, 3072),
            time_embedding=time_emb,
        )

        # 计算停止概率
        H_cumulative += delta_H
        F = 1 - torch.exp(-H_cumulative)
        h = 1 - torch.exp(-delta_H)

        print(f"Step {step}:")
        print(f"  delta_H: {delta_H.item():.3f}")
        print(f"  H: {H_cumulative.item():.3f}")
        print(f"  F: {F.item():.3f}")
        print(f"  h: {h.item():.3f}")
        print(f"  alpha: {alpha.item():.2f}, beta: {beta.item():.2f}")
        print(f"  jump_mean: {(alpha/(alpha+beta)).item():.3f}")
```

### 6.2 监控权重衰减

```python
# 训练时记录
wandb.log({
    'heuristic/weight': model.scheduler_head.heuristic_weight,
    'heuristic/hazard_contribution': w * delta_H_heuristic.mean(),
    'network/hazard_contribution': (1-w) * delta_H_network.mean(),
})
```

### 6.3 对比启发式 vs 网络

```python
# 在验证集上对比
# 纯启发式
model.scheduler_head.heuristic_weight = 1.0
heuristic_results = evaluate(model, val_dataset)

# 纯网络
model.scheduler_head.heuristic_weight = 0.0
network_results = evaluate(model, val_dataset)

print(f"Heuristic: SR={heuristic_results['success_rate']:.2f}")
print(f"Network: SR={network_results['success_rate']:.2f}")
```

---

## 七、方案优势

### 7.1 相比随机初始化

| 特性 | 随机初始化 | 启发式初始化 |
|------|-----------|-------------|
| 初始策略质量 | ❌ 完全随机 | ✅ 合理 |
| 探索效率 | ❌ 低 | ✅ 高 |
| 训练稳定性 | ❌ 不稳定 | ✅ 稳定 |
| Reward 信号 | ❌ 弱 | ✅ 强 |
| 收敛速度 | ❌ 慢 | ✅ 快 |

### 7.2 相比监督学习预训练

| 特性 | 监督预训练 | 启发式初始化 |
|------|-----------|-------------|
| 需要扫描数据 | ❌ 需要 | ✅ 不需要 |
| 分布偏移 | ❌ 有（固定跳步 vs 自适应跳步） | ✅ 无 |
| 实现复杂度 | ❌ 高 | ✅ 低 |
| 训练时间 | ❌ 长（两阶段） | ✅ 短（一阶段） |
| 照顾跳步策略 | ❌ 不照顾 | ✅ 照顾 |

---

## 八、常见问题

### Q1: 启发式会不会限制模型的学习能力？

**A**: 不会。启发式只在训练前期起作用（前 50%），后期完全依赖网络学习。启发式只是提供一个好的起点，不会限制最终性能。

### Q2: 如果启发式策略本身不好怎么办？

**A**: 即使启发式不是最优的，只要比随机初始化好，就能加速训练。PPO 会自动调整策略，逐渐偏离启发式，找到更好的策略。

### Q3: 能否跳过启发式，直接用 PPO？

**A**: 可以，但风险较高。如果你有丰富的 RL 经验，可以尝试。但对于大多数情况，启发式初始化能显著降低训练难度。

### Q4: 启发式权重衰减太快/太慢怎么办？

**A**: 根据训练曲线调整：
- 如果训练不稳定：延长衰减时间（70% 或 80%）
- 如果训练过于保守：缩短衰减时间（30% 或 40%）

### Q5: 如何验证启发式是否有效？

**A**: 对比实验：
1. 训练两个模型：一个用启发式，一个不用
2. 对比训练曲线（reward、停止步数、成功率）
3. 对比收敛速度和最终性能

---

## 九、实施清单

### 阶段 1：实现（1-2 天）

- [ ] 实现 `HazardSchedulerHead` 类
- [ ] 添加启发式计算方法
- [ ] 添加权重衰减方法
- [ ] 单元测试

### 阶段 2：集成（1 天）

- [ ] 集成到训练循环
- [ ] 添加日志记录
- [ ] 添加可视化

### 阶段 3：调试（1-2 天）

- [ ] 验证启发式输出合理性
- [ ] 验证权重衰减正确性
- [ ] 对比启发式 vs 网络输出

### 阶段 4：训练（1-2 周）

- [ ] 小规模训练验证
- [ ] 调整超参数
- [ ] 全量训练
- [ ] 评估效果

---

## 十、参考资料

### 相关文档

- `LingBot-VA_Hazard_停止控制方案.md`：完整的 Hazard 方案说明
- `LingBot-VA_TPDM_统一训练方案_累计Hazard版.docx`：原始方案文档

### 相关代码

- `wan_va/modules/hazard_scheduler.py`：Scheduler Head 实现
- `wan_va/train.py`：训练循环
- `wan_va/utils/scheduler.py`：Flow Matching Scheduler

---

## 附录：完整配置示例

```yaml
# train_profiles/hazard_ppo_with_heuristic.yaml

# 模型配置
model:
  enable_hazard_scheduler: true
  hidden_dim: 3072
  time_embed_dim: 256
  condition_dim: 128

# Scheduler 配置
scheduler:
  K_min: 2
  K_max: 10
  eta: 0.85

# 启发式配置
heuristic:
  hazard_scale: 0.5
  hazard_growth: 2.0
  jump_mean: 0.5
  jump_concentration: 5.0
  decay_strategy: 'linear'  # 'linear', 'cosine', 'extended'
  decay_progress: 0.5  # 前 50% 训练衰减到 0

# PPO 配置
ppo:
  learning_rate: 1e-4
  ppo_epochs: 4
  ppo_clip_ratio: 0.2
  ppo_kl_coef: 0.01
  value_loss_coef: 0.5
  entropy_coef: 0.01

# Reward 配置
reward:
  lambda_cost: 0.01  # 计算代价权重
  baseline_policy: 'uniform'  # 参考策略

# 训练配置
training:
  num_epochs: 20
  batch_size: 8
  gradient_accumulation_steps: 4
  max_grad_norm: 1.0
  warmup_steps: 500

# 日志配置
logging:
  log_interval: 100
  eval_interval: 1000
  save_interval: 5000
  wandb_project: 'hazard_scheduler'
  wandb_run_name: 'ppo_with_heuristic'
```

---

**文档结束**
