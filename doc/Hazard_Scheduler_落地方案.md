# Hazard Scheduler 落地方案

## 1. 文档定位

本文档是面向当前 `lingbot-va` 仓库的实际落地方案，不再把“连续时间 jump + 完整 PPO + 在线微调”作为第一阶段目标，而是采用一条更稳、更符合当前代码结构的路线：

- 不做离线扫描
- 不做 stop/cutoff 标签监督学习
- V1 不把在线微调当作必须项
- 先做 `Hazard stop-only`
- 先在当前固定离散视频步数网格上训练
- 用离线 reward 做策略优化

这条路线的核心思想是：

> 不告诉模型“正确答案应该停在第几步”，而是让模型自己决定何时停止，然后根据最终动作效果和计算代价给 reward，再更新 scheduler。

---

## 2. 最终建议

### 2.1 V1 推荐方案

推荐先实现下面这个版本：

- **策略形式**：累计 Hazard 停止控制
- **动作空间**：只学 `stop / continue`，不学 jump
- **时间空间**：只在现有离散视频步数网格上决策
- **训练方式**：离线 reward 驱动的策略优化
- **状态来源**：现有 demonstration 数据集
- **reward**：动作质量减去视频计算代价
- **可训练参数**：只训练 scheduler head，主干 backbone 冻结

### 2.2 V1 明确不做的事

第一阶段明确不做：

- 不做离线扫描生成 oracle cutoff
- 不做 stop 标签监督学习
- 不做连续时间插值版 jump
- 不做完整 online RL / 在线微调
- 不直接改造现有 `wan_va/train.py` 主训练流程

### 2.3 为什么这样收缩

这是因为当前仓库最成熟的是：

- 固定 scheduler 的监督式训练链
- 固定步数的视频推理链
- 离线精确 handoff 的 cutoff runtime

而最不成熟的是：

- 策略 rollout 训练
- 路径级 PPO 基础设施
- 连续时间 jump
- 在线环境闭环更新

因此，V1 应该优先证明：

> 在不使用 stop 标签监督学习、也不依赖离线扫描的情况下，仅通过 reward，Hazard stop scheduler 是否能够学到比固定步数更好的计算分配策略。

---

## 3. 为什么不建议第一阶段直接做原方案

## 3.1 不建议先做连续时间 jump

当前 `FlowMatchScheduler` 的 `step()` 实现会先把输入 `timestep` 映射到最近的离散 `timestep_id`，然后再取下一个离散 sigma 更新。[`wan_va/utils/scheduler.py`](../wan_va/utils/scheduler.py)

这意味着当前仓库并没有真正意义上的“连续时间积分器”。

因此如果一开始就做：

```python
t_next = t_current + r * (1 - t_current)
```

在现有实现里并不能自然落到真正连续的 flow 时间控制上，反而会把问题变成：

- 连续时间定义
- 离散 scheduler 对齐
- cache handoff 正确性

三件事同时调。

对于 V1，这个复杂度没有必要承担。

## 3.2 不建议先做 stop + jump 联合学习

当前已经被验证有研究价值的是：

- 不同 video cutoff 会显著影响动作误差

但还没有被直接验证的是：

- 中间 jump 路径本身一定比只学 stop 更值得先做

因此 V1 更合理的最小化设计应是：

- 只学习何时停止
- 不改变当前视频分支单步更新公式
- 只改变“要不要继续跑下一步”

## 3.3 不建议先做在线微调

在线微调当然更接近最终任务目标，但它要求同时具备：

- 稳定的 RoboTwin 环境链路
- 稳定的 server/client 推理链
- 稳定的策略训练器
- 稀疏成功 reward 下的 RL 调参能力

对当前阶段而言，这不是必须项，也不是最优先项。

更合理的顺序是：

1. 先做离线 reward 训练版 scheduler
2. 先验证 action quality 与平均视频步数的 tradeoff
3. 只有在离线 reward 版已经有效，但 RoboTwin 成功率仍不理想时，再考虑在线微调

---

## 4. 与当前仓库的对应关系

## 4.1 当前训练主线

当前 `wan_va/train.py` 是标准监督式 flow matching 训练：

- 视频和动作各自加噪
- Transformer 做一次联合前向
- 计算视频损失和动作损失

它并不包含：

- rollout buffer
- log_prob
- advantage
- value function
- PPO 更新

因此 scheduler 训练不应强行塞进当前 `train.py`，而应新建独立 trainer。

## 4.2 当前在线推理主线

当前 `wan_va/wan_va_server.py` 的视频推理 loop 是固定步数循环，且 video cache 只在最后一步写入。

这和 scheduler 训练所需的语义并不完全一致。scheduler 需要的是：

- 在任意 stop 发生时
- 立刻把当前 video 表征 handoff 给动作分支

因此 V1 更适合复用已有的“精确 handoff”思路，而不是一开始直接重写在线 server 主流程。

## 4.3 当前最值得复用的基础设施

当前仓库里最接近 scheduler runtime 需求的是：

- `wan_va/utils/cutoff_scan.py`
- `wan_va/utils/history_cutoff_scan.py`

它们已经具备：

- 视频分支推进到指定 cutoff
- cutoff 后精确刷新 video cache
- 动作分支在该 cache 上生成动作

V1 不做离线扫描，但可以复用这里面的 runtime 语义。

---

## 5. V1 技术路线

## 5.1 核心定义

在第 `k` 个视频去噪步，scheduler 输出一个非负 Hazard 增量：

```python
delta_H_k = softplus(g(z_k))
```

累计 Hazard：

```python
H_k = H_(k-1) + delta_H_k
```

条件停止概率：

```python
h_k = 1 - exp(-delta_H_k)
```

V1 只使用：

- `delta_H_k`
- `H_k`
- `h_k`

不输出 jump 参数。

## 5.2 状态定义

V1 的 scheduler 状态建议包含：

- 当前 video 分支 hidden feature 的 pooled 表示
  - 推荐使用 **mean pooling** over spatial tokens
  - 特征维度：与 backbone hidden dim 一致（如 768 或 1024）
  - 或使用 learnable query token 做 cross-attention pooling
- 当前离散 step id（归一化为 k/K_max，范围 [0, 1]）
- 当前累计 video 步数（用于计算代价）
- 可选的 chunk 级上下文特征（如任务描述 embedding）

注意：

当前模型默认前向并不直接暴露 pooled hidden feature，因此需要新增中间特征返回接口，或在 runtime 中显式抽取用于 scheduler 的特征。

**具体实现建议**：
```python
# 在 model.py 中添加
def forward(..., return_video_features=False):
    ...
    if return_video_features:
        video_pooled = video_hidden.mean(dim=1)  # [B, D]
        return output, video_pooled
```

## 5.3 决策空间

V1 采用离散视频步数网格：

- 当前 `robotwin` 默认视频步数为 `25`
- scheduler 在 `k = 1 ... 25` 的网格上决定 stop / continue

如果：

- `sample(stop) == True`

则立即停止视频分支，并 handoff 到动作分支。

如果：

- `sample(stop) == False`

则继续跑下一个固定视频 step。

---

## 6. reward 设计

## 6.1 V1 reward

V1 采用下面这个 reward：

```python
R = -L_act - lambda_cost * video_steps
```

其中：

- `L_act`：动作误差
- `video_steps`：本次 rollout 使用的视频步数
- `lambda_cost`：计算代价惩罚系数

## 6.2 为什么这个 reward 合理

它满足三个要求：

- 不需要 stop 标签
- 不需要离线扫描出最优 cutoff
- 不需要在线环境就能得到较密的反馈

这本质上是：

- 模型自己选择停止点
- 系统只根据最终效果给分

而不是告诉模型“应该停在哪里”。

## 6.3 `L_act` 的来源

V1 直接复用当前仓库已有的动作误差口径，优先使用当前训练里已经稳定使用的动作 loss。

**具体计算方式**：
- 使用 demonstration 数据集中的 GT action
- 在 rollout 时，动作分支生成预测动作
- 计算 `L_act = MSE(predicted_action, GT_action)`
- 或使用当前训练中已有的 action loss 函数（如加权 MSE、Huber loss 等）

这样有三个好处：

- 工程上最简单
- reward 与现有模型训练目标保持一致
- 不需要在线环境就能得到密集反馈

**注意**：这里的 GT action 来自 demonstration 数据，不是环境交互产生的。因此这是离线 reward，不是在线 RL。

## 6.4 V1 不直接使用 RoboTwin success 作为主 reward

原因是：

- success/fail 对策略学习来说过于稀疏
- 训练方差大
- 成本高

因此 V1 不把环境成功率作为主训练 reward，而把它留给后续评估，必要时再作为第二阶段在线微调的目标。

---

## 7. V1 训练流程

## 7.1 数据来源

V1 直接使用当前 demonstration 数据集中的 chunk/state。

这些数据只承担下面的作用：

- 提供起始状态
- 提供文本条件
- 提供动作监督信号以计算 reward

它们**不承担**：

- 提供 stop 标签
- 提供最优 cutoff 标签

## 7.2 rollout 流程

对于每个训练样本，执行：

1. 从数据集中取一个 chunk/state（包含 GT action）
2. 初始化视频噪声与动作噪声
3. 按固定视频 step 网格运行 video branch
4. 每一步通过 scheduler 计算 `delta_H_k` 和 `h_k`
5. 由 `h_k = 1 - exp(-delta_H_k)` 采样 stop / continue
   - **关键**：记录 `log_prob = log(h_k)` if stop else `log(1 - h_k)`
   - 存储到 rollout buffer：`(state, delta_H_k, h_k, action, log_prob)`
6. 若 stop，则立刻 handoff 到动作分支
7. 动作分支完整生成动作（去噪得到预测动作）
8. 计算 reward：
   - `L_act = MSE(predicted_action, GT_action)`
   - `R = -L_act - lambda_cost * video_steps`
9. 用 REINFORCE 更新 scheduler head：
   - `loss = -log_prob * (R - baseline)`
   - 只更新 scheduler head，backbone 冻结

**关键说明**：
- `L_act` 使用 demonstration 数据的 GT action 计算
- 梯度只通过 `log_prob` 反向传播，不通过 stop 决策本身（REINFORCE 技巧）
- 每个 rollout 只产生一条轨迹（从开始到 stop）

## 7.3 更新算法

V1 推荐从轻量策略梯度开始，而不是一开始就上完整 actor-critic PPO。

推荐顺序：

1. `REINFORCE + baseline`
2. `PPO-lite`
3. 如果确实需要，再扩展到完整 PPO

原因：

- V1 的 action 空间很小，只是 stop / continue
- rollout horizon 短
- 先跑通低复杂度版本更稳

## 7.4 baseline 建议

为了减小方差，建议使用 reference baseline：

**方案 A（推荐）**：固定步数策略
- 固定 `K=10` 或 `K=15` 作为 reference policy
- 在相同样本上运行 reference policy 得到 `R_ref`
- Advantage：`A = R_sampled - R_ref`

**方案 B**：固定 Hazard 启发式策略
- 使用简单的指数增长 Hazard：`delta_H_k = 0.3 * exp(0.15 * k)`
- 这会让模型倾向于在 k=8-12 步左右停止
- 同样计算 `R_ref` 作为 baseline

**方案 C**：移动平均 baseline
- 维护一个 running mean：`baseline = 0.99 * baseline + 0.01 * R`
- 不需要额外 rollout，但方差略大

最终 advantage 计算：

```python
A = R_sampled - R_reference
```

这样可以显著降低训练方差，加速收敛。

## 7.5 KL 正则

V1 仍建议加 KL 正则，约束当前策略不要在初期偏离 reference policy 过快。

例如：

```python
L = L_policy + beta_kl * KL(pi_theta || pi_ref)
```

---

## 8. V1 实现拆分

## 8.1 推荐新增文件

- `wan_va/modules/hazard_scheduler.py`
  - 定义 `HazardSchedulerHead`
- `wan_va/utils/hazard_runtime.py`
  - 封装 video rollout、stop、handoff、action decode
- `wan_va/hazard_trainer.py`
  - 独立的 reward-based trainer
- `train_profiles/hazard_stop_reward_v1.yaml`
  - 训练配置
- `script/run_hazard_stop_reward.sh`
  - 启动脚本
- `tests/test_hazard_scheduler.py`
  - HazardSchedulerHead 单元测试
- `tests/test_hazard_runtime.py`
  - rollout 流程集成测试

## 8.2 推荐修改文件

- `wan_va/modules/model.py`
  - 增加 scheduler 所需的中间特征返回接口
- `wan_va/modules/utils.py`
  - 支持加载带 scheduler head 的模型
- `wan_va/utils/cutoff_scan.py`
  - 抽离可复用的精确 handoff runtime

## 8.3 明确不建议直接修改的文件

V1 不建议一开始就深改：

- `wan_va/train.py`
- `wan_va/wan_va_server.py`

原因是：

- 这两条链分别是当前主训练链和主在线推理链
- 过早侵入会增加调试成本
- V1 更适合先用独立 runtime 证明效果

---

## 9. 实施阶段

**总时间估算**：3-4 周（阶段 1-3 必做，阶段 4 可选）

各阶段可部分并行：阶段 1 完成后，可同时进行阶段 2 的代码开发和阶段 3 的评估准备。

---

## 阶段 1：基础设施（约 1 周）

### 目标

完成 scheduler head、特征接口和独立 runtime。

### 任务

- [ ] 实现 `HazardSchedulerHead`
- [ ] 在模型中暴露 scheduler 所需中间特征
- [ ] 抽出精确 handoff runtime
- [ ] 编写基础单元测试

### 验收标准

- 能拿到中间特征
- 能对单个样本完成：video rollout -> stop -> handoff -> action generation
- 能输出合理的 `delta_H_k` 与 `h_k`

## 阶段 2：离线 reward 训练（约 1-2 周）

### 目标

跑通 stop-only 策略训练。

### 任务

- [ ] 实现 reward 计算
- [ ] 实现 REINFORCE / PPO-lite 更新
- [ ] 加入 reward 归一化
- [ ] 加入 KL 正则与 entropy bonus
- [ ] 跑通小规模训练

### 验收标准

- 训练稳定，无明显 collapse
- 平均视频步数出现可控变化
- 动作误差不劣于固定步数 baseline

## 阶段 3：离线评估与部署验证（约 1 周）

### 目标

验证 scheduler 是否值得进入在线部署。

### 任务

- [ ] 对比固定 `K=25`
- [ ] 对比固定 `K=10`
- [ ] 记录平均视频步数
- [ ] 记录动作误差
- [ ] 验证 stop 分布是否合理

### 验收标准

- 平均视频步数下降
- 动作误差不明显变差
- stop 分布不是全部过早停或全部跑满

## 阶段 4：可选在线微调（非必需）

### 进入条件

只有满足下面任一条件时，才建议考虑在线微调：

- 离线 reward 指标改善了，但 RoboTwin 成功率没有改善
- 发现离线 `action_loss` 与真实 success 存在明显偏差
- 需要进一步贴近最终环境目标

### 这阶段才做的事

- 在 RoboTwin 环境中运行策略
- 用 success / fail / latency 做环境 reward
- 小学习率微调 scheduler head

注意：

在线微调是可选增强项，不是 V1 必选项。

---

## 10. 风险与缓解

## 风险 1：策略全部过早停

### 表现

- 平均视频步数极低
- 动作误差明显变差

### 缓解

- 降低 Hazard 初始偏置
- 加大 KL 正则
- 设置最小停止步数 `K_min`

## 风险 2：策略始终跑满

### 表现

- 平均视频步数接近 25
- 学不到预算分配能力

### 缓解

- 增大 `lambda_cost`
- 增强 entropy / exploration
- 调整 reference policy

## 风险 3：reward 方差过大

### 表现

- 训练不稳定
- 曲线大幅震荡

### 缓解

- 使用 reward normalization
- 使用 reference baseline
- 从 REINFORCE 起步，不急着完整 PPO

## 风险 4：runtime 与在线 server 语义不一致

### 表现

- 训练有效，但部署失效

### 缓解

- 先复用精确 handoff 语义
- scheduler 生效后再把同样的 handoff 逻辑迁到 server

---

## 11. 成功指标

## 11.1 V1 必须达成

- [ ] 不使用 stop/cutoff 标签监督学习
- [ ] 不依赖离线扫描生成 oracle
- [ ] scheduler 能稳定训练
- [ ] 相比固定 `K=25`，平均视频步数下降
- [ ] 动作误差不明显劣化

## 11.2 V1 期望达成

- [ ] 平均视频步数下降 20% 以上
- [ ] 动作误差持平或小幅改善
- [ ] 不同状态出现不同停止分布

## 11.3 V2 可选目标

- [ ] RoboTwin 成功率提升
- [ ] 在线推理时延进一步下降
- [ ] 再考虑加入 jump 或在线微调

---

## 12. FAQ

### Q1：这算监督学习吗？

**不是。**

因为系统并没有使用 stop 标签或最优 cutoff 标签，只是根据最终动作效果给 reward。

### Q2：这还需要离线扫描吗？

**不需要。**

V1 的目标就是在不做离线扫描的前提下，直接用 reward 驱动 scheduler 学习。

### Q3：这还需要在线微调吗？

**V1 不需要。**

在线微调是可选增强项，不是第一阶段前提。

### Q4：为什么 V1 不直接做 jump？

因为当前仓库的 scheduler 与 runtime 更适合先在离散时间网格上学习 stop。先把 stop 跑稳，再决定 jump 是否值得做。

### Q5：为什么不建议一上来完整 PPO？

因为当前仓库没有现成 PPO 基础设施，且 V1 的 action 空间较小。先从更轻量的策略梯度版本起步，更符合当前工程现实。

---

## 14. 关键技术细节补充

### 14.1 梯度流动问题

**问题**：scheduler 的 stop 决策是离散的，如何反向传播？

**解决方案**：使用 REINFORCE 策略梯度

```python
# 前向：采样 stop 决策
h_k = 1 - torch.exp(-delta_H_k)
stop = torch.bernoulli(h_k)  # 离散采样，无梯度

# 记录 log_prob
log_prob = torch.log(h_k) if stop else torch.log(1 - h_k)

# 反向：通过 log_prob 传递梯度
loss = -log_prob * (R - baseline)  # REINFORCE
loss.backward()  # 梯度只流向 delta_H_k，不流向 stop
```

**关键点**：
- stop 决策本身不参与梯度计算
- 梯度通过 `log_prob` 反向传播到 `delta_H_k`
- 这是标准的 REINFORCE 技巧，适用于离散动作空间

### 14.2 为什么不用 Gumbel-Softmax？

Gumbel-Softmax 可以让离散采样可微，但不适合这个场景：

- Gumbel-Softmax 需要 temperature annealing，增加调参复杂度
- REINFORCE 更简单，且在小动作空间（stop/continue）上效果足够好
- 当前方案已经有 baseline 和 KL 正则来降低方差

### 14.3 训练稳定性保证

**方差控制**：
- 使用 reference baseline（固定 K=10）
- Reward normalization：`R_norm = (R - R_mean) / (R_std + 1e-8)`
- Advantage clipping：`A_clip = torch.clamp(A, -10, 10)`

**策略稳定性**：
- KL 正则：约束策略不要偏离 reference 过快
- Entropy bonus：鼓励探索，防止过早收敛
- 梯度裁剪：`torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)`

**初始化**：
- Hazard head 的输出层初始化为接近 0（让初始 `delta_H_k` 较小）
- 这样初始策略倾向于多跑几步，避免过早停止

---

## 13. 一句话总结

当前仓库最合理的 Hazard Scheduler 落地路线是：

> 不做离线扫描，不做 stop 标签监督学习，V1 也不把在线微调当作前提；先在现有离散视频步数网格上训练一个 `Hazard stop-only` scheduler，用离线 reward 直接优化“动作质量 - 视频计算代价”。

