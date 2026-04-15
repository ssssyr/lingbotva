# Hazard Scheduler 跳步策略集成方案（结合 TPDM）

## 1. 文档目的

本文档用于说明如何将 `/home/syr/code/TPDM` 中的连续跳步策略迁移到当前 LingBot-VA 的 Hazard Scheduler 训练与推理框架中，形成一套“累计 Hazard 停止 + 连续相对跳步”的统一调度方案。

本文档重点回答 4 个问题：

1. 当前 `lingbot-va` 的 Hazard 实现到什么程度。
2. TPDM 的哪些设计值得直接借鉴。
3. 跳步策略应该如何落到当前代码结构中。
4. 推荐的实施顺序、兼容策略与风险控制方式是什么。

---

## 2. 当前仓库现状

### 2.1 当前实现能力

当前 `hazard-schedular` 分支已经完成的是 V1 版“停止策略”：

- `wan_va/modules/hazard_scheduler.py`
  - 仅输出单个 `delta_H_k`。
  - 仅支持累计 Hazard 停止，不支持继续时的跳步预测。
- `wan_va/utils/hazard_runtime.py`
  - rollout 按固定离散 video timestep 逐步前进。
  - 每轮只执行相邻一步 denoise，不支持一次跨越多个 timestep。
- `wan_va/utils/hazard_trainer.py`
  - 当前训练目标是 REINFORCE 路径奖励。
  - 尚未实现 PPO clipped update。
- `wan_va/wan_va_server.py`
  - 在线推理支持 `fixed` 和 `hazard` 两种模式。
  - `hazard` 仍然是“逐步 denoise + 累计 Hazard 停止”。

### 2.2 当前实现与统一方案的差距

与 `LingBot-VA_TPDM_统一训练方案_累计Hazard版.docx` 相比，当前代码缺失的核心部分是：

- 跳步动作头 `alpha/beta` 或等价参数化。
- continue 动作下的 `Beta` 采样。
- 连续相对时间推进，而不是固定离散 step 递进。
- 训练期“stop + jump”联合路径概率。
- 面向 PPO 的旧策略 logprob 缓存与重算接口。

### 2.3 Git 一致性结论

截至 2026-04-13，本地 `HEAD` 与 `origin/hazard-schedular` 的已提交代码一致，提交哈希均为 `b30e685`。

但工作区不是完全干净状态，当前还有：

- 一个未提交删除：`doc/Hazard_Scheduler_奖励函数问题分析.md`
- 一个未跟踪目录：`.codex`

因此：

- 如果只看已提交代码：本地与 GitHub `origin/hazard-schedular` 一致。
- 如果看当前工作区状态：不完全一致，因为存在未提交本地修改。

---

## 3. TPDM 中最值得迁移的设计

## 3.1 连续相对跳步，而不是离散 step 分类

TPDM 的时间预测头不是直接分类“下一个 step 是多少”，而是预测一个 `Beta(alpha, beta)` 分布，并从中采样相对比例 `r`，然后更新：

```text
sigma_next = sigma_cur * r
```

这里的 `r` 更准确地说是“剩余噪声保留比例”或 `noise retention ratio`：

- `r` 接近 1：小跳步，较保守
- `r` 接近 0：大跳步，较激进

因此日志和可视化中建议同时输出：

```text
jump_distance = 1 - r
```

这比离散 step 分类更适合 LingBot-VA，原因是：

- 与文档中的“跳步比例”定义一致。
- 更容易表达“剩余去噪进度压缩多少”。
- 能与累计 Hazard 停止自然组合。
- 更方便从固定 schedule 推导 reference distribution。

TPDM 对应实现位置：

- `src/models/stable_diffusion_3/modeling_sd3_pnt.py`
- `src/models/model_utilis.py`

## 3.2 自定义 scheduler step

TPDM 不是只在预定义 timestep 列表里取相邻元素，而是直接执行：

```text
prev_sample = sample + (sigma_next - sigma_cur) * model_output
```

这意味着只要知道当前 sigma 和目标 sigma，就能进行一次连续跳步更新。

这一点对 LingBot-VA 很关键，因为当前 `wan_va/utils/scheduler.py` 只支持从当前 timestep 前进到离散表中的下一个 timestep。

## 3.3 rollout 时缓存策略输入，PPO 时只重算 head

TPDM 在采样阶段会缓存：

- 每步 hidden states / 特征
- 每步 temb
- 每步旧策略输出的 `alpha/beta`
- 每步旧策略 logprob
- 每步有效 mask

然后 PPO 更新时只重算策略头 logprob，不重跑整条主干采样链。

这对 LingBot-VA 更合适，因为当前 backbone 默认冻结，最昂贵的其实是 video/action rollout，而不是 scheduler head。

## 3.4 reference Beta KL

TPDM 的 KL 不是简单对一个固定常数分布，而是根据基准 schedule 计算 step-conditioned 的参考 `Beta` 分布。

这对 LingBot-VA 的 jump 部分很有价值，因为我们已经有固定 25-step video schedule，可以直接从它导出参考跳步比例：

```text
r_ref_k = sigma_{k+1} / sigma_k
```

然后给一个固定 concentration，得到：

```text
alpha_ref_k = r_ref_k * (c - 2) + 1
beta_ref_k = (1 - r_ref_k) * (c - 2) + 1
```

这里需要特别注意：

- TPDM 的 `reference_distributions.py` 中有一套针对 SD3 schedule 的 `t-space` 变换。
- LingBot-VA 的 shift 公式与 TPDM 不同，不应直接套用 TPDM 的 `t-space` 参考公式。
- LingBot-VA 应直接从当前 `self.video_scheduler.sigmas` 数组计算 `r_ref_k`，保证 reference policy 与本项目真实 schedule 一致。

## 3.5 PPO / RLOO 训练框架

TPDM 使用的是一套 RLOO + PPO clipped objective 的在线训练逻辑，本质上说明两点：

- 只要 rollout 输出了旧 logprob、奖励和可重算的新 logprob，就可以做 PPO。
- 若 reward 方差较大，可以用同一输入重复采样多条轨迹，使用 leave-one-out baseline 降低方差。

对 LingBot-VA 而言，这比当前单样本 REINFORCE 更适合“stop + jump”联合策略。

---

## 4. 对 LingBot-VA 的总体设计选择

## 4.1 总体目标

将当前 stop-only 的 Hazard Scheduler 升级为：

```text
统一策略 = 累计 Hazard 停止 + 连续相对跳步
```

每一步策略输出同时包含：

- `delta_H_k`：当前步新增停止风险
- `jump distribution`：继续时的跳步分布

训练期：

- 先基于 Hazard 采样 stop/continue
- 若 continue，再从 jump 分布采样跳步比例
- 最终基于整条路径的动作质量和计算代价训练策略

部署期：

- stop 使用累计 Hazard 的确定阈值规则
- jump 使用分布的 mode 或 mean 进行确定性推进

## 4.2 跳步参数化建议

不建议直接让 head 仅通过 `softplus` 生硬输出 `alpha/beta`。

TPDM 的原始做法是直接输出两个实数，然后通过：

```text
alpha = exp(raw_alpha) + 1
beta  = exp(raw_beta) + 1
```

从而保证：

- `alpha > 1`
- `beta > 1`

这样 `Beta(alpha, beta)` 一定是单峰分布，部署期可以稳定使用 mode。

对 LingBot-VA，我更推荐采用与 TPDM 等价但更易控的 `mode + concentration` 参数化，输出：

- `m_k`：jump mode，范围 `(eps, 1 - eps)`
- `c_k`：jump concentration，范围 `(2 + eps, +inf)`

推荐具体形式：

```text
mode_k = sigmoid(raw_m) * (1 - 2 * eps) + eps
concentration_k = softplus(raw_c) + 2.0 + eps
```

再换算为：

```text
alpha_k = m_k * (c_k - 2) + 1
beta_k  = (1 - m_k) * (c_k - 2) + 1
```

这样有两个好处：

- mode 永远存在，部署期可以直接用 mode。
- 避免 `alpha <= 1` 或 `beta <= 1` 导致 mode 不稳定。

## 4.3 时间变量建议

LingBot-VA 当前 scheduler 本质上已经在使用 Flow Matching 的 `sigma / timestep` 表。

因此 V2 方案建议内部统一维护连续 `sigma`，而不是继续把离散 `step_idx` 当作主状态变量。

推荐定义：

- `sigma_k`：当前 video denoising 所处噪声强度
- `r_k`：继续时的相对压缩比例，也就是剩余噪声保留比例
- `sigma_{k+1} = sigma_k * r_k`
- `jump_distance_k = 1 - r_k`

这样：

- 和 TPDM 的 `relative=True` 逻辑一致。
- 和你文档中的“对剩余未去噪量做比例更新”一致。
- 更适合加入自定义 step。

## 4.4 停止策略建议

停止部分维持当前累计 Hazard 设计，不需要改成 TPDM 的纯跳步式收敛终止。

具体建议：

- 若 `sigma_k < sigma_min`，无论 Hazard 值如何都强制停止
- 训练期：`b_k ~ Bernoulli(h_k)`，其中 `h_k = 1 - exp(-delta_H_k)`
- 部署期：
  - 若 `sigma_k < sigma_min`，强制停止
  - 否则若 `executed_video_steps >= K_max`，强制停止
  - 否则若 `executed_video_steps >= K_min` 且 `F_k >= eta`，停止
  - 否则继续

其中：

- `executed_video_steps` 指实际执行了多少次 video 前向
- 在当前 V2 设计中，`executed_video_steps` 与 `decision_count` 实际上始终相等，因为每次调度决策都需要一次 video forward
- 真正需要区分的是：
  - `executed_video_steps`：真实计算开销
  - `equivalent_fixed_steps`：在固定 schedule 上等效跨过了多少步，用于分析覆盖范围

---

## 5. 核心代码改动方案

## 5.1 `wan_va/modules/hazard_scheduler.py`

### 当前问题

当前 head 只支持：

- 输入：`video_feature + step_id`
- 输出：`delta_H_k`

### 建议改动

新增 V2 版 head，推荐命名：

- `HazardJumpSchedulerHead`

建议输出：

- `delta_H_k`
- `jump_mode_k`
- `jump_concentration_k`

并提供以下方法：

- `compute_stop_probability(delta_H_k)`
- `compute_jump_distribution(mode, concentration)`
- `sample_jump_ratio(...)`
- `recompute_step_logprob(...)`

### 兼容建议

不要直接覆盖旧的 stop-only head 语义。

推荐做法：

- 保留旧 `HazardSchedulerHead` 作为 V1 兼容类
- 新增 `HazardJumpSchedulerHead` 作为 V2 类
- checkpoint 中加入 `schema_version` 和 `policy_variant`

这样可以避免旧 checkpoint 在 loader 中直接失效。

## 5.2 `wan_va/utils/scheduler.py`

### 当前问题

当前 `step()` 只能从当前 timestep 前进到离散表中的下一个 timestep。

### 建议改动

新增一个 TPDM 风格接口：

```python
custom_step(model_output, sigma_cur, sigma_next, sample)
```

核心逻辑：

```text
prev_sample = sample + (sigma_next - sigma_cur) * model_output
```

并增加辅助函数：

- `find_nearest_sigma(...)`
- `find_nearest_timestep(...)`
- `sigma_from_timestep(...)`

这样 rollout 可以完全由连续 `sigma` 驱动，而不是绑定离散 index。

## 5.3 `wan_va/utils/hazard_runtime.py`

### 当前问题

当前 rollout 是固定离散循环：

- 每轮固定做一步 video denoise
- 每轮只判断 stop
- 没有 jump 分支

### 建议改动

将主循环改为连续 `sigma` 驱动：

1. 维护当前 `sigma_cur`
2. 执行一次 video branch forward，得到：
   - `video_noise_pred`
   - `video_feature`
3. scheduler head 输出：
   - `delta_H_k`
   - jump 分布参数
4. 若 `sigma_cur < sigma_min`，直接强制 stop
5. 否则采样或判定 stop
6. 若 continue：
   - 采样 `r_k`
   - 计算 `sigma_next = sigma_cur * r_k`
   - 用 `custom_step` 更新 latents
7. 若 stop：
   - 刷新 cache
   - handoff 到 action branch

对 `r_k` 的日志建议同时记录：

- `retention_ratio = r_k`
- `jump_distance = 1 - r_k`

### rollout 结果新增字段

建议在 `RolloutResult` 中新增：

- `executed_video_steps`
- `equivalent_fixed_steps`
- `sigma_cur_history`
- `sigma_next_history`
- `jump_ratio_history`
- `jump_distance_history`
- `policy_features`
- `policy_times`
- `old_logprobs`
- `trajectory_mask`

其中，PPO 重算最关键的不是只保存 `r_k`，而是保存每步：

- `sigma_cur`
- `sigma_next`

因为 rollout 时会对 ratio 做 clamp，后续重算 logprob 时应优先通过：

```text
r_obs = clamp(saved_sigma_next / saved_sigma_cur, eps, 1 - eps)
```

反算观测到的 ratio，再用新策略分布评估其 logprob。

## 5.4 `wan_va/utils/hazard_reward.py`

### 当前问题

当前 reward 默认把 `video_steps` 同时当作：

- 计算代价
- 停止位置

但引入跳步后，这两个语义会分离。

### 建议改动

明确拆开：

- `executed_video_steps`
  - 实际执行了多少次 video forward
  - 用于 cost
- `equivalent_fixed_steps`
  - 在固定 25-step schedule 上等效跨越了多少步
  - 用于分析 jump 的覆盖范围
- `terminal_sigma` 或 `terminal_time`
  - 视频分支停止时的连续位置
  - 用于统计和分析

V2 第一版 reward 建议仍然沿用：

```text
reward = quality - lambda_cost * executed_video_steps
```

先不要把跳步距离本身单独加到 reward 中，避免训练信号过于复杂。

## 5.5 `wan_va/utils/hazard_trainer.py`

### 当前问题

当前 trainer 是 REINFORCE：

```text
loss = -sum(log_prob * advantage)
```

对于联合 stop+jump 策略，方差会明显上升。

### 建议改动

建议升级为 PPO-lite：

1. rollout 阶段保存 `old_logprob`
2. PPO epoch 中只重算 scheduler head 的 `new_logprob`
3. 使用 clipped objective：

```text
ratio = exp(new_logprob - old_logprob)
loss = -min(ratio * A, clip(ratio) * A)
```

其中 jump 部分应按 TPDM 的思路重算：

1. rollout 保存每步 `sigma_cur` 和 `sigma_next`
2. PPO 重算时，通过：

```text
r_obs = clamp(saved_sigma_next / saved_sigma_cur, eps, 1 - eps)
```

反算旧轨迹中真实执行过的 ratio
3. 用新策略头输出的 `Beta(new_alpha, new_beta)` 计算：

```text
new_logprob_jump = log Beta(new_alpha, new_beta).pdf(r_obs)
```

4. stop 部分同样使用缓存的 stop/continue 动作，按新 `h_k` 重算：

```text
new_logprob_stop =
  log(h_k)         if action=stop
  log(1 - h_k)     if action=continue
```

5. 路径级 `new_logprob` 为 stop 和 jump 两部分之和

### 优势估计建议

建议分两阶段：

- 阶段 A：保留当前 EMA baseline，快速完成 V2 联调
- 阶段 B：增加 `rloo_k`，对同一 chunk 采样多条路径，使用 leave-one-out baseline

这部分可以直接借鉴 TPDM 的训练结构，但不必一开始就完全照搬其 trainer 框架。

若阶段 A 仍临时保留 REINFORCE，则建议同时加入以下稳定化措施：

- 对 jump 分支的 logprob 乘以较小系数，如 `jump_logprob_scale = 0.3`
- 提高 `gradient_accumulation_steps`，如从 4 提到 8 或 16
- 对 advantage 做 zero-mean / unit-variance 归一化
- 初期只训练 scheduler head，不放开 backbone

## 5.6 `wan_va/utils/hazard_loader.py`

需要根据 checkpoint 的 `schema_version / policy_variant` 决定加载：

- V1 stop-only head
- V2 stop+jump head

同时建议允许：

- 旧 checkpoint 严格兼容加载
- 新 checkpoint 返回 jump 相关配置

## 5.7 `wan_va/wan_va_server.py`

在线推理必须和训练期 runtime 对齐。

需要做的改动：

- 新增 jump-aware runtime 分支
- 内部维护连续 `sigma`
- stop 使用累计 Hazard 的确定阈值规则
- continue 时 jump 使用 mode 或 mean
- cache refresh 不再依赖离散 `video_steps_used` 下标

建议 metadata 返回：

- `executed_video_steps`
- `equivalent_fixed_steps`
- `sigma_history`
- `jump_ratio_history`
- `jump_distance_history`
- `hazard_h_cumulative`

## 5.8 配置文件

建议在 `train_profiles/robotwin_hazard_*.yaml` 中增加：

```yaml
hazard:
  policy_variant: stop_jump_v2
  jump_parameterization: mode_concentration
  jump_relative: true
  sigma_min: 0.01
  jump_ref_concentration: 20.0
  heuristic_jump_mode: 0.75
  heuristic_jump_concentration: 10.0
  ppo_num_epochs: 4
  cliprange: 0.2
  rloo_k: 1
  use_rloo: false
  jump_logprob_scale: 0.3
  advantage_normalize: true
```

---

## 6. 推荐的训练与部署细节

## 6.1 训练期路径概率

若第 `T` 次决策停止，则路径概率建议写为：

```text
log pi(tau) =
  sum_{k<T} [ log(1 - h_k) + log p(r_k | alpha_k, beta_k) ]
  + log(h_T)
```

其中：

- `h_k = 1 - exp(-delta_H_k)`
- `r_k` 仅在 continue 时存在
- 若某一步是因为 `sigma_k < sigma_min` 被硬停止，则该步应通过 mask 从策略损失中排除，不参与 stop/jump logprob 累积

## 6.2 部署期 jump 取值

推荐优先使用分布 mode，而不是随机采样。

若数值不稳定，则回退到 mean：

```text
mode = (alpha - 1) / (alpha + beta - 2)
mean = alpha / (alpha + beta)
```

由于本文建议使用 `mode + concentration` 参数化，因此 mode 通常是稳定可用的。

日志和可视化中建议不要只看 `r_k`，而要同时输出：

```text
retention_ratio = r_k
jump_distance = 1 - r_k
```

否则容易把“小 ratio = 大跳步”误读为“跳步比例很小所以跳得近”。

## 6.3 jump KL 参考分布

推荐从当前固定 25-step schedule 生成参考分布，而不是使用常数 `Beta(2.5, 2.5)` 作为全局参考。

这会比纯常数 heuristic 更平滑，也更接近现有系统行为。

实现上应直接使用 LingBot-VA 当前 scheduler 的 `sigmas`：

```text
ref_ratios = sigmas[1:] / sigmas[:-1]
```

而不是复用 TPDM 为 SD3 写的 `t-space` reference 公式。

## 6.4 启发式初始化

TPDM 的启发是对 `alpha/beta` 偏置做初始化，而不是只靠随机网络自学。

LingBot-VA 建议：

- stop 部分继续使用指数增长式启发
- jump 部分初始化为偏保守的小步前进，而不是一开始就大跳步
- 推荐起点：
  - `mode = 0.75`
  - `concentration = 10.0`
- 更好的做法是直接从固定 25-step schedule 推导初始 `r_ref_k`，并将其作为 step-conditioned jump 初始化

后续再线性衰减启发权重。

---

## 7. 推荐实施顺序

## 阶段 1：runtime 改造

目标：先把“连续 jump + stop”链路跑通。

改动范围：

- `wan_va/utils/scheduler.py`
- `wan_va/modules/hazard_scheduler.py`
- `wan_va/utils/hazard_runtime.py`

验收标准：

- rollout 能产生 `stop + jump` 轨迹
- 能正确 handoff 到 action branch
- reward 能正常计算

## 阶段 2：特征缓存与 logprob 重算

目标：为 PPO 做准备。

改动范围：

- `wan_va/utils/hazard_runtime.py`
- `wan_va/utils/hazard_reward.py`

验收标准：

- rollout 结果中包含策略头重算所需缓存
- 不重新跑 backbone 也能重算新 logprob

## 阶段 3：PPO-lite 替换 REINFORCE

目标：降低联合策略训练方差。

改动范围：

- `wan_va/utils/hazard_trainer.py`
- `train_profiles/robotwin_hazard_*.yaml`

验收标准：

- 能进行多 epoch PPO 更新
- 日志中能看到 `ratio`、`approx_kl`、`clipfrac`

## 阶段 4：在线推理与评估对齐

目标：训练和部署使用一致调度逻辑。

改动范围：

- `wan_va/wan_va_server.py`
- `wan_va/utils/hazard_loader.py`
- `evaluation/config/*.yaml`

验收标准：

- server 支持 V2 jump runtime
- metadata 能返回 jump 轨迹
- 旧 stop-only checkpoint 仍可加载

---

## 8. 主要风险与对应建议

## 8.1 风险：连续 jump 后 cache refresh 语义不清

原因：

- 当前 refresh 逻辑依赖离散 cutoff index。

建议：

- 统一改成基于当前 `sigma/t` 刷新 cache。
- 不要再把 cache refresh 绑定到离散 step 数。

## 8.2 风险：REINFORCE 方差过大

原因：

- 动作空间从 stop-only 变成 stop+jump 联合策略。

建议：

- 尽快升级到 PPO-lite。
- 若阶段 1 仍暂时使用 REINFORCE：
  - 对 jump logprob 使用较小系数，如 `0.3`
  - 增大 `gradient_accumulation_steps`
  - 对 advantage 做标准化
  - 初期降低 jump 头学习率或冻结一部分辅助参数
- 若仍不稳定，再加入 `rloo_k > 1`。

## 8.3 风险：旧 checkpoint 失效

原因：

- 新 head 参数量和输出结构变化。

建议：

- 引入 `schema_version`
- loader 按 `policy_variant` 分流
- V1/V2 并存一段时间

## 8.4 风险：训练和部署不一致

原因：

- 训练期使用连续 jump，部署期仍按离散 `video_steps_used` 处理。

建议：

- runtime、server、eval 必须一起升级。
- 不建议只改 trainer，不改 server。

## 8.5 风险：`latent_cond` 在大跳步下出现 artifact

原因：

- 当前实现每步 video 更新后都会重新覆盖第一帧条件。
- 在 jump 很大时，`sigma` 下降更快，可能放大条件帧覆盖带来的局部 artifact。

建议：

- 阶段 1 联调时，分别构造：
  - 小跳步样本：`r_k ≈ 0.9`
  - 大跳步样本：`r_k ≈ 0.3`
- 对比 action 质量和视频特征稳定性
- 若出现异常，再单独检查 `latent_cond` 的 overwrite 位置是否需要跟随 `sigma` 做特殊处理

---

## 9. 最终建议

结合 TPDM 后，LingBot-VA 的跳步策略不应设计成“离散 step 分类器”，而应采用：

```text
累计 Hazard 停止 + 连续相对跳步 Beta 分布
```

最合理的落地路线是：

1. 先引入 TPDM 风格 `custom_step`。
2. 再引入 `mode + concentration` 跳步头。
3. rollout 阶段缓存策略特征。
4. trainer 从 REINFORCE 升级到 PPO-lite。
5. 最后完成 server / eval / checkpoint 的整体对齐。

这样做的优点是：

- 与你的统一方案文档保持一致。
- 与 TPDM 的工程模式高度一致。
- 能避免把 jump 设计成不自然的离散动作。
- 为后续做 PPO 和参考分布 KL 留出清晰接口。
