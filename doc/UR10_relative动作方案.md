# UR10 relative 动作方案

## 目标

当前 UR10 `xyzgripper` 路线使用的是 absolute 动作语义：

- `tcp.x`
- `tcp.y`
- `tcp.z`
- `gripper.pos`

训练时模型直接回归未来若干步的绝对 TCP 位置。在线推理时，server 一次生成整块 chunk，client 再逐步执行。这种做法在真机上暴露出两个明显问题：

1. absolute `xyz` 对首块多步外推非常敏感，中间步容易直接掉出数据集工作空间范围。
2. 训练时 chunk 内动作可以利用更强的 chunk 上下文，而推理首块只有当前观测，absolute 目标点语义会放大这种时序失配。

RobotWin 路线之所以更稳定，一个关键原因是它在训练时使用了 relative pose 语义。UR10 可以沿这个方向改造。

本文给出一版面向当前代码库的 UR10 relative 动作方案。

## 设计原则

改造目标不是简单“把数值减掉初始值”，而是让训练和在线推理语义一致，并尽量复用现有代码。

原则如下：

1. 只改动作语义，不改观测相机和整体 server/client 通信接口。
2. 第一阶段只做 `xyz + gripper` 的 relative 版本，不恢复姿态学习。
3. relative 基准优先使用 chunk 起点 TCP，而不是 episode 固定初始位。
4. 推理时模型输出 relative 动作，client 在发送给机器人前恢复成 absolute TCP。
5. 数据集范围保护仍保留，但范围统计应切换到 relative 动作分布。

## 推荐语义

推荐把 UR10 动作改成：

- `delta_x = tcp.x_t - tcp.x_anchor`
- `delta_y = tcp.y_t - tcp.y_anchor`
- `delta_z = tcp.z_t - tcp.z_anchor`
- `gripper.pos_t`

其中：

- `tcp.*_anchor` 是当前 chunk 的锚点 TCP
- 对训练数据而言，锚点定义为该样本切片的起始时刻 TCP
- 对在线推理而言，锚点定义为当前 chunk 开始执行前机器人的实时 TCP

这样可以保证：

- 训练的 `frame_idx=1` 对应“相对当前起点的前 4 步”
- 推理首块 `frame_idx=1` 也对应“相对当前机器人状态的前 4 步”

这是和现有 chunk rollout 最一致的做法。

## 不推荐的语义

### 1. 相对 episode 初始位

不推荐：

- `tcp.x_t - tcp.x_episode_init`
- `tcp.y_t - tcp.y_episode_init`
- `tcp.z_t - tcp.z_episode_init`

原因：

- 对长轨迹来说，relative 值会越来越大
- 不能缓解 chunk 内多步外推问题
- 也不能像 RobotWin 那样形成稳定的局部控制序列

### 2. 纯一步 delta

即：

- `tcp_t - tcp_{t-1}`

这是另一条可行路线，但它会改变训练和在线回放逻辑更多：

- 需要在推理时递推积分
- cache 回写语义也要更谨慎处理

相比之下，“相对 chunk anchor”更适合先落地。

## 代码改造点

### 1. 数据集动作预处理

当前入口：

- `wan_va/dataset/lerobot_latent_dataset.py`

当前 UR10 路线没有 special case，只有 RobotWin 在 `_action_post_process()` 里做 relative pose 变换。

建议新增一个 UR10 relative 分支，例如：

- `config.action_representation = "absolute"` 或 `"relative_chunk_anchor"`

在 `_action_post_process()` 里，对 `env_type == "none"` 且 `action_representation == "relative_chunk_anchor"` 时：

1. 从原始 action 中取 `x/y/z/gripper`
2. 选取 anchor：
   - anchor 使用 `action[0, :3]`
3. 构造：
   - `action[:, 0] -= anchor_x`
   - `action[:, 1] -= anchor_y`
   - `action[:, 2] -= anchor_z`
   - `gripper` 保持原值
4. 再继续现有：
   - 前置 pad
   - 通道对齐
   - quantile 归一化
   - reshape 成 `(C, F, 4, 1)`

注意：

- relative 化必须发生在 quantile 归一化之前
- 新的 `norm_stat` 也必须基于 relative 数据重新统计

### 2. 配置文件

建议新增一份独立配置，不覆盖现有 absolute 版本：

- `wan_va/configs/va_ur10_follower_safe_xyzgripper_relative_cfg.py`

建议字段：

```python
action_representation = "relative_chunk_anchor"
relative_action_base = "chunk_anchor"
```

并重新提供 relative 动作的 `q01/q99`。

### 3. 训练配置

如果后续恢复本地 train cfg 文件，建议新增：

- `va_ur10_follower_safe_xyzgripper_relative_train_cfg.py`

训练 profile 也单独起名，避免和 absolute 混淆。

### 4. 在线推理 client

当前入口：

- `lerobot/examples/rtc/eval_ur10_lingbot_va_xyzgripper.py`

当前 `action_chunk_to_robot_action()` 是把模型输出直接当 absolute `x/y/z`：

```python
"tcp.x": float(step[0])
"tcp.y": float(step[1])
"tcp.z": float(step[2])
```

relative 版本应改为：

```python
"tcp.x": anchor_x + float(step[0])
"tcp.y": anchor_y + float(step[1])
"tcp.z": anchor_z + float(step[2])
```

其中 `anchor_x/y/z` 用当前 chunk 开始前观测到的 `chunk_anchor_state`。

也就是说：

- relative -> absolute 的恢复发生在 client 里
- server 仍然只负责预测 action chunk

### 5. 执行期间的 anchor 是否更新

建议：

- 一个 chunk 内 anchor 固定不变
- 下一 chunk 开始时再更新为新的实时 TCP

这样与训练语义一致：

- 一个训练样本的整段动作都相对同一个切片起点

不要在 chunk 内每一步都更新 anchor，否则训练/推理又会不一致。

### 6. 动作范围保护

当前 clamp 用的是 absolute 动作数据集统计。

如果切到 relative 语义，必须同步切换为：

- relative `delta_x/y/z`
- absolute `gripper`

否则在线 clamp 会把 relative 动作错误地当 absolute 工作空间去裁剪。

因此需要：

1. 为 relative 数据重新统计动作 `min/max` 和 `q01/q99`
2. 真机 client 在 relative 模式下读 relative 统计
3. clamp 发生在“模型输出 relative 动作”这一层
4. clamp 后再加回 anchor 变成 absolute TCP

## 训练与推理时序关系

改成 relative 之后，并不能自动消除所有时序问题，但会显著缓解：

- absolute 模式：
  - 预测的是未来绝对点
  - 中间一步错了就可能直接掉出工作空间

- relative 模式：
  - 预测的是相对当前起点的小位移
  - 更像局部控制序列
  - 更适合 chunk rollout

但是以下问题仍然存在：

- 训练时 `frame_idx=0` 是 pad 占位 frame
- 训练 mask 仍然是 chunk-level，不是 frame-level
- 推理首块 `frame_idx=1` 仍然缺少未来真实观测

所以 relative 方案是“显著缓解”，不是“完全根治”。

## 推荐实施顺序

### Phase 1

先做最小可行版本：

1. 新增 UR10 relative 配置
2. 在数据集里实现 `relative_chunk_anchor`
3. 重算 relative 动作统计
4. 在 UR10 client 里把 relative 恢复成 absolute TCP
5. 跑 shadow
6. 跑短时真机

### Phase 2

如果 relative 版本仍然有明显 chunk 内漂移，再做：

1. 训练 mask 从 chunk-level causal 改为 frame-level causal
2. 或缩小 `frame_chunk_size`

## 验证标准

relative 方案上线后，至少看这几项：

1. shadow 首块 `frame_idx=1` 的动作是否仍大量掉出 relative 数据分布
2. 真机首块是否明显减少：
   - dataset clamp
   - servo guard
3. 动作 `0~3` 是否仍然塌缩
4. 中间 frame slot 是否仍然整体掉到底部工作空间附近

## 预期效果

如果 relative 方案有效，应该看到：

1. 首块动作更接近局部平滑位移，而不是远距离 absolute 跳变
2. 中间几个动作不再集体掉到 `x/z` 下界以下
3. clamp/guard 触发频率下降
4. 真机轨迹更像连续局部控制，而不是离散目标点串

## 风险

1. 如果 relative 统计做错，归一化会失真。
2. 如果 client 用错 anchor，真机会直接跑偏。
3. 如果 relative 只在推理侧改而训练侧不改，会比现在更糟。

## 结论

UR10 当前 absolute `xyzgripper` 路线与 chunk rollout 的兼容性较差。  
最合理的下一步是把动作语义改成：

- 相对 chunk 起点 TCP 的 `delta xyz`
- 保留 absolute `gripper`

这条方案与 RobotWin 当前成功经验最接近，同时对现有 server/client 改动相对可控。
