# Cutoff Stage 1 Code Change Log

## 1. 背景

本文件记录 `shared_backbone_action_only_adapter_cutoff_plan_v2.docx` 在当前仓库中的第一阶段代码落地情况。

第一阶段的目标不是训练 scheduler，也不是训练 action-only adapter，而是先回答一个更基础的问题：

> 在当前 LingBot-VA checkpoint 上，视频分支停在不同离散 cutoff 位置时，动作误差是否会发生可观测变化？

如果这个问题没有正信号，后续再做 adapter、oracle、scheduler 的价值就会很弱。

因此，第一阶段的实现重点是：

1. 提供一条离线 cutoff 扫描链路。
2. 保证 cutoff 的 handoff 语义是正确的。
3. 输出可以直接观察趋势的数值和图像结果。
4. 为后续 oracle 标注保留稳定的 state 级样本标识。

---

## 2. 第一阶段范围

第一阶段只做下面这些事：

- 离线扫描 `cutoff_idx -> action loss` 曲线。
- 使用固定 checkpoint、固定动作预算、固定噪声种子比较不同视频 cutoff。
- 生成 `summary.csv` 和 `per_state_loss.pt`。
- 生成曲线图，辅助快速观察趋势。

第一阶段明确 **没有** 做下面这些事：

- 没有训练 cutoff scheduler。
- 没有训练 action-only adapter。
- 没有改 Transformer 主干结构。
- 没有改训练主流程。
- 没有改在线 websocket server 推理链。

---

## 3. 术语约定

为了避免把概念混在一起，当前实现中统一使用下面这组术语：

- `cutoff c`
  - 研究概念，表示“视频去噪到什么程度停下”。
- `cutoff_idx`
  - 工程概念，表示“在当前固定 scheduler 离散网格上停在第几个位置”。
- `full cutoff`
  - 当前固定视频推理总步数对应的完整视频预算。

在当前 `robotwin` 配置下，视频总步数是 `25`，因此：

- `cutoff_idx=1` 表示非常早停。
- `cutoff_idx=25` 表示完整视频预算跑满。

第一阶段扫描的是离散 `cutoff_idx`，不是直接回归连续 `cutoff c`。

---

## 4. 已修改的已有文件

### 4.1 `wan_va/dataset/lerobot_latent_dataset.py`

这是第一阶段唯一修改的已有代码文件。

#### 改动 A：补样本级元信息

为了让扫描结果后续可以直接复用到 oracle 标注阶段，给每个样本补充了稳定标识：

- `global_index`
- `source_dataset_id`
- `sample_index`
- `episode_index`
- `local_start_frame`
- `local_end_frame`
- `dataset_repo_id`
- `state_uid`

这样 `per_state_loss.pt` 里的每条记录都能稳定对应回具体状态。

#### 改动 B：支持小规模 smoke test

新增了 `max_dataset_repos` 配置项读取逻辑，用于限制初始化多少个 LeRobot 子数据集。

原因是原始 `MultiLatentLeRobotDataset` 初始化时会把整个 `dataset_path` 下的所有 `meta/info.json` 子数据集全部加载一遍。即使只想测试 `1` 个样本，启动时也会把 `100` 个子数据集都扫一遍，冷启动非常慢。

这次改动后，可以通过扫描脚本传入 `--max-datasets 1`，只初始化前 `1` 个子数据集，用于快速 smoke test。

---

## 5. 新增文件

### 5.1 `wan_va/utils/cutoff_scan.py`

这是第一阶段的核心执行模块，负责离线 cutoff 扫描。

主要职责：

- 构造视频和动作分支各自的 scheduler。
- 让视频分支只积分到指定 `cutoff_idx`。
- 以“cutoff 后的真实视频状态”刷新 cache。
- 让动作分支在该 cache 条件下完整生成动作。
- 计算动作误差指标。
- 汇总样本级和 cutoff 级结果。

#### 为什么要单独写这个模块

因为当前在线推理路径里，视频早停时的 cache handoff 语义并不适合作为研究用定义。

在研究语义上，我们需要的是：

1. 视频先真正更新到目标 cutoff。
2. 然后把 **cutoff 后** 的视频表示交给动作端。

因此在 `cutoff_scan.py` 中专门实现了“精确 handoff”逻辑。

#### 核心函数含义

- `_run_video_to_cutoff(...)`
  - 把视频分支推进到目标 `cutoff_idx`。
- `_refresh_video_cache_exact(...)`
  - 用 cutoff 后的视频 latent 刷新 cache。
- `_run_action_from_cache(...)`
  - 在当前 video cache 条件下完整跑动作推理。
- `compute_action_metrics(...)`
  - 计算 `masked_action_mse`、`first_pred_action_mse`、`denorm_action_l1`。
- `scan_cutoff_curve(...)`
  - 外层扫描函数，负责逐样本、逐 cutoff 汇总结果。

#### 当前指标解释

- `masked_action_mse`
  - 主指标，和训练动作损失口径最接近。
- `denorm_action_l1`
  - 更易解释的人类可读误差。
- `first_pred_action_mse`
  - 表示第一个真正由模型预测且有有效监督标签的动作帧误差。

---

### 5.2 `script/scan_robotwin_cutoff_curve.py`

这是第一阶段的命令行入口脚本。

主要职责：

- 读取 config。
- 解析 checkpoint 和 dataset 路径。
- 解析 `cutoff_idx` 列表。
- 调用 `CutoffScanner` 执行离线扫描。
- 输出结构化结果文件。

输出文件包括：

- `summary.csv`
  - cutoff 级聚合统计结果。
- `per_state_loss.pt`
  - state 级原始结果。
- `scan_args.json`
  - 本次扫描所用参数记录。

#### 后续补充改动

在实际 smoke test 过程中，这个脚本又补了两个实用改动：

- 自动把仓库根目录加入 `sys.path`
  - 解决直接运行 `script/...py` 时 `wan_va` 包无法导入的问题。
- 新增 `--max-datasets`
  - 用于限制初始化多少个子数据集。

---

### 5.3 `script/plot_cutoff_scan.py`

这是第一阶段新增的绘图脚本，用于把扫描结果直接变成图片，方便快速判断趋势。

输入：

- `summary.csv`
- `per_state_loss.pt`
- `scan_args.json`

输出：

- `plots/cutoff_mean_curves.png`
- `plots/cutoff_per_state_curves.png`

#### 为什么不只画一张均值曲线

如果只画均值曲线，很容易被少数离群样本误导。

因此当前脚本默认画两张图：

- `cutoff_mean_curves.png`
  - 看平均趋势。
  - 带 `mean ± SEM` 误差带。
- `cutoff_per_state_curves.png`
  - 看每个 state 自己的趋势。
  - 防止平均值掩盖个体差异。

这是比“只画均值折线”更稳妥的处理方式。

---

## 6. 运行与环境处理过程

第一阶段在实际落地时，除代码改动外，还处理了若干运行问题：

### 6.1 直接运行脚本时找不到 `wan_va`

问题：

- 直接执行 `python3 script/scan_robotwin_cutoff_curve.py ...` 时，入口目录是 `script/`，Python 不会自动把仓库根目录加入模块搜索路径。

处理：

- 在脚本中手动把 repo root 插入 `sys.path`。

### 6.2 `lerobot` 依赖链不完整

问题：

- 当前 `lingbot-va` 环境已经装了 `lerobot==0.3.3`，但缺少它依赖的 `datasets`、`pyarrow`、`jsonlines`。

处理：

- 在当前环境补装了最小必需依赖：
  - `datasets`
  - `pyarrow`
  - `jsonlines`

说明：

- 这一步是为了让第一阶段离线扫描链路跑起来，不是对整个环境进行完整重装。

### 6.3 GPU0 显存不足

问题：

- 默认跑在 `cuda:0` 时，GPU0 已有其他大进程占用，导致加载模型时 OOM。

处理：

- 使用 `CUDA_VISIBLE_DEVICES=1`，把物理 GPU1 暴露给脚本。
- 脚本内部继续使用 `--device cuda:0` 即可，因为对进程而言可见设备只有 GPU1。

### 6.4 smoke test 冷启动太慢

问题：

- 即使只想测试 `1` 个样本，原始数据集也会初始化全部 `100` 个子数据集。

处理：

- 新增 `--max-datasets`，让 smoke test 只初始化前 `1` 个子数据集。

---

## 7. 已完成的 smoke test

### 7.1 最小 smoke test

已实际跑通下面这条命令：

```bash
CUDA_VISIBLE_DEVICES=1 python3 script/scan_robotwin_cutoff_curve.py \
  --config robotwin_train \
  --checkpoint /mnt/sda/syr/models/lingbot-va-posttrain-robotwin \
  --dataset-path /mnt/sda/syr/lerobot_datasets/robotwin-clean-and-aug-lerobot \
  --cutoff-indices 1 \
  --max-datasets 1 \
  --max-samples 1 \
  --seed 42 \
  --num-workers 0 \
  --init-workers 1 \
  --device cuda:0 \
  --output-dir train_out/cutoff_scan/robotwin_stage1_smoke_gpu1
```

结果：

- 成功生成 `summary.csv`
- 成功生成 `per_state_loss.pt`
- 成功生成 `scan_args.json`

说明第一阶段链路已经具备最小可运行能力。

### 7.2 小规模趋势测试

随后又实际跑通了这条命令：

```bash
CUDA_VISIBLE_DEVICES=1 python3 script/scan_robotwin_cutoff_curve.py \
  --config robotwin_train \
  --checkpoint /mnt/sda/syr/models/lingbot-va-posttrain-robotwin \
  --dataset-path /mnt/sda/syr/lerobot_datasets/robotwin-clean-and-aug-lerobot \
  --cutoff-indices 1,4,8 \
  --max-datasets 1 \
  --max-samples 8 \
  --seed 42 \
  --num-workers 0 \
  --init-workers 1 \
  --device cuda:0 \
  --output-dir train_out/cutoff_scan/robotwin_stage1_small_gpu1
```

结果目录：

- `train_out/cutoff_scan/robotwin_stage1_small_gpu1/summary.csv`
- `train_out/cutoff_scan/robotwin_stage1_small_gpu1/per_state_loss.pt`
- `train_out/cutoff_scan/robotwin_stage1_small_gpu1/scan_args.json`
- `train_out/cutoff_scan/robotwin_stage1_small_gpu1/plots/cutoff_mean_curves.png`
- `train_out/cutoff_scan/robotwin_stage1_small_gpu1/plots/cutoff_per_state_curves.png`

---

## 8. 当前阶段观察到的结果

在 `adjust_bottle-aloha-agilex_randomized_500-1000` 这个单任务的 8 个样本小测试中：

- `cutoff_idx=1 -> 4 -> 8` 时
  - `masked_action_mse` 下降
  - `denorm_action_l1` 下降

这说明当前 checkpoint 下，视频 budget 变大时，动作误差确实存在改善趋势。

这属于第一阶段想要的“正信号”。

但目前结论仍然很弱，原因有三个：

1. 样本数只有 `8`。
2. 只覆盖了 `1` 个任务。
3. 只扫了 `1,4,8` 三个较早的 cutoff。

因此现在能得出的正确结论是：

> 这个方向值得继续做。

而不是：

> 已经足够支持后续 scheduler 训练结论。

---

## 9. 当前已知问题

### 9.1 `first_action_mse` 问题已修正

在早期版本里，动作序列第一个 frame 会被 `action_cond=0` 覆盖，因此 `first_action_mse` 会出现“所有 cutoff 下都完全相同”的现象。

后续已经把这个指标替换为：

- `first_pred_action_mse`

新指标的定义是：

- 跳过第 0 帧
- 找到第一个真正由模型预测且有有效监督标签的动作帧
- 只在该帧上计算 MSE

这比旧指标更合理，但它依然只是一个局部指标，因此当前阶段仍建议把下面两个指标作为主判断依据：

- `masked_action_mse`
- `denorm_action_l1`

### 9.2 当前 `--max-datasets` 只是截取前 N 个 repo

这适合 smoke test，但不适合作为正式实验抽样方式。

后续如果要更系统地跑第一阶段，应该继续补：

- 按任务名过滤
- 只测 clean 子集 / aug 子集
- 指定某个任务模式

### 9.3 环境仍然不是完全干净的官方组合

当前为了先打通第一阶段链路，只补了最小必需依赖。

这意味着：

- 现在这套环境足够跑第一阶段离线扫描
- 但它不等于“官方标准完整环境”

如果后续要在这台机器上长期重复做更大规模实验，仍建议整理一份明确的环境锁定方案。

---

## 10. 当前最合理的后续动作

从工程优先级看，下一步最合理的是：

1. 扩大小规模验证
   - 例如 `--max-datasets 3~5`
   - `--max-samples 50~100`
   - `cutoff-indices 1,4,8,12,16,20,25`
2. 使用任务级过滤，而不是只取前 N 个子数据集
3. 对 clean / aug 子集分开验证
4. 在确认不同任务上普遍存在 tradeoff 之后，再进入 adapter 训练和 oracle 标注阶段

---

## 11. 第一阶段代码改动总结

一句话总结当前已经完成的第一阶段代码工作：

> 已经为当前仓库补齐了一条可运行的离线 cutoff 扫描与绘图链路，并通过小规模 GPU1 smoke test 验证了“更大的视频 cutoff 会改善动作误差”的初步正信号。

---

## 12. 阶段 1 补充更新

本节记录在初版第一阶段实现完成之后，围绕“让实验更可控、更可解释”追加的改动。

### 12.1 新增任务级过滤能力

扫描脚本新增了：

- `--dataset-substring`

它允许通过 repo 路径子串筛选 LeRobot 子数据集，例如：

- `--dataset-substring adjust_bottle`
- `--dataset-substring adjust_bottle,aug_500`

这样做的目的，是让第一阶段验证从“只取前几个 repo”升级为“明确指定任务或子数据子集”，从而让结论更干净、更易解释。

### 12.2 `first_pred_action_mse` 替换旧指标

为了修复旧版 `first_action_mse` 恒定不变的问题，扫描脚本已将该指标替换为：

- `first_pred_action_mse`

输出文件 `summary.csv` 的字段也同步更新为：

- `masked_action_mse_mean/std`
- `first_pred_action_mse_mean/std`
- `denorm_action_l1_mean/std`

### 12.3 已完成的单任务 50 样本验证

使用下面这条命令，对 `adjust_bottle` 单任务跑通了 50 个样本、7 个 cutoff 的验证：

```bash
CUDA_VISIBLE_DEVICES=1 python3 script/scan_robotwin_cutoff_curve.py   --config robotwin_train   --checkpoint /mnt/sda/syr/models/lingbot-va-posttrain-robotwin   --dataset-path /mnt/sda/syr/lerobot_datasets/robotwin-clean-and-aug-lerobot   --dataset-substring adjust_bottle   --cutoff-indices 1,4,8,12,16,20,25   --max-datasets 1   --max-samples 50   --seed 42   --num-workers 0   --init-workers 1   --device cuda:0   --output-dir train_out/cutoff_scan/adjust_bottle_stage1_gpu1
```

对应结果目录：

- `train_out/cutoff_scan/adjust_bottle_stage1_gpu1/summary.csv`
- `train_out/cutoff_scan/adjust_bottle_stage1_gpu1/per_state_loss.pt`
- `train_out/cutoff_scan/adjust_bottle_stage1_gpu1/plots/cutoff_mean_curves.png`
- `train_out/cutoff_scan/adjust_bottle_stage1_gpu1/plots/cutoff_per_state_curves.png`

### 12.4 这次 50 样本结果的研究含义

在 `adjust_bottle` 单任务上，50 个样本的结果表明：

- `masked_action_mse` 在 `cutoff_idx=8` 最低
- `denorm_action_l1` 在 `cutoff_idx=12` 左右进入平台
- `full cutoff=25` 并不是统一最优

这说明当前模型下存在下面这个重要现象：

> 视频预算不是越大越一定更好；对这个任务来说，中等 cutoff 往往已经足够，继续增加到 full cutoff 的收益很小，甚至会回退。

这比早期 8 样本小测试更强，因为它已经展示出：

- 中间 cutoff 的平均表现优于 full cutoff
- 不同状态的最优 cutoff 分布并不集中在 `25`

这为后续做 oracle cutoff 和 scheduler 学习提供了更直接的动机。

### 12.5 当前阶段的正确结论

截至目前，第一阶段可以支持的结论是：

1. 在单任务内部，不同 cutoff 确实会显著影响动作误差。
2. full cutoff 不是统一最优，存在明显的中等 cutoff 区间。
3. 不同状态对 cutoff 的需求不同，因此后续学习 state-dependent cutoff 是有研究价值的。

但当前仍然不能直接支持：

- “多任务下都已经验证成立”
- “可以跳过 adapter，直接进入最终 scheduler”

因此更合理的后续动作仍然是：

- 扩大到多任务验证
- clean / aug 分开验证
- 再决定是否进入 adapter / oracle 阶段



---

## 13. 阶段 2 增量更新：action residual adapter 接入

本节记录在第一阶段 cutoff 可行性验证之后，为后续“只微调动作分支”的实验所做的第二阶段增量改动。

这次改动的目标不是直接训练 cutoff scheduler，而是先把下面这条实验链路补齐：

1. 在现有 Transformer 结构上增量接入 action residual adapter。
2. 支持“只训练动作头 + adapter，冻结其他模块”。
3. 把 adapter 与 trainable 开关统一纳入 profile 配置体系。
4. 让 cutoff scan 脚本也能做 baseline / adapter shell / trained adapter 的 A/B 对照。

### 13.1 这次改动的研究目的

第二阶段想先回答的是：

> 如果只给动作 token 增加一个轻量 residual adapter，并且只训练动作头与 adapter，本身是否可行，是否会明显破坏原模型能力？

因此第二阶段的第一优先级不是“先提升指标”，而是：

- 先保证结构接入方式正确。
- 先保证初始化是 no-op。
- 先保证冻结范围与实验设定一致。
- 先保证扫描脚本可以直接做结构接入前后的对照。

### 13.2 已修改的已有文件

#### 13.2.1 `wan_va/modules/model.py`

这是第二阶段最核心的结构改动文件。

本次新增了 `ResidualAdapter`，其结构为：

- `LayerNorm`
- `down`
- `SiLU`
- `Dropout`
- `up`
- residual add

并在 `WanTransformer3DModel` 中新增了：

- `enable_action_residual_adapter`
- `action_adapter_dim`
- `action_adapter_dropout`

当开关打开时，会为每个 Transformer block 创建一个动作专属 adapter。

#### 关键实现语义

1. adapter 插入位置在每个 block 完成 FFN 残差之后
   - 也就是 block 返回之后再额外接一层动作专属 residual adapter。
2. 训练态只对 `action_hidden_states` 应用 adapter
   - 不再对 `condition_action_hidden_states` 应用 adapter。
3. 推理态仅在 `action_mode=True` 时应用 adapter
   - 视频分支不会经过这条 adapter 路径。

#### 为什么训练态不再改 `condition_action_hidden_states`

在早期接入版本中，训练态 adapter 同时作用于：

- `action_hidden_states`
- `condition_action_hidden_states`

后续复核后把它收窄为：

- 只作用于 `action_hidden_states`

原因是第二阶段当前想验证的是：

> 只对“需要被去噪生成的动作 token”增加一条轻量可训练残差路径。

而 `condition_action_hidden_states` 更接近条件上下文，不是本轮需要直接建模的生成目标分支。为了让可行性实验更干净、变量更少，当前版本不再对条件动作 token 使用 adapter。

#### 初始化修正：为什么 `alpha` 必须从 `0` 改成 `1`

在最初版本里，adapter 使用了：

- `alpha = 0`
- `up.weight = 0`
- `up.bias = 0`

这会导致前向虽然是 no-op，但反向梯度也全部变成 `0`，adapter 分支会被直接“置死”，无法学习。

后续已修正为：

- `alpha = 1`
- `up.weight = 0`
- `up.bias = 0`

这样仍然满足：

- 初始前向输出与原模型一致

同时又满足：

- `up` 在第一步就能拿到非零梯度
- adapter 可以真正开始学习

这一步已经在 `lingbot-va` conda 环境里做过最小梯度实验验证。

#### 13.2.2 `wan_va/train.py`

这是第二阶段冻结逻辑的核心位置。

本次新增了：

- `configure_trainable_modules()`
- `log_trainable_params()`

训练逻辑改为：

1. 先对整个 transformer 执行 `requires_grad_(False)`。
2. 再根据 config 开关有选择地放开模块。

当前支持的 trainable 开关包括：

- `freeze_backbone`
- `freeze_embeddings`
- `train_action_adapter`
- `train_action_head`
- `train_video_heads`
- `train_time_embedder`

在当前建议 profile 下，实验语义是：

- 冻结 backbone
- 冻结 embeddings
- 只训练 `action_residual_adapters`
- 只训练 `action_proj_out`

这样就实现了“只微调动作头和 adapter”的实验设定。

#### 13.2.3 `wan_va/modules/utils.py`

`load_transformer(...)` 新增了：

- `model_overrides`

原因是旧 checkpoint 的 `config.json` 里并没有 adapter 字段，如果不支持 overrides，就无法做到：

> 加载旧 checkpoint 权重，同时在当前代码里显式启用 adapter 结构

#### 13.2.4 `wan_va/configs/shared_config.py`

新增了 adapter 与 trainable 开关的统一默认值，包括：

- `enable_action_residual_adapter`
- `action_adapter_dim`
- `action_adapter_dropout`
- `freeze_backbone`
- `freeze_embeddings`
- `train_action_adapter`
- `train_action_head`
- `train_video_heads`
- `train_time_embedder`

这样所有下游配置都可以从统一默认值出发。

#### 13.2.5 `wan_va/configs/va_robotwin_train_cfg.py`

新增了从环境变量到训练 config 的映射，负责把：

- adapter 结构开关
- adapter 超参数
- trainable 冻结开关

统一注入最终训练配置。

这里保持了“纯 env -> config 映射层”的职责，没有引入额外复杂逻辑。

#### 13.2.6 `script/run_va_posttrain_profile.sh`

profile loader 新增了两个 YAML section：

- `model`
- `trainable`

并把它们映射为环境变量：

- `LINGBOT_VA_ENABLE_ACTION_ADAPTER`
- `LINGBOT_VA_ACTION_ADAPTER_DIM`
- `LINGBOT_VA_ACTION_ADAPTER_DROPOUT`
- `LINGBOT_VA_FREEZE_BACKBONE`
- `LINGBOT_VA_FREEZE_EMBEDDINGS`
- `LINGBOT_VA_TRAIN_ACTION_ADAPTER`
- `LINGBOT_VA_TRAIN_ACTION_HEAD`
- `LINGBOT_VA_TRAIN_VIDEO_HEADS`
- `LINGBOT_VA_TRAIN_TIME_EMBEDDER`

这一步的作用是把 profile 配置真正打通到训练系统。

#### 13.2.7 `script/scan_robotwin_cutoff_curve.py`

第二阶段还补了一个很关键的对照能力：

扫描脚本新增了：

- `--enable-action-adapter`
- `--action-adapter-dim`
- `--action-adapter-dropout`

并在加载 transformer 时把它们通过 `model_overrides` 传入。

这样 cutoff scan 就可以直接做三组对照：

1. `baseline`
   - 原模型，不启用 adapter。
2. `zero-init adapter shell`
   - 启用 adapter 结构，但不加载训练后的 adapter 权重，只使用 no-op 初始化外壳。
3. `trained adapter`
   - 加载训练过 adapter 的 checkpoint。

这对第二阶段是必要的，因为它可以把“结构接入问题”和“训练效果问题”区分开。

### 13.3 新增文件

#### 13.3.1 `train_profiles/robotwin_local_4gpu_action_adapter.yaml`

这是第二阶段新增的训练 profile。

之所以选择新建 profile，而不是直接污染原有 `robotwin_local_4gpu.yaml`，原因是：

- baseline 与 adapter 实验应当明确分离
- 避免后续回看实验记录时把两种设定混在一起

当前 profile 的核心设定是：

- 打开 action residual adapter
- `action_adapter_dim = 256`
- `action_adapter_dropout = 0.0`
- 冻结 backbone
- 冻结 embeddings
- 训练 action adapter
- 训练 action head
- 不训练 video head
- 不训练 time embedder

### 13.4 第二阶段当前推荐的验证顺序

当前最合理的验证顺序是：

1. `baseline cutoff scan`
   - 不启用 adapter，先记录原模型表现。
2. `zero-init adapter shell cutoff scan`
   - 只启用 adapter 结构，确认结构接入本身不会明显伤害模型。
3. `adapter + action head` 小步训练
   - 保持主干冻结，先做可行性验证。
4. 对训练后 checkpoint 再跑 cutoff scan
   - 观察 full cutoff 与中低 cutoff 是否变得更稳。

### 13.5 目前阶段能支持的正确结论

截至这次代码改动完成，第二阶段能支持的结论是：

1. 当前仓库已经具备 action residual adapter 的最小可运行结构。
2. 已经具备“只训练动作头和 adapter”的冻结控制能力。
3. 已经具备 profile 级统一配置能力。
4. 已经具备 baseline / adapter shell / trained adapter 的 cutoff scan 对照能力。

但当前还不能支持：

- adapter 一定能提升指标
- adapter 一定能改善 full cutoff 表现
- 可以跳过 baseline / shell 对照直接进入最终结论

因此当前最合理的下一步仍然是：

- 先跑 baseline cutoff scan
- 再跑 zero-init adapter shell cutoff scan
- 确认结构接入本身安全后，再进入小步训练验证
