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
  - 计算 `masked_action_mse`、`first_action_mse`、`denorm_action_l1`。
- `scan_cutoff_curve(...)`
  - 外层扫描函数，负责逐样本、逐 cutoff 汇总结果。

#### 当前指标解释

- `masked_action_mse`
  - 主指标，和训练动作损失口径最接近。
- `denorm_action_l1`
  - 更易解释的人类可读误差。
- `first_action_mse`
  - 当前实现里这个指标还不可靠，见“已知问题”。

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

### 9.1 `first_action_mse` 当前不可靠

在当前实现里，动作序列第一个 frame 会被 `action_cond=0` 覆盖，因此当前 `first_action_mse` 会出现“所有 cutoff 下都完全相同”的现象。

这不是模型真的对第一个动作完全不敏感，而是当前指标定义和动作初始化方式叠在一起导致的。

因此现阶段更可靠的两个指标是：

- `masked_action_mse`
- `denorm_action_l1`

后续如果继续完善第一阶段，应该把 `first_action_mse` 改成“第一个可预测动作位”的误差，而不是当前这个定义。

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
2. 修正 `first_action_mse` 指标定义
3. 增加任务级过滤，而不是只取前 N 个子数据集
4. 在确认不同任务上普遍存在 tradeoff 之后，再进入 adapter 训练和 oracle 标注阶段

---

## 11. 第一阶段代码改动总结

一句话总结当前已经完成的第一阶段代码工作：

> 已经为当前仓库补齐了一条可运行的离线 cutoff 扫描与绘图链路，并通过小规模 GPU1 smoke test 验证了“更大的视频 cutoff 会改善动作误差”的初步正信号。
