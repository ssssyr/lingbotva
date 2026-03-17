# DSW 驱动、CUDA 与四卡训练排查说明

本文档基于当前仓库 `/home/syr/code/lingbot-va` 和 2026-03-14 这次远端排查结果整理，目标是说明：

1. 为什么当前 DSW 实例上 `LingBot-VA` 的四卡训练起不来。
2. 为什么单卡仍然能跑，而四卡会在很早阶段崩掉。
3. 在 DSW 平台里，镜像、CUDA、PyTorch、驱动分别属于哪一层。
4. 后续应该优先排查什么，而不是盲目改训练超参数。

## 1. 本次实例的实际观测结果

当前远端实例观测到的关键信息如下：

| 项目 | 当前值 |
|---|---|
| GPU | 4 x NVIDIA A100-SXM4-80GB |
| NVIDIA 驱动 | `470.199.02` |
| `nvidia-smi` 显示 CUDA | `12.1` |
| `torch` | `2.9.0+cu126` |
| PyTorch 编译 CUDA | `12.6` |
| pip NCCL | `nvidia-nccl-cu12==2.27.5` |
| pip CUDA runtime | `nvidia-cuda-runtime-cu12==12.6.77` |
| 训练环境 | `lingbot-va` |

本次排查已经确认：

- 远端损坏的源码文件已经修复。
- 训练配置、模型路径、数据路径都已经打通。
- 单卡训练路径能走到数据集初始化阶段。
- 四卡训练不是配置错误，而是多卡分布式底层直接崩溃。

## 2. 当前问题的真正根因

当前四卡训练失败，不是因为：

- `robotwin` 数据集路径错了
- `lingbot-va-base` 模型目录不完整
- 训练 YAML 配错
- `wandb` 没配置

真正的根因是：

- 当前实例的宿主机驱动版本过旧
- 但训练环境使用的是 `torch 2.9.0+cu126`
- 四卡 `torchrun + NCCL` 初始化时直接 `SIGSEGV`

这说明当前问题位于：

- 驱动 / CUDA 兼容层
- 多卡 NCCL 通信层
- DSW 宿主机 GPU 栈

而不是当前项目的训练逻辑层。

## 3. 为什么单卡还能跑，四卡却不行

这个问题最容易让人误判成“模型本身没问题，所以环境也没问题”。实际上不是。

### 3.1 单卡路径绕开了最脆弱的分布式部分

当前训练入口在 [train.py](/home/syr/code/lingbot-va/wan_va/train.py#L532) 到 [train.py](/home/syr/code/lingbot-va/wan_va/train.py#L554)：

- 从环境变量读取 `WORLD_SIZE`
- 调用 `init_distributed(world_size, local_rank, rank)`
- 然后创建 `Trainer`

分布式初始化在 [util.py](/home/syr/code/lingbot-va/wan_va/distributed/util.py#L24) 到 [util.py](/home/syr/code/lingbot-va/wan_va/distributed/util.py#L33)：

- 如果 `world_size <= 1`，函数直接返回
- 不会初始化 `nccl` process group

也就是说，单卡时：

- 不走 `dist.init_process_group("nccl", ...)`
- 不需要 GPU 之间互联
- 不需要 NCCL collective
- 不需要多进程 rendezvous

这就让单卡绕开了当前实例最容易崩的那一层。

### 3.2 单卡也不会进入 FSDP 分片逻辑

模型配置逻辑在 [util.py](/home/syr/code/lingbot-va/wan_va/distributed/util.py#L6) 到 [util.py](/home/syr/code/lingbot-va/wan_va/distributed/util.py#L21)：

- 如果 `dist` 没初始化，就只执行 `model.to(param_dtype)` 和 `model.to(device)`
- 如果 `dist` 已初始化，才会调用 `shard_fn(model)`

而 `shard_fn` 对应的是 FSDP fully sharding，定义在 [fsdp.py](/home/syr/code/lingbot-va/wan_va/distributed/fsdp.py#L18) 到 [fsdp.py](/home/syr/code/lingbot-va/wan_va/distributed/fsdp.py#L35)。

这意味着单卡时：

- 模型只是正常搬到一张卡上
- 不做 FSDP 分片
- 不做跨卡参数同步
- 不依赖 NCCL 建立稳定的 4 卡通信组

所以单卡能继续往下走，并不代表四卡环境没有问题，只能说明：

- 单卡本地 CUDA kernel 执行还能工作
- 模型和数据路径基本是对的

### 3.3 四卡一上来就依赖 NCCL 和驱动兼容性

四卡时，训练一开始就依赖：

- `torchrun`
- `nccl` process group
- 4 个进程各自绑定 GPU
- GPU 间 collective 通信
- NVLink / PCIe / NUMA 拓扑
- 驱动与 CUDA runtime 的兼容

只要这层有问题，就会发生：

- 还没进入真正训练循环
- 还没开始 forward / backward
- 还没开始 optimizer step
- 进程就直接 `SIGSEGV`

这正是当前实例的实际现象。

## 4. 镜像、CUDA、PyTorch、驱动分别是什么

在 DSW 里，这几层一定要分清：

### 4.1 驱动

驱动是宿主机层的 NVIDIA driver。

它决定：

- GPU 能否被内核正确驱动
- 最高能兼容到哪一代 CUDA 用户态库
- 多卡 NCCL 初始化是否有机会稳定运行

你通常不能在 DSW 容器里自己修改宿主机驱动。

### 4.2 CUDA runtime / NCCL / torch

这些属于用户态环境：

- 来自你选择的镜像
- 或来自 conda / pip 安装

这些你通常可以自己改。

但是：

用户态环境版本不能超过宿主机驱动的兼容范围。

### 4.3 镜像来源

“镜像来源”一般控制的是：

- 容器里的软件环境
- 预装包
- Python / CUDA runtime / torch 版本组合

它不等于宿主机驱动版本。

所以即使你导入自定义镜像，也不能保证绕过底层旧驱动限制。

## 5. 为什么当前实例很可疑

当前组合是：

- 驱动：`470.199.02`
- PyTorch：`2.9.0+cu126`

这套组合风险非常高，原因很简单：

- 训练环境已经明确走的是 CUDA 12.6 家族
- 但宿主机驱动还是 470 系列

在这种情况下，单卡有时还能做一部分本地执行，是因为它只触发了更窄的一部分 CUDA 路径。

但四卡 NCCL 是更敏感、更依赖驱动完整兼容的一层，所以会在最小通信测试时直接崩。

## 6. 本次已经做过哪些验证

本次已经做了下面这些验证，结果都指向同一个结论：

1. 真实四卡训练启动
结果：在 `torchrun` 阶段直接 `SIGSEGV`

2. 脱离训练代码的最小四卡 `torchrun + NCCL all_reduce`
结果：同样 `SIGSEGV`

3. 关闭一些常见 NCCL 开关尝试规避
包括：
- `NCCL_P2P_DISABLE=1`
- `NCCL_IB_DISABLE=1`
- `NCCL_SHM_DISABLE=1`
- `NCCL_CUMEM_ENABLE=0`

结果：仍然 `SIGSEGV`

4. 强制优先使用 pip 安装的 `nccl/cudart/nvjitlink`
结果：仍然 `SIGSEGV`

这些结果说明：

- 当前问题不是训练脚本里的某一个参数
- 不是 `wandb`
- 不是数据集
- 不是模型 checkpoint
- 而是四卡底层分布式栈本身不稳定

## 7. 后续应该怎么判断新实例是否值得申请

如果你要重新申请 DSW 实例，不要只看“最高 CUDA 版本”。

你真正要关注的是：

1. 宿主机驱动版本是多少
2. 这个资源池是否支持你想用的 CUDA 家族
3. 该资源池上 4 卡 NCCL 是否稳定

最理想的方式是：

- 优先申请更高驱动的节点
- 再让镜像/conda 环境去贴合作者的 `torch 2.9.0 + cu126`

而不是反过来。

## 8. 如果平台驱动不能改，应该怎么办

如果 DSW 的驱动就是平台固定的，用户不能改，那么通常只有两条路：

### 路线 A：换实例 / 换资源池 / 换节点环境

目标是拿到：

- 更新的宿主机驱动
- 能稳定支持 CUDA 12.x 的节点

这条路线最贴近作者公开环境。

### 路线 B：重建一个与当前驱动兼容的旧环境

也就是：

- 不再坚持 `cu126`
- 改成和旧驱动兼容的 CUDA / torch 组合

但这条路线也有代价：

- 需要重新验证项目里用到的新 PyTorch API 是否仍然可用
- 速度和稳定性可能和作者环境不同
- 会偏离作者公开配置

## 9. 结论

一句话总结：

- 当前实例上，单卡能跑不代表环境没问题
- 它只说明单卡路径绕开了 NCCL 多卡通信层
- 四卡直接崩，说明真正坏的是底层分布式栈
- 目前最值得优先确认的是“宿主机驱动是否足够新”，而不是继续微调训练超参数

在当前证据下，更合理的策略是：

- 如果目标是尽快复现作者四卡训练，优先换到更新驱动的 DSW 实例
- 如果驱动无法变更，再考虑重建一个与旧驱动兼容的降级环境

