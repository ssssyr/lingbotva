# 新服务器 LIBERO 训练交接

日期：2026-04-17（CST）

## 机器识别
- SSH：`ssh -i ~/.ssh/cfff_ed25519 -p 30522 ct_24210860031@10.193.2.99`
- Hostname：`dsw-33117-bb879d944-ww6bj`
- GPU：`4 x NVIDIA A100-SXM4-80GB`

## 存储结论
- `/home` 还有大量可用空间，适合放代码、环境和后续训练输出。
- 共享盘 `cpfs` 已满，不适合继续作为新训练工作目录。
- 因此新的工作目录切到：`/home/ct_24210860031/812/SYR/code/lingbot-va-libero`

## 当前可复用资产
共享盘上仍可直接复用：
- 旧代码目录：`/cpfs01/projects-HDD/cfff-4a2485d4a88d_HDD/ct_24210860031/812/SYR/code/lingbot-va`
- 基础模型：`/cpfs01/projects-HDD/cfff-4a2485d4a88d_HDD/ct_24210860031/812/SYR/models/lingbot-va-posttrain-robotwin`
- 现有数据：`robotwin`、`ur10` 相关数据都还在共享盘

新的 LIBERO 训练数据已放到 `/home/ct_24210860031/812/SYR/data/libero_10`。
- 可直接作为训练根目录使用的路径：`/home/ct_24210860031/812/SYR/data/libero_10/0.0.0/libero_10_0.0.0_lerobot_part_0`
- 该目录已确认包含 LeRobot 数据、latents、videos、meta；`empty_emb.pt` 已补生成。

## 已完成事项
- 新服务器已配置 SSH 公钥，可以从本机直接登录。
- 远端已安装 `rsync`。
- Ubuntu apt 源已切到清华镜像；原始 `/etc/apt/sources.list` 备份为：`/etc/apt/sources.list.bak-codex-rsync`
- 已把一份“仅用于 LIBERO 训练的最小代码工作区”同步到：
  - `/home/ct_24210860031/812/SYR/code/lingbot-va-libero`
- 之前那条误把大目录一并同步到 `/home/.../lingbot-va` 的宽同步已经停掉。

## 当前注意点
- 这份最小工作区里已经有：
  - `wan_va/configs/va_libero_cfg.py`
  - `wan_va/configs/va_libero_i2va.py`
  - `wan_va/configs/va_libero_train_cfg.py`
- `wan_va/configs/__init__.py` 的 `libero / libero_i2av / libero_train` 注册问题已经修好。
  - 本地仓库已修。
  - 新服务器工作区 `/home/ct_24210860031/812/SYR/code/lingbot-va-libero` 已同步该修复。

## 训练环境状态
- 新服务器已经在 `/home/ct_24210860031/.venvs/torch29check` 下建立了可用的 LingBot-VA 训练 venv。
- 这套环境是给 `lingbot-va-libero` 工作区准备的。
- 当前状态：`torch / diffusers / transformers / lerobot / flex_attention / wan_va.train` 都已经可以 import。
- 目前剩下的主要问题已经不是 Python 环境，而是 `LIBERO` 数据本身。

## 还没完成的事
1. 用 1 GPU 做一次更干净的 smoke test，确认训练入口能稳定跑过 `Setting up datasets...` 并打出第一步 loss。
2. 再决定正式多卡训练命令与输出目录。

## 建议下一步
1. 直接使用数据根目录：`/home/ct_24210860031/812/SYR/data/libero_10/0.0.0/libero_10_0.0.0_lerobot_part_0`
2. 在新 venv 里做：
   - `import wan_va.train`
   - `import lerobot`
   - 小步数单卡训练测试
3. smoke test 通过后，再起正式训练。
