# LingBot-VA Docker 镜像使用说明

## 镜像信息

- **lingbot-va**: 主训练环境 (约 8.7GB)
- **robotwin**: RoboTwin 评估环境 (约 8.4GB)

## 用户使用方法

### 1. 拉取镜像

```bash
# 拉取 lingbot-va 环境
docker pull <your-dockerhub-username>/lingbot-va:latest

# 拉取 robotwin 环境
docker pull <your-dockerhub-username>/lingbot-va-robotwin:latest
```

### 2. 运行容器

```bash
# 训练 (带 GPU 支持)
docker run --gpus all -it --rm \
    -v /path/to/your/data:/workspace/data \
    -v /path/to/your/output:/workspace/output \
    <your-dockerhub-username>/lingbot-va:latest

# 评估 (带 GPU 支持)
docker run --gpus all -it --rm \
    -v /path/to/your/data:/workspace/data \
    <your-dockerhub-username>/lingbot-va-robotwin:latest
```

### 3. 交互式使用

```bash
# 进入容器
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    <your-dockerhub-username>/lingbot-va:latest \
    /bin/bash

# 在容器内运行训练
python -m wan_va.train --config-name robotwin_train
```

## 维护者操作说明

### 构建和推送镜像

```bash
# 设置 Docker Hub 用户名并执行构建
DOCKER_HUB_USERNAME=yourname ./docker/build.sh
```

### 手动构建单个镜像

```bash
# lingbot-va
docker build -f docker/Dockerfile.lingbot-va -t yourname/lingbot-va:20260316 .

# robotwin
docker build -f docker/Dockerfile.robotwin -t yourname/lingbot-va-robotwin:20260316 .
```

### 推送到 Docker Hub

```bash
docker login
docker push yourname/lingbot-va:20260316
docker push yourname/lingbot-va-robotwin:20260316
```

## 注意事项

1. **GPU 支持**: 需要安装 nvidia-docker2
2. **数据挂载**: 使用 `-v` 参数挂载本地数据目录
3. **网络**: 如果使用 WandB 等服务，确保网络连接正常
