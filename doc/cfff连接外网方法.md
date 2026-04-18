# CFFF 连接外网方法

本文记录当前已经在 LingBot-VA 工作流里实际跑通的方案：

- CFFF / DSW 远端机器本身不能直接访问公网
- 本机可以通过 Clash / Mihomo 代理访问公网
- 通过 SSH 反向端口转发，把本机代理借给远端
- 远端通过这个代理执行 `curl`、`pip install`、`hf download`
- 如果 `hf-xet` 卡住，直接改用 `curl -C -` 断点续传下载大文件

这不是“让远端自己联网”，而是：

- 代理实际跑在本机
- 远端只访问自己的 `127.0.0.1:<port>`
- 这个端口通过 SSH 隧道回到本机代理端口

---

## 1. 适用场景

适合下面这些情况：

- 本机可以 SSH 到 CFFF / DSW 远端
- 本机已有可用代理
- 远端直接访问 `huggingface.co`、`pypi.org`、`drive.google.com` 会超时
- 需要在远端直接下载大文件，避免先下到本机再上传

尤其适合：

- 远端 `pip install`
- 远端 `hf download`
- 远端 `curl` / `wget`
- 远端下载 Hugging Face 上的大模型或大数据集

---

## 2. 核心原理

假设：

- 本机代理端口：`127.0.0.1:7890`
- 远端要访问的本地映射端口：`127.0.0.1:17890`

本机执行：

```bash
ssh -fN \
  -o ExitOnForwardFailure=yes \
  -R 17890:127.0.0.1:7890 \
  <remote>
```

含义是：

- 远端监听 `127.0.0.1:17890`
- 远端发到这个端口的流量
- 会经由 SSH 隧道转发到本机的 `127.0.0.1:7890`

于是远端只要设置：

```bash
HTTP_PROXY=http://127.0.0.1:17890
HTTPS_PROXY=http://127.0.0.1:17890
```

就能借本机代理出网。

---

## 3. 本次实际使用的配置

本次实际跑通时使用的是：

- 本机代理程序：`/home/syr/clash/clash`
- 本机 Mihomo 配置：`/home/syr/.config/mihomo/proxies.yaml`
- 本机代理端口：`127.0.0.1:7890`
- 本地 controller：`127.0.0.1:9090`
- 远端暴露端口：`127.0.0.1:17890`

远端 SSH 命令：

```bash
ssh -i ~/.ssh/cfff_ed25519 -p 30522 ct_24210860031@10.193.2.99
```

如果本地已经配置好 `cfff` alias，可以把后面示例里的 `<remote>` 直接替换成 `cfff`。

---

## 4. 启动步骤

### 4.1 本机启动代理

如果本机已有稳定运行的 Clash / Mihomo，可以跳过。

当前环境里可用的启动方式：

```bash
/home/syr/clash/clash -f /home/syr/.config/mihomo/proxies.yaml >/tmp/mihomo-session.log 2>&1 &
```

验证代理是否真的起来：

```bash
ss -ltnp | grep -E ':(7890|9090)\b'
curl --max-time 10 http://127.0.0.1:9090/version
```

正常情况下应看到：

- `127.0.0.1:9090` 在监听
- `*:7890` 在监听

### 4.2 本机建立到远端的反向隧道

```bash
ssh -fN \
  -o ExitOnForwardFailure=yes \
  -o StrictHostKeyChecking=accept-new \
  -o UserKnownHostsFile=/tmp/cfff_known_hosts \
  -i ~/.ssh/cfff_ed25519 \
  -p 30522 \
  -R 17890:127.0.0.1:7890 \
  ct_24210860031@10.193.2.99
```

如果使用 SSH alias：

```bash
ssh -fN -o ExitOnForwardFailure=yes -R 17890:127.0.0.1:7890 cfff
```

确认隧道进程还在：

```bash
ps -ef | grep 'R 17890:127.0.0.1:7890' | grep -v grep
```

### 4.3 远端验证代理是否可用

在远端执行：

```bash
HTTP_PROXY='http://127.0.0.1:17890' \
HTTPS_PROXY='http://127.0.0.1:17890' \
curl -I -L --max-time 20 https://huggingface.co
```

如果能返回 `HTTP/2 200` 或其他正常响应头，说明代理链路已经打通。

---

## 5. 远端如何使用

### 5.1 `curl` / `wget`

```bash
HTTP_PROXY='http://127.0.0.1:17890' \
HTTPS_PROXY='http://127.0.0.1:17890' \
curl -L -O https://example.com/file
```

```bash
HTTP_PROXY='http://127.0.0.1:17890' \
HTTPS_PROXY='http://127.0.0.1:17890' \
wget https://example.com/file
```

### 5.2 `pip install`

```bash
HTTP_PROXY='http://127.0.0.1:17890' \
HTTPS_PROXY='http://127.0.0.1:17890' \
python3 -m pip install --user huggingface_hub
```

### 5.3 `hf download`

普通情况可以这样用：

```bash
PATH="$HOME/.local/bin:$PATH" \
HTTP_PROXY='http://127.0.0.1:17890' \
HTTPS_PROXY='http://127.0.0.1:17890' \
hf download --repo-type dataset --local-dir /path/to/save <repo_id>
```

例如：

```bash
PATH="$HOME/.local/bin:$PATH" \
HTTP_PROXY='http://127.0.0.1:17890' \
HTTPS_PROXY='http://127.0.0.1:17890' \
hf download --repo-type dataset \
  --local-dir /home/ct_24210860031/812/SYR/data/libero_plus_assets \
  Sylvest/LIBERO-plus
```

### 5.4 Hugging Face 大文件优先使用 `curl -C -`

这次实际排查得到的结论是：

- `hf download` 在很多普通场景下可以正常使用
- 但对于 Hugging Face 的超大文件，尤其会走 `hf-xet` / `xet` 的时候
- 进程可能还活着，但会长时间卡在 `Fetching ... files`
- 这时更稳的办法通常不是继续等，而是直接绕开 `hf-xet`

推荐兜底方案：

- 使用 `curl -L -C -` 下载真实文件 URL
- `-C -` 保证断点续传
- 配合 `--retry`、`--speed-time`、`--speed-limit` 做自动重试

最小示例：

```bash
HTTP_PROXY='http://127.0.0.1:17890' \
HTTPS_PROXY='http://127.0.0.1:17890' \
curl -L -C - \
  --retry 100 \
  --retry-delay 5 \
  --retry-all-errors \
  --connect-timeout 10 \
  --speed-time 60 \
  --speed-limit 1024 \
  -o /path/to/assets.zip \
  'https://huggingface.co/datasets/Sylvest/LIBERO-plus/resolve/main/assets.zip'
```

这个方案已经在本次 `LIBERO-plus` 下载里实际跑通。

---

## 6. 推荐的后台下载写法

### 6.1 `hf download` 版

如果文件不大，或者 `hf download` 表现稳定，可以后台跑：

```bash
mkdir -p /home/ct_24210860031/812/SYR/logs

LOG=/home/ct_24210860031/812/SYR/logs/download_$(date +%Y%m%d_%H%M%S).log

nohup bash -lc '
export PATH="$HOME/.local/bin:$PATH"
export HTTP_PROXY=http://127.0.0.1:17890
export HTTPS_PROXY=http://127.0.0.1:17890

hf download --repo-type dataset \
  --local-dir /home/ct_24210860031/812/SYR/data/libero_plus_assets \
  Sylvest/LIBERO-plus
' > "$LOG" 2>&1 < /dev/null &

echo "$LOG"
```

### 6.2 `curl -C -` 版

如果是 Hugging Face 大文件，当前更推荐写成独立脚本顺序下载：

```bash
cat > /home/ct_24210860031/812/SYR/logs/libero_plus_curl_download.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

export HTTP_PROXY=http://127.0.0.1:17890
export HTTPS_PROXY=http://127.0.0.1:17890

urls=(
  "https://huggingface.co/datasets/Sylvest/LIBERO-plus/resolve/main/assets.zip /home/ct_24210860031/812/SYR/data/libero_plus_assets/assets.zip"
  "https://huggingface.co/datasets/Sylvest/libero_plus_rlds/resolve/main/libero_plus_mixdata.z01 /home/ct_24210860031/812/SYR/data/libero_plus_rlds/libero_plus_mixdata.z01"
  "https://huggingface.co/datasets/Sylvest/libero_plus_rlds/resolve/main/libero_plus_mixdata.z02 /home/ct_24210860031/812/SYR/data/libero_plus_rlds/libero_plus_mixdata.z02"
  "https://huggingface.co/datasets/Sylvest/libero_plus_rlds/resolve/main/libero_plus_mixdata.zip /home/ct_24210860031/812/SYR/data/libero_plus_rlds/libero_plus_mixdata.zip"
)

echo "[START] $(date)"
for item in "${urls[@]}"; do
  url=${item%% *}
  out=${item#* }
  echo "[FILE] $(date) $out"
  curl -L -C - \
    --retry 100 \
    --retry-delay 5 \
    --retry-all-errors \
    --connect-timeout 10 \
    --speed-time 60 \
    --speed-limit 1024 \
    -o "$out" \
    "$url"
  echo "[DONE_FILE] $(date) $out"
done
echo "[DONE_ALL] $(date)"
SH

chmod +x /home/ct_24210860031/812/SYR/logs/libero_plus_curl_download.sh
```

后台启动：

```bash
LOG=/home/ct_24210860031/812/SYR/logs/libero_plus_curl_download_$(date +%Y%m%d_%H%M%S).log
nohup bash /home/ct_24210860031/812/SYR/logs/libero_plus_curl_download.sh > "$LOG" 2>&1 < /dev/null &
echo "$LOG"
```

当前这个版本的优点：

- 完全绕开 `hf-xet`
- 日志直观
- 可直接断点续传
- 对超大文件更稳

---

## 7. 推荐的本地保持方式

这个方法能否持续工作，取决于两件事都不能断：

- 本机代理进程不能退出
- 本机到远端的 SSH 反向隧道不能断

因此推荐：

- 代理用后台进程启动
- SSH 隧道用 `ssh -fN` 启动
- 不要随手杀掉本机上的 `clash` / `ssh -R`

如果担心当前 shell 退出影响后台任务：

```bash
disown -a
```

或者直接用 `nohup` / `systemd --user` / `tmux` 托管。

---

## 8. 停止方法

### 8.1 停止反向隧道

本机执行：

```bash
ps -ef | grep 'R 17890:127.0.0.1:7890' | grep -v grep
kill <pid>
```

或者：

```bash
pkill -f 'ssh .* -R 17890:127.0.0.1:7890'
```

### 8.2 停止本机代理

```bash
pkill -f '/home/syr/clash/clash'
```

---

## 9. 常见问题与调试方法

### 9.1 远端 `curl` 还是超时

依次检查：

- 本机代理是否真的监听 `7890`
- 反向隧道进程是否还在
- 远端是否正确使用了 `HTTP_PROXY` / `HTTPS_PROXY`

优先排查：

```bash
ss -ltnp | grep 7890
ps -ef | grep 'R 17890:127.0.0.1:7890' | grep -v grep
ssh <remote> "HTTP_PROXY=http://127.0.0.1:17890 HTTPS_PROXY=http://127.0.0.1:17890 curl -I https://huggingface.co"
```

### 9.2 `hf download` 报 SOCKS 相关错误

本次实际踩到过：

- `huggingface_hub` 底层走 `httpx`
- 如果设置了 `ALL_PROXY=socks5h://...`
- 远端没装 `socksio`
- 可能报 `Using SOCKS proxy, but the 'socksio' package is not installed`

因此对 `hf download` 更稳的做法是：

- 只设置 `HTTP_PROXY`
- 只设置 `HTTPS_PROXY`
- 不设置 `ALL_PROXY`

即：

```bash
HTTP_PROXY='http://127.0.0.1:17890' \
HTTPS_PROXY='http://127.0.0.1:17890' \
hf download ...
```

### 9.3 日志不刷新，但文件大小在增长

这在 `hf download` 里很常见。

Hugging Face 下载大文件时，可能：

- 先写到 `.cache/huggingface/download/*.incomplete`
- 或先写到隐藏缓存目录
- 再回写到目标文件

因此不要只盯日志，最好同时看：

```bash
ps -ef | grep hf | grep -v grep
du -sb /path/to/local-dir
find /path/to/local-dir/.cache/huggingface/download -type f -name '*.incomplete' -printf '%p %s\n'
```

### 9.4 `hf download` 卡在 `Fetching ... files`

本次实际遇到过：

- `hf download` 进程存在
- 日志停在 `Fetching 4 files`
- 目标文件不增长，或者 `.incomplete` 长时间不动

优先检查：

```bash
ps -ef | grep 'hf download' | grep -v grep
ps -f --ppid <hf_pid>
cat /proc/<hf_pid>/wchan
ls -l /proc/<hf_pid>/fd
```

如果看到：

- 进程还在
- 打开了 `~/.cache/huggingface/xet/logs/...`
- `wchan` 长时间停在等待

通常说明卡在 `hf-xet`，而不是 SSH / 代理完全没通。

这时建议：

1. 保留本地代理和反向隧道
2. 杀掉卡住的 `hf download`
3. 切换到 `curl -C -`

### 9.5 读取 `hf-xet` 日志

如果要确认是不是 `xet` 在卡，可以直接看：

```bash
sed -n '1,200p' ~/.cache/huggingface/xet/logs/xet_*.log
tail -n 100 ~/.cache/huggingface/xet/logs/xet_*.log
```

本次实际观测到的典型现象：

- `xet` 日志在持续打印并发控制信息
- 但目标文件几乎不增长

这种情况下继续等通常收益不大，直接换 `curl` 更有效。

### 9.6 先做小探针，再决定是否正式下载

如果怀疑“首页能通，但大文件不通”，先做探针。

远端探测大文件 URL：

```bash
HTTP_PROXY='http://127.0.0.1:17890' \
HTTPS_PROXY='http://127.0.0.1:17890' \
curl -I -L --max-time 30 \
  'https://huggingface.co/datasets/Sylvest/LIBERO-plus/resolve/main/assets.zip'
```

如果想测试真实字节流，再拉前 `1MB`：

```bash
HTTP_PROXY='http://127.0.0.1:17890' \
HTTPS_PROXY='http://127.0.0.1:17890' \
curl -L --range 0-1048575 -o /tmp/probe.bin \
  'https://huggingface.co/datasets/Sylvest/LIBERO-plus/resolve/main/assets.zip'
```

如果 `HEAD` 和 `1MB` 探针都成功：

- 当前代理链路是通的
- 后续大文件可以直接交给 `curl -C -`

### 9.7 本地 Clash 节点可用，但自动选择不一定最稳

这次还有一个重要结论：

- `自动选择` 能访问 `google.com`
- 不代表它对 `assets.zip` 这种大文件入口也稳定

所以排查大文件失败时，可以直接查本地 Clash controller：

```bash
curl -s http://127.0.0.1:9090/proxies
curl -s http://127.0.0.1:9090/proxies/自动选择
curl -s http://127.0.0.1:9090/proxies/一元机场.VIP
```

如果要固定某个节点：

```bash
curl -X PUT http://127.0.0.1:9090/proxies/一元机场.VIP \
  -H 'Content-Type: application/json' \
  -d '{"name":"<节点名>"}'
```

然后在本地重新直测大文件入口：

```bash
curl --proxy http://127.0.0.1:7890 -I -L --max-time 20 \
  'https://huggingface.co/datasets/Sylvest/LIBERO-plus/resolve/main/assets.zip'
```

如果能稳定返回：

- `HTTP/1.1 200 OK`
- `Content-Type: application/zip`
- `Content-Length: ...`

说明这个节点至少能打通大文件入口。

### 9.8 拿到的是现成 YAML，而不是订阅链接

本次还实际处理过：

- 拿到的是一份现成 Mihomo YAML
- 但里面自带的 `proxy-groups` 有坏引用
- 直接 `clash -t -f your.yml` 会失败

这时不要急着放弃，先做两步：

1. 先校验

```bash
/home/syr/clash/clash -t -f /path/to/your.yml
```

2. 如果坏的是分组，但 `proxies` 本身是好的，就保留：

- 原有 `proxies`
- 原有 `rules`

只重写一个最小可用的 `proxy-groups`。例如：

```yaml
proxy-groups:
  - name: 一元机场.VIP
    type: select
    proxies:
      - 自动选择
      - DIRECT
      - <所有节点名>
  - name: 自动选择
    type: url-test
    proxies:
      - <所有节点名>
    url: http://www.gstatic.com/generate_204
    interval: 300
```

然后再切换到修正后的 YAML。

---

## 10. 最小可复用模板

如果只想快速复用，最少记住：

### 本机启动代理

```bash
/home/syr/clash/clash -f /home/syr/.config/mihomo/proxies.yaml >/tmp/mihomo-session.log 2>&1 &
```

### 本机建立反向隧道

```bash
ssh -fN \
  -o ExitOnForwardFailure=yes \
  -R 17890:127.0.0.1:7890 \
  <remote>
```

### 远端使用代理

```bash
HTTP_PROXY='http://127.0.0.1:17890' \
HTTPS_PROXY='http://127.0.0.1:17890' \
<your_command>
```

### 远端大文件下载优先用 `curl`

```bash
HTTP_PROXY='http://127.0.0.1:17890' \
HTTPS_PROXY='http://127.0.0.1:17890' \
curl -L -C - --retry 100 --retry-delay 5 --retry-all-errors \
  --connect-timeout 10 --speed-time 60 --speed-limit 1024 \
  -o /path/to/file \
  <url>
```

---

## 11. 当前结论

当前这套“本机代理 + SSH 反向隧道 + 远端 HTTP(S)_PROXY”的方法已经实际跑通了：

- 远端 `curl https://huggingface.co`
- 远端 `pip install`
- 远端 `hf download`
- 远端下载 `LIBERO-plus`
- 当 `hf-xet` 卡住时，远端切到 `curl -C -` 继续大文件下载
- 本地通过 Clash controller 查询和固定节点
- 远端通过 `HEAD` / `--range` 探针验证大文件链路

因此在 CFFF / DSW 远端“本身不通公网，但本机可代理出网”的前提下，这就是当前推荐方法。
