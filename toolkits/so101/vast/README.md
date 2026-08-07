# SO-101 RECAP on vast.ai (RTX PRO 6000 WS / 96 GB)

一键在 vast.ai 云实例上跑 RECAP 管道（returns → value → advantages → CFG）。
本目录脚本把"代码 + 物料 + 训练 + 推送 + 上传 + 停实例"全部串起来。

## 脚本一览

| 脚本 | 作用 |
|------|------|
| `upload_assets.sh` | **一次性**：把本地已标注数据集 ×2、PyTorch 基模、可选 value checkpoint 上传到 HF（云端 `hf download` 的源头） |
| `launch.sh` | 本机执行：把代码送到云端（默认 git clone 你的 fork，备选 rsync）→ 注入 secrets → 云端后台跑管道 |
| `run_recap_pipeline.sh` | 云端执行：装 venv → 下载物料 → 按 `RECAP_STAGES` 逐阶段训练 → Serverchan 推送 → 产物上传 HF → 自动停实例 |
| `status.sh` | 本机执行：查云端 status.json + 日志 + GPU 占用 |

## 一次性准备（本机）

### 1. 上传物料到 HF

先准备好 4 个仓库（`recap_vast.env.example` 里的名字）：

| 仓库 | 内容 | 来源 |
|------|------|------|
| `SFT_DATASET_REPO` (dataset) | 60+30 条成功演示 + RECAP 标注（`meta/` 下 returns/advantages/stats） | `/mnt/pqssd/so101/datasets/merged_lerobot_dataset_with_dagger30_trimmed` |
| `ROLLOUT_DATASET_REPO` (dataset) | 50 条策略 rollout + 同标注 | `/mnt/pqssd/so101/datasets/pi05_jax_rtc_rollouts_50_v3` |
| `OPENPI_CHECKPOINT_REPO` (model) | **PyTorch RLinf 格式**基模（`model.safetensors` + `physical-intelligence/`），不是 JAX 权重 | `/mnt/pqssd/so101/train_outputs/openpi_pi05/openpi-so101-pi05-60-30000/rlinf_ckpt` |
| `VALUE_CHECKPOINT_REPO` (model, 可选) | 本机训好的 value 模型（云端复用场景） | `outputs/so101_recap/value/.../checkpoints/global_step_*`（脚本自动取最新） |

```bash
export HF_TOKEN=hf_xxx
# 默认上传全部（value 仅当设置了 VALUE_CHECKPOINT_REPO）
bash toolkits/so101/vast/upload_assets.sh
# 或只传部分：
bash toolkits/so101/vast/upload_assets.sh --only sft,openpi
```

仓库默认**公开**（`PRIVATE=0`）；如要私有：`export PRIVATE=1` 后再跑。

> 说明：`so101_grab_blue_pen_90`（HF 上那个）是**无标注**的原始 90 条，管道需要的是**本地已加 Return/Advantage 标注**的版本，两者不是一个仓库。`rlinf_ckpt` 约 13.4G，上传时间取决于上行带宽。

### 2. 配置 secrets

```bash
cp toolkits/so101/vast/recap_vast.env.example /path/to/recap.env
# 编辑：HF_TOKEN、4 个 repo 名、SERVERCHAN_SENDKEY、VAST_* 等
chmod 600 /path/to/recap.env
```

## 代码部署方式（SOURCE_MODE，默认 git）

默认 `SOURCE_MODE=git`：云端直接 clone 你的 **GitHub fork**，VAST 访问 GitHub 速度很快，且不再依赖本机在线。推荐做法：

1. 网页上 Fork https://github.com/RLinf/RLinf 到你的账号；
2. 本机把魔改推送到 fork 的 `so101-recap` 分支（见下文「推送修改到 fork」）；
3. `launch.sh` 设三个变量：

```bash
export SOURCE_MODE=git
# 公开 fork:
export GIT_REPO="https://github.com/<you>/RLinf.git"
# 私有 fork 则内嵌 token:
# export GIT_REPO="https://<user>:<token>@github.com/<you>/RLinf.git"
export GIT_BRANCH=so101-recap
```

备选 `SOURCE_MODE=rsync`：把本地整个仓库（含未提交改动）直接 rsync 上云，适合不想走 GitHub 或临时调试。

### 推送修改到 fork

```bash
cd /home/larry/RLinf
git switch -c so101-recap                     # 从干净的 main 建分支
git add -A                                    # .gitignore 已排除 .agents/.reasonix/.secrets/outputs/.venv*
git commit -s -m "feat(so101): RECAP offline-RL pipeline and OpenPI PyTorch CFG"
# origin 改指自己的 fork,upstream 保留官方:
git remote set-url origin https://github.com/<you>/RLinf.git
git remote add upstream https://github.com/RLinf/RLinf.git 2>/dev/null || true
git push -u origin so101-recap
```

之后想同步官方更新：`git fetch upstream && git merge upstream/main`。

## 两种运行模式

### 模式 A：云端全管道（默认）

`RECAP_STAGES="returns,value,advantages,cfg"`——value 模型在云端重训（96G 跑 0.7B 很轻松），无需上传 value checkpoint。

### 模式 B：复用本机 value 模型

本机已训好 value（`outputs/so101_recap/value/.../global_step_8000`），云端只跑后半段：

```bash
export RECAP_STAGES="advantages,cfg"
export VALUE_CHECKPOINT_REPO="<owner/value-checkpoint-repo>"   # upload_assets.sh 传上去的
```

## 启动与监控

```bash
export VAST_SSH_HOST=... VAST_SSH_PORT=... VAST_ENV_FILE=/path/to/recap.env
bash toolkits/so101/vast/launch.sh        # 推代码 + 后台跑管道
bash toolkits/so101/vast/status.sh        # 随时查进度（status.json + 日志 + GPU）
```

训练产物（value/cfg checkpoint、tensorboard 等）会随阶段完成自动上传到 `OUTPUT_MODEL_REPO`，训练中每 `CHECKPOINT_UPLOAD_INTERVAL` 秒增量同步新 `global_step_*`；成功/失败均 Serverchan 推送；默认完成后自动停实例（`AUTO_STOP_INSTANCE=1`）。

## 断点续训

实例挂了/超时了，换新实例继续：

```bash
export RESUME_RUN_ID="so101_recap_<之前的RUN_ID>"
```

管道启动时从 `OUTPUT_MODEL_REPO/<RUN_ID>` 拉取已上传的 checkpoint，value/cfg 阶段自动匹配 `runner.resume_dir` 续训。

## 96G 提速（FAST_MODE）

`cfg_rl_openpi_pytorch_so101.yaml` 的 `cpu_offload: true` + `gradient_checkpointing: true` 是为 24G 卡准备的。96G 上 `FAST_MODE=1`（默认）会自动给 CFG 阶段传
`actor.fsdp_config.cpu_offload=false` + `actor.fsdp_config.gradient_checkpointing=false`，显著提速。如遇 OOM 改回 `FAST_MODE=0`。

## 进阶优化：vast.ai snapshot 加速冷启动

每次新实例都要重装 venv（约 20-40 分钟）+ 重新下载 siglip2/gemma-3。想秒级恢复：

1. 第一次租实例跑完 `PREPARE_ENV=1` 的"prepare environment"阶段后（或干脆跑完整管道），在 vast.ai 控制台对磁盘做 **snapshot**；
2. 之后租机时直接选该 snapshot 模板（`launch.sh` 的 `PREPARE_ENV` 会自动跳过已存在的 venv，HF 缓存目录 `WORK_ROOT/cache` 也一并保留）。

## 备注

- 云端代码默认来自你的 GitHub fork（git 模式），**代码不进 recap.env、不依赖本机在线**；rsync 模式会带全部未提交改动，适合本地魔改还没推 GitHub 时。
- siglip2 / gemma-3 由管道自动从 HF 下载（`google/siglip2-so400m-patch14-224`、`google/gemma-3-270m`）。
- 本机根文件系统只读的问题在云端不存在（磁盘可写），无需 `LIBERO_CONFIG_PATH` hack。
