# SO-101 RECAP on vast.ai (RTX PRO 6000 WS / 96 GB)

一键在 vast.ai 云实例上跑 RECAP 管道（returns → value → advantages → CFG）。
本目录脚本把"代码 + 物料 + 训练 + 推送 + 上传 + 停实例"全部串起来。

## 脚本一览

| 脚本 | 作用 |
|------|------|
| `upload_assets.sh` | **一次性**：把本地已标注数据集 ×2、PyTorch 基模、可选 value checkpoint 上传到 HF（云端 `hf download` 的源头） |
| `launch.sh` | 本机执行：把代码送到云端（默认 git clone 你的 fork，备选 rsync）→ 注入 secrets → 启动管道（`START_PIPELINE=0` 则只部署不启动） |
| `remote_start.sh` | **云端执行**：`ssh` 上去后一键启动训练（source recap.env + nohup 跑管道） |
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

## 三种运行模式

### 模式 A：云端全管道（默认）

`RECAP_STAGES="returns,value,advantages,cfg"`——value 模型在云端重训（96G 跑 0.7B 很轻松），无需上传 value checkpoint。

### 模式 B：复用本机 value 模型

本机已训好 Value，云端只跑后半段：

```bash
export RECAP_STAGES="advantages,cfg"
export VALUE_CHECKPOINT_REPO="<owner/value-checkpoint-repo>"   # upload_assets.sh 传上去的
export VALUE_SRC="/absolute/path/to/global_step_3000/actor"
export ADVANTAGE_TAG="so101_v2_gs3000_q30"
```

### 模式 C：本机完成 Advantage，云端只训练 CFG

这是已经在本机验证 Value 后最稳妥的交接方式。两个 dataset repo 必须包含
`meta/advantages_${ADVANTAGE_TAG}.parquet`，且 `mixture_config.yaml` 中存在同名 tag：

```bash
export RECAP_STAGES="cfg"
export ADVANTAGE_TAG="so101_v2_gs3000_q30"
export REQUIRE_ADVANTAGES="1"
export DATASET_UPLOAD_MODE="meta"  # 仅当远端与本地 Episode/帧数一致

# VALUE_CHECKPOINT_REPO/VALUE_SRC 仅用于归档；cfg-only 不会下载 Value。
export VALUE_CHECKPOINT_REPO="<owner/so101-value-v2-gs3000>"
export VALUE_SRC="/home/larry/RLinf/logs/value_sft/recap_so101_value_model_sft_v2-20260812-02:24:53/recap_so101_value_sft_v2/checkpoints/global_step_3000/actor"

bash toolkits/so101/vast/upload_assets.sh
```

`DATASET_UPLOAD_MODE=meta` 只更新已有 dataset repo 的 `meta/`，保留原有
data/videos，适合在相同 90+50 数据上新增 Advantage tag。新建 dataset repo
时应使用默认的 `full`。

上传前脚本会验证完整帧覆盖、timeout/failure 负终奖、精确 q01/q99、
Advantage tag、Value checkpoint 布局，以及 OpenPI checkpoint 内的 norm stats。
CFG 的 state/action normalization 始终使用 OpenPI checkpoint 中的
`physical-intelligence/behavior/norm_stats.json`；LeRobot `meta/stats.json`
保持精确是为了其他数据工具和后续训练一致性，但不会替换基模坐标系。

## 启动与监控

方式一（自动）：本机 `launch.sh` 部署代码+env 并直接后台启动。
方式二（手动控制）：本机只部署、云端再启动：

```bash
# 本机：只部署代码 + recap.env，不启动
START_PIPELINE=0 bash toolkits/so101/vast/launch.sh

# 然后 ssh 到云端，随时手动启动训练
ssh root@<host> -p <port>
bash /workspace/RLinf/toolkits/so101/vast/remote_start.sh
# 日志 tail -f /workspace/RLinf/logs/recap-launcher.log
```

状态查询：`bash toolkits/so101/vast/status.sh`（status.json + 日志 + GPU）。

训练产物（value/cfg checkpoint、tensorboard 等）在**每个阶段完成时**自动上传到 `OUTPUT_MODEL_REPO`（`UPLOAD_CHECKPOINTS=0` 默认，训练中不增量上传）；训练中**只保留最新 1 个 checkpoint**（`PRUNE_CHECKPOINTS=1` / `KEEP_CHECKPOINTS=1`，3.35B 的 CFG checkpoint 约 7G/个，不清理会爆盘）；成功/失败均 Serverchan 推送；默认完成后自动停实例（`AUTO_STOP_INSTANCE=1`）。

## 断点续训

实例挂了/超时了，换新实例继续：

```bash
export RESUME_RUN_ID="so101_recap_<之前的RUN_ID>"
```

管道启动时从 `OUTPUT_MODEL_REPO/<RUN_ID>` 拉取已上传的 checkpoint，value/cfg 阶段自动匹配 `runner.resume_dir` 续训。

## 96G 提速（FAST_MODE）

`cfg_rl_openpi_pytorch_so101.yaml` 的 `cpu_offload: true` + `gradient_checkpointing: true` 是为 24G 卡准备的。96G 上 `FAST_MODE=1`（默认）会自动给 CFG 阶段传
`actor.fsdp_config.cpu_offload=false` + `actor.fsdp_config.gradient_checkpointing=false`，显著提速。如遇 OOM 改回 `FAST_MODE=0`。

VAST 脚本会把 scheduler 的 `total_training_steps` 绑定到实际的
`VALUE_MAX_STEPS` / `CFG_MAX_STEPS`。默认 CFG 为 300 steps warmup、3000
steps cosine，避免短训练错误地全程停留在 YAML 的 5000-step warmup 中。

## 进阶优化：vast.ai snapshot 加速冷启动

每次新实例都要重装 venv（约 20-40 分钟）+ 重新下载 siglip2/gemma-3。想秒级恢复：

1. 第一次租实例跑完 `PREPARE_ENV=1` 的"prepare environment"阶段后（或干脆跑完整管道），在 vast.ai 控制台对磁盘做 **snapshot**；
2. 之后租机时直接选该 snapshot 模板（`launch.sh` 的 `PREPARE_ENV` 会自动跳过已存在的 venv，HF 缓存目录 `WORK_ROOT/cache` 也一并保留）。

## 备注

- 云端代码默认来自你的 GitHub fork（git 模式），**代码不进 recap.env、不依赖本机在线**；rsync 模式会带全部未提交改动，适合本地魔改还没推 GitHub 时。
- **`INSTALL_ENV=dummy`（默认）**：RECAP 是 offline RL/SFT，`install.sh` 不再安装 libero/maniskill（跳过 `libero-download-assets` 与 maniskill assets 下载，装 venv 快很多）；需要真实环境交互时才改回 `maniskill_libero`。openpi tokenizer（几 MB）仍会自动下载。
- siglip2 / gemma-3 由管道自动从 HF 下载（`google/siglip2-so400m-patch14-224`、`google/gemma-3-270m`）。
- 本机根文件系统只读的问题在云端不存在（磁盘可写），无需 `LIBERO_CONFIG_PATH` hack。
