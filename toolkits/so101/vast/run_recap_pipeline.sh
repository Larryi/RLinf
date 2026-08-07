#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
: "${HF_TOKEN:?Set HF_TOKEN}"
: "${SFT_DATASET_REPO:?Set SFT_DATASET_REPO}"
: "${ROLLOUT_DATASET_REPO:?Set ROLLOUT_DATASET_REPO}"
: "${OPENPI_CHECKPOINT_REPO:?Set OPENPI_CHECKPOINT_REPO}"
: "${OUTPUT_MODEL_REPO:?Set OUTPUT_MODEL_REPO}"
: "${WORK_ROOT:=/workspace/rlinf-so101-recap}"
: "${RUN_ID:=so101_recap_$(date +%Y%m%d_%H%M%S)}"
: "${RECAP_STAGES:=returns,value,advantages,cfg}"
: "${GPU_IDS:=0}"
: "${GPU_COUNT:=1}"
: "${PREPARE_ENV:=1}"
: "${OUTPUT_MODEL_PRIVATE:=1}"
: "${AUTO_STOP_INSTANCE:=1}"
: "${AUTO_STOP_ON_FAILURE:=0}"
: "${VALUE_MICRO_BATCH:=4}"
: "${VALUE_GLOBAL_BATCH:=32}"
: "${VALUE_MAX_STEPS:=8000}"
: "${VALUE_SAVE_INTERVAL:=500}"
: "${CFG_MICRO_BATCH:=1}"
: "${CFG_GLOBAL_BATCH:=8}"
: "${CFG_MAX_STEPS:=3000}"
: "${CFG_SAVE_INTERVAL:=250}"
: "${TRAIN_EXPERT_ONLY:=1}"
: "${HF_DOWNLOAD_WORKERS:=16}"
: "${CHECKPOINT_UPLOAD_INTERVAL:=300}"
: "${FAST_MODE:=1}"
# RECAP trains offline on recorded datasets, so no simulator environment is
# needed: use the lightweight "dummy" env (skips libero/maniskill asset
# downloads). Switch to "maniskill_libero" only if rollout/RL envs are needed.
: "${INSTALL_ENV:=dummy}"
: "${RESUME_RUN_ID:=}"

IFS=',' read -r -a gpu_ids <<<"${GPU_IDS}"
(( ${#gpu_ids[@]} == GPU_COUNT )) || {
  echo "GPU_IDS selects ${#gpu_ids[@]} GPUs, expected GPU_COUNT=${GPU_COUNT}" >&2
  exit 2
}
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export HF_TOKEN HF_XET_HIGH_PERFORMANCE=1 PYTHONUNBUFFERED=1
export WANDB_PROJECT="${WANDB_PROJECT:-rlinf-so101-recap}"
[[ -z "${WANDB_API_KEY:-}" ]] || export WANDB_API_KEY WANDB_MODE=online

RUN_ROOT="${WORK_ROOT}/runs/${RUN_ID}"
LOG_DIR="${RUN_ROOT}/logs"
STATUS_FILE="${RUN_ROOT}/status.json"
VENV_DIR="${WORK_ROOT}/venvs/openpi-recap-v3"
SFT_ROOT="${WORK_ROOT}/datasets/sft"
ROLLOUT_ROOT="${WORK_ROOT}/datasets/rollout"
POLICY_ROOT="${WORK_ROOT}/models/openpi-pytorch"
SIGLIP_ROOT="${WORK_ROOT}/models/siglip2-so400m-patch14-224"
GEMMA_ROOT="${WORK_ROOT}/models/gemma-3-270m"
VALUE_ROOT="${RUN_ROOT}/value"
CFG_ROOT="${RUN_ROOT}/cfg"
mkdir -p "${LOG_DIR}" "${VALUE_ROOT}" "${CFG_ROOT}"
exec > >(tee -a "${LOG_DIR}/pipeline.log") 2>&1

PHASE="bootstrap"
UPLOAD_STATUS="not_started"
CHECKPOINT_SYNC_PID=""

write_status() {
  local state="$1"
  STATUS_STATE="${state}" STATUS_PHASE="${PHASE}" STATUS_UPLOAD="${UPLOAD_STATUS}" \
  STATUS_FILE="${STATUS_FILE}" STATUS_RUN_ID="${RUN_ID}" python3 - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

path = Path(os.environ["STATUS_FILE"])
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps({
    "state": os.environ["STATUS_STATE"],
    "phase": os.environ["STATUS_PHASE"],
    "upload": os.environ["STATUS_UPLOAD"],
    "run_id": os.environ["STATUS_RUN_ID"],
    "updated_at": datetime.now(timezone.utc).isoformat(),
}, indent=2) + "\n")
temporary.replace(path)
PY
}

set_phase() {
  PHASE="$1"
  write_status running
  echo "[phase] ${PHASE}"
}

serverchan_url() {
  if [[ -n "${SERVERCHAN_URL:-}" ]]; then
    printf '%s' "${SERVERCHAN_URL}"
  elif [[ "${SERVERCHAN_SENDKEY:-}" =~ ^sctp([0-9]+)t ]]; then
    printf 'https://%s.push.ft07.com/send/%s.send' "${BASH_REMATCH[1]}" "${SERVERCHAN_SENDKEY}"
  elif [[ -n "${SERVERCHAN_SENDKEY:-}" ]]; then
    printf 'https://sctapi.ftqq.com/%s.send' "${SERVERCHAN_SENDKEY}"
  fi
}

notify() {
  local title="$1" body="${2:-}" url
  url="$(serverchan_url)"
  [[ -n "${url}" ]] || { echo "[notify disabled] ${title}"; return 0; }
  curl --fail --silent --show-error --max-time 20 --retry 3 \
    --request POST "${url}" \
    --header 'Content-Type: application/x-www-form-urlencoded' \
    --data-urlencode "text=${title}" --data-urlencode "desp=${body}" >/dev/null || true
}

stop_instance() {
  local rc="$1"
  [[ "${AUTO_STOP_INSTANCE}" == "1" ]] || return 0
  if (( rc != 0 )) && [[ "${AUTO_STOP_ON_FAILURE}" != "1" ]]; then
    echo "Failure policy keeps the instance running for debugging."
    return 0
  fi
  [[ -n "${VAST_API_KEY:-}" && -n "${VAST_INSTANCE_ID:-}" ]] || return 0
  command -v uvx >/dev/null || return 0
  VAST_API_KEY="${VAST_API_KEY}" uvx --from vastai vastai stop instance \
    "${VAST_INSTANCE_ID}" --raw || true
}

on_exit() {
  local rc=$?
  trap - EXIT
  set +e
  [[ -z "${CHECKPOINT_SYNC_PID}" ]] || kill "${CHECKPOINT_SYNC_PID}" 2>/dev/null
  if (( rc == 0 )); then
    PHASE="complete"
    write_status success
    notify "RLInf SO101 RECAP 完成" "Run: ${RUN_ID}\nRepo: https://huggingface.co/${OUTPUT_MODEL_REPO}"
  else
    write_status failed
    notify "RLInf SO101 RECAP 失败" "Run: ${RUN_ID}\nPhase: ${PHASE}\nExit: ${rc}"
  fi
  stop_instance "${rc}"
  exit "${rc}"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
write_status running

has_stage() {
  [[ ",${RECAP_STAGES}," == *",$1,"* ]]
}

download_hf() {
  local repo_id="$1" destination="$2" repo_type="${3:-model}"
  mkdir -p "${destination}"
  hf download "${repo_id}" --repo-type "${repo_type}" \
    --local-dir "${destination}" --max-workers "${HF_DOWNLOAD_WORKERS}"
}

upload_run() {
  set_phase "upload run artifacts"
  hf upload "${OUTPUT_MODEL_REPO}" "${RUN_ROOT}" "${RUN_ID}" \
    --repo-type model --commit-message "train: SO101 RECAP ${RUN_ID}"
  UPLOAD_STATUS="success"
  write_status running
}

upload_dataset_meta() {
  local repo_id="$1" root="$2"
  hf upload "${repo_id}" "${root}/meta" meta --repo-type dataset \
    --commit-message "data: add RECAP labels ${RUN_ID}"
}

start_checkpoint_sync() {
  local training_root="$1" label="$2"
  (
    last_uploaded=""
    while true; do
      latest="$(find "${training_root}" -type d -name 'global_step_*' 2>/dev/null | sort -V | tail -1)"
      if [[ -n "${latest}" && "${latest}" != "${last_uploaded}" ]]; then
        size_before="$(du -sb "${latest}" | cut -f1)"
        sleep 15
        size_after="$(du -sb "${latest}" | cut -f1)"
        if [[ "${size_before}" == "${size_after}" ]]; then
          step="$(basename "${latest}")"
          hf upload "${OUTPUT_MODEL_REPO}" "${latest}" \
            "${RUN_ID}/${label}/checkpoints/${step}" --repo-type model \
            --commit-message "checkpoint: ${label} ${step}"
          last_uploaded="${latest}"
        fi
      fi
      sleep "${CHECKPOINT_UPLOAD_INTERVAL}"
    done
  ) &
  CHECKPOINT_SYNC_PID=$!
}

stop_checkpoint_sync() {
  [[ -z "${CHECKPOINT_SYNC_PID}" ]] || kill "${CHECKPOINT_SYNC_PID}" 2>/dev/null || true
  [[ -z "${CHECKPOINT_SYNC_PID}" ]] || wait "${CHECKPOINT_SYNC_PID}" 2>/dev/null || true
  CHECKPOINT_SYNC_PID=""
}

stop_ray() {
  ray stop --force >/dev/null 2>&1 || true
}

set_phase "prepare environment"
if [[ "${PREPARE_ENV}" == "1" && ! -x "${VENV_DIR}/bin/python" ]]; then
  python3 -m pip install --upgrade uv huggingface_hub
  bash "${ROOT}/requirements/install.sh" embodied \
    --model openpi --env "${INSTALL_ENV}" --lerobot-v3 \
    --no-flash-attn --no-root --venv "${VENV_DIR}"
fi
[[ -x "${VENV_DIR}/bin/python" ]] || { echo "Missing venv: ${VENV_DIR}" >&2; exit 4; }
source "${VENV_DIR}/bin/activate"
export REPO_PATH="${ROOT}"
export HF_HOME="${WORK_ROOT}/cache/huggingface"
export HF_DATASETS_CACHE="${WORK_ROOT}/cache/datasets"
export TRANSFORMERS_CACHE="${WORK_ROOT}/cache/transformers"
export UV_CACHE_DIR="${WORK_ROOT}/cache/uv"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${UV_CACHE_DIR}"

set_phase "validate GPU"
python - <<'PY'
import av
import datasets
import lerobot
import torch
from packaging.version import Version

assert Version(datasets.__version__) >= Version("4.0.0"), datasets.__version__
assert torch.cuda.is_available(), "CUDA is unavailable"
print({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    "memory_gib": [round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1) for i in range(torch.cuda.device_count())],
})
PY

set_phase "download datasets and models"
download_hf "${SFT_DATASET_REPO}" "${SFT_ROOT}" dataset
download_hf "${ROLLOUT_DATASET_REPO}" "${ROLLOUT_ROOT}" dataset
download_hf "${OPENPI_CHECKPOINT_REPO}" "${POLICY_ROOT}" model
download_hf "google/siglip2-so400m-patch14-224" "${SIGLIP_ROOT}" model
download_hf "google/gemma-3-270m" "${GEMMA_ROOT}" model
export RECAP_SIGLIP_PATH="${SIGLIP_ROOT}"
export RECAP_GEMMA_PATH="${GEMMA_ROOT}"
export SO101_RECAP_DEMO_DATASET="${SFT_ROOT}"
export SO101_RECAP_ROLLOUT_DATASET="${ROLLOUT_ROOT}"
export SO101_RECAP_POLICY_CHECKPOINT="${POLICY_ROOT}"
create_args=(repo create "${OUTPUT_MODEL_REPO}" --repo-type model --exist-ok)
[[ "${OUTPUT_MODEL_PRIVATE}" == "1" ]] && create_args+=(--private)
hf "${create_args[@]}"

RESUME_ROOT=""
if [[ -n "${RESUME_RUN_ID}" ]]; then
  set_phase "download resume state"
  RESUME_ROOT="${WORK_ROOT}/resume/${RESUME_RUN_ID}"
  mkdir -p "${RESUME_ROOT}"
  hf download "${OUTPUT_MODEL_REPO}" --repo-type model \
    --include "${RESUME_RUN_ID}/**" --local-dir "${RESUME_ROOT}" \
    --max-workers "${HF_DOWNLOAD_WORKERS}"
fi

if has_stage returns; then
  set_phase "compute returns"
  python "${ROOT}/examples/offline_rl/advantage_labeling/recap/process/compute_returns.py" \
    --config-path "${ROOT}/examples/offline_rl/config" \
    --config-name recap_so101_compute_returns \
    "data.train_data_paths=[{dataset_path:${SFT_ROOT},type:sft}]" data.tag=so101
  python "${ROOT}/examples/offline_rl/advantage_labeling/recap/process/compute_returns.py" \
    --config-path "${ROOT}/examples/offline_rl/config" \
    --config-name recap_so101_compute_returns \
    "data.train_data_paths=[{dataset_path:${ROLLOUT_ROOT},type:rollout}]" data.tag=so101
  upload_dataset_meta "${SFT_DATASET_REPO}" "${SFT_ROOT}"
  upload_dataset_meta "${ROLLOUT_DATASET_REPO}" "${ROLLOUT_ROOT}"
fi

sharding="no_shard"
[[ "${GPU_COUNT}" == "1" ]] || sharding="full_shard"
if has_stage value; then
  set_phase "train value model"
  value_args=(
    --config-path "${ROOT}/examples/offline_rl/config"
    --config-name recap_so101_value_model_sft
    "runner.logger.log_path=${VALUE_ROOT}"
    "runner.max_steps=${VALUE_MAX_STEPS}"
    "runner.save_interval=${VALUE_SAVE_INTERVAL}"
    "runner.val_check_interval=-1"
    "actor.micro_batch_size=${VALUE_MICRO_BATCH}"
    "actor.global_batch_size=${VALUE_GLOBAL_BATCH}"
    "actor.model.siglip_path=${SIGLIP_ROOT}"
    "actor.model.gemma3_path=${GEMMA_ROOT}"
    "actor.model.tokenizer_path=${GEMMA_ROOT}"
    "actor.fsdp_config.sharding_strategy=${sharding}"
  )
  if [[ -n "${RESUME_ROOT}" ]]; then
    value_resume="$(find "${RESUME_ROOT}" -type d -name 'global_step_*' | grep '/value/' | sort -V | tail -1 || true)"
    [[ -z "${value_resume}" ]] || value_args+=("runner.resume_dir=${value_resume}")
  fi
  start_checkpoint_sync "${VALUE_ROOT}" value
  python "${ROOT}/examples/offline_rl/advantage_labeling/recap/train_value.py" \
    "${value_args[@]}"
  stop_checkpoint_sync
  stop_ray
  upload_run
fi

VALUE_CHECKPOINT_DIR="$(find "${VALUE_ROOT}" -type d -name 'global_step_*' 2>/dev/null | sort -V | tail -1)"
if has_stage advantages && [[ -z "${VALUE_CHECKPOINT_DIR}" && -n "${RESUME_ROOT}" ]]; then
  VALUE_CHECKPOINT_DIR="$(find "${RESUME_ROOT}" -type d -name 'global_step_*' | grep '/value/' | sort -V | tail -1 || true)"
fi
if has_stage advantages && [[ -z "${VALUE_CHECKPOINT_DIR}" ]]; then
  : "${VALUE_CHECKPOINT_REPO:?Set VALUE_CHECKPOINT_REPO when starting at advantages}"
  VALUE_CHECKPOINT_DIR="${WORK_ROOT}/models/value-checkpoint"
  download_hf "${VALUE_CHECKPOINT_REPO}" "${VALUE_CHECKPOINT_DIR}" model
fi

if has_stage advantages; then
  set_phase "compute advantages"
  export RECAP_VALUE_CHECKPOINT="${VALUE_CHECKPOINT_DIR}"
  bash "${ROOT}/examples/offline_rl/advantage_labeling/recap/process/run_compute_advantages.sh" \
    recap_so101_compute_advantages --nproc "${GPU_COUNT}" \
    "advantage.value_checkpoint=${VALUE_CHECKPOINT_DIR}" \
    "advantage.model.siglip_path=${SIGLIP_ROOT}" \
    "advantage.model.gemma3_path=${GEMMA_ROOT}" \
    "advantage.model.tokenizer_path=${GEMMA_ROOT}"
  upload_dataset_meta "${SFT_DATASET_REPO}" "${SFT_ROOT}"
  upload_dataset_meta "${ROLLOUT_DATASET_REPO}" "${ROLLOUT_ROOT}"
fi

if has_stage cfg; then
  set_phase "train CFG policy"
  cfg_args=(
    --config-path "${ROOT}/examples/offline_rl/config"
    --config-name cfg_rl_openpi_pytorch_so101
    "runner.logger.log_path=${CFG_ROOT}"
    "runner.max_steps=${CFG_MAX_STEPS}"
    "runner.save_interval=${CFG_SAVE_INTERVAL}"
    "actor.micro_batch_size=${CFG_MICRO_BATCH}"
    "actor.global_batch_size=${CFG_GLOBAL_BATCH}"
    "actor.model.model_path=${POLICY_ROOT}"
    "actor.model.openpi.train_expert_only=$([[ "${TRAIN_EXPERT_ONLY}" == "1" ]] && echo true || echo false)"
    "actor.fsdp_config.sharding_strategy=${sharding}"
  )
  if [[ -n "${RESUME_ROOT}" ]]; then
    cfg_resume="$(find "${RESUME_ROOT}" -type d -name 'global_step_*' | grep '/cfg/' | sort -V | tail -1 || true)"
    [[ -z "${cfg_resume}" ]] || cfg_args+=("runner.resume_dir=${cfg_resume}")
  fi
  if [[ "${FAST_MODE}" == "1" ]]; then
    # RTX PRO 6000 WS (96 GB): the 3.35B Pi0 fits fully on GPU, so drop the
    # 24 GB-era CPU offload and activation checkpointing for higher throughput.
    # Set FAST_MODE=0 to restore the memory-saving defaults from the YAML.
    cfg_args+=(
      "actor.fsdp_config.cpu_offload=false"
      "actor.fsdp_config.gradient_checkpointing=false"
    )
  fi
  start_checkpoint_sync "${CFG_ROOT}" cfg
  python "${ROOT}/examples/offline_rl/policy_optimization/cfg_rl/train_cfg.py" \
    "${cfg_args[@]}"
  stop_checkpoint_sync
  stop_ray
  upload_run
fi

UPLOAD_STATUS="success"
write_status running
