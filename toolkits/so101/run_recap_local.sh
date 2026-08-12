#!/usr/bin/env bash
# Run the SO-101 RECAP pipeline against local LeRobot v3 datasets.
#
# The default inputs are the local 90-demo and labeled 50-rollout datasets.
# Override any path with the SO101_RECAP_* variables; the same variables are
# exported by the VastAI runner after it downloads the Hugging Face repos.
set -Eeuo pipefail

# DataLoader workers exchange image tensors via torch shared memory (one FD per
# tensor); shells with a low `ulimit -n` (e.g. 1024) hit "Too many open files".
# Raise the limit when the hard limit allows; lower ADV_WORKERS if it does not.
ulimit -n 65535 2>/dev/null || true

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${RECAP_PYTHON:=${ROOT}/.venv-openpi-recap-v3/bin/python}"
: "${RECAP_STAGES:=returns,value,advantages,cfg}"
: "${SO101_RECAP_DEMO_DATASET:=/mnt/pqssd/so101/datasets/merged_lerobot_dataset_with_dagger30_trimmed}"
: "${SO101_RECAP_ROLLOUT_DATASET:=/mnt/pqssd/so101/datasets/pi05_jax_rtc_rollouts_50_v3}"
: "${SO101_RECAP_POLICY_CHECKPOINT:=/mnt/pqssd/so101/train_outputs/openpi_pi05/openpi-so101-pi05-60-30000/rlinf_ckpt}"
: "${SO101_RECAP_RUN_ROOT:=${ROOT}/outputs/so101_recap}"
: "${RECAP_NPROC:=1}"
: "${VALUE_MICRO_BATCH:=8}"
: "${VALUE_GLOBAL_BATCH:=64}"
: "${VALUE_MAX_STEPS:=8000}"
: "${VALUE_SAVE_INTERVAL:=500}"
: "${VALUE_GRAD_CHECKPOINT:=false}"
: "${ADV_WORKERS:=4}"
: "${ADV_PREFETCH:=2}"
: "${ADV_BATCH_SIZE:=128}"
: "${SO101_RECAP_ADVANTAGE_TAG:=so101_q30}"
: "${CFG_MICRO_BATCH:=2}"
: "${CFG_GLOBAL_BATCH:=8}"
: "${CFG_MAX_STEPS:=3000}"
: "${CFG_SAVE_INTERVAL:=250}"

[[ -x "${RECAP_PYTHON}" ]] || {
  echo "Python environment not found: ${RECAP_PYTHON}" >&2
  echo "Activate the RLInf environment or set RECAP_PYTHON=/path/to/python." >&2
  exit 2
}
if ! "${RECAP_PYTHON}" -c \
  'import av, datasets, hydra, lerobot; from packaging.version import Version; assert Version(datasets.__version__) >= Version("4.0.0")' \
  >/dev/null 2>&1; then
  echo "${RECAP_PYTHON} is not the complete LeRobot-v3 RECAP environment." >&2
  echo "Rebuild it with requirements/install.sh --lerobot-v3; do not use lerobot_hil for training." >&2
  exit 2
fi
[[ -d "${SO101_RECAP_DEMO_DATASET}" ]] || {
  echo "Missing 90-demo dataset: ${SO101_RECAP_DEMO_DATASET}" >&2
  exit 2
}
[[ -d "${SO101_RECAP_ROLLOUT_DATASET}" ]] || {
  echo "Missing 50-rollout dataset: ${SO101_RECAP_ROLLOUT_DATASET}" >&2
  exit 2
}

export REPO_PATH="${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export SO101_RECAP_DEMO_DATASET SO101_RECAP_ROLLOUT_DATASET
export SO101_RECAP_POLICY_CHECKPOINT
export SO101_RECAP_ADVANTAGE_TAG
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HOME}/.cache/transformers}"

CONFIG_ROOT="${ROOT}/examples/offline_rl/config"
VALUE_ROOT="${SO101_RECAP_RUN_ROOT}/value"
CFG_ROOT="${SO101_RECAP_RUN_ROOT}/cfg"
mkdir -p "${VALUE_ROOT}" "${CFG_ROOT}"

has_stage() {
  [[ ",${RECAP_STAGES}," == *",$1,"* ]]
}

latest_checkpoint() {
  find "$1" -type d -name 'global_step_*' 2>/dev/null | sort -V | tail -1
}

echo "SO-101 RECAP: ${RECAP_STAGES}"
echo "  demo90:   ${SO101_RECAP_DEMO_DATASET}"
echo "  rollout50:${SO101_RECAP_ROLLOUT_DATASET}"
echo "  output:   ${SO101_RECAP_RUN_ROOT}"

if has_stage returns; then
  echo "[1/4] Compute returns (timeout == failure == -300 terminal reward)"
  "${RECAP_PYTHON}" \
    "${ROOT}/examples/offline_rl/advantage_labeling/recap/process/compute_returns.py" \
    --config-path "${CONFIG_ROOT}" --config-name recap_so101_compute_returns
fi

if has_stage value; then
  echo "[2/4] Train value model"
  value_args=(
    --config-path "${CONFIG_ROOT}"
    --config-name recap_so101_value_model_sft
    "runner.logger.log_path=${VALUE_ROOT}"
    "runner.max_steps=${VALUE_MAX_STEPS}"
    "runner.save_interval=${VALUE_SAVE_INTERVAL}"
    "actor.micro_batch_size=${VALUE_MICRO_BATCH}"
    "actor.global_batch_size=${VALUE_GLOBAL_BATCH}"
  )
  [[ "${VALUE_GRAD_CHECKPOINT}" == "true" ]] && value_args+=("actor.fsdp_config.gradient_checkpointing=true")
  [[ -z "${VALUE_RESUME_DIR:-}" ]] || value_args+=("+runner.resume_dir=${VALUE_RESUME_DIR}")
  "${RECAP_PYTHON}" \
    "${ROOT}/examples/offline_rl/advantage_labeling/recap/train_value.py" \
    "${value_args[@]}"
  ray stop --force >/dev/null 2>&1 || true
fi

value_checkpoint="${RECAP_VALUE_CHECKPOINT:-}"
if [[ -z "${value_checkpoint}" ]]; then
  value_checkpoint="$(latest_checkpoint "${VALUE_ROOT}")"
fi

if has_stage advantages; then
  [[ -n "${value_checkpoint}" ]] || {
    echo "No value checkpoint found; set RECAP_VALUE_CHECKPOINT." >&2
    exit 3
  }
  echo "[3/4] Compute advantages from ${value_checkpoint}"
  advantage_common=(
    --config-path "${CONFIG_ROOT}"
    --config-name recap_so101_compute_advantages
    "advantage.value_checkpoint=${value_checkpoint}"
    "advantage.tag=${SO101_RECAP_ADVANTAGE_TAG}"
    "advantage.batch_size=${ADV_BATCH_SIZE}"
    "advantage.num_dataloader_workers_per_gpu=${ADV_WORKERS}"
    "advantage.prefetch_factor=${ADV_PREFETCH}"
  )
  advantage_args=(
    -m torch.distributed.run
    "--nproc_per_node=${RECAP_NPROC}"
    "${ROOT}/examples/offline_rl/advantage_labeling/recap/process/compute_advantages.py"
    "${advantage_common[@]}"
  )
  if [[ "${RECAP_NPROC}" == "1" ]]; then
    advantage_args=(
      "${ROOT}/examples/offline_rl/advantage_labeling/recap/process/compute_advantages.py"
      "${advantage_common[@]}"
    )
  fi
  "${RECAP_PYTHON}" "${advantage_args[@]}"
fi

if has_stage cfg; then
  [[ -d "${SO101_RECAP_POLICY_CHECKPOINT}" ]] || {
    echo "Missing converted Pi0.5 checkpoint: ${SO101_RECAP_POLICY_CHECKPOINT}" >&2
    exit 2
  }
  echo "[4/4] Train RECAP-CFG policy"
  cfg_args=(
    --config-path "${CONFIG_ROOT}"
    --config-name cfg_rl_openpi_pytorch_so101
    "runner.logger.log_path=${CFG_ROOT}"
    "runner.max_steps=${CFG_MAX_STEPS}"
    "runner.save_interval=${CFG_SAVE_INTERVAL}"
    "actor.micro_batch_size=${CFG_MICRO_BATCH}"
    "actor.global_batch_size=${CFG_GLOBAL_BATCH}"
    "data.advantage_tag=${SO101_RECAP_ADVANTAGE_TAG}"
  )
  [[ -z "${CFG_RESUME_DIR:-}" ]] || cfg_args+=("+runner.resume_dir=${CFG_RESUME_DIR}")
  "${RECAP_PYTHON}" \
    "${ROOT}/examples/offline_rl/policy_optimization/cfg_rl/train_cfg.py" \
    "${cfg_args[@]}"
  ray stop --force >/dev/null 2>&1 || true
fi

echo "SO-101 RECAP stages complete: ${RECAP_STAGES}"
