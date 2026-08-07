#!/usr/bin/env bash
# Upload SO-101 RECAP assets to the Hugging Face Hub so a vast.ai instance can
# fetch them with `hf download` (see run_recap_pipeline.sh).
#
# Four assets exist. Datasets and the OpenPI base model are REQUIRED by the
# pipeline; the value checkpoint is only needed when reusing a locally trained
# value model (RECAP_STAGES=advantages,cfg + VALUE_CHECKPOINT_REPO).
#
#   asset   repo (env)                 local source (env)                   repo-type
#   -----   -------------------------- ------------------------------------ ----------
#   sft     SFT_DATASET_REPO            SFT_SRC                              dataset
#   rollout ROLLOUT_DATASET_REPO        ROLLOUT_SRC                          dataset
#   openpi  OPENPI_CHECKPOINT_REPO      OPENPI_SRC                           model
#   value   VALUE_CHECKPOINT_REPO       VALUE_SRC (auto = latest global_step) model
#
# The OpenPI source must be the RLinf PyTorch checkpoint layout
# (model.safetensors + physical-intelligence/), NOT the JAX training output.
#
# Usage:
#   export HF_TOKEN=...
#   bash toolkits/so101/vast/upload_assets.sh                # upload all
#   bash toolkits/so101/vast/upload_assets.sh --only sft,openpi
#
# Env overrides: SFT_SRC ROLLOUT_SRC OPENPI_SRC VALUE_SRC VALUE_ROOT, PRIVATE
# (1 = private repos, default 0 = public).
set -Eeuo pipefail
umask 077

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"

: "${HF_TOKEN:?Set HF_TOKEN}"
: "${SFT_DATASET_REPO:?Set SFT_DATASET_REPO (e.g. owner/so101-demo90-dataset)}"
: "${ROLLOUT_DATASET_REPO:?Set ROLLOUT_DATASET_REPO (e.g. owner/so101-rollout-dataset)}"
: "${OPENPI_CHECKPOINT_REPO:?Set OPENPI_CHECKPOINT_REPO (e.g. owner/pi05-so101-rlinf-checkpoint)}"
: "${VALUE_CHECKPOINT_REPO:=}"

# Local source directories (overridable).
: "${SFT_SRC:=/mnt/pqssd/so101/datasets/merged_lerobot_dataset_with_dagger30_trimmed}"
: "${ROLLOUT_SRC:=/mnt/pqssd/so101/datasets/pi05_jax_rtc_rollouts_50_v3}"
: "${OPENPI_SRC:=/mnt/pqssd/so101/train_outputs/openpi_pi05/openpi-so101-pi05-60-30000/rlinf_ckpt}"
: "${VALUE_ROOT:=${ROOT}/outputs/so101_recap/value}"

: "${PRIVATE:=0}"

# Comma-separated subset to upload: sft,rollout,openpi,value. Default: all
# assets whose target repo is set (value only when VALUE_CHECKPOINT_REPO set).
ONLY="${ONLY:-}"

# Resolve the value source to the latest global_step_* unless overridden.
: "${VALUE_SRC:=}"
if [[ -z "${VALUE_SRC}" && -n "${VALUE_CHECKPOINT_REPO}" && -d "${VALUE_ROOT}" ]]; then
  VALUE_SRC="$(find "${VALUE_ROOT}" -type d -name 'global_step_*' 2>/dev/null | sort -V | tail -1 || true)"
fi

ensure_hf() {
  command -v hf >/dev/null 2>&1 || {
    echo "[setup] installing huggingface_hub CLI" >&2
    python3 -m pip install -q --upgrade "huggingface_hub[cli]"
  }
}

create_repo() {
  local repo_id="$1" repo_type="$2"
  local args=(repo create "${repo_id}" --repo-type "${repo_type}" --exist-ok)
  [[ "${PRIVATE}" == "1" ]] && args+=(--private)
  hf "${args[@]}" >/dev/null
}

upload_dir() {
  local repo_id="$1" src="$2" repo_type="$3"
  [[ -d "${src}" ]] || {
    echo "[skip] source missing: ${src}" >&2
    return 0
  }
  echo "[repo] ${repo_type} ${repo_id}"
  create_repo "${repo_id}" "${repo_type}"
  echo "[upload] ${src} -> ${repo_id} (this may take a while for large files)"
  hf upload "${repo_id}" "${src}" . --repo-type "${repo_type}" \
    --commit-message "assets: upload ${src##*/}"
  echo "[done] https://huggingface.co/${repo_id}"
}

wants() {
  [[ -z "${ONLY}" ]] && return 0
  [[ ",${ONLY}," == *",$1,"* ]]
}

ensure_hf

if wants sft; then
  upload_dir "${SFT_DATASET_REPO}" "${SFT_SRC}" dataset
fi
if wants rollout; then
  upload_dir "${ROLLOUT_DATASET_REPO}" "${ROLLOUT_SRC}" dataset
fi
if wants openpi; then
  upload_dir "${OPENPI_CHECKPOINT_REPO}" "${OPENPI_SRC}" model
fi
if wants value; then
  if [[ -n "${VALUE_CHECKPOINT_REPO}" ]]; then
    [[ -n "${VALUE_SRC}" ]] || {
      echo "VALUE_CHECKPOINT_REPO is set but no local value checkpoint found under ${VALUE_ROOT}" >&2
      exit 3
    }
    upload_dir "${VALUE_CHECKPOINT_REPO}" "${VALUE_SRC}" model
  else
    echo "[skip] value: set VALUE_CHECKPOINT_REPO to upload the locally trained value model"
  fi
fi

echo "[ok] all requested assets uploaded"
