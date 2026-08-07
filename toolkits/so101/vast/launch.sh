#!/usr/bin/env bash
# Provision code on the vast.ai instance and start the RECAP pipeline.
#
# Code provisioning (SOURCE_MODE):
#   git  (default)  - clone/pull GIT_REPO@GIT_BRANCH on the instance. VAST's
#                     network reaches GitHub at high speed, and the instance
#                     no longer depends on this machine being up.
#   rsync           - push the local checkout (including uncommitted changes)
#                     from this machine to the instance.
#
# Private GIT_REPO: embed a token in the URL, e.g.
#   https://<user>:<token>@github.com/<you>/RLinf.git
# (the URL lands in the instance's .git/config; instances are ephemeral).
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
: "${VAST_SSH_HOST:?Set VAST_SSH_HOST}"
: "${VAST_SSH_PORT:?Set VAST_SSH_PORT}"
: "${VAST_ENV_FILE:?Set VAST_ENV_FILE to a chmod-600 private env file}"
: "${VAST_SSH_USER:=root}"
: "${REMOTE_ROOT:=/workspace/RLinf}"
: "${DETACH:=1}"
: "${SOURCE_MODE:=git}"
: "${GIT_REPO:=}"
: "${GIT_BRANCH:=so101-recap}"
# START_PIPELINE=0: only deploy code + recap.env, do not start training
# (ssh in and run toolkits/so101/vast/remote_start.sh when you are ready).
: "${START_PIPELINE:=1}"
[[ -f "${VAST_ENV_FILE}" ]] || { echo "Missing ${VAST_ENV_FILE}" >&2; exit 2; }
mode="$(stat -c '%a' "${VAST_ENV_FILE}")"
[[ "${mode}" == "600" || "${mode}" == "400" ]] || {
  echo "VAST_ENV_FILE must have mode 600 or 400" >&2
  exit 2
}
remote="${VAST_SSH_USER}@${VAST_SSH_HOST}"
ssh_args=(-p "${VAST_SSH_PORT}" -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
ssh "${ssh_args[@]}" "${remote}" "mkdir -p '${REMOTE_ROOT}'"

if [[ "${SOURCE_MODE}" == "git" ]]; then
  : "${GIT_REPO:?Set GIT_REPO when SOURCE_MODE=git}"
  echo "[code] git ${GIT_BRANCH} <- ${GIT_REPO}"
  ssh "${ssh_args[@]}" "${remote}" \
    "REMOTE_ROOT='${REMOTE_ROOT}' GIT_REPO='${GIT_REPO}' GIT_BRANCH='${GIT_BRANCH}'" \
    bash -s <<'REMOTE'
set -euo pipefail
if [[ -d "${REMOTE_ROOT}/.git" ]]; then
  git -C "${REMOTE_ROOT}" fetch origin "${GIT_BRANCH}"
  git -C "${REMOTE_ROOT}" reset --hard "origin/${GIT_BRANCH}"
else
  # Any leftover files (previous rsync/aborted run) make git clone refuse the
  # directory, so wipe it and clone fresh. .secrets is recreated below and the
  # env file is re-scp'd on every launch, so nothing of value is lost.
  rm -rf "${REMOTE_ROOT}"
  git clone --depth 1 --branch "${GIT_BRANCH}" "${GIT_REPO}" "${REMOTE_ROOT}"
fi
REMOTE
else
  echo "[code] rsync ${ROOT}/ -> ${remote}:${REMOTE_ROOT}/"
  rsync -az --info=progress2 -e "ssh ${ssh_args[*]}" \
    --exclude '.git/' --exclude '.venv*/' --exclude '.secrets/' \
    --exclude 'outputs/' --exclude 'logs/' --exclude '*.env' \
    "${ROOT}/" "${remote}:${REMOTE_ROOT}/"
fi

ssh "${ssh_args[@]}" "${remote}" "mkdir -p '${REMOTE_ROOT}/.secrets' '${REMOTE_ROOT}/logs'"
scp -P "${VAST_SSH_PORT}" "${VAST_ENV_FILE}" "${remote}:${REMOTE_ROOT}/.secrets/recap.env"
ssh "${ssh_args[@]}" "${remote}" "chmod 600 '${REMOTE_ROOT}/.secrets/recap.env'"
command="cd '${REMOTE_ROOT}' && set -a && source .secrets/recap.env && set +a"
if [[ "${START_PIPELINE}" == "1" ]]; then
  if [[ "${DETACH}" == "1" ]]; then
    ssh "${ssh_args[@]}" "${remote}" \
      "${command} && nohup bash toolkits/so101/vast/run_recap_pipeline.sh >>logs/recap-launcher.log 2>&1 </dev/null & echo \$!"
  else
    ssh -t "${ssh_args[@]}" "${remote}" \
      "${command} && exec bash toolkits/so101/vast/run_recap_pipeline.sh"
  fi
else
  echo "[skip] pipeline not started (START_PIPELINE=0). To start it on the instance:"
  echo "  ssh root@${VAST_SSH_HOST} -p ${VAST_SSH_PORT}"
  echo "  bash ${REMOTE_ROOT}/toolkits/so101/vast/remote_start.sh"
fi
