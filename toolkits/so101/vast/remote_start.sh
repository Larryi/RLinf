#!/usr/bin/env bash
# Run ON the vast.ai instance to start training manually, after launch.sh has
# deployed the code and recap.env (or after scp'ing recap.env to .secrets/):
#
#   ssh root@<host> -p <port>
#   bash /workspace/RLinf/toolkits/so101/vast/remote_start.sh
#
# Equivalently from your machine: bash toolkits/so101/vast/launch.sh (DETACH=1
# default) starts the pipeline automatically. Use this script when you want to
# control when training starts from the instance itself.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"
: "${SECRETS_FILE:=.secrets/recap.env}"
[[ -f "${SECRETS_FILE}" ]] || {
  echo "Missing ${SECRETS_FILE}. Run launch.sh from your machine once (it" >&2
  echo "deploys code + recap.env), or scp the env file here and retry." >&2
  exit 2
}
set -a && source "${SECRETS_FILE}" && set +a
mkdir -p logs
nohup bash toolkits/so101/vast/run_recap_pipeline.sh >> logs/recap-launcher.log 2>&1 </dev/null &
echo "pipeline PID=$!"
echo "log:    ${ROOT}/logs/recap-launcher.log"
echo "status: ${ROOT}/runs/*/status.json (appears after bootstrap)"
