#!/usr/bin/env bash
set -euo pipefail

: "${VAST_SSH_HOST:?Set VAST_SSH_HOST}"
: "${VAST_SSH_PORT:?Set VAST_SSH_PORT}"
: "${VAST_SSH_USER:=root}"
: "${WORK_ROOT:=/workspace/rlinf-so101-recap}"
: "${TAIL_LINES:=30}"
remote="${VAST_SSH_USER}@${VAST_SSH_HOST}"
ssh -p "${VAST_SSH_PORT}" -o ServerAliveInterval=30 "${remote}" bash -s -- \
  "${WORK_ROOT}" "${TAIL_LINES}" <<'REMOTE'
set -euo pipefail
work_root="$1"
tail_lines="$2"
status="$(find "${work_root}/runs" -name status.json -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)"
if [[ -n "${status}" ]]; then
  cat "${status}"
  log="${status%/status.json}/logs/pipeline.log"
  [[ ! -f "${log}" ]] || tail -n "${tail_lines}" "${log}"
else
  echo "No status file yet"
fi
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader 2>/dev/null || true
REMOTE
