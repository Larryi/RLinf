#!/usr/bin/env bash
set -euo pipefail

# Two-process RECAP -> LeRobot episodic DAgger launcher.
#
# Terminal 1: bash toolkits/so101/run_recap_dagger_local.sh server
# Terminal 2: bash toolkits/so101/run_recap_dagger_local.sh collect

REPO_ROOT="${REPO_ROOT:-/home/larry/RLinf}"
LEROBOT_ROOT="${LEROBOT_ROOT:-/home/larry/lerobot_latest}"
RECAP_CHECKPOINT="${RECAP_CHECKPOINT:-/mnt/pqssd/so101/recap_cfg_gs3000_infer}"
RECAP_HOST="${RECAP_HOST:-127.0.0.1}"
RECAP_PORT="${RECAP_PORT:-8001}"
TASK_PROMPT="${TASK_PROMPT:-Grab the blue pen and place it into the black box}"

FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/ttyACM0}"
FOLLOWER_ID="${FOLLOWER_ID:-so101}"
LEADER_PORT="${LEADER_PORT:-/dev/ttyACM1}"
LEADER_ID="${LEADER_ID:-so101leader}"
FRONT_CAMERA="${FRONT_CAMERA:-2}"
WRIST_CAMERA="${WRIST_CAMERA:-0}"
# video2 was previously observed to negotiate at 60 Hz. The control and
# dataset rates remain 30 Hz; only the camera capture request is 60 Hz.
FRONT_CAMERA_FPS="${FRONT_CAMERA_FPS:-60}"
WRIST_CAMERA_FPS="${WRIST_CAMERA_FPS:-30}"

DATASET_ROOT="${DATASET_ROOT:-/mnt/pqssd/so101/datasets/recap_v1_dagger_iter1}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/rollout_recap_v1_dagger_iter1}"
NUM_EPISODES="${NUM_EPISODES:-50}"
EPISODE_TIME_S="${EPISODE_TIME_S:-10}"
RESET_TIME_S="${RESET_TIME_S:-5}"
CONTROL_FPS="${CONTROL_FPS:-30}"
RESUME_DATASET="${RESUME_DATASET:-false}"

usage() {
  echo "Usage: $0 server|collect|doctor"
  echo "  server   Load the RECAP checkpoint and serve action chunks on localhost"
  echo "  collect  Run LeRobot episodic DAgger (robot/cameras/leader owner)"
  echo "  doctor   Print paths and currently visible devices without moving hardware"
}

doctor() {
  test -d "${RECAP_CHECKPOINT}" || { echo "Missing checkpoint: ${RECAP_CHECKPOINT}"; return 1; }
  test -x "${REPO_ROOT}/.venv-openpi-recap-v3/bin/python" || {
    echo "Missing RECAP Python: ${REPO_ROOT}/.venv-openpi-recap-v3/bin/python"
    return 1
  }
  test -d "${LEROBOT_ROOT}/src/lerobot" || { echo "Missing LeRobot source: ${LEROBOT_ROOT}"; return 1; }
  echo "checkpoint=${RECAP_CHECKPOINT}"
  echo "server=ws://${RECAP_HOST}:${RECAP_PORT}"
  echo "follower=${FOLLOWER_PORT} leader=${LEADER_PORT}"
  echo "front=/dev/video${FRONT_CAMERA}@${FRONT_CAMERA_FPS} wrist=/dev/video${WRIST_CAMERA}@${WRIST_CAMERA_FPS}"
  echo "dataset=${DATASET_ROOT} episodes=${NUM_EPISODES} max=${EPISODE_TIME_S}s reset=${RESET_TIME_S}s"
  ls -l "${FOLLOWER_PORT}" "${LEADER_PORT}" \
    "/dev/video${FRONT_CAMERA}" "/dev/video${WRIST_CAMERA}" 2>/dev/null || true
}

serve() {
  doctor
  cd "${REPO_ROOT}"
  export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  exec .venv-openpi-recap-v3/bin/python -u toolkits/so101/serve_recap_cfg.py \
    --checkpoint "${RECAP_CHECKPOINT}" \
    --host "${RECAP_HOST}" \
    --port "${RECAP_PORT}" \
    --task "${TASK_PROMPT}" \
    --advantage-condition positive
}

collect() {
  doctor
  if [[ "${CONFIRM_MOTION:-}" != "YES" ]]; then
    echo
    echo "This process will command ${FOLLOWER_PORT}. Keep the emergency stop ready."
    read -r -p "Type YES to enable robot motion: " answer
    [[ "${answer}" == "YES" ]] || { echo "Cancelled."; return 1; }
  fi

  camera_config="{ front: {type: opencv, index_or_path: ${FRONT_CAMERA}, width: 640, height: 480, fps: ${FRONT_CAMERA_FPS}}, wrist: {type: opencv, index_or_path: ${WRIST_CAMERA}, width: 640, height: 480, fps: ${WRIST_CAMERA_FPS}, fourcc: YUYV}}"
  export PYTHONPATH="${REPO_ROOT}/toolkits/so101:${LEROBOT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
  exec conda run -n lerobot_hil --no-capture-output \
    python -u "${REPO_ROOT}/toolkits/so101/run_lerobot_recap_dagger.py" \
    --strategy.type=dagger \
    --strategy.episodic=true \
    --strategy.record_autonomous=false \
    --strategy.enable_policy_override=true \
    --strategy.override_mode=snap_then_relative_joint \
    --strategy.start_with_override=false \
    --strategy.rewind_buffer_s=5 \
    --strategy.rewind_jump_frames=30 \
    --strategy.reset_on_intervention=true \
    --inference.type=rtc \
    --inference.queue_threshold=30 \
    --inference.rtc.execution_horizon=20 \
    --inference.rtc.max_guidance_weight=10 \
    --policy.path="${REPO_ROOT}/toolkits/so101/lerobot_policy_recap_remote/model" \
    --policy.host="${RECAP_HOST}" \
    --policy.port="${RECAP_PORT}" \
    --robot.type=so101_follower \
    --robot.port="${FOLLOWER_PORT}" \
    --robot.id="${FOLLOWER_ID}" \
    --robot.disable_torque_on_disconnect=true \
    --robot.cameras="${camera_config}" \
    --teleop.type=so101_leader \
    --teleop.port="${LEADER_PORT}" \
    --teleop.id="${LEADER_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.single_task="${TASK_PROMPT}" \
    --dataset.fps="${CONTROL_FPS}" \
    --dataset.num_episodes="${NUM_EPISODES}" \
    --dataset.episode_time_s="${EPISODE_TIME_S}" \
    --dataset.reset_time_s="${RESET_TIME_S}" \
    --dataset.streaming_encoding=false \
    --dataset.push_to_hub=false \
    --fps="${CONTROL_FPS}" \
    --interpolation_multiplier=1 \
    --resume="${RESUME_DATASET}" \
    --return_to_initial_position=false \
    --play_sounds=true
}

case "${1:-}" in
  server) serve ;;
  collect) collect ;;
  doctor) doctor ;;
  *) usage; exit 2 ;;
esac
