#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Closed-loop deployment of a converted RECAP-CFG checkpoint on the real SO-101 arm.

The checkpoint must first be converted with ``sft2new`` (see
``infer_recap_cfg.py`` docstring). lerobot 0.4.4 ships the ``so101_follower``
driver; the only extra dependency is the Feetech servo SDK::

    .venv-openpi-recap-v3/bin/pip install "feetech-servo-sdk>=1.0.0,<2.0.0"

Usage::

    PYTHONPATH=/home/larry/RLinf .venv-openpi-recap-v3/bin/python \\
        toolkits/so101/deploy_recap_cfg.py \\
        --checkpoint <sft2new 输出目录> \\
        --port /dev/ttyACM0

Loop: ``get_observation()`` -> ``predict_action_batch()`` -> execute the
action chunk on the arm at ``--frequency``. The model's ``STATE_ACTION_NAMES``
(shoulder_pan..gripper) match the lerobot SO-101 motor names 1:1, and SO-101
data is in degrees (``use_degrees=True`` default). Safety (joint-step clip)
is OFF by default; enable with ``--safety``.
"""

from __future__ import annotations

import argparse
import collections
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.openpi_pytorch import get_model

STATE_ACTION_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)


class RTCInference:
    """Asynchronous policy inference with a rolling action queue (openpi-style RTC).

    A background thread keeps re-inferring 50-step chunks from the latest
    observation while the control loop consumes actions at 30 Hz, so inference
    overlaps execution instead of blocking it.
    """

    def __init__(self, policy, queue_threshold: int = 30, guidance_weight: float = 10.0):
        self.policy = policy
        self.threshold = queue_threshold
        # 10 -> follow predictions exactly; lower values smooth toward the
        # previously executed action.
        self.guidance = guidance_weight / 10.0
        self._obs = None
        self._obs_lock = threading.Lock()
        self._actions: collections.deque = collections.deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._infer_loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def update_observation(self, env_obs: dict) -> None:
        with self._obs_lock:
            self._obs = env_obs

    def pop_action(self, last: np.ndarray | None) -> np.ndarray | None:
        with self._lock:
            if not self._actions:
                return None
            raw = np.asarray(self._actions.popleft(), dtype=np.float32)
        if last is not None and self.guidance < 1.0:
            return last + self.guidance * (raw - last)
        return raw

    def _infer_loop(self) -> None:
        while not self._stop.is_set():
            with self._obs_lock:
                obs = self._obs
            if obs is None:
                time.sleep(0.01)
                continue
            with self._lock:
                need = len(self._actions) < self.threshold
            if not need:
                time.sleep(0.01)
                continue
            with torch.no_grad():
                actions, _ = self.policy.predict_action_batch(obs)
            chunk = actions[0].float().cpu().numpy()
            with self._lock:
                if len(self._actions) < self.threshold:
                    self._actions.extend(t for t in chunk)


def build_model_cfg(checkpoint: Path, num_steps: int) -> OmegaConf:
    """Build the eval model config (same shapes as the CFG training config)."""
    return OmegaConf.create(
        {
            "model_type": "openpi_pytorch",
            "model_path": str(checkpoint),
            "precision": "bf16",
            "num_action_chunks": 50,
            "action_dim": 6,
            "num_steps": num_steps,
            "openpi": {
                "task": "eval",
                "config_name": "pi05_so101",
                "model_action_dim": 32,
                "paligemma_variant": "gemma_2b",
                "action_expert_variant": "gemma_300m",
                "max_token_len": 200,
                "discrete_state_input": True,
                "action_chunk": 50,
                "action_env_dim": 6,
            },
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--port", default="/dev/ttyUSB0", help="SO-101 MotorBus serial port")
    parser.add_argument("--task", default="", help="task description; defaults to a generic prompt")
    parser.add_argument("--num-steps", type=int, default=10, help="sampling steps per inference")
    parser.add_argument("--execute-steps", type=int, default=20,
                        help="steps executed per inference cycle; the model always predicts 50 and we re-infer"
                             " after these. 0 = execute the whole 50-step chunk")
    parser.add_argument("--frequency", type=float, default=30.0, help="action execution rate in Hz")
    parser.add_argument("--front-camera", type=int, default=0, help="front camera device index")
    parser.add_argument("--wrist-camera", type=int, default=2, help="wrist camera device index")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--front-fourcc", default="", help="e.g. YUYV; empty = auto-detect")
    parser.add_argument("--wrist-fourcc", default="YUYV", help="e.g. YUYV; empty = auto-detect")
    parser.add_argument("--no-cameras", action="store_true", help="debug: state-only, no images")
    parser.add_argument("--use-radians", action="store_true",
                        help="default False: SO-101 data is in degrees")
    parser.add_argument("--safety", action="store_true",
                        help="optional: clip each joint step to --max-relative-target")
    parser.add_argument("--max-relative-target", type=float, default=10.0,
                        help="max per-step joint movement in degrees (used with --safety)")
    parser.add_argument("--no-calibrate", action="store_true", help="skip lerobot calibration on connect")
    parser.add_argument("--calibration-dir", default=None,
                        help="custom lerobot calibration dir; default = ~/.cache/huggingface/lerobot/calibration "
                             "(so_follower/so101.json is auto-loaded when id=so101)")
    parser.add_argument("--max-segments", type=int, default=0,
                        help="stop after N inference cycles (0 = run forever)")
    parser.add_argument("--rtc", action="store_true",
                        help="asynchronous real-time control (openpi-style RTC): a background thread keeps "
                             "inferring 50-step chunks while the loop executes at --frequency")
    parser.add_argument("--rtc-queue-threshold", type=int, default=30,
                        help="re-infer when queued actions fall below this")
    parser.add_argument("--rtc-execution-horizon", type=int, default=20,
                        help="accepted for openpi parity; scheduling is threshold-driven")
    parser.add_argument("--rtc-max-guidance-weight", type=float, default=10.0,
                        help="10 = follow predictions exactly; lower values smooth toward the last executed action")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Pi0.5 inference")

    print(f"[deploy] loading model from {args.checkpoint}")
    policy = get_model(build_model_cfg(args.checkpoint, args.num_steps)).cuda().eval()

    # --- cameras ---
    cameras: dict = {}
    if not args.no_cameras:
        from lerobot.cameras.opencv import OpenCVCameraConfig

        cameras["front"] = OpenCVCameraConfig(
            index_or_path=args.front_camera, fps=args.fps, width=args.width, height=args.height,
            fourcc=args.front_fourcc or None,
        )
        cameras["wrist"] = OpenCVCameraConfig(
            index_or_path=args.wrist_camera, fps=args.fps, width=args.width, height=args.height,
            fourcc=args.wrist_fourcc or None,
        )

    # --- robot ---
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    config = SO101FollowerConfig(
        id="so101",  # matches so_follower/so101.json in the lerobot calibration dir
        port=args.port,
        cameras=cameras,
        calibration_dir=args.calibration_dir,
        use_degrees=not args.use_radians,
        max_relative_target=args.max_relative_target if args.safety else None,
    )
    robot = SO101Follower(config)
    robot.connect(calibrate=not args.no_calibrate)
    print(f"[deploy] connected on {args.port} (cameras={list(cameras)} "
          f"use_degrees={not args.use_radians} safety={'on' if args.safety else 'off'})")

    def _stop(*_):
        print("\n[deploy] interrupt: disconnecting")
        try:
            robot.disconnect()
        finally:
            sys.exit(0)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    task = args.task or "Grab the blue pen and place it into the black box"
    step_dt = 1.0 / args.frequency
    execute = args.execute_steps or 50
    segment = 0
    rtc = None
    if args.rtc:
        rtc = RTCInference(policy, args.rtc_queue_threshold, args.rtc_max_guidance_weight)
        rtc.start()
        print(f"[deploy] RTC on (queue_threshold={args.rtc_queue_threshold} "
              f"guidance={args.rtc_max_guidance_weight})")
    last_cmd = None
    try:
        while True:
            obs = robot.get_observation()
            state = np.asarray([obs[name] for name in STATE_ACTION_NAMES], dtype=np.float32)
            env_obs: dict = {
                "main_images": np.asarray(obs["front"])[None],
                "states": state[None],
                "task_descriptions": [task],
            }
            if not args.no_cameras:
                env_obs["wrist_images"] = np.asarray(obs["wrist"])[None]

            if rtc is not None:
                rtc.update_observation(env_obs)
                cmd = rtc.pop_action(last_cmd)
                if cmd is not None:
                    last_cmd = cmd
                if last_cmd is not None:
                    robot.send_action(
                        {name: float(last_cmd[i]) for i, name in enumerate(STATE_ACTION_NAMES)}
                    )
                time.sleep(step_dt)
            else:
                with torch.no_grad():
                    actions, _ = policy.predict_action_batch(env_obs)
                chunk = actions[0].float().cpu().numpy()  # model always predicts 50 steps
                print(f"[deploy] segment={segment} state={state.tolist()} "
                      f"pred[0]={chunk[0].tolist()}")
                for t in range(min(execute, len(chunk))):
                    robot.send_action(
                        {name: float(chunk[t, i]) for i, name in enumerate(STATE_ACTION_NAMES)}
                    )
                    time.sleep(step_dt)

            segment += 1
            if args.max_segments and segment >= args.max_segments:
                break
    finally:
        if rtc is not None:
            rtc.stop()
        print("[deploy] disconnecting")
        robot.disconnect()


if __name__ == "__main__":
    main()
