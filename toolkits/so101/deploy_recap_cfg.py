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
import signal
import sys
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
    parser.add_argument("--chunk", type=int, default=50, help="action steps executed per inference")
    parser.add_argument("--frequency", type=float, default=20.0, help="action execution rate in Hz")
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
    parser.add_argument("--max-episodes", type=int, default=0, help="stop after N inferences (0 = run forever)")
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
        port=args.port,
        cameras=cameras,
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
    episode = 0
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

            with torch.no_grad():
                actions, _ = policy.predict_action_batch(env_obs)
            chunk = actions[0].float().cpu().numpy()[: args.chunk]

            print(f"[deploy] episode={episode} state={state.tolist()} "
                  f"pred[0]={chunk[0].tolist()}")
            for t in range(len(chunk)):
                robot.send_action(
                    {name: float(chunk[t, i]) for i, name in enumerate(STATE_ACTION_NAMES)}
                )
                time.sleep(step_dt)

            episode += 1
            if args.max_episodes and episode >= args.max_episodes:
                break
    finally:
        print("[deploy] disconnecting")
        robot.disconnect()


if __name__ == "__main__":
    main()
