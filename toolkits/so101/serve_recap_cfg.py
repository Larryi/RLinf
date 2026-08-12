#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Serve an RLInf Pi0.5 RECAP checkpoint through the OpenPI WebSocket API.

This process owns only the GPU model.  Cameras, robot serial ports, RTC queues,
DAgger intervention, and LeRobot dataset writing remain in the ``lerobot_hil``
client process.  Keeping RTC on the client avoids nested action queues.
"""

from __future__ import annotations

import argparse
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from openpi.serving.websocket_policy_server import WebsocketPolicyServer

from rlinf.models.embodiment.openpi_pytorch import get_model

STATE_ACTION_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)


def condition_task_prompt(task: str, advantage_condition: str) -> str:
    """Apply the exact advantage suffix used by RECAP CFG training."""
    if advantage_condition == "none":
        return task
    return f"{task}\nAdvantage: {advantage_condition}"


def build_model_cfg(checkpoint: Path, num_steps: int) -> OmegaConf:
    """Build the same strict eval model used by ``deploy_recap_cfg.py``."""
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


def _image(request: dict[str, Any], key: str) -> np.ndarray:
    image = np.asarray(request[key])
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"{key} must have shape [H,W,3], got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


class RecapWebsocketPolicy:
    """Adapt RLInf's batched eval API to OpenPI's request/response protocol."""

    def __init__(
        self,
        policy: Any,
        default_task: str,
        advantage_condition: str = "positive",
    ) -> None:
        self._policy = policy
        self._default_task = default_task
        self._advantage_condition = advantage_condition
        self._lock = threading.Lock()

    def infer(self, request: dict[str, Any]) -> dict[str, np.ndarray]:
        """Return one physical six-joint action chunk."""
        front = _image(request, "observation.images.front")
        wrist = _image(request, "observation.images.wrist")
        state = np.asarray(request["observation.state"], dtype=np.float32).reshape(-1)
        if state.shape != (6,):
            raise ValueError(
                f"observation.state must have shape [6], got {state.shape}"
            )
        task = str(request.get("prompt") or self._default_task).strip()
        if not task:
            raise ValueError("A task prompt is required")
        task = condition_task_prompt(task, self._advantage_condition)

        env_obs = {
            "main_images": front[None],
            "wrist_images": wrist[None],
            "states": state[None],
            "task_descriptions": [task],
        }
        # The WebSocket server may accept more than one client.  The model is
        # intentionally serialized so concurrent requests cannot race CUDA.
        with self._lock, torch.inference_mode():
            actions, _ = self._policy.predict_action_batch(env_obs)
        chunk = actions[0].float().cpu().numpy()
        if chunk.shape != (50, 6):
            raise RuntimeError(f"RECAP returned {chunk.shape}, expected (50, 6)")
        return {"actions": chunk.astype(np.float32, copy=False)}


class SmokePolicy:
    """Dependency-light deterministic model for cross-environment bridge tests."""

    def predict_action_batch(self, env_obs: dict[str, Any]):
        # MessagePack may decode a read-only NumPy view.  Copy it before
        # handing it to torch so even test policies never alias that buffer.
        state = torch.tensor(
            np.array(env_obs["states"], copy=True), dtype=torch.float32
        )
        return state[:, None, :].repeat(1, 50, 1), {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument(
        "--task",
        default="Grab the blue pen and place it into the black box",
    )
    parser.add_argument(
        "--advantage-condition",
        choices=("positive", "none", "negative"),
        default="positive",
        help="append the exact RECAP training suffix to the task prompt",
    )
    parser.add_argument(
        "--smoke-policy",
        action="store_true",
        help="serve deterministic repeated-state actions without loading CUDA",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    if args.smoke_policy:
        policy = SmokePolicy()
    else:
        if args.checkpoint is None:
            parser.error("--checkpoint is required unless --smoke-policy is set")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Pi0.5 RECAP inference")
        logging.info("Loading strict RECAP eval model from %s", args.checkpoint)
        policy = (
            get_model(build_model_cfg(args.checkpoint, args.num_steps)).cuda().eval()
        )

    adapter = RecapWebsocketPolicy(
        policy,
        default_task=args.task,
        advantage_condition=args.advantage_condition,
    )
    metadata = {
        "policy_type": "rlinf_recap_pi05",
        "action_shape": [50, 6],
        "action_names": list(STATE_ACTION_NAMES),
        "rtc_owner": "client",
        "advantage_condition": args.advantage_condition,
    }
    logging.info("Serving RECAP policy at ws://%s:%d", args.host, args.port)
    try:
        WebsocketPolicyServer(
            policy=adapter,
            host=args.host,
            port=args.port,
            metadata=metadata,
        ).serve_forever()
    except KeyboardInterrupt:
        logging.info("RECAP policy server stopped")


if __name__ == "__main__":
    main()
