# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""OpenPI observation/action transforms for the SO-101 follower arm."""

from __future__ import annotations

import dataclasses

import einops
import numpy as np
import openpi.transforms as transforms

STATE_ACTION_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
CAMERA_KEYS = (
    "observation.images.front",
    "observation.images.wrist",
)


def _parse_image(image: np.ndarray) -> np.ndarray:
    """Return one RGB image as uint8 HWC."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        scale = 255.0 if image.size == 0 or float(np.nanmax(image)) <= 1.0 else 1.0
        image = np.clip(image * scale, 0, 255).astype(np.uint8)
    if image.ndim != 3:
        raise ValueError(f"Expected an unbatched RGB image, got {image.shape}")
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image, got {image.shape}")
    return image


@dataclasses.dataclass(frozen=True)
class SO101Inputs(transforms.DataTransformFn):
    """Map a two-camera SO-101 LeRobot observation to OpenPI model inputs."""

    def __call__(self, data: dict) -> dict:
        # Accept both the raw LeRobot keys (used by value training and
        # so101_dataconfig) and the repacked keys produced by
        # compute_advantages' build_obs (KEY_MAPPINGS["so101"]).
        front_key = (
            "observation.images.front"
            if "observation.images.front" in data
            else "observation/image"
        )
        wrist_key = (
            "observation.images.wrist"
            if "observation.images.wrist" in data
            else "observation/wrist_image"
        )
        front = _parse_image(data[front_key])
        wrist = _parse_image(data[wrist_key])
        state_key = (
            "observation.state" if "observation.state" in data else "observation/state"
        )
        result = {
            "image": {
                "base_0_rgb": front,
                "left_wrist_0_rgb": np.zeros_like(front),
                "right_wrist_0_rgb": wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.True_,
            },
            "state": np.asarray(data[state_key], dtype=np.float32),
        }
        if "action" in data:
            result["actions"] = np.asarray(data["action"], dtype=np.float32)
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        elif "task" in data:
            result["prompt"] = data["task"]
        return result


@dataclasses.dataclass(frozen=True)
class SO101Outputs(transforms.DataTransformFn):
    """Trim model padding and return absolute SO-101 position targets."""

    action_dim: int = len(STATE_ACTION_NAMES)

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., : self.action_dim]}
