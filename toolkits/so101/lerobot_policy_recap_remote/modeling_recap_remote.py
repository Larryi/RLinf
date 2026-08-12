# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""LeRobot policy wrapper that delegates Pi0.5 inference over WebSocket."""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.policies.pretrained import PreTrainedPolicy
from openpi_client.websocket_client_policy import WebsocketClientPolicy

from .configuration_recap_remote import RecapRemoteConfig


def _image_to_uint8_hwc(value: Any, name: str) -> np.ndarray:
    tensor = torch.as_tensor(value).detach().cpu()
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError(f"{name} only supports batch size 1, got {tensor.shape}")
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(
            f"{name} must be a three-dimensional image, got {tensor.shape}"
        )
    if tensor.shape[0] == 3:
        tensor = tensor.permute(1, 2, 0)
    if tensor.shape[-1] != 3:
        raise ValueError(f"{name} must have three channels, got {tensor.shape}")
    array = tensor.numpy()
    if np.issubdtype(array.dtype, np.floating):
        if array.size and float(np.nanmax(array)) <= 1.5:
            array = array * 255.0
        array = np.rint(array)
    return np.clip(array, 0, 255).astype(np.uint8)


def batch_to_request(batch: dict[str, Any]) -> dict[str, Any]:
    """Convert LeRobot's CPU inference batch to the proven RLInf eval schema."""
    state = torch.as_tensor(batch["observation.state"]).detach().cpu()
    if state.ndim == 2:
        if state.shape[0] != 1:
            raise ValueError(
                f"Remote RECAP only supports batch size 1, got {state.shape}"
            )
        state = state[0]
    state_array = state.float().numpy().reshape(-1)
    if state_array.shape != (6,):
        raise ValueError(
            f"observation.state must have shape [6], got {state_array.shape}"
        )

    task = batch.get("task", "")
    if isinstance(task, (list, tuple)):
        task = task[0] if task else ""
    return {
        "observation.images.front": _image_to_uint8_hwc(
            batch["observation.images.front"], "observation.images.front"
        ),
        "observation.images.wrist": _image_to_uint8_hwc(
            batch["observation.images.wrist"], "observation.images.wrist"
        ),
        "observation.state": state_array.astype(np.float32, copy=False),
        "prompt": str(task),
    }


class RecapRemotePolicy(PreTrainedPolicy):
    """Inference-only policy whose model lives in the RLInf Python process."""

    config_class = RecapRemoteConfig
    name = "recap_remote"

    def __init__(self, config: RecapRemoteConfig) -> None:
        super().__init__(config)
        self._client = WebsocketClientPolicy(host=config.host, port=config.port)
        self._actions: collections.deque[torch.Tensor] = collections.deque()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: RecapRemoteConfig | None = None,
        **_: Any,
    ) -> "RecapRemotePolicy":
        """Connect to the server; the local directory contains no model weights."""
        if config is None:
            config = RecapRemoteConfig.from_pretrained(pretrained_name_or_path)
        return cls(config)

    def reset(self) -> None:
        """Discard client-side sync cache; LeRobot clears its RTC queue separately."""
        self._actions.clear()

    def get_optim_params(self) -> dict:
        return {}

    def forward(self, batch: dict[str, torch.Tensor]):
        raise RuntimeError("RecapRemotePolicy is inference-only")

    @torch.inference_mode()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Request one 50-step absolute six-joint chunk from RLInf."""
        response = self._client.infer(batch_to_request(batch))
        actions = np.asarray(response["actions"], dtype=np.float32)
        if actions.shape != (self.config.chunk_size, 6):
            raise RuntimeError(
                f"Remote RECAP returned {actions.shape}, "
                f"expected ({self.config.chunk_size}, 6)"
            )
        return torch.from_numpy(actions.copy()).unsqueeze(0)

    @torch.inference_mode()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Synchronous fallback; episodic DAgger should normally use RTC."""
        if not self._actions:
            self._actions.extend(self.predict_action_chunk(batch)[0])
        return self._actions.popleft().unsqueeze(0)
