# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Configuration for the lightweight remote RECAP policy."""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig

ACTION_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def _input_features() -> dict[str, PolicyFeature]:
    return {
        "observation.images.front": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 480, 640)
        ),
        "observation.images.wrist": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 480, 640)
        ),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
    }


def _output_features() -> dict[str, PolicyFeature]:
    return {"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))}


@PreTrainedConfig.register_subclass("recap_remote")
@dataclass
class RecapRemoteConfig(PreTrainedConfig):
    """Connection and feature schema for an RLInf RECAP inference server."""

    host: str = "127.0.0.1"
    port: int = 8001
    chunk_size: int = 50
    action_feature_names: list[str] = field(default_factory=lambda: list(ACTION_NAMES))
    input_features: dict[str, PolicyFeature] = field(default_factory=_input_features)
    output_features: dict[str, PolicyFeature] = field(default_factory=_output_features)
    device: str | None = "cpu"
    use_amp: bool = False

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> None:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None

    def get_optimizer_preset(self) -> AdamWConfig:
        """Satisfy the LeRobot policy interface; this policy is inference-only."""
        return AdamWConfig(lr=0.0)

    def get_scheduler_preset(self):
        return None

    def validate_features(self) -> None:
        expected = {
            "observation.images.front",
            "observation.images.wrist",
            "observation.state",
        }
        missing = expected - set(self.input_features)
        if missing:
            raise ValueError(
                f"Remote RECAP config is missing features: {sorted(missing)}"
            )
