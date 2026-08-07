# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit coverage for the SO-101 RECAP bridge."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from rlinf.data.datasets.recap.utils import (
    load_episode_outcomes,
    load_task_descriptions,
)
from rlinf.models.embodiment.openpi.policies.so101_policy import (
    SO101Inputs,
    SO101Outputs,
)
from rlinf.models.embodiment.openpi_pytorch.cfg_action_model import (
    CFGObservation,
    OpenPiPytorchCFGActionModel,
    compute_cfg_routing_masks,
)


def test_lerobot_v3_task_descriptions_from_parquet_index(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    pd.DataFrame(
        {"task_index": [0]},
        index=["Grab the blue pen and place it into the black box"],
    ).to_parquet(meta / "tasks.parquet")

    assert load_task_descriptions(tmp_path) == {
        0: "Grab the blue pen and place it into the black box"
    }


def test_rollout_timeout_and_failure_are_both_unsuccessful(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    pd.DataFrame(
        {
            "episode_index": [0, 1, 2],
            "outcome": ["success", "timeout", "failure"],
            "is_success": [True, False, False],
        }
    ).to_parquet(meta / "episode_outcomes.parquet", index=False)

    assert load_episode_outcomes(tmp_path) == {0: True, 1: False, 2: False}


def test_rollout_outcome_rejects_conflicting_success_flag(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    pd.DataFrame(
        {
            "episode_index": [0],
            "outcome": ["timeout"],
            "is_success": [True],
        }
    ).to_parquet(meta / "episode_outcomes.parquet", index=False)

    with pytest.raises(ValueError, match="timeout and failure must be false"):
        load_episode_outcomes(tmp_path)


def test_so101_transforms_preserve_absolute_six_dimensional_actions():
    front = np.full((8, 10, 3), 17, dtype=np.uint8)
    wrist = np.full((3, 8, 10), 23, dtype=np.uint8)
    state = np.arange(6, dtype=np.float32)
    actions = np.arange(30, dtype=np.float32).reshape(5, 6)

    transformed = SO101Inputs()(
        {
            "observation.images.front": front,
            "observation.images.wrist": wrist,
            "observation.state": state,
            "action": actions,
            "prompt": "grab the blue pen",
        }
    )

    np.testing.assert_array_equal(transformed["image"]["base_0_rgb"], front)
    np.testing.assert_array_equal(
        transformed["image"]["right_wrist_0_rgb"], wrist.transpose(1, 2, 0)
    )
    assert not transformed["image_mask"]["left_wrist_0_rgb"]
    np.testing.assert_array_equal(transformed["state"], state)
    np.testing.assert_array_equal(transformed["actions"], actions)

    padded = np.pad(actions, ((0, 0), (0, 26)))
    np.testing.assert_array_equal(
        SO101Outputs()({"actions": padded})["actions"], actions
    )


def test_positive_only_cfg_routing_never_conditions_negative_samples():
    routing = compute_cfg_routing_masks(
        torch.tensor([True, False, True, False]),
        positive_only_conditional=True,
        unconditional_prob=0.25,
        random_values=torch.tensor([0.5, 0.9, 0.1, 0.0]),
    )
    assert routing["conditional_mask"].tolist() == [True, False, False, False]
    assert routing["negative_conditional_mask"].tolist() == [False] * 4


class _FakePi0(nn.Module):
    action_dim = 32

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.seen_tokens = None

    def compute_loss(self, observation, actions, **kwargs):
        self.seen_tokens = observation.tokenized_prompt.detach().clone()
        return actions[..., 0].square() + self.anchor

    def gradient_checkpointing_enable(self, **kwargs):
        return None

    def gradient_checkpointing_disable(self):
        return None


def test_openpi_pytorch_cfg_selects_guidance_tokens_per_advantage():
    core = _FakePi0()
    model = OpenPiPytorchCFGActionModel(
        core,
        num_steps=10,
        action_env_dim=6,
        unconditional_prob=0.0,
        positive_only_conditional=True,
    )
    batch_size = 2
    observation = CFGObservation(
        images={"base_0_rgb": torch.zeros(batch_size, 2, 2, 3)},
        image_masks={"base_0_rgb": torch.ones(batch_size, dtype=torch.bool)},
        state=torch.zeros(batch_size, 32),
        tokenized_prompt=torch.tensor([[1, 1], [1, 1]]),
        tokenized_prompt_mask=torch.ones(batch_size, 2, dtype=torch.bool),
        tokenized_positive_guidance_prompt=torch.tensor([[2, 2], [2, 2]]),
        tokenized_positive_guidance_prompt_mask=torch.ones(
            batch_size, 2, dtype=torch.bool
        ),
        tokenized_negative_guidance_prompt=torch.tensor([[3, 3], [3, 3]]),
        tokenized_negative_guidance_prompt_mask=torch.ones(
            batch_size, 2, dtype=torch.bool
        ),
    )
    loss, metrics = model(
        data={
            "observation": observation,
            "actions": torch.ones(batch_size, 50, 32),
            "advantage": torch.tensor([True, False]),
        },
        random_values=torch.ones(batch_size),
    )

    assert loss.item() == 1.0
    assert core.seen_tokens.tolist() == [[2, 2], [1, 1]]
    assert metrics["conditional_count"] == 1
    assert metrics["unconditional_count"] == 1
