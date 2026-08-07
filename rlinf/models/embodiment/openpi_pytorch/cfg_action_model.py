# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Advantage-conditioned CFG training wrapper for self-contained OpenPI."""

from __future__ import annotations

import dataclasses
from typing import Any

import torch

from rlinf.models.embodiment.openpi_pytorch.pi0_model.model import Observation
from rlinf.models.embodiment.openpi_pytorch.sft_action_model import (
    OpenPiPytorchSFTActionModel,
)


@dataclasses.dataclass
class CFGObservation(Observation):
    """OpenPI observation carrying base and advantage-guidance prompt tokens."""

    tokenized_positive_guidance_prompt: torch.Tensor | None = None
    tokenized_positive_guidance_prompt_mask: torch.Tensor | None = None
    tokenized_negative_guidance_prompt: torch.Tensor | None = None
    tokenized_negative_guidance_prompt_mask: torch.Tensor | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CFGObservation":
        base = Observation.from_dict(data)
        pairs = (
            (
                "tokenized_positive_guidance_prompt",
                "tokenized_positive_guidance_prompt_mask",
            ),
            (
                "tokenized_negative_guidance_prompt",
                "tokenized_negative_guidance_prompt_mask",
            ),
        )
        for tokens_key, mask_key in pairs:
            if (tokens_key in data) != (mask_key in data):
                raise ValueError(f"{tokens_key} and {mask_key} must be provided together")
        return cls(
            images=base.images,
            image_masks=base.image_masks,
            state=base.state,
            tokenized_prompt=base.tokenized_prompt,
            tokenized_prompt_mask=base.tokenized_prompt_mask,
            token_ar_mask=base.token_ar_mask,
            token_loss_mask=base.token_loss_mask,
            pcd_xyz=base.pcd_xyz,
            tokenized_positive_guidance_prompt=data.get(
                "tokenized_positive_guidance_prompt"
            ),
            tokenized_positive_guidance_prompt_mask=data.get(
                "tokenized_positive_guidance_prompt_mask"
            ),
            tokenized_negative_guidance_prompt=data.get(
                "tokenized_negative_guidance_prompt"
            ),
            tokenized_negative_guidance_prompt_mask=data.get(
                "tokenized_negative_guidance_prompt_mask"
            ),
        )


def compute_cfg_routing_masks(
    advantage: torch.Tensor,
    *,
    positive_only_conditional: bool,
    unconditional_prob: float,
    random_values: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Route advantage labels to conditional or unconditional prompt branches."""
    if not 0.0 <= unconditional_prob <= 1.0:
        raise ValueError("unconditional_prob must be in [0, 1]")
    advantage = advantage.to(dtype=torch.bool)
    if random_values is None:
        random_values = torch.rand(advantage.shape[0], device=advantage.device)
    else:
        random_values = random_values.to(device=advantage.device)
    positive_mask = advantage
    negative_mask = ~positive_mask
    if positive_only_conditional:
        positive_conditional = positive_mask & (
            random_values > unconditional_prob
        )
        negative_conditional = torch.zeros_like(positive_mask)
    else:
        conditional = random_values > unconditional_prob
        positive_conditional = positive_mask & conditional
        negative_conditional = negative_mask & conditional
    conditional_mask = positive_conditional | negative_conditional
    return {
        "positive_mask": positive_mask,
        "negative_mask": negative_mask,
        "conditional_mask": conditional_mask,
        "positive_conditional_mask": positive_conditional,
        "positive_unconditional_mask": positive_mask & ~positive_conditional,
        "negative_conditional_mask": negative_conditional,
        "negative_unconditional_mask": negative_mask & ~negative_conditional,
    }


class OpenPiPytorchCFGActionModel(OpenPiPytorchSFTActionModel):
    """Train the self-contained Pi0.5 policy with RECAP CFG routing."""

    def __init__(
        self,
        pi0_model,
        *,
        num_steps: int,
        action_env_dim: int,
        unconditional_prob: float,
        positive_only_conditional: bool,
    ):
        super().__init__(
            pi0_model,
            num_steps=num_steps,
            action_env_dim=action_env_dim,
        )
        if not 0.0 <= unconditional_prob <= 1.0:
            raise ValueError("unconditional_prob must be in [0, 1]")
        self.unconditional_prob = unconditional_prob
        self.positive_only_conditional = positive_only_conditional

    @staticmethod
    def _masked_loss_sum(loss: torch.Tensor, mask: torch.Tensor) -> float:
        if not torch.any(mask):
            return 0.0
        return (loss * mask.float()).sum().detach().item()

    def forward(self, data: dict[str, Any], **kwargs):
        """Compute flow-matching loss after routing each sample's prompt."""
        observation = data["observation"]
        if isinstance(observation, dict):
            observation = CFGObservation.from_dict(observation)
        if not isinstance(observation, CFGObservation):
            raise TypeError(
                "CFG observation must be CFGObservation or dict, got "
                f"{type(observation)!r}"
            )
        actions = self._actions_to_device(data["actions"])
        advantage = torch.as_tensor(data["advantage"], device=self.device).bool()
        routing = compute_cfg_routing_masks(
            advantage,
            positive_only_conditional=self.positive_only_conditional,
            unconditional_prob=self.unconditional_prob,
            random_values=kwargs.get("random_values"),
        )
        positive_tokens = observation.tokenized_positive_guidance_prompt
        positive_masks = observation.tokenized_positive_guidance_prompt_mask
        negative_tokens = observation.tokenized_negative_guidance_prompt
        negative_masks = observation.tokenized_negative_guidance_prompt_mask
        if any(
            value is None
            for value in (
                positive_tokens,
                positive_masks,
                negative_tokens,
                negative_masks,
            )
        ):
            raise ValueError("CFG observation is missing guidance prompt tokens")

        positive_mask = routing["positive_mask"]
        conditional_mask = routing["conditional_mask"]
        if self.positive_only_conditional:
            guided_tokens = positive_tokens
            guided_masks = positive_masks
        else:
            guided_tokens = torch.where(
                positive_mask[:, None], positive_tokens, negative_tokens
            )
            guided_masks = torch.where(
                positive_mask[:, None], positive_masks, negative_masks
            )
        final_tokens = torch.where(
            conditional_mask[:, None], guided_tokens, observation.tokenized_prompt
        )
        final_masks = torch.where(
            conditional_mask[:, None], guided_masks, observation.tokenized_prompt_mask
        )
        selected = Observation(
            images=observation.images,
            image_masks=observation.image_masks,
            state=observation.state,
            tokenized_prompt=final_tokens,
            tokenized_prompt_mask=final_masks,
            token_ar_mask=observation.token_ar_mask,
            token_loss_mask=observation.token_loss_mask,
            pcd_xyz=observation.pcd_xyz,
        )
        selected = self._observation_to_device(selected)
        per_timestep = self.model.compute_loss(
            selected,
            actions,
            train=True,
            noise=kwargs.get("noise"),
            time=kwargs.get("time"),
        )
        per_sample = per_timestep.mean(dim=-1)
        metrics = {
            "conditional_count": conditional_mask.sum().item(),
            "unconditional_count": (~conditional_mask).sum().item(),
            "conditional_loss_sum": self._masked_loss_sum(
                per_sample, conditional_mask
            ),
            "unconditional_loss_sum": self._masked_loss_sum(
                per_sample, ~conditional_mask
            ),
            "positive_label_count": positive_mask.sum().item(),
            "negative_label_count": routing["negative_mask"].sum().item(),
        }
        for name in (
            "positive_conditional",
            "positive_unconditional",
            "negative_conditional",
            "negative_unconditional",
        ):
            mask = routing[f"{name}_mask"]
            metrics[f"{name}_count"] = mask.sum().item()
            metrics[f"{name}_loss_sum"] = self._masked_loss_sum(per_sample, mask)
        return per_timestep.mean(), metrics
