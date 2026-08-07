# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Run one full Pi0.5 RECAP-CFG training forward from parity inputs."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.openpi_pytorch import get_model


def _tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array).copy())[None].to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parity-inputs", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full-model CFG smoke test")

    cfg = OmegaConf.create(
        {
            "model_type": "openpi_pytorch",
            "model_path": str(args.checkpoint),
            "precision": "bf16",
            "num_action_chunks": 50,
            "action_dim": 6,
            "num_steps": 10,
            "openpi": {
                "task": "cfg",
                "config_name": "pi05_so101",
                "model_action_dim": 32,
                "paligemma_variant": "gemma_2b",
                "action_expert_variant": "gemma_300m",
                "max_token_len": 200,
                "discrete_state_input": True,
                "action_chunk": 50,
                "action_env_dim": 6,
                "unconditional_prob": 0.1,
                "positive_only_conditional": True,
                "train_expert_only": True,
            },
        }
    )
    model = get_model(cfg).cuda().eval()
    device = model.device
    source = np.load(args.parity_inputs)
    base = source["0/base_image"]
    tokens = _tensor(source["0/tokens"], device)
    token_mask = _tensor(source["0/token_mask"], device)
    observation = {
        "image": {
            "base_0_rgb": _tensor(base, device),
            "left_wrist_0_rgb": _tensor(np.zeros_like(base), device),
            "right_wrist_0_rgb": _tensor(source["0/wrist_image"], device),
        },
        "image_mask": {
            "base_0_rgb": torch.ones(1, dtype=torch.bool, device=device),
            "left_wrist_0_rgb": torch.zeros(1, dtype=torch.bool, device=device),
            "right_wrist_0_rgb": torch.ones(1, dtype=torch.bool, device=device),
        },
        "state": _tensor(source["0/state"], device),
        "tokenized_prompt": tokens,
        "tokenized_prompt_mask": token_mask,
        "tokenized_positive_guidance_prompt": tokens,
        "tokenized_positive_guidance_prompt_mask": token_mask,
        "tokenized_negative_guidance_prompt": tokens,
        "tokenized_negative_guidance_prompt_mask": token_mask,
    }
    generator = torch.Generator(device=device).manual_seed(0)
    noise = torch.randn((1, 50, 32), generator=generator, device=device)
    time = torch.full((1,), 0.5, device=device)
    with torch.no_grad():
        loss, metrics = model(
            {
                "observation": observation,
                "actions": torch.zeros((1, 50, 32), device=device),
                "advantage": torch.ones(1, dtype=torch.bool, device=device),
            },
            noise=noise,
            time=time,
            random_values=torch.ones(1, device=device),
        )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    result = {
        "loss": float(loss),
        "trainable_params": trainable,
        "total_params": total,
        **metrics,
    }
    if not np.isfinite(result["loss"]):
        raise RuntimeError(f"non-finite CFG loss: {result['loss']}")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
