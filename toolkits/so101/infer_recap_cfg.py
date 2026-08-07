#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Inference for a converted RECAP-CFG checkpoint on real SO-101 dataset frames.

The training checkpoint (``global_step_<N>/actor/model_state_dict/full_weights.pt``)
must first be converted to the new-format layout (``model.safetensors``) with::

    python -m rlinf.utils.ckpt_convertor.openpi.convert --mode sft2new \
        --ckpt <global_step_N dir> \\
        --input-norm-stats <so101 norm_stats.json> \\
        --output-model <out_dir> \\
        --output-norm-stats <out_dir>/physical-intelligence/behavior/norm_stats.json

Usage::

    PYTHONPATH=/home/larry/RLinf .venv-openpi-recap-v3/bin/python \\
        toolkits/so101/infer_recap_cfg.py \\
        --checkpoint <out_dir> \\
        --dataset /mnt/pqssd/so101/datasets/merged_lerobot_dataset_with_dagger30_trimmed \\
        --episode 0 --frame 50

Prints the sampled action chunk (first frames) and, when the dataset has an
``action`` column, the ground-truth action for that frame plus a simple MAE.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.openpi_pytorch import get_model


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


def episode_frame_indices(dataset, episode: int) -> tuple[int, int]:
    """Return (absolute_start, absolute_end) of an episode (0.4.x compatible)."""
    try:
        edi = dataset.episode_data_index
        start = int(edi.loc[episode, "from"])
        end = int(edi.loc[episode, "to"])
    except (KeyError, AttributeError):
        hf = dataset.hf_dataset
        ep_indices = [
            i for i in range(len(hf)) if int(hf[i]["episode_index"]) == episode
        ]
        if not ep_indices:
            raise ValueError(f"episode {episode} not found in dataset")
        start, end = min(ep_indices), max(ep_indices) + 1
    return start, end


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=50)
    parser.add_argument("--task", default="", help="task description; defaults to dataset task")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--print-frames", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Pi0.5 inference")

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    print(f"[infer] loading model from {args.checkpoint}")
    policy = get_model(build_model_cfg(args.checkpoint, args.num_steps)).cuda().eval()

    print(f"[infer] loading dataset {args.dataset}")
    dataset = LeRobotDataset(args.dataset.name, root=args.dataset, download_videos=False)
    start, end = episode_frame_indices(dataset, args.episode)
    abs_index = start + args.frame
    if not (start <= abs_index < end):
        raise ValueError(
            f"frame {args.frame} out of range for episode {args.episode} "
            f"({end - start} frames)"
        )
    sample = dataset.hf_dataset[abs_index]

    image = np.asarray(sample["observation.images.front"])  # H,W,C
    state = np.asarray(sample["observation.state"], dtype=np.float32)
    task = args.task or str(sample.get("task", "")).strip()
    if not task:
        raise ValueError("no task description; pass --task")

    env_obs: dict = {
        "main_images": image[None],
        "states": state[None],
        "task_descriptions": [task],
    }
    if "observation.images.wrist" in sample:
        env_obs["wrist_images"] = np.asarray(sample["observation.images.wrist"])[None]

    print(f"[infer] episode={args.episode} frame={args.frame} task={task!r} "
          f"state={state.shape} image={image.shape}")

    with torch.no_grad():
        actions, info = policy.predict_action_batch(env_obs)
    predicted = actions[0].float().cpu().numpy()  # [action_chunk, action_dim]

    print(f"[infer] sampled action chunk: {predicted.shape}")
    np.set_printoptions(precision=4, suppress=True)
    for t in range(min(args.print_frames, predicted.shape[0])):
        print(f"  t={t:3d} action={predicted[t]}")

    gt = sample.get("action")
    if gt is not None:
        gt = np.asarray(gt, dtype=np.float32)
        gt = gt[0] if gt.ndim == 2 else gt
        mae = float(np.mean(np.abs(predicted[0] - gt)))
        print(f"[infer] GT action (t=0)      ={gt}")
        print(f"[infer] MAE(pred[0], GT)     ={mae:.4f}")


if __name__ == "__main__":
    main()
