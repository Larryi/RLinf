#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Verify SO-101 JAX OpenPI and RLinf PyTorch OpenPI with fixed noise.

The two models run in separate processes so a 24 GiB GPU never holds both.
Use ``all`` for orchestration or an individual stage for debugging.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _parse_frames(value: str) -> list[int]:
    frames = [int(item) for item in value.split(",") if item.strip()]
    if not frames:
        raise argparse.ArgumentTypeError("at least one frame is required")
    return frames


def _raw_observation(frame: dict[str, Any]) -> dict[str, Any]:
    return {
        "observation.images.front": np.asarray(
            frame["observation.images.front"]
        ),
        "observation.images.wrist": np.asarray(
            frame["observation.images.wrist"]
        ),
        "observation.state": np.asarray(
            frame["observation.state"], dtype=np.float32
        ),
        "prompt": str(frame["task"]),
    }


def _noise(seed: int, frame: int) -> np.ndarray:
    return np.random.default_rng(seed + frame).standard_normal(
        (50, 32), dtype=np.float32
    )


def _save_stage(path: Path, records: list[dict[str, Any]]) -> None:
    arrays = {}
    for idx, record in enumerate(records):
        for key, value in record.items():
            arrays[f"{idx}/{key}"] = np.asarray(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def run_jax(args: argparse.Namespace) -> None:
    import jax
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from openpi.policies import policy_config
    from openpi.training import config as openpi_config

    config = openpi_config.get_config(args.jax_config)
    policy = policy_config.create_trained_policy(config, args.jax_checkpoint)
    dataset = LeRobotDataset(args.repo_id, root=args.dataset)
    records = []
    for frame_index in args.frames:
        raw = _raw_observation(dataset[frame_index])
        transformed = policy._input_transform(dict(raw))  # parity probe
        result = policy.infer(raw, noise=_noise(args.seed, frame_index))
        records.append(
            {
                "frame": frame_index,
                "actions": np.asarray(result["actions"], dtype=np.float32),
                "state": np.asarray(transformed["state"]),
                "tokens": np.asarray(transformed["tokenized_prompt"]),
                "token_mask": np.asarray(transformed["tokenized_prompt_mask"]),
                "base_image": np.asarray(transformed["image"]["base_0_rgb"]),
                "wrist_image": np.asarray(
                    transformed["image"]["right_wrist_0_rgb"]
                ),
                "left_mask": np.asarray(
                    transformed["image_mask"]["left_wrist_0_rgb"]
                ),
            }
        )
    _save_stage(args.output / "jax_outputs.npz", records)
    print(f"jax_device={jax.devices()} frames={len(records)}")


def _batch_torch(value, torch, device):
    array = np.asarray(value)
    return torch.from_numpy(array.copy())[None].to(device)


def run_pytorch(args: argparse.Namespace) -> None:
    import torch
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.openpi_pytorch import get_model
    from rlinf.models.embodiment.openpi_pytorch.pi0_model.model import Observation

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is required for the full Pi0.5 parity gate")
    cfg = OmegaConf.create(
        {
            "model_type": "openpi_pytorch",
            "model_path": str(args.pytorch_checkpoint),
            "precision": args.pytorch_precision,
            "num_action_chunks": 50,
            "action_dim": 6,
            "num_steps": args.num_steps,
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
    policy = get_model(cfg).cuda().eval()
    jax_outputs = np.load(args.output / "jax_outputs.npz")
    records = []
    with torch.no_grad():
        for record_index, frame_index in enumerate(args.frames):
            transformed = {
                "image": {
                    "base_0_rgb": jax_outputs[f"{record_index}/base_image"],
                    "left_wrist_0_rgb": np.zeros_like(
                        jax_outputs[f"{record_index}/base_image"]
                    ),
                    "right_wrist_0_rgb": jax_outputs[
                        f"{record_index}/wrist_image"
                    ],
                },
                "image_mask": {
                    "base_0_rgb": np.asarray(True),
                    "left_wrist_0_rgb": jax_outputs[
                        f"{record_index}/left_mask"
                    ],
                    "right_wrist_0_rgb": np.asarray(True),
                },
                "state": jax_outputs[f"{record_index}/state"],
                "tokenized_prompt": jax_outputs[f"{record_index}/tokens"],
                "tokenized_prompt_mask": jax_outputs[
                    f"{record_index}/token_mask"
                ],
            }
            batched = {
                "image": {
                    key: _batch_torch(value, torch, policy.device)
                    for key, value in transformed["image"].items()
                },
                "image_mask": {
                    key: _batch_torch(value, torch, policy.device)
                    for key, value in transformed["image_mask"].items()
                },
                "state": _batch_torch(transformed["state"], torch, policy.device),
                "tokenized_prompt": _batch_torch(
                    transformed["tokenized_prompt"], torch, policy.device
                ),
                "tokenized_prompt_mask": _batch_torch(
                    transformed["tokenized_prompt_mask"], torch, policy.device
                ),
            }
            observation = Observation.from_dict(batched)
            noise = _batch_torch(_noise(args.seed, frame_index), torch, policy.device)
            model_actions = policy.model.sample_actions(
                observation,
                num_steps=args.num_steps,
                noise=noise,
            )
            physical = policy._output_transform_fn(
                {
                    "actions": model_actions[0].float().cpu().numpy(),
                    "state": np.asarray(transformed["state"]),
                }
            )["actions"]
            records.append(
                {
                    "frame": frame_index,
                    "actions": np.asarray(physical, dtype=np.float32),
                    "state": np.asarray(transformed["state"]),
                    "tokens": np.asarray(transformed["tokenized_prompt"]),
                    "token_mask": np.asarray(transformed["tokenized_prompt_mask"]),
                    "base_image": np.asarray(transformed["image"]["base_0_rgb"]),
                    "wrist_image": np.asarray(
                        transformed["image"]["right_wrist_0_rgb"]
                    ),
                    "left_mask": np.asarray(
                        transformed["image_mask"]["left_wrist_0_rgb"]
                    ),
                }
            )
    _save_stage(args.output / "pytorch_outputs.npz", records)
    print(f"torch_device={policy.device} frames={len(records)} strict_load=true")


def _norm_action_range(norm_stats: Path) -> np.ndarray:
    payload = json.loads(norm_stats.read_text())
    stats = payload.get("norm_stats", payload)
    action = stats.get("actions", stats.get("action"))
    if action is None:
        raise KeyError("norm_stats has neither actions nor action")
    q01 = np.asarray(action["q01"], dtype=np.float64)[:6]
    q99 = np.asarray(action["q99"], dtype=np.float64)[:6]
    return np.maximum(q99 - q01, 1e-6)


def run_compare(args: argparse.Namespace) -> None:
    jax_data = np.load(args.output / "jax_outputs.npz")
    torch_data = np.load(args.output / "pytorch_outputs.npz")
    action_range = _norm_action_range(args.norm_stats)
    frame_reports = []
    all_relative = []
    preprocessing_ok = True
    for idx, frame in enumerate(args.frames):
        for key in ("tokens", "token_mask", "left_mask"):
            preprocessing_ok &= np.array_equal(
                jax_data[f"{idx}/{key}"], torch_data[f"{idx}/{key}"]
            )
        for key in ("state", "base_image", "wrist_image"):
            preprocessing_ok &= np.allclose(
                jax_data[f"{idx}/{key}"],
                torch_data[f"{idx}/{key}"],
                atol=1e-6,
                rtol=1e-6,
            )
        jax_actions = jax_data[f"{idx}/actions"][:, :6]
        torch_actions = torch_data[f"{idx}/actions"][:, :6]
        error = np.abs(jax_actions - torch_actions)
        relative = error / action_range[None]
        all_relative.append(relative.reshape(-1))
        frame_reports.append(
            {
                "frame": frame,
                "mae": float(error.mean()),
                "max_abs": float(error.max()),
                "relative_mae": float(relative.mean()),
                "relative_p99": float(np.quantile(relative, 0.99)),
            }
        )
    relative = np.concatenate(all_relative)
    report = {
        "preprocessing_equal": bool(preprocessing_ok),
        "relative_mae": float(relative.mean()),
        "relative_p99": float(np.quantile(relative, 0.99)),
        "relative_max": float(relative.max()),
        "thresholds": {
            "max_relative_mae": args.max_relative_mae,
            "max_relative_p99": args.max_relative_p99,
        },
        "frames": frame_reports,
    }
    report["passed"] = bool(
        preprocessing_ok
        and report["relative_mae"] <= args.max_relative_mae
        and report["relative_p99"] <= args.max_relative_p99
    )
    (args.output / "parity_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "frames"}))
    if not report["passed"]:
        raise SystemExit(1)


def _tail(path: Path, lines: int = 30) -> str:
    content = path.read_text(errors="replace").splitlines()
    return "\n".join(content[-lines:])


def run_all(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    common = [
        "--dataset",
        str(args.dataset),
        "--repo-id",
        args.repo_id,
        "--frames",
        ",".join(map(str, args.frames)),
        "--seed",
        str(args.seed),
        "--num-steps",
        str(args.num_steps),
        "--output",
        str(args.output),
        "--jax-checkpoint",
        str(args.jax_checkpoint),
        "--pytorch-checkpoint",
        str(args.pytorch_checkpoint),
        "--norm-stats",
        str(args.norm_stats),
    ]
    stages = (
        ("jax", args.jax_python, args.openpi_repo),
        ("pytorch", args.pytorch_python, args.rlinf_repo),
    )
    for stage, python, cwd in stages:
        log_path = args.output / f"{stage}.log"
        env = os.environ.copy()
        source_root = cwd / ("src" if stage == "jax" else "")
        env["PYTHONPATH"] = f"{source_root}:{env.get('PYTHONPATH', '')}"
        with log_path.open("w") as log:
            result = subprocess.run(
                [str(python), str(script), stage, *common],
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            print(_tail(log_path), file=sys.stderr)
            raise SystemExit(f"{stage} stage failed; full log: {log_path}")
        print(_tail(log_path, 2))
    run_compare(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("all", "jax", "pytorch", "compare"))
    parser.add_argument("--jax-checkpoint", type=Path, required=True)
    parser.add_argument("--pytorch-checkpoint", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--repo-id", default="so101_grab_blue_pen_60")
    parser.add_argument("--frames", type=_parse_frames, default=[0, 100, 500])
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("outputs/so101_parity"))
    parser.add_argument("--jax-config", default="pi05_so101_60")
    parser.add_argument("--pytorch-precision", default="bf16")
    parser.add_argument("--jax-python", type=Path, default=Path("/home/larry/openpi-kuavo/.venv/bin/python"))
    parser.add_argument("--pytorch-python", type=Path, default=Path(".venv-openpi-recap/bin/python"))
    parser.add_argument("--openpi-repo", type=Path, default=Path("/home/larry/openpi-kuavo"))
    parser.add_argument("--rlinf-repo", type=Path, default=Path.cwd())
    parser.add_argument("--max-relative-mae", type=float, default=0.02)
    parser.add_argument("--max-relative-p99", type=float, default=0.05)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output = args.output.expanduser().resolve()
    if args.stage == "jax":
        run_jax(args)
    elif args.stage == "pytorch":
        run_pytorch(args)
    elif args.stage == "compare":
        run_compare(args)
    else:
        run_all(args)


if __name__ == "__main__":
    main()
