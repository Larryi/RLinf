#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Validate SO-101 ReCap datasets and checkpoints before local/cloud stages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rlinf.data.datasets.recap.stats import repair_numeric_quantiles


def _frame_keys(dataset_root: Path) -> pd.DataFrame:
    files = sorted((dataset_root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No data parquet files under {dataset_root / 'data'}")
    frame = pd.concat(
        [
            pd.read_parquet(path, columns=["episode_index", "frame_index"])
            for path in files
        ],
        ignore_index=True,
    )
    if frame.duplicated(["episode_index", "frame_index"]).any():
        raise ValueError(f"Duplicate frame keys in {dataset_root}")
    return frame


def _validate_sidecar(
    dataset_root: Path, filename: str, expected_keys: pd.DataFrame
) -> pd.DataFrame:
    path = dataset_root / "meta" / filename
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    keys = ["episode_index", "frame_index"]
    if not set(keys).issubset(frame.columns):
        raise ValueError(f"{path} is missing frame-key columns")
    if frame.duplicated(keys).any():
        raise ValueError(f"Duplicate frame keys in {path}")
    expected = set(map(tuple, expected_keys[keys].to_numpy()))
    actual = set(map(tuple, frame[keys].to_numpy()))
    if expected != actual:
        raise ValueError(
            f"Frame coverage mismatch in {path}: missing={len(expected - actual)}, "
            f"extra={len(actual - expected)}"
        )
    return frame


def validate_dataset(
    dataset_root: Path,
    *,
    dataset_type: str,
    returns_tag: str,
    advantage_tag: str | None,
    require_advantages: bool,
) -> dict:
    """Validate one LeRobot v3 dataset and its ReCap sidecars."""
    info_path = dataset_root / "meta" / "info.json"
    stats_path = dataset_root / "meta" / "stats.json"
    if not info_path.exists() or not stats_path.exists():
        raise FileNotFoundError(
            f"Missing LeRobot metadata under {dataset_root / 'meta'}"
        )
    info = json.loads(info_path.read_text())
    if not str(info.get("codebase_version", "")).startswith("v3"):
        raise ValueError(f"Expected LeRobot v3 dataset at {dataset_root}")

    quantile_changes = repair_numeric_quantiles(dataset_root, write=False)
    max_quantile_error = max(
        value for feature in quantile_changes.values() for value in feature.values()
    )
    if max_quantile_error > 1e-6:
        raise ValueError(
            f"Inexact q01/q99 statistics in {stats_path}; "
            "run toolkits/so101/repair_lerobot_quantiles.py --write"
        )

    data_keys = _frame_keys(dataset_root)
    returns = _validate_sidecar(
        dataset_root, f"returns_{returns_tag}.parquet", data_keys
    )
    if not np.isfinite(returns[["return", "reward"]].to_numpy()).all():
        raise ValueError(f"Non-finite return/reward values in {dataset_root}")

    outcome_counts: dict[str, int] = {}
    if dataset_type == "rollout":
        outcome_path = dataset_root / "meta" / "episode_outcomes.parquet"
        outcomes = pd.read_parquet(outcome_path)
        allowed = {"success", "timeout", "failure"}
        labels = outcomes["outcome"].astype(str).str.lower()
        if not set(labels).issubset(allowed):
            raise ValueError(f"Unknown outcomes in {outcome_path}")
        expected_success = labels.eq("success")
        if not np.array_equal(
            expected_success.to_numpy(), outcomes["is_success"].astype(bool)
        ):
            raise ValueError(f"Conflicting outcome/is_success values in {outcome_path}")
        terminal = (
            returns.sort_values(["episode_index", "frame_index"])
            .groupby("episode_index", as_index=False)
            .tail(1)
            .merge(outcomes[["episode_index", "outcome"]], on="episode_index")
        )
        unsuccessful = terminal[terminal["outcome"].isin(["timeout", "failure"])]
        if unsuccessful.empty or not (unsuccessful["reward"] < 0).all():
            raise ValueError("timeout/failure terminal rewards must all be negative")
        outcome_counts = labels.value_counts().sort_index().to_dict()

    positive_rate = None
    if require_advantages:
        if not advantage_tag:
            raise ValueError("--advantage-tag is required with --require-advantages")
        advantages = _validate_sidecar(
            dataset_root, f"advantages_{advantage_tag}.parquet", data_keys
        )
        required = {"advantage", "advantage_continuous", "value_current", "value_next"}
        if not required.issubset(advantages.columns):
            raise ValueError(
                f"Advantage sidecar is missing {sorted(required - set(advantages))}"
            )
        if not np.isfinite(
            advantages[
                ["advantage_continuous", "value_current", "value_next"]
            ].to_numpy()
        ).all():
            raise ValueError(f"Non-finite advantage/value fields in {dataset_root}")
        if dataset_type == "sft" and not advantages["advantage"].astype(bool).all():
            raise ValueError("All SFT advantage labels must be positive")
        positive_rate = float(advantages["advantage"].astype(bool).mean())

        mixture_path = dataset_root / "meta" / "mixture_config.yaml"
        mixture = yaml.safe_load(mixture_path.read_text()) or {}
        if advantage_tag not in mixture.get("tags", {}):
            raise ValueError(f"Tag {advantage_tag!r} missing from {mixture_path}")

    return {
        "path": str(dataset_root),
        "type": dataset_type,
        "episodes": int(data_keys["episode_index"].nunique()),
        "frames": len(data_keys),
        "outcomes": outcome_counts,
        "advantage_positive_rate": positive_rate,
        "max_quantile_error": max_quantile_error,
    }


def validate_policy(checkpoint: Path) -> dict:
    """Validate the PyTorch OpenPI base checkpoint and authoritative norm stats."""
    required = [
        checkpoint / "config.json",
        checkpoint / "model.safetensors",
        checkpoint / "physical-intelligence" / "behavior" / "norm_stats.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete OpenPI checkpoint: {missing}")
    norm_path = required[-1]
    stats = json.loads(norm_path.read_text())
    stats = stats.get("norm_stats", stats)
    for key in ("state", "actions"):
        if key not in stats:
            raise ValueError(f"{key!r} missing from {norm_path}")
        q01 = np.asarray(stats[key]["q01"], dtype=float)
        q99 = np.asarray(stats[key]["q99"], dtype=float)
        if (
            not np.isfinite(q01).all()
            or not np.isfinite(q99).all()
            or not (q99 > q01).all()
        ):
            raise ValueError(f"Invalid OpenPI quantile norm for {key!r} in {norm_path}")
    return {
        "path": str(checkpoint),
        "model_bytes": (checkpoint / "model.safetensors").stat().st_size,
        "norm_stats": str(norm_path),
        "norm_source": "OpenPI checkpoint (authoritative for CFG train/inference)",
    }


def validate_value(checkpoint: Path) -> dict:
    """Validate a Value checkpoint consumable by Advantage computation."""
    full_weights = checkpoint / "model_state_dict" / "full_weights.pt"
    dcp_metadata = checkpoint / "dcp_checkpoint" / ".metadata"
    if not full_weights.exists() or not dcp_metadata.exists():
        raise FileNotFoundError(
            f"Incomplete Value actor checkpoint at {checkpoint}; expected "
            "model_state_dict/full_weights.pt and dcp_checkpoint/.metadata"
        )
    return {"path": str(checkpoint), "full_weights_bytes": full_weights.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--value", type=Path)
    parser.add_argument("--returns-tag", default="so101")
    parser.add_argument("--advantage-tag")
    parser.add_argument("--require-advantages", action="store_true")
    args = parser.parse_args()

    report = {
        "sft": validate_dataset(
            args.sft,
            dataset_type="sft",
            returns_tag=args.returns_tag,
            advantage_tag=args.advantage_tag,
            require_advantages=args.require_advantages,
        ),
        "rollout": validate_dataset(
            args.rollout,
            dataset_type="rollout",
            returns_tag=args.returns_tag,
            advantage_tag=args.advantage_tag,
            require_advantages=args.require_advantages,
        ),
        "policy": validate_policy(args.policy),
    }
    if args.value:
        report["value"] = validate_value(args.value)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
