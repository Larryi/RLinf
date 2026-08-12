# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Exact numeric-feature statistics for LeRobot datasets.

LeRobot v3 currently aggregates per-episode quantiles by averaging them. That
operation is not equivalent to a quantile over all frames. The helpers here
compute quantiles directly from the numeric frame columns and can safely patch
only those fields in ``meta/stats.json``.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
import pyarrow.parquet as pq

QUANTILES = {
    "q01": 0.01,
    "q10": 0.10,
    "q50": 0.50,
    "q90": 0.90,
    "q99": 0.99,
}


def exact_quantiles(values: np.ndarray) -> dict[str, list[float]]:
    """Return exact global quantiles for a frame-major numeric array."""
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(
            f"Expected a non-empty [frames, dims] array, got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("Cannot compute normalization stats with NaN/Inf values")
    return {
        name: np.quantile(values, probability, axis=0).astype(float).tolist()
        for name, probability in QUANTILES.items()
    }


def _read_feature(dataset_root: Path, feature: str) -> np.ndarray:
    parquet_files = sorted((dataset_root / "data").rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No data parquet files found under {dataset_root / 'data'}"
        )

    chunks = []
    for parquet_file in parquet_files:
        schema = pq.ParquetFile(parquet_file).schema_arrow
        if feature not in schema.names:
            raise KeyError(f"Feature {feature!r} missing from {parquet_file}")
        column = pq.read_table(parquet_file, columns=[feature]).column(feature)
        chunks.append(np.asarray(column.to_pylist()))
    return np.concatenate(chunks, axis=0)


def repair_numeric_quantiles(
    dataset_root: str | Path,
    *,
    features: Sequence[str] = ("observation.state", "action"),
    write: bool = False,
) -> dict[str, dict[str, float]]:
    """Compute exact quantiles and optionally update ``meta/stats.json``.

    Returns the maximum absolute change for each quantile and feature. When
    writing, the original file is preserved once as
    ``meta/stats.pre_exact_quantiles.json``.
    """
    dataset_root = Path(dataset_root)
    stats_path = dataset_root / "meta" / "stats.json"
    with open(stats_path) as file:
        stats = json.load(file)

    changes: dict[str, dict[str, float]] = {}
    for feature in features:
        if feature not in stats:
            raise KeyError(f"Feature {feature!r} missing from {stats_path}")
        computed = exact_quantiles(_read_feature(dataset_root, feature))
        changes[feature] = {}
        for key, new_values in computed.items():
            old_values = np.asarray(stats[feature].get(key, new_values), dtype=float)
            changes[feature][key] = float(
                np.max(np.abs(old_values - np.asarray(new_values, dtype=float)))
            )
            stats[feature][key] = new_values

    if write:
        backup_path = stats_path.with_name("stats.pre_exact_quantiles.json")
        if not backup_path.exists():
            shutil.copy2(stats_path, backup_path)
        output_mode = (
            (backup_path if backup_path.exists() else stats_path).stat().st_mode
        )
        fd, temp_name = tempfile.mkstemp(
            prefix="stats.exact_quantiles.", suffix=".json", dir=stats_path.parent
        )
        try:
            with os.fdopen(fd, "w") as file:
                json.dump(stats, file, indent=4)
                file.write("\n")
            os.chmod(temp_name, output_mode)
            os.replace(temp_name, stats_path)
        except BaseException:
            Path(temp_name).unlink(missing_ok=True)
            raise

    return changes
