# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lightweight utility helpers for ReCap datasets."""

from __future__ import annotations

import importlib.metadata
import io
import json
import logging
from pathlib import Path

import numpy as np

try:  # lerobot >= 0.2 layout
    from lerobot.datasets.utils import hf_transform_to_torch
except ModuleNotFoundError:  # lerobot < 0.2
    from lerobot.common.datasets.utils import hf_transform_to_torch
from PIL import Image as PILImage

logger = logging.getLogger(__name__)


def validate_lerobot_v3_stack(dataset_path: str | Path) -> None:
    """Fail early with an actionable message for an incompatible v3 stack."""
    info_path = Path(dataset_path) / "meta" / "info.json"
    if not info_path.exists():
        return
    with open(info_path, "r") as f:
        codebase_version = str(json.load(f).get("codebase_version", ""))
    if not codebase_version.startswith("v3"):
        return

    from packaging.version import Version

    datasets_version = Version(importlib.metadata.version("datasets"))
    if datasets_version < Version("4.0.0"):
        raise RuntimeError(
            f"LeRobot {codebase_version} dataset at {dataset_path} requires "
            f"datasets>=4.0, but {datasets_version} is installed. Recreate or "
            "repair the dedicated venv with: bash requirements/install.sh "
            "embodied --model openpi --env maniskill_libero --lerobot-v3 "
            "--no-flash-attn --venv .venv-openpi-recap"
        )


def cast_image_features(hf_dataset):
    """Cast image columns from struct to Image type for proper decoding."""
    from datasets import Image

    features = hf_dataset.features
    needs_cast = False
    new_features = features.copy()

    for key, feat in features.items():
        if isinstance(feat, dict) and "bytes" in feat:
            new_features[key] = Image()
            needs_cast = True

    if needs_cast:
        hf_dataset = hf_dataset.cast(new_features)
        hf_dataset.set_transform(hf_transform_to_torch)

    return hf_dataset


def decode_image_struct_batch(batch: dict) -> dict:
    """Decode Image-feature struct dicts before ``hf_transform_to_torch``."""
    for key in list(batch.keys()):
        vals = batch[key]
        if vals and isinstance(vals[0], dict) and "bytes" in vals[0]:
            batch[key] = [PILImage.open(io.BytesIO(v["bytes"])) for v in vals]
    return hf_transform_to_torch(batch)


def load_return_stats_from_dataset(
    dataset_path: str | Path,
) -> tuple[float | None, float | None]:
    """Load return min/max from a dataset's ``meta/stats.json``."""
    stats_path = Path(dataset_path) / "meta" / "stats.json"
    if not stats_path.exists():
        return None, None

    try:
        with open(stats_path, "r") as f:
            stats = json.load(f)
        return_stats = stats.get("return", {})
        return return_stats.get("min"), return_stats.get("max")
    except (json.JSONDecodeError, KeyError):
        return None, None


def load_returns_sidecar(
    dataset_path: str | Path,
    returns_tag: str | None = None,
) -> dict[int, dict[str, np.ndarray]] | None:
    """Load ``meta/returns_{tag}.parquet`` sidecar written by compute_returns.py."""
    import pyarrow.parquet as pq

    dataset_path = Path(dataset_path)
    sidecar_filename = (
        f"returns_{returns_tag}.parquet" if returns_tag else "returns.parquet"
    )
    sidecar_path = dataset_path / "meta" / sidecar_filename
    if not sidecar_path.exists():
        if returns_tag:
            raise FileNotFoundError(
                f"Returns sidecar not found: {sidecar_path}. "
                f"Run compute_returns.py with tag='{returns_tag}' first."
            )
        return None

    table = pq.read_table(str(sidecar_path))
    ep_col = table.column("episode_index").to_numpy()
    frame_col = table.column("frame_index").to_numpy()
    ret_col = table.column("return").to_numpy()
    rew_col = table.column("reward").to_numpy()

    sidecar: dict[int, dict[str, np.ndarray]] = {}
    for ep in np.unique(ep_col):
        mask = ep_col == ep
        frames = frame_col[mask]
        order = np.argsort(frames)
        sidecar[int(ep)] = {
            "return": ret_col[mask][order].astype(np.float32),
            "reward": rew_col[mask][order].astype(np.float32),
        }

    logger.info(f"Loaded returns sidecar: {sidecar_path} ({len(sidecar)} episodes)")
    return sidecar


def load_episode_outcomes(dataset_path: str | Path) -> dict[int, bool] | None:
    """Load optional episode-level success labels for rollout datasets.

    When the recorder's ``outcome`` column is present, only ``success`` is a
    successful terminal state. Both ``timeout`` and ``failure`` are failures.
    A conflicting ``is_success`` value is rejected instead of silently
    assigning an incorrect terminal reward.
    """
    import pyarrow.parquet as pq

    path = Path(dataset_path) / "meta" / "episode_outcomes.parquet"
    if not path.exists():
        return None
    schema_names = set(pq.ParquetFile(path).schema_arrow.names)
    required = {"episode_index", "is_success"}
    if not required.issubset(schema_names):
        raise ValueError(f"{path} must contain columns {sorted(required)}")
    columns = ["episode_index", "is_success"]
    if "outcome" in schema_names:
        columns.append("outcome")
    table = pq.read_table(path, columns=columns)
    episodes = table.column("episode_index").to_numpy()
    successes = table.column("is_success").to_numpy()
    if len(np.unique(episodes)) != len(episodes):
        raise ValueError(f"Duplicate episode_index values in {path}")
    if "outcome" in table.column_names:
        labels = [str(value).strip().lower() for value in table["outcome"].to_pylist()]
        allowed = {"success", "timeout", "failure"}
        unknown = sorted(set(labels) - allowed)
        if unknown:
            raise ValueError(f"Unsupported outcome labels in {path}: {unknown}")
        expected = np.asarray([label == "success" for label in labels])
        inconsistent = np.flatnonzero(expected != successes.astype(bool))
        if inconsistent.size:
            bad_episodes = [int(episodes[index]) for index in inconsistent[:10]]
            raise ValueError(
                f"Conflicting outcome/is_success labels in {path}; "
                f"episodes={bad_episodes}. timeout and failure must be false."
            )
        successes = expected
    outcomes = {
        int(episode): bool(success) for episode, success in zip(episodes, successes)
    }
    logger.info("Loaded episode outcomes: %s (%d episodes)", path, len(outcomes))
    return outcomes


def load_task_descriptions(dataset_path: str | Path) -> dict[int, str]:
    """Load task descriptions from ``meta/tasks.jsonl`` or ``meta/tasks.parquet``."""
    meta = Path(dataset_path) / "meta"
    jsonl = meta / "tasks.jsonl"
    if jsonl.exists():
        tasks = {}
        with open(jsonl, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    tasks[data.get("task_index", len(tasks))] = data.get("task", "")
        return tasks

    parquet = meta / "tasks.parquet"
    if parquet.exists():
        import pandas as pd

        df = pd.read_parquet(parquet)
        if "task_index" in df.columns and "task" in df.columns:
            return {int(r["task_index"]): str(r["task"]) for _, r in df.iterrows()}
        # LeRobot v3 stores task strings in the parquet index.
        if "task_index" in df.columns and not isinstance(df.index, pd.RangeIndex):
            return {
                int(task_index): str(task)
                for task, task_index in zip(df.index, df["task_index"])
            }

    return {}
