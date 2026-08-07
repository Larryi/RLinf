# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Create non-destructive episode success labels for LeRobot v3 rollouts."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import pandas as pd


def parse_episode_set(spec: str) -> set[int]:
    """Parse comma-separated episode IDs and inclusive ranges."""
    result: set[int] = set()
    if not spec:
        return result
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"Invalid descending range: {token}")
            result.update(range(start, end + 1))
        else:
            result.add(int(token))
    if any(index < 0 for index in result):
        raise ValueError("Episode indices must be non-negative")
    return result


def load_episode_indices(dataset_path: Path) -> list[int]:
    """Read episode IDs from v3 metadata, falling back to data parquet files."""
    episode_files = sorted((dataset_path / "meta" / "episodes").rglob("*.parquet"))
    if episode_files:
        values = pd.concat(
            [pd.read_parquet(path, columns=["episode_index"]) for path in episode_files],
            ignore_index=True,
        )["episode_index"]
    else:
        data_files = sorted((dataset_path / "data").rglob("*.parquet"))
        if not data_files:
            raise FileNotFoundError(f"No episode or data parquet files under {dataset_path}")
        values = pd.concat(
            [pd.read_parquet(path, columns=["episode_index"]) for path in data_files],
            ignore_index=True,
        )["episode_index"]
    return sorted({int(value) for value in values})


def parse_success(value: object) -> bool:
    """Parse common CSV representations of a success label."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "success", "s", "yes", "y"}:
        return True
    if text in {"0", "false", "failure", "fail", "f", "no", "n"}:
        return False
    raise ValueError(f"Invalid is_success value: {value!r}")


def load_labels_csv(path: Path) -> dict[int, tuple[bool, str]]:
    """Load episode_index,is_success[,notes] labels from CSV."""
    frame = pd.read_csv(path)
    required = {"episode_index", "is_success"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} must contain columns {sorted(required)}")
    result = {}
    for row in frame.to_dict(orient="records"):
        episode = int(row["episode_index"])
        notes = str(row.get("notes", ""))
        result[episode] = (parse_success(row["is_success"]), notes)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--success", default="", help="IDs/ranges, e.g. 0-9,12")
    parser.add_argument("--failure", default="", help="IDs/ranges, e.g. 10-11,13")
    parser.add_argument("--labels-csv", type=Path)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_path = args.dataset.expanduser().resolve()
    episodes = load_episode_indices(dataset_path)
    valid = set(episodes)
    output = dataset_path / "meta" / "episode_outcomes.parquet"
    labels: dict[int, tuple[bool, str]] = {}
    if output.exists():
        existing = pd.read_parquet(output)
        labels.update(
            {
                int(row["episode_index"]): (
                    bool(row["is_success"]),
                    str(row.get("notes", "")),
                )
                for row in existing.to_dict(orient="records")
            }
        )
    if args.labels_csv:
        labels.update(load_labels_csv(args.labels_csv))

    success = parse_episode_set(args.success)
    failure = parse_episode_set(args.failure)
    overlap = success & failure
    if overlap:
        raise ValueError(f"Episodes labeled both success and failure: {sorted(overlap)}")
    for episode in success:
        labels[episode] = (True, labels.get(episode, (True, ""))[1])
    for episode in failure:
        labels[episode] = (False, labels.get(episode, (False, ""))[1])

    unknown = set(labels) - valid
    if unknown:
        raise ValueError(f"Labels reference unknown episodes: {sorted(unknown)}")
    if args.interactive:
        for episode in episodes:
            current = labels.get(episode)
            suffix = "success" if current and current[0] else "failure" if current else "unset"
            while True:
                answer = input(f"episode {episode} [{suffix}] s/f/Enter: ").strip().lower()
                if not answer:
                    break
                if answer in {"s", "success"}:
                    labels[episode] = (True, current[1] if current else "")
                    break
                if answer in {"f", "fail", "failure"}:
                    labels[episode] = (False, current[1] if current else "")
                    break

    missing = valid - set(labels)
    if missing and not args.allow_partial:
        raise ValueError(
            f"{len(missing)} episodes remain unlabeled (first 20: {sorted(missing)[:20]}). "
            "Label all rollout episodes or pass --allow-partial while annotating."
        )
    frame = pd.DataFrame(
        [
            {
                "episode_index": episode,
                "is_success": labels[episode][0],
                "notes": labels[episode][1],
            }
            for episode in sorted(labels)
        ]
    )
    successes = int(frame["is_success"].sum()) if len(frame) else 0
    print(
        f"episodes={len(episodes)} labeled={len(frame)} success={successes} "
        f"failure={len(frame) - successes} missing={len(missing)}"
    )
    if args.dry_run:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="episode_outcomes.", suffix=".parquet", dir=output.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(output)


if __name__ == "__main__":
    main()
