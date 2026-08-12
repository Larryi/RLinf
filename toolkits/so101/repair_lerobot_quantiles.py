#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Check or repair exact global q01/q10/q50/q90/q99 in LeRobot stats."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rlinf.data.datasets.recap.stats import repair_numeric_quantiles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_paths", nargs="+")
    parser.add_argument(
        "--features", nargs="+", default=["observation.state", "action"]
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Atomically update stats.json; default is a read-only audit.",
    )
    args = parser.parse_args()

    for dataset_path in args.dataset_paths:
        changes = repair_numeric_quantiles(
            dataset_path, features=args.features, write=args.write
        )
        print(
            json.dumps(
                {
                    "dataset": dataset_path,
                    "written": args.write,
                    "max_abs_change": changes,
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
