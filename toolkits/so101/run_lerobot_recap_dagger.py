#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Register the remote RECAP plugin and launch LeRobot's native rollout CLI.

Run this file with the ``lerobot_hil`` Conda environment (Python 3.12).  It
does not import RLInf or load the GPU model; ``serve_recap_cfg.py`` owns that.
"""

from lerobot_policy_recap_remote.configuration_recap_remote import RecapRemoteConfig


def main() -> None:
    # Importing the config above registers ``recap_remote`` before draccus
    # parses --policy.path. LeRobot's third-party fallback then imports the
    # matching RecapRemotePolicy class by naming convention.
    assert RecapRemoteConfig.get_choice_name(RecapRemoteConfig) == "recap_remote"
    from lerobot.scripts.lerobot_rollout import main as lerobot_rollout_main

    lerobot_rollout_main()


if __name__ == "__main__":
    main()
