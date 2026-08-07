# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from toolkits.so101.record_pi05_rollouts import (
    ACTION_Q99,
    disable_robot_torque,
    enable_robot_torque_at_current_pose,
    live_disposition_from_key,
    safe_target,
    validate_action_chunk,
    validate_args,
    write_episode_outcome,
)


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("s", "success"),
        ("t", "timeout"),
        ("f", "failure"),
        ("r", "redo"),
        ("q", "quit"),
        ("\n", "review"),
        ("x", None),
    ],
)
def test_live_disposition_keys(key: str, expected: str | None):
    assert live_disposition_from_key(key) == expected


def test_torque_enable_reanchors_goal_before_enabling():
    class FakeBus:
        def __init__(self):
            self.calls = []

        def sync_read(self, register):
            self.calls.append(("read", register))
            return {"joint": 12.5}

        def sync_write(self, register, values):
            self.calls.append(("write", register, values))

        def enable_torque(self):
            self.calls.append(("enable",))

        def disable_torque(self):
            self.calls.append(("disable",))

    robot = type("Robot", (), {"bus": FakeBus()})()
    enable_robot_torque_at_current_pose(robot)
    disable_robot_torque(robot)
    assert robot.bus.calls == [
        ("read", "Present_Position"),
        ("write", "Goal_Position", {"joint": 12.5}),
        ("enable",),
        ("disable",),
    ]


def test_validate_action_chunk_rejects_bad_shape_and_nonfinite():
    with pytest.raises(ValueError, match="shape"):
        validate_action_chunk(np.zeros((49, 6)))
    actions = np.zeros((50, 6), dtype=np.float32)
    actions[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_action_chunk(actions)


def test_safe_target_applies_training_envelope_then_delta_limit():
    state = ACTION_Q99 - 1
    target = np.full(6, 1e4, dtype=np.float32)
    result = safe_target(target, state, max_delta=0.25)
    np.testing.assert_allclose(result, state + 0.25)
    assert np.all(result <= ACTION_Q99)


def test_safe_target_can_explicitly_disable_all_software_limits():
    state = np.zeros(6, dtype=np.float32)
    target = np.full(6, 1e4, dtype=np.float32)
    result = safe_target(target, state, max_delta=0.25, disable_limits=True)
    np.testing.assert_array_equal(result, target)


def test_write_episode_outcome_upserts_atomically(tmp_path: Path):
    output = write_episode_outcome(tmp_path, 2, False)
    write_episode_outcome(tmp_path, 1, True, notes="clean")
    write_episode_outcome(tmp_path, 2, False, notes="late", outcome="timeout")

    frame = pd.read_parquet(output)
    assert frame["episode_index"].tolist() == [1, 2]
    assert frame["is_success"].tolist() == [True, False]
    assert frame["outcome"].tolist() == ["success", "timeout"]
    assert frame["notes"].tolist() == ["clean", "late"]


def test_validate_args_accepts_independent_camera_rates():
    from argparse import Namespace

    args = Namespace(
        execute=False,
        confirm_motion="",
        disable_action_limits=False,
        confirm_unlimited_motion="",
        fps=30,
        episode_seconds=20,
        reset_seconds=5,
        camera_fps=30,
        front_camera_fps=30,
        wrist_camera_fps=60,
        open_loop_horizon=5,
        rtc=False,
        max_step_delta=5,
        max_relative_target=5,
    )
    validate_args(args)
