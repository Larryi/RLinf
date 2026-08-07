#!/usr/bin/env python3
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

"""Collect labeled Pi0.5 rollouts from an SO-101 into LeRobot v3.

Two policy backends are supported:

* ``openpi-websocket`` connects to an OpenPI policy server (JAX stays in its
  own environment/process).
* ``rlinf-pytorch`` loads the converted ``model.safetensors`` directly.

Motion is locked unless ``--execute --confirm-motion SO101`` is supplied.
Every completed episode is reviewed before it is saved: success/failure is
written to ``meta/episode_outcomes.parquet`` for the RECAP data pipeline.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import queue
import select
import sys
import tempfile
import termios
import threading
import time
import tty
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

ACTION_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)

# q01/q99 of the 60-episode SO-101 SFT dataset. These are a deployment
# envelope, not the robot's mechanical limits.
ACTION_Q01 = np.array(
    [-23.05217, -99.54145, -51.98219, 33.97929, -42.88221, 0.63827],
    dtype=np.float32,
)
ACTION_Q99 = np.array(
    [36.92817, 49.96341, 99.96682, 84.16307, 13.03409, 29.86958],
    dtype=np.float32,
)


class PolicyBackend(Protocol):
    """Minimal policy interface used by the hardware loop."""

    def infer(self, observation: dict[str, Any], prompt: str) -> np.ndarray:
        """Return one action chunk with shape ``(50, 6)``."""


def camera_source(value: str) -> int | Path:
    """Turn a numeric V4L index into int and a device path into Path."""
    return int(value) if value.isdigit() else Path(value)


def validate_device_paths(args: argparse.Namespace) -> None:
    """Fail early when a requested serial/camera device is absent or inaccessible."""
    requested = {"robot": Path(args.robot_port)}
    for name, source in (
        ("front camera", args.front_camera),
        ("wrist camera", args.wrist_camera),
    ):
        parsed = camera_source(source)
        requested[name] = (
            Path(f"/dev/video{parsed}") if isinstance(parsed, int) else parsed
        )
    for name, path in requested.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} device does not exist: {path}")
        if not os.access(path, os.R_OK | os.W_OK):
            raise PermissionError(
                f"No read/write permission for {name} device {path}; "
                "check udev ACLs or membership of the dialout/video groups."
            )


def current_state(observation: dict[str, Any]) -> np.ndarray:
    """Extract the ordered six-joint state expected by Pi0.5."""
    return np.asarray([observation[name] for name in ACTION_NAMES], dtype=np.float32)


def validate_action_chunk(actions: Any) -> np.ndarray:
    """Validate and normalize a Pi0.5 action chunk."""
    chunk = np.asarray(actions, dtype=np.float32)
    if chunk.shape != (50, 6):
        raise ValueError(
            f"Expected Pi0.5 actions with shape (50, 6), got {chunk.shape}"
        )
    if not np.isfinite(chunk).all():
        raise ValueError("Policy returned NaN or Inf actions")
    return chunk


def safe_target(
    raw_target: Any,
    state: Any,
    max_delta: float,
    *,
    disable_limits: bool = False,
) -> np.ndarray:
    """Apply the training envelope followed by a per-control-step delta cap."""
    target = np.asarray(raw_target, dtype=np.float32)
    if disable_limits:
        return target
    current = np.asarray(state, dtype=np.float32)
    target = np.clip(target, ACTION_Q01, ACTION_Q99)
    return np.clip(target, current - max_delta, current + max_delta)


def action_dict(target: np.ndarray) -> dict[str, float]:
    """Convert an ordered target vector to the LeRobot hardware action dict."""
    return {
        name: float(value) for name, value in zip(ACTION_NAMES, target, strict=True)
    }


def disable_robot_torque(robot) -> None:
    """Release every SO101 joint between episodes."""
    robot.bus.disable_torque()
    logging.info("SO101 torque disabled; arm is released for scene reset")


def enable_robot_torque_at_current_pose(robot) -> None:
    """Enable torque without snapping back to the previous episode's goal."""
    present = robot.bus.sync_read("Present_Position")
    robot.bus.sync_write("Goal_Position", present)
    robot.bus.enable_torque()
    logging.info("SO101 goal re-anchored to current pose; torque enabled")


class OpenPIWebsocketBackend:
    """JAX OpenPI served through the lightweight OpenPI websocket client."""

    def __init__(self, host: str, port: int):
        try:
            from openpi_client import image_tools, websocket_client_policy
        except ImportError as exc:
            raise RuntimeError(
                "openpi-websocket needs openpi-client in this environment. "
                "Install packages/openpi-client from openpi, or add it to PYTHONPATH."
            ) from exc
        self._image_tools = image_tools
        self.host = host
        self.port = port
        self._client = websocket_client_policy.WebsocketClientPolicy(
            host=host, port=port
        )

    def build_request(self, observation: dict[str, Any], prompt: str) -> dict[str, Any]:
        """Build the OpenPI SO101 websocket request."""
        resize = self._image_tools.resize_with_pad
        as_uint8 = self._image_tools.convert_to_uint8
        return {
            "observation.images.front": as_uint8(
                resize(np.asarray(observation["front"]), 224, 224)
            ),
            "observation.images.wrist": as_uint8(
                resize(np.asarray(observation["wrist"]), 224, 224)
            ),
            "observation.state": current_state(observation),
            "prompt": prompt,
        }

    def infer_request(self, request: dict[str, Any]) -> np.ndarray:
        """Send a prepared normal or RTC request and validate its action chunk."""
        return validate_action_chunk(self._client.infer(request)["actions"])

    def infer(self, observation: dict[str, Any], prompt: str) -> np.ndarray:
        return self.infer_request(self.build_request(observation, prompt))

    def build_rtc_request(
        self,
        observation: dict[str, Any],
        prompt: str,
        prev_actions: np.ndarray,
        *,
        inference_delay: int,
        execution_horizon: int,
        max_guidance_weight: float,
    ) -> dict[str, Any]:
        """Build an OpenPI JAX denoising-level RTC guidance request."""
        request = self.build_request(observation, prompt)
        request["rtc"] = {
            "prev_actions": np.asarray(prev_actions, dtype=np.float32),
            "inference_delay": inference_delay,
            "execution_horizon": execution_horizon,
            "max_guidance_weight": max_guidance_weight,
        }
        return request


@dataclass(frozen=True)
class AsyncInferenceResult:
    """One background OpenPI inference result."""

    actions: np.ndarray | None
    latency_s: float
    error: BaseException | None = None


class OpenPIAsyncWorker:
    """Own a separate websocket client for one-at-a-time RTC requests."""

    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port
        # Unbounded single-producer queues avoid a shutdown deadlock when an
        # episode ends while a request/result is in flight. ``_busy`` still
        # enforces at most one inference request at a time.
        self._requests: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._results: queue.Queue[AsyncInferenceResult] = queue.Queue()
        self._busy = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="openpi-rollout-rtc", daemon=True
        )

    @property
    def busy(self) -> bool:
        return self._busy.is_set()

    def start(self) -> None:
        self._thread.start()

    def submit(self, request: dict[str, Any]) -> bool:
        if self.busy:
            return False
        self._busy.set()
        try:
            self._requests.put_nowait(request)
        except queue.Full:
            self._busy.clear()
            return False
        return True

    def poll(self) -> AsyncInferenceResult | None:
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None

    def close(self, timeout_s: float = 10.0) -> None:
        self._requests.put_nowait(None)
        self._thread.join(timeout=timeout_s)
        if self._thread.is_alive():
            raise RuntimeError(
                "RTC worker did not stop; refusing to start another episode"
            )

    def _run(self) -> None:
        backend = OpenPIWebsocketBackend(self._host, self._port)
        while True:
            request = self._requests.get()
            if request is None:
                return
            started = time.perf_counter()
            try:
                actions = backend.infer_request(request)
                result = AsyncInferenceResult(
                    actions=actions,
                    latency_s=time.perf_counter() - started,
                )
            except BaseException as error:
                result = AsyncInferenceResult(
                    actions=None,
                    latency_s=time.perf_counter() - started,
                    error=error,
                )
            self._results.put(result)
            self._busy.clear()


class RLInfPyTorchBackend:
    """Converted RLInf PyTorch Pi0.5 eval policy."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        precision: str,
        num_steps: int,
        seed: int,
    ):
        import torch
        from omegaconf import OmegaConf

        from rlinf.models.embodiment.openpi_pytorch import get_model

        if not torch.cuda.is_available():
            raise RuntimeError("rlinf-pytorch rollout requires a CUDA GPU")
        cfg = OmegaConf.create(
            {
                "model_type": "openpi_pytorch",
                "model_path": str(checkpoint),
                "precision": precision,
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
        self._torch = torch
        self._generator = torch.Generator(device="cuda").manual_seed(seed)
        self._policy = get_model(cfg).cuda().eval()

    def infer(self, observation: dict[str, Any], prompt: str) -> np.ndarray:
        env_obs = {
            "states": current_state(observation)[None],
            "main_images": np.asarray(observation["front"])[None],
            "wrist_images": np.asarray(observation["wrist"])[None],
            "task_descriptions": [prompt],
        }
        with self._torch.no_grad():
            actions, _ = self._policy.predict_action_batch(
                env_obs, mode="eval", rng=self._generator
            )
        return validate_action_chunk(actions[0].float().cpu().numpy())


def make_policy(args: argparse.Namespace) -> PolicyBackend:
    """Construct the selected policy backend."""
    if args.backend == "openpi-websocket":
        return OpenPIWebsocketBackend(args.host, args.port)
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for --backend rlinf-pytorch")
    return RLInfPyTorchBackend(
        args.checkpoint.expanduser().resolve(),
        precision=args.precision,
        num_steps=args.num_steps,
        seed=args.seed,
    )


def make_robot(args: argparse.Namespace):
    """Create the SO-101 and two OpenCV cameras using LeRobot 0.4 or 0.5."""
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    try:
        # LeRobot 0.5+ unified SO-100/SO-101 under ``so_follower``.
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    except ImportError:
        # LeRobot 0.4 and the Python 3.11-compatible v3 dataset checkout.
        from lerobot.robots.so101_follower.config_so101_follower import (
            SO101FollowerConfig,
        )
        from lerobot.robots.so101_follower.so101_follower import SO101Follower

    cameras = {
        "front": OpenCVCameraConfig(
            index_or_path=camera_source(args.front_camera),
            fps=args.front_camera_fps or args.camera_fps,
            width=args.camera_width,
            height=args.camera_height,
            fourcc=args.camera_fourcc,
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path=camera_source(args.wrist_camera),
            fps=args.wrist_camera_fps or args.camera_fps,
            width=args.camera_width,
            height=args.camera_height,
            fourcc=args.camera_fourcc,
        ),
    }
    config = SO101FollowerConfig(
        port=args.robot_port,
        id=args.robot_id,
        calibration_dir=args.calibration_dir.expanduser(),
        cameras=cameras,
        use_degrees=True,
        max_relative_target=(
            None if args.disable_action_limits else args.max_relative_target
        ),
        disable_torque_on_disconnect=True,
    )
    return SO101Follower(config)


def make_dataset(args: argparse.Namespace, robot):
    """Create or resume a LeRobot v3 writer compatible with the robot."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    try:
        from lerobot.utils.feature_utils import hw_to_dataset_features
    except ImportError:
        from lerobot.datasets.utils import hw_to_dataset_features

    root = args.dataset_root.expanduser().resolve()
    if args.resume:
        if not root.exists():
            raise FileNotFoundError(f"Cannot resume missing dataset: {root}")
        if hasattr(LeRobotDataset, "resume"):
            return LeRobotDataset.resume(
                args.repo_id,
                root=root,
                image_writer_threads=args.image_writer_threads,
            )
        dataset = LeRobotDataset(args.repo_id, root=root)
        dataset.start_image_writer(num_threads=args.image_writer_threads)
        return dataset
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"Dataset root is not empty: {root}. Pass --resume to append."
        )
    features = {
        **hw_to_dataset_features(robot.action_features, "action", args.video),
        **hw_to_dataset_features(robot.observation_features, "observation", args.video),
    }
    return LeRobotDataset.create(
        args.repo_id,
        args.fps,
        root=root,
        robot_type=robot.name,
        features=features,
        use_videos=args.video,
        image_writer_threads=args.image_writer_threads,
    )


def add_dataset_frame(
    dataset, observation: dict[str, Any], sent_action: dict[str, Any], task: str
) -> None:
    """Add one observation and the action actually sent to the LeRobot buffer."""
    try:
        from lerobot.utils.feature_utils import build_dataset_frame
    except ImportError:
        from lerobot.datasets.utils import build_dataset_frame

    observation_frame = build_dataset_frame(
        dataset.features, observation, prefix="observation"
    )
    action_frame = build_dataset_frame(dataset.features, sent_action, prefix="action")
    dataset.add_frame({**observation_frame, **action_frame, "task": task})


def write_episode_outcome(
    dataset_root: Path,
    episode_index: int,
    is_success: bool,
    *,
    notes: str = "",
    outcome: str | None = None,
) -> Path:
    """Atomically insert or replace one outcome label in the sidecar parquet."""
    import pandas as pd

    output = dataset_root / "meta" / "episode_outcomes.parquet"
    if output.exists():
        frame = pd.read_parquet(output)
        frame = frame[frame["episode_index"] != episode_index]
    else:
        frame = pd.DataFrame(
            columns=["episode_index", "is_success", "outcome", "notes"]
        )
    if "outcome" not in frame.columns:
        frame["outcome"] = np.where(frame["is_success"], "success", "failure")
    frame = pd.concat(
        [
            frame,
            pd.DataFrame(
                [
                    {
                        "episode_index": int(episode_index),
                        "is_success": bool(is_success),
                        "outcome": outcome or ("success" if is_success else "failure"),
                        "notes": notes,
                    }
                ]
            ),
        ],
        ignore_index=True,
    ).sort_values("episode_index")
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
    return output


def review_episode() -> str:
    """Prompt until a supported episode disposition is entered."""
    while True:
        answer = (
            input("结果 [s]成功 [t]超时 [f]失败 [r]丢弃重录 [q]丢弃并结束: ")
            .strip()
            .lower()
        )
        aliases = {
            "s": "success",
            "success": "success",
            "t": "timeout",
            "timeout": "timeout",
            "f": "failure",
            "fail": "failure",
            "failure": "failure",
            "r": "redo",
            "redo": "redo",
            "q": "quit",
            "quit": "quit",
        }
        if answer in aliases:
            return aliases[answer]


@contextlib.contextmanager
def live_key_mode():
    """Enable immediate single-key reads for an interactive POSIX terminal."""
    if not sys.stdin.isatty():
        yield
        return
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)


def live_disposition_from_key(key: str) -> str | None:
    """Map one terminal key to an immediate episode disposition."""
    return {
        "s": "success",
        "t": "timeout",
        "f": "failure",
        "r": "redo",
        "q": "quit",
        "\r": "review",
        "\n": "review",
    }.get(key.lower())


def poll_live_disposition() -> str | None:
    """Return an immediate episode label, or ``review`` for the Enter key."""
    if not sys.stdin.isatty() or not select.select([sys.stdin], [], [], 0)[0]:
        return None
    return live_disposition_from_key(sys.stdin.read(1))


def collect_episode(
    args: argparse.Namespace, robot, policy: PolicyBackend, dataset
) -> tuple[int, str | None]:
    """Run one episode and return the saved-frame count plus any live label."""
    start = time.perf_counter()
    chunk = None
    chunk_index = 0
    executed_since_inference = args.open_loop_horizon
    frame_count = 0
    disposition = None
    while time.perf_counter() - start < args.episode_seconds:
        disposition = poll_live_disposition()
        if disposition is not None:
            break
        loop_start = time.perf_counter()
        observation = robot.get_observation()
        if chunk is None or executed_since_inference >= args.open_loop_horizon:
            infer_start = time.perf_counter()
            chunk = policy.infer(observation, args.prompt)
            logging.info("inference %.2fs", time.perf_counter() - infer_start)
            chunk_index = 0
            executed_since_inference = 0

        state = current_state(observation)
        target = safe_target(
            chunk[chunk_index],
            state,
            args.max_step_delta,
            disable_limits=args.disable_action_limits,
        )
        sent = robot.send_action(action_dict(target))
        add_dataset_frame(dataset, observation, sent, args.prompt)
        frame_count += 1
        chunk_index += 1
        executed_since_inference += 1

        elapsed = time.perf_counter() - loop_start
        remaining = 1.0 / args.fps - elapsed
        if remaining > 0:
            time.sleep(remaining)
    return frame_count, disposition


def warmup_openpi_rtc(
    policy: OpenPIWebsocketBackend,
    observation: dict[str, Any],
    args: argparse.Namespace,
) -> float:
    """Compile the JAX RTC path and measure one hot-request latency."""
    actions = policy.infer(observation, args.prompt)
    request = policy.build_rtc_request(
        observation,
        args.prompt,
        actions[: args.rtc_execution_horizon],
        inference_delay=0,
        execution_horizon=args.rtc_execution_horizon,
        max_guidance_weight=args.rtc_max_guidance_weight,
    )
    logging.info("Compiling/warming the JAX RTC request before motor execution")
    policy.infer_request(request)
    started = time.perf_counter()
    policy.infer_request(request)
    latency_s = time.perf_counter() - started
    logging.info(
        "JAX RTC hot latency=%.3fs estimated_delay=%d control periods",
        latency_s,
        int(np.ceil(latency_s * args.fps)),
    )
    return latency_s


def collect_episode_rtc(
    args: argparse.Namespace,
    robot,
    policy: OpenPIWebsocketBackend,
    dataset,
    *,
    initial_latency_s: float,
) -> tuple[int, str | None]:
    """Record one episode using OpenPI's asynchronous JAX RTC queue replacement."""
    observation = robot.get_observation()
    actions = policy.infer(observation, args.prompt)[: args.open_loop_horizon]
    worker = OpenPIAsyncWorker(policy.host, policy.port)
    worker.start()

    action_index = 0
    sent = 0
    frame_count = 0
    request_sent_at = None
    estimated_delay = int(np.ceil(initial_latency_s * args.fps))
    queue_wait_started = None
    start = time.perf_counter()
    disposition = None

    try:
        while time.perf_counter() - start < args.episode_seconds:
            disposition = poll_live_disposition()
            if disposition is not None:
                break

            loop_start = time.perf_counter()
            observation = robot.get_observation()
            completed = worker.poll()
            if completed is not None:
                if completed.error is not None:
                    raise RuntimeError(
                        "Background RTC inference failed"
                    ) from completed.error
                if request_sent_at is None or completed.actions is None:
                    raise RuntimeError("RTC result has no matching request or actions")
                actual_delay = sent - request_sent_at
                new_actions = completed.actions[: args.open_loop_horizon]
                if actual_delay >= len(new_actions):
                    raise RuntimeError(
                        f"RTC consumed {actual_delay} control periods for a "
                        f"{len(new_actions)}-step chunk"
                    )
                actions = new_actions[actual_delay:]
                action_index = 0
                request_sent_at = None
                queue_wait_started = None
                estimated_delay = int(np.ceil(completed.latency_s * args.fps))
                logging.info(
                    "RTC merged latency=%.3fs delay=%d retained=%d",
                    completed.latency_s,
                    actual_delay,
                    len(actions),
                )

            remaining = len(actions) - action_index
            if (
                request_sent_at is None
                and not worker.busy
                and 0 < remaining <= args.rtc_queue_threshold
            ):
                prefix = actions[
                    action_index : action_index + args.rtc_execution_horizon
                ]
                request = policy.build_rtc_request(
                    observation,
                    args.prompt,
                    prefix,
                    inference_delay=min(estimated_delay, args.open_loop_horizon),
                    execution_horizon=min(args.rtc_execution_horizon, len(prefix)),
                    max_guidance_weight=args.rtc_max_guidance_weight,
                )
                if worker.submit(request):
                    request_sent_at = sent
                    logging.info(
                        "RTC request remaining=%d estimated_delay=%d prefix=%d",
                        remaining,
                        estimated_delay,
                        len(prefix),
                    )

            if action_index >= len(actions):
                if queue_wait_started is None:
                    queue_wait_started = time.perf_counter()
                    logging.warning("RTC queue underrun; holding the last target")
                if time.perf_counter() - queue_wait_started > args.rtc_result_timeout:
                    raise TimeoutError(
                        "Timed out waiting for the next RTC action chunk"
                    )
            else:
                state = current_state(observation)
                target = safe_target(
                    actions[action_index],
                    state,
                    args.max_step_delta,
                    disable_limits=args.disable_action_limits,
                )
                sent_action = robot.send_action(action_dict(target))
                add_dataset_frame(dataset, observation, sent_action, args.prompt)
                action_index += 1
                sent += 1
                frame_count += 1

            remaining_period = 1.0 / args.fps - (time.perf_counter() - loop_start)
            if remaining_period > 0:
                time.sleep(remaining_period)
    finally:
        worker.close(timeout_s=args.rtc_result_timeout)
        logging.info("RTC worker stopped; episode action queue discarded")
    return frame_count, disposition


def reset_countdown(seconds: float) -> None:
    """Show a compact terminal countdown while the operator resets the scene."""
    if seconds <= 0:
        return
    print(f"请恢复场景；{seconds:g} 秒后可开始下一条。")
    deadline = time.perf_counter() + seconds
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        print(f"\r恢复环境剩余 {remaining:4.1f}s", end="", flush=True)
        time.sleep(min(0.2, remaining))
    print("\r恢复环境完成，请确认场景后按 Enter。")


def run_smoke(robot, policy: PolicyBackend, prompt: str) -> None:
    """Read both cameras and run one inference without sending an action."""
    observation = robot.get_observation()
    state = current_state(observation)
    actions = policy.infer(observation, prompt)
    first = safe_target(actions[0], state, max_delta=5.0)
    print(
        f"smoke_ok state_shape={state.shape} front={np.asarray(observation['front']).shape} "
        f"wrist={np.asarray(observation['wrist']).shape} actions={actions.shape}"
    )
    print(f"first_safe_target={np.array2string(first, precision=2)} motion_sent=false")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("openpi-websocket", "rlinf-pytorch"),
        default="openpi-websocket",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--robot-port", default="/dev/ttyACM0")
    parser.add_argument("--robot-id", default="so101")
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path("~/.cache/huggingface/lerobot/calibration/robots/so_follower"),
    )
    parser.add_argument("--front-camera", default="/dev/video0")
    parser.add_argument("--wrist-camera", default="/dev/video2")
    parser.add_argument(
        "--camera-fps",
        type=int,
        default=30,
        help="Fallback FPS used by both cameras unless overridden below.",
    )
    parser.add_argument(
        "--front-camera-fps",
        type=int,
        help="Head/front camera capture FPS (defaults to --camera-fps).",
    )
    parser.add_argument(
        "--wrist-camera-fps",
        type=int,
        help="Wrist camera capture FPS (defaults to --camera-fps).",
    )
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fourcc", default="MJPG")

    parser.add_argument("--repo-id", default="local/so101_pi05_rollouts")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-writer-threads", type=int, default=8)
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=20,
        help="Target total number of saved episodes, including resumed episodes.",
    )
    parser.add_argument("--episode-seconds", type=float, default=20.0)
    parser.add_argument(
        "--reset-seconds",
        type=float,
        default=5.0,
        help="Scene-reset countdown between episodes.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--open-loop-horizon", type=int, default=5)
    parser.add_argument(
        "--rtc",
        action="store_true",
        help="Use OpenPI JAX denoising-level Real-Time Chunking.",
    )
    parser.add_argument("--rtc-queue-threshold", type=int, default=30)
    parser.add_argument("--rtc-execution-horizon", type=int, default=20)
    parser.add_argument("--rtc-max-guidance-weight", type=float, default=10.0)
    parser.add_argument("--rtc-result-timeout", type=float, default=10.0)
    parser.add_argument("--max-step-delta", type=float, default=5.0)
    parser.add_argument("--max-relative-target", type=float, default=5.0)
    parser.add_argument(
        "--disable-action-limits",
        action="store_true",
        help=(
            "Disable q01/q99 clipping, per-step delta clipping, and LeRobot "
            "max_relative_target. Hardware-risky."
        ),
    )
    parser.add_argument(
        "--prompt", default="Grab the blue pen and place it into the black box"
    )

    parser.add_argument(
        "--execute", action="store_true", help="Allow motor commands and recording."
    )
    parser.add_argument(
        "--confirm-motion",
        default="",
        help="Must be exactly SO101 together with --execute.",
    )
    parser.add_argument(
        "--confirm-unlimited-motion",
        default="",
        help="Must be exactly UNLIMITED when --disable-action-limits is used.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Reject unsafe or nonsensical runtime settings before hardware access."""
    if args.execute and args.confirm_motion != "SO101":
        raise ValueError("--execute requires --confirm-motion SO101")
    if args.disable_action_limits and args.confirm_unlimited_motion != "UNLIMITED":
        raise ValueError(
            "--disable-action-limits requires --confirm-unlimited-motion UNLIMITED"
        )
    if args.fps <= 0 or args.episode_seconds <= 0 or args.reset_seconds < 0:
        raise ValueError(
            "--fps/--episode-seconds must be positive and --reset-seconds non-negative"
        )
    camera_fps_values = (
        args.camera_fps,
        args.front_camera_fps or args.camera_fps,
        args.wrist_camera_fps or args.camera_fps,
    )
    if any(value <= 0 for value in camera_fps_values):
        raise ValueError("camera FPS values must be positive")
    if not 1 <= args.open_loop_horizon <= 50:
        raise ValueError("--open-loop-horizon must be in [1, 50]")
    if args.rtc:
        if args.backend != "openpi-websocket":
            raise ValueError("--rtc currently requires --backend openpi-websocket")
        if not 1 <= args.rtc_execution_horizon <= args.open_loop_horizon:
            raise ValueError(
                "--rtc-execution-horizon must be in [1, open-loop-horizon]"
            )
        if not (
            args.rtc_execution_horizon
            <= args.rtc_queue_threshold
            <= args.open_loop_horizon
        ):
            raise ValueError(
                "--rtc-queue-threshold must be in "
                "[rtc-execution-horizon, open-loop-horizon]"
            )
        if args.rtc_max_guidance_weight <= 0 or args.rtc_result_timeout <= 0:
            raise ValueError("RTC guidance weight and result timeout must be positive")
    if not args.disable_action_limits and (
        args.max_step_delta <= 0 or args.max_relative_target <= 0
    ):
        raise ValueError("action delta limits must be positive")


def main() -> None:
    """Connect hardware, collect reviewed episodes, and always disconnect safely."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    validate_args(args)
    validate_device_paths(args)
    policy = make_policy(args)
    robot = make_robot(args)
    dataset = None
    try:
        robot.connect()
        if not args.execute:
            run_smoke(robot, policy, args.prompt)
            return
        rtc_hot_latency_s = 0.0
        if args.rtc:
            if not isinstance(policy, OpenPIWebsocketBackend):
                raise TypeError("RTC requires the OpenPI websocket backend")
            rtc_hot_latency_s = warmup_openpi_rtc(policy, robot.get_observation(), args)
        dataset = make_dataset(args, robot)
        disable_robot_torque(robot)
        while dataset.num_episodes < args.num_episodes:
            input(
                f"\n布置场景后按 Enter 开始 episode {dataset.num_episodes} "
                f"(最长 {args.episode_seconds:g}s): "
            )
            print(
                "运行中单键即时打标并停止: "
                "[s]成功 [t]超时 [f]失败 [r]重录 [q]结束；Enter=停止后再选择"
            )
            try:
                enable_robot_torque_at_current_pose(robot)
                with live_key_mode():
                    if args.rtc:
                        frames, disposition = collect_episode_rtc(
                            args,
                            robot,
                            policy,
                            dataset,
                            initial_latency_s=rtc_hot_latency_s,
                        )
                    else:
                        frames, disposition = collect_episode(
                            args, robot, policy, dataset
                        )
            finally:
                disable_robot_torque(robot)
                if not args.rtc:
                    logging.info("Episode action chunk discarded")
            print(f"episode finished: frames={frames}")
            if disposition in {None, "review"}:
                disposition = review_episode()
            else:
                print(f"即时标签: {disposition}")
            if disposition in {"redo", "quit"}:
                dataset.clear_episode_buffer()
                if disposition == "quit":
                    break
                reset_countdown(args.reset_seconds)
                continue

            notes = ""
            if disposition in {"failure", "timeout"}:
                notes = input(
                    "备注(可空，例如 missed_grasp/collision/late_success): "
                ).strip()
            episode_index = dataset.num_episodes
            dataset.save_episode()
            output = write_episode_outcome(
                args.dataset_root.expanduser().resolve(),
                episode_index,
                disposition == "success",
                notes=notes,
                outcome=disposition,
            )
            print(
                f"saved episode={episode_index} outcome={disposition} labels={output}"
            )
            if dataset.num_episodes < args.num_episodes:
                reset_countdown(args.reset_seconds)
    except KeyboardInterrupt:
        print("\nInterrupted: the current unsaved episode is discarded.")
        has_pending = False
        if dataset is not None and hasattr(dataset, "has_pending_frames"):
            has_pending = dataset.has_pending_frames()
        elif dataset is not None:
            has_pending = getattr(dataset, "episode_buffer", None) is not None
        if has_pending:
            dataset.clear_episode_buffer()
    finally:
        if dataset is not None:
            dataset.finalize()
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
