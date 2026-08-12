# SO-101 RECAP → LeRobot episodic DAgger

The bridge intentionally uses two Python processes. The RLInf environment owns
the GPU checkpoint and serves raw 50×6 action chunks. The `lerobot_hil` Conda
environment owns the follower, leader, cameras, RTC action queue, rewind logic,
and LeRobot v3 dataset writer. There is only one RTC queue: the LeRobot client.

The server defaults to the exact positive prompt used during RECAP training:
`<task>\nAdvantage: positive`. This is positive-conditioned single-pass
inference. `--inference.rtc.max_guidance_weight` only controls RTC action-chunk
splicing; it is not a classifier-free-guidance scale. Use server option
`--advantage-condition none` only for an explicit ablation.

## Preflight

```bash
cd /home/larry/RLinf
bash toolkits/so101/run_recap_dagger_local.sh doctor
```

The defaults are follower `/dev/ttyACM0`, leader `/dev/ttyACM1`, front camera
`video2`, and wrist camera `video0`. Override any value without editing files,
for example `LEADER_PORT=/dev/ttyACM2 ... doctor`.

## Run

Terminal 1:

```bash
cd /home/larry/RLinf
bash toolkits/so101/run_recap_dagger_local.sh server
```

Wait for `Serving RECAP policy at ws://127.0.0.1:8001`, then use terminal 2:

```bash
cd /home/larry/RLinf
bash toolkits/so101/run_recap_dagger_local.sh collect
```

The collector asks for `YES` immediately before it can command the arm. No
`robot.max_relative_target` is configured. Each episode is at most 10 seconds;
reset time is 5 seconds; 50 successful clean episodes are written by default.
Timeout is treated exactly as failure and the episode is discarded.

Controls during an episode:

- `Space`: stop policy, clear RTC/action state, and align the leader.
- `Left` / `Backspace`: rewind one / 30 still-rewindable frames.
- `Enter`: start recording the corrected branch from the selected point.
- `Space`: finish correction and resume the reset policy.
- `s`: save immediately as success; no need to wait for 10 seconds.
- `q` or `r`: discard the episode and re-record it.
- `Esc`: stop the session. Shutdown disconnects with torque release and does
  not command an automatic return-to-initial movement.
- `Right` or `n`: end the five-second manual reset phase early.

The default output is
`/mnt/pqssd/so101/datasets/recap_v1_dagger_iter1`. Set a new `DATASET_ROOT`
for a new collection, or set `RESUME_DATASET=true` only to resume the same one.

This dataset is a clean SFT/DAgger set: bad frames removed by rewind are not
available to the value model. Collect a separate autonomous, outcome-labeled
rollout set from the current RECAP policy for the next return/value/advantage
iteration.
