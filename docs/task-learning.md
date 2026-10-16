# Task learning without hardware

The **Task Learning** tab contains an independent, fixed-wrist MuJoCo right hand and a
40 mm, 40 g block. It runs in its own process, including rendering, and has no robot/HAL
connection. You can use it with the panel's mock backend. This adds object interaction
and task outcomes to the mapping workflow.

## Local and lab-server options

Both options run the same simulator and write the same episode format. The location of
the Python panel determines where physics, rendering, training and episode files live.
The camera tracker always runs in your browser.

### Local MacBook

```bash
scripts/run_task_panel.sh local
```

This uses local rendering and opens `http://localhost:8081`. It preserves the existing
`r1h-panel --backend mock --sides right` workflow as well.

### Headless Linux lab server

Clone this repository and install `.[panel,sim]` in a virtual environment on the server.
Then run:

```bash
# On the server: GPU rendering, no browser window or display required
scripts/run_task_panel.sh server

# Alternative on a server configured for CPU software rendering
R1H_TASK_RENDERER=osmesa scripts/run_task_panel.sh server
```

EGL requires a compatible GPU driver/EGL runtime. OSMesa requires an installed OSMesa
library. These are the [headless rendering backends supported by MuJoCo](https://mujoco.readthedocs.io/en/3.3.7/programming.html).
The launcher does not install system graphics drivers or change the machine configuration.

On your MacBook, establish a tunnel and open the localhost URL:

```bash
ssh -N -L 8081:127.0.0.1:8081 USER@LAB_SERVER
open http://localhost:8081
```

The panel remains bound to server loopback. This also gives your Mac browser a localhost
origin for webcam access. Camera images stay in the browser; tracked landmarks travel
through the tunnel. Network round-trip latency affects teleoperation, and the existing
250 ms input expiry still applies. Episodes are saved on the server and can be downloaded
through the panel. Keep the server process running in your usual terminal/session manager.

Use `R1H_PANEL_PORT=8090` to change the launcher port (and match the SSH tunnel), or pass
panel options after the mode. The underlying CLI option is
`--task-renderer auto|glfw|egl|osmesa`; the choice applies only to the isolated task process.
`auto` preserves any existing `MUJOCO_GL` environment setting. Restart the panel to change
rendering backend. Local rendering has been exercised on macOS. The lab deployment on
[iceberg](lab-server.md) has also passed headless EGL rendering, all 47 Python tests,
and the browser task recording/download checks.

## Start and collect

Install the panel and simulation dependencies if needed:

```bash
uv pip install --python .venv/bin/python -e '.[panel,sim]'
.venv/bin/r1h-panel --backend mock --sides right --port 8081
```

Open the printed localhost URL. The panel has four tabs:

- **Hand Control:** motor sliders, grasp presets, routines, speed, mirror and motor recording.
- **Mapping Lab:** camera, calibration, 3D mapping preview, landmark clips and reviewed mapping labels.
- **Task Learning:** physics scene, object contacts, task input and aligned episode recording.
- **Logs:** panel event messages.

Space remains the global stop/resume shortcut. Grasp number shortcuts and O apply only
in Hand Control. Changing tabs disarms robot teleoperation. Task input and robot
teleoperation cannot be armed together.

1. Select **Task Learning → Start task simulation**. This requires MuJoCo and a working
   local OpenGL context; the regular panel and mapping lab do not require task startup.
2. Set a reset seed and click **Reset scene**. A seed gives a reproducible initial block
   position, jittered by up to 4 mm in each horizontal direction.
3. Enter an operator ID and click **Start episode**, before moving the fingers.
4. Close the simulated hand with the six sliders. For a pipeline check, use **Example
   grasp target** and wait about 1.5 seconds. This preset is tagged `scripted`, not human data.
5. Click **Withdraw support**. The task succeeds after one continuous second of a stable
   grasp with opposing thumb/finger contacts. Success saves the recording automatically.
6. For unsuccessful attempts, click **Finish & save episode**, or wait for the 20 second
   recording limit. Download the NPZ using the link. The server also saves NPZ and JSON
   metadata under `logs/tasks/`.
7. Reset with a different seed before the next episode. Recording blocks scene resets and
   calibration changes so that each episode has consistent provenance.

For webcam demonstrations, start and calibrate the camera in **Mapping Lab**, leave robot
control disarmed, then select **Use live mapping input** in Task Learning. The camera keeps
running across tabs. Only fresh live preview samples drive this simulated hand; replay
clips do not. Tracking loss or input older than 250 ms holds measured joint positions.
Manual slider changes and the example preset turn mapping input off. **Hold hand** also
turns it off. Global stop locks task input until resumed.

The input is finger closure only: this scene has a fixed wrist and no arm, so a human
cannot demonstrate reaching or moving the object to another location yet.

## Task success and scope

The support must be removed. The block must have opposing thumb/other-finger contact
normals, no support/floor contact, a maximum contact normal force of at most 25 N, height
of at least 160 mm, translational speed below 40 mm/s, and position within 25 mm of the
start of the continuous stable hold. Those conditions must persist for one second.
The initial capture can move the block before this stable interval starts.

These are configurable simulation criteria, not measured hardware limits. Dynamics come
from the Revo2 URDF; object interaction uses primitive capsule/box contact shapes. Hand
self-contact is not enabled/validated in this task. A successful rollout demonstrates
this simplified grasp task, not real-world collision safety or sim-to-real transfer.

## Episode format

Schema: `r1hands.task-episode.v1`. NPZ files load with `allow_pickle=False`.

| Array | Shape | Meaning |
|---|---|---|
| `observation_state` | T+1 × 26 | Measured hand q/dq, object xyz/quaternion, object linear/angular velocity, support phase |
| `observation_rgb` | T+1 × 240 × 320 × 3 | uint8 RGB images from the simulated task camera |
| `time_s` | T+1 | Simulation seconds relative to recording start |
| `action` | T × 6 | Issued, rate-limited normalized position targets |
| `desired_action` | T × 6 | Requested targets before rate limiting or invalid-input hold |
| `action_source` | T | `manual`, `mapping`, `scripted` or `hold` |
| `input_valid`, `input_age_s` | T | Input quality/expiry information |
| `human_landmarks` | T × 21 × 3 | Mapping input landmarks; NaN where absent |
| `contact_fingers` | T+1 × 5 | Thumb/index/middle/ring/pinky contact flags |
| `max_contact_force_n` | T+1 | Maximum object-contact normal force |
| `metadata_json` | scalar string | Scene/config hash, seed, outcomes, mapper/profile/checkpoint provenance and events |

Each transition is `observation[i] → action[i] → observation[i+1]` at 20 Hz simulation
control frequency. Wall time can run slower on a busy machine. Images are simulator RGB;
raw human webcam images are not recorded. Raw human landmarks remain available for auditing.
Support withdrawal refreshes the current observation and is logged as an event at that
observation index, so the next action sees the correct support phase.

The measured success flag is separate from an optional operator outcome. Source counts
make scripted examples distinguishable from human demonstrations. Do not report preset
rollouts as human imitation data.

## Train the first task policy

Collect successful episodes across several reset seeds; at least two distinct seeds are
required. Prefer 10 or more seeds before interpreting generalization results.

```bash
.venv/bin/python -m r1hands.tasks.policy train logs/tasks/*.npz --output models/task.npz
.venv/bin/python -m r1hands.tasks.policy evaluate models/task.npz --seeds 101 102 103
```

Training excludes failed episodes and invalid input rows by default. `--include-failures`
is an explicit override, useful for inspection but not a substitute for an appropriate
learning objective. Episodes must have the same scene hash and task configuration.

The baseline is ridge regression from 13 values (six measured finger positions, object
position/linear velocity, support phase) to six targets. It uses no human landmarks at
inference. Normalization uses training rows only; entire reset seeds are held out, and
the report includes held-out action error and input source counts. Duplicate episode IDs
are rejected. Closed-loop evaluation rejects training seeds and uses the recorded task
configuration. It withdraws support at 1.5 seconds and measures success over five seconds.

A low action error does not establish task success; inspect the rollout success rate.
The CLI evaluation is simulation only. Policy execution is not connected to hardware or
the panel's hand-control interface.

A **more complex learning model** would consume camera images and a history of observations,
and possibly predict several future actions together. For example, a visual behavior
cloning policy with an image encoder and action-sequence head could handle ambiguity and
temporal coordination that a small state-to-action regression cannot. The RGB episodes
provide a starting dataset for that later stage. No visual policy is trained here yet.
