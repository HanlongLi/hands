# Webcam mapping lab without robot hardware

The control panel now has a **3D Revo2 right-hand preview next to the webcam**, plus portable
landmark recording and replay. You do not need the Jetson, Revo2 SDK or MuJoCo for this lab.
Use a source checkout containing the robot assets and run:

```sh
.venv/bin/r1h-panel --backend mock --sides right
```

Open `http://localhost:8080`. Restart an older panel process and refresh the page after updating.
All model, mesh, rendering and tracking files are served locally; no runtime CDN is required.

## Try the example first

1. Click **Load example** in Webcam teleop.
2. Click **Replay in preview**. The example is a synthetic six-second open/close sequence,
   explicitly labeled; it is not evidence of webcam tracking accuracy.
3. Drag the 3D view to rotate it and scroll to zoom. **Reset view** restores the camera.
4. **Pause replay** holds the current frame. **Rewind** resets the timeline; press replay again.

The solid teal hand uses the Revo2 URDF's visual meshes, joint origins/axes, driving ranges and
coupled distal-joint relationships. It is a **kinematic mapping preview**, not contact or motor
physics. MuJoCo still runs when you choose `--backend sim`.

## Use your own hand

1. Start camera and verify the right-hand label. Swap left/right if needed.
2. The mapped preview follows without arming robot control. Calibrate open, fist and optionally
   thumb pinch to improve the mapping; default ranges work before calibration.
3. Click **Record landmarks**, perform gestures for up to two minutes, then click
   **Stop landmark recording**.
4. Click **Download clip** to save a JSON file. **Import clip** loads it in a later session.
5. Replay with **Use recorded calibration** checked to reproduce that clip's mapping. Uncheck
   it to compare the same observations with your current operator calibration. Changing this
   option pauses and rewinds replay.

Clips include raw world/image landmarks, detector handedness scores, detection gaps, sample
intervals, the left/right swap setting and calibration endpoints frozen at recording start.
They contain **no camera images or video**. Recalibrate before starting a new recording if you
want its saved endpoints to reflect your new calibration. Existing top-bar **Record** remains
separate: it saves motor-state/target telemetry as NPZ, not portable landmark clips.

Starting replay stops camera capture, ends landmark recording and disarms live control from
that browser. Replay messages go to a separate read-only mapper and never issue hand targets.
The server rejects replay while live control is armed. Starting the camera, hiding the page or
pressing STOP pauses replay. Actual robot motion still requires the existing explicit arm flow.

## What to compare

- Open, fist, point, pinch and individual fingers: look for unwanted coupling or thumb direction.
- Hold each pose still: **Target variation (2 s)** shows the largest per-axis standard deviation
  over the recent two seconds. This includes intentional motion, so it measures jitter only
  when the operator holds still; it does not measure tracking accuracy against ground truth.
- **Show → Panel target** displays the backend command instead of the independently mapped
  preview. Enable **Measured pose (amber)** to overlay reported backend positions. In mock mode,
  the virtual block under the index finger can explain a target/position difference.

Replay filtering advances on original sample timestamps, not rendering cadence. Every frame,
including tracking gaps, is remapped in order. Rewind resets the mapping/filter state. A large
browser timing stall or connection loss resets replay rather than silently skipping input.
Recordings are capped at 3600 frames/two minutes or approximately 18 MiB, whichever comes
first, so exported recordings fit the 20 MiB import limit. Imports are validated.
The saved format is `r1hands.landmarks.v1`.

Use one active preview/camera browser at a time: the panel shares preview state across viewers.
A replay should reproduce its mapping, but it cannot evaluate mechanical travel, stall-current
limits, physical contact or camera-to-motor latency. Those remain hardware bench checks.

## Regression checks

```sh
.venv/bin/python -m pytest -q
node tests/test_webcam_ui.cjs
node tests/test_landmark_session.cjs
```

For the browser checks, use an isolated mock panel:

```sh
# Terminal 1
.venv/bin/r1h-panel --backend mock --sides right --no-browser --port 8097
# Terminal 2: Chrome installed; temporary test dependency only
npm install --prefix /tmp/r1hands-browser-test playwright-core
NODE_PATH=/tmp/r1hands-browser-test/node_modules node tests/preview_browser_smoke.cjs
NODE_PATH=/tmp/r1hands-browser-test/node_modules node tests/webcam_browser_smoke.cjs
```

The first browser test checks real 3D rendering, preview-only replay, pause/rewind, JSON roundtrip,
invalid imports, landmark capture using a synthetic camera, responsive layout and local-only
asset loading. The second runs real MediaPipe initialization and inference on synthetic frames.
Neither substitutes for human-hand tracking accuracy measurements.
