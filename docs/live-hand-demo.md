# Live right-hand demo

This demo uses the MacBook camera to drive the robot's **right Revo2 hand**. Camera images
stay in the browser; only 21 3D landmarks per detected hand leave it for the local Python panel.
Tracking uses a pinned MediaPipe Tasks Vision 0.10.14 model in a Web Worker. The initial
model/WASM download requires internet access to jsDelivr and Google Storage. This release
has automated mock/simulation and WebSocket checks; the webcam, real-hand mapping and
physical latency still require the bench evaluation below.

## Run

From a source checkout with the project extras installed:

```sh
.venv/bin/r1h-panel --backend mock --sides right
# Or view the hand in the existing simulation:
.venv/bin/r1h-panel --backend sim --sides right
```

Open `http://localhost:8080` in a current Chrome browser and allow camera access. The mock
backend includes a virtual block under the right index finger; its contact hold is deliberate.

For real hands, deploy the updated client **and server together**: motion packets now require
server-derived deadlines. On the Jetson, with the existing hand configuration verified:

```sh
.venv/bin/r1h-hand-server --backend real --sides right
```

On the Mac:

```sh
.venv/bin/r1h-panel --remote ws://192.168.123.164:8765/ws --sides right
```

1. Start camera. Keep only your right hand in view, with even lighting and a plain background.
2. Confirm the overlay labels your anatomical right hand **right**. The preview is mirrored.
   Use Swap if needed; swapping clears calibration and disarms motion.
3. Hold an open palm, click **Capture open**, and hold for one second. Repeat with a fist.
   Keep the whole hand inside the image, including fingertips. Failed or collapsed calibration
   is rejected. Calibrate again after changing operator or viewpoint substantially.
4. For the thumb, capture a thumb–index pinch with visible thumb bend and opposition.
   If those two features did not change enough, the capture is rejected. The UI identifies when
   the thumb still uses its default range. Calibration is session-local.
5. Click **Arm right-hand demo**. Three consecutive valid observations acquire tracking.
   Slowly open and close the hand, then move each finger separately. The left robot hand
   receives no webcam targets, even if manual Mirror is enabled.
6. **Disarm / hold** holds the measured right-hand pose. **Space / STOP** opens both served
   hands at the existing stop speed and locks commands; resuming does not re-arm teleop.
   Starting manual movement, calibration, stopping the camera, hiding the page or disconnecting
   the driving browser disarms teleop. Re-arm explicitly to resume.

## Mapping across embodiments

The Revo2 has one coupled actuator per finger and two thumb actuators. It cannot reproduce
independent MCP/PIP/DIP angles, finger spread, or wrist rotation. The mapper sums finger
flexion angles in a local palm frame, then maps operator open/closed endpoints to normalized
motor targets. This makes it invariant to rigid hand rotation, translation and scale; it is not
a task-space fingertip optimizer. Thumb flexion/opposition can be calibrated separately with
a pinch rather than inferred entirely from a fist.

The result passes through a time-aware One Euro filter (default minimum cutoff 1.5 Hz,
beta 0.7), then the existing rate/stall guard. This trades steady-pose jitter against response
speed. The motor command range remains the driver's normalized range: the real hand's
comfortable travel, thumb-axis direction and current threshold must be checked on hardware.
The configured 600 mA stall threshold is still a placeholder, not a validated safe limit.

## Tracking loss and latency

- The browser processes one camera frame at a time, skips duplicate frames and never queues
  camera frames behind inference. Camera request: 640×480, up to 30 fps. Inference uses CPU
  in a classic worker so the stop controls remain responsive while detection runs.
- Detection/presence/tracking thresholds are 0.7. Handedness classification must score at
  least 0.8; **this is an identity score, not a per-landmark accuracy estimate**. Clipped images,
  nonfinite/collapsed geometry and ambiguous duplicate labels are rejected. These checks
  cannot detect every plausible but incorrect landmark estimate, especially during occlusion.
- Low-quality or missing observations hold the measured pose. A stopped frame stream expires
  the Mac target after 250 ms plus one runner tick (normally 10 ms). Existing stall/thermal
  monitoring continues while holding. After loss, three valid frames reacquire tracking.
- The panel rejects duplicate/reordered frames, frames older than 200 ms since browser sampling,
  and echoed panel state older than 300 ms. No cross-machine clock synchronization is needed.
- Mac-to-Jetson motion delivery coalesces pending targets per hand. Packets expire against a
  deadline derived from the last **Jetson** state timestamp. The robot target expires within
  300 ms plus one runner tick after updates cease. Recovery does not replay a motion backlog.
- These are software target expiry bounds, not guarantees of physical stopping time. The Mac
  and robot timeouts are separate; camera failure still needs the hold target to cross the LAN.
  Mechanical inertia, bus timing and firmware behavior need bench measurement. Stale hand
  telemetry older than 500 ms is refused by the guard and reported as a control error.

The panel shows inference time, sampled-frame age, browser↔Mac RTT and Mac↔Jetson RTT.
Sampled-frame age starts when JavaScript samples the decoded video, not at sensor exposure.
RTT is round-trip network time, not camera-to-motor delay. Neither number alone measures
end-to-end response.

Recording samples at approximately 30 Hz independently of the number of connected browsers.
Saved NPZ files include targets, measured positions, currents, mapping endpoints, armed state,
inference time, sampled-frame age and robot RTT. Calibration changes during a recording mean
its saved endpoint arrays describe the final calibration; start a fresh recording after calibration.

## Bench acceptance before a public demo

Record the software metrics and use an external high-frame-rate camera to film your hand and
the robot together. Do not claim an end-to-end latency number from inference timing alone.
These are initial targets to validate, not measured results:

| Check | Procedure | Initial acceptance target |
|---|---|---|
| Identity | Present left hand alone, then right hand | Only the intended right hand drives |
| Individual motion | Open/fist, each finger, pinch, point; 10 repetitions | No unintended finger closure; thumb direction correct |
| Steady pose | Hold open, half-closed and pinch for 10 s each | Normalized target standard deviation <0.03 per axis |
| Physical delay | Film repeated open/close transitions | p95 onset delay <200 ms; report motion duration separately |
| Loss | Remove/occlude hand; stop camera; hide page | Targets hold within documented software bounds |
| Network loss | Disconnect LAN during closing, then reconnect | Target expires; no queued closing sequence replays |
| Stop | STOP during tracking, loss and a routine | Opening/lockout works; resume leaves teleop disarmed |
| Contact | Use the existing current audit and a foam object | Measured stall protection works before longer demos |
| Endurance | 10 min demo with recordings | No target queue growth or unexplained identity switches |

A monocular camera cannot reliably reconstruct fingers hidden behind the palm. Keep the palm
or a slight three-quarter view visible for this first demo. If failures cluster around occlusion,
use the recordings and filmed trials to decide whether to constrain the demo pose range,
change the estimator, or add a second viewpoint.

## Automated checks

```sh
.venv/bin/python -m pytest -q
node tests/test_webcam_ui.cjs
node --check src/r1hands/panel/static/hand-worker.js
```

The Python suite includes real localhost WebSocket transport, packet deadlines, tracking
reacquisition, right-only mapping, ownership, calibration rejection, rigid-transform invariance,
filter response, stale telemetry and measured-pose hold behavior. The JavaScript contract test
uses mocked DOM/transport objects; it does not run MediaPipe or obtain camera permissions.
