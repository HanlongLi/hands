# r1hands: Unitree R1 EDU + BrainCo Revo2 Basic

Hardware layer, safety, a Mac control panel, and the start of the data + learning pipeline
(imitation learning, RL, code-as-policy) for loco-manipulation on an R1 with two Revo2 Basic hands.

## Levels

| Level | Scope | Status |
|---|---|---|
| **1** | Hands move: bring-up, telemetry, grasps, Mac control panel, webcam teleop | **ready for hardware** (verified on mock + sim, and the SDK on Linux) |
| 2 | Arms + hands: IK reach, arm_sdk blending | code written, sim-tested, **not run on the robot** |
| 3 | Whole body: LocoClient walking + upper body, RL locomotion | driver written, **untested** |

## Architecture

```
 Mac                                         R1 Jetson (192.168.123.164)             Revo2 hands
 ┌───────────────────────────┐   WebSocket   ┌─────────────────────────────┐  RS485   ┌────────┐
 │ r1h-panel  (browser UI)   │ ────────────▶ │ r1h-hand-server             │ ───────▶ │ left   │ 0x7E
 │  sliders, presets, E-stop │ ◀──────────── │  HandRunner 100 Hz          │ 460800   │ right  │ 0x7F
 │  current charts, record   │   state 50 Hz │  HandGuard (rate, stall)    │          └────────┘
 │  webcam teleop (MediaPipe)│               │  Revo2Hand (bc-stark-sdk)   │
 └───────────────────────────┘               └─────────────────────────────┘
```

All controllers go through one interface. Panel sliders, teleop, `Skills`, LLM code and policies talk to
`hal.base.Hand` / `Body`. Backends: `mock` (numpy), `sim` (MuJoCo R1+Revo2), `real` (SDK), and `remote`
(hands on another machine via the hand server). Nothing reaches a motor without passing `safety.guard`:
- Rate limit: 2.0/s normalized by default.
- Stall latch: a finger drawing high current for 1.5 s is backed off 0.05 from the contact point and held
  there, so it grips without squeezing until it is commanded open.
- E-stop: opens both hands slowly and locks out commands.

Conventions: hand joints are normalized **0 = open … 1 = closed** in the order
`thumb, thumb_aux, index, middle, ring, pinky`. This is the same order and scale as Unitree's G1_Brainco
datasets, so they map 1:1. Body joints are in radians, using R1.urdf names.

## Level 1 bring-up (today)

1. **Wire and power.** Use one USB-RS485 adapter per hand into the Jetson (this is Unitree's own layout for
   G1 + BrainCo). Hand supply: check BrainCo's spec for the Basic version; retailers list 12–28 V, which is
   unverified. The R1 upper-body panel has 24 V XT30 outputs.
2. **Deploy from the Mac.** Your Mac must be on the robot network (Ethernet to the R1, static `192.168.123.x`).
   ```bash
   scripts/deploy_jetson.sh unitree@192.168.123.164
   ```
3. **Identify the ports** on the Jetson, then set `hands.left.port` / `hands.right.port` in `configs/r1.yaml`
   (use `/dev/revo2_left` / `/dev/revo2_right`):
   ```bash
   cd ~/r1hands && .venv/bin/python scripts/revo2_scan.py            # which port has which hand
   sudo .venv/bin/python scripts/revo2_scan.py --write               # stable /dev/revo2_left|right
   ```
4. **Bench checks** on the Jetson, one hand at a time. Keep the hand clear of the body and of fingers.
   ```bash
   .venv/bin/python scripts/l1_hand_check.py probe  --side right
   .venv/bin/python scripts/l1_hand_check.py sweep  --side right     # each finger alone, slowly
   .venv/bin/python scripts/l1_hand_check.py rate   --side right     # achievable telemetry rate
   .venv/bin/python scripts/l1_hand_check.py audit  --side right     # foam block in hand: stall current
   ```
   `audit` deliberately bypasses the guard. It holds a power grasp on a foam block and logs current until the
   firmware flags a stall. Put the suggested value in `hand_limits.stall_current` (`configs/r1.yaml`). The
   current default of 600 mA is a placeholder.
5. **Hand server** on the Jetson (leave it running):
   ```bash
   .venv/bin/r1h-hand-server --backend real          # ws://0.0.0.0:8765/ws, no auth: robot LAN only
   ```
6. **Control panel** on the Mac:
   ```bash
   .venv/bin/r1h-panel                                   # uses hand_server from configs/r1.yaml
   .venv/bin/r1h-panel --remote ws://<jetson-ip>:8765/ws
   .venv/bin/r1h-panel --backend sim                     # no hardware: MuJoCo R1+Revo2
   .venv/bin/r1h-panel --backend mock                    # no hardware: right index finger blocked at 0.6 (demo of the stall latch)
   ```
   Panel features:
   - Sliders per motor, with the actual position as a coloured fill behind each slider.
   - Grasp presets 1–8, and "Save as preset" (written to `configs/grasps.yaml`).
   - **Space = E-stop / resume.** O opens the active hand.
   - Mirror L↔R.
   - Sweep and grasp-cycle routines.
   - Live motor-current charts with the stall threshold.
   - Recording to `logs/panel/*.npz`.
   - Webcam teleop. MediaPipe runs in the browser and the retargeter runs in Python. Calibrate open and fist
     once per operator.

   The grasp preset values in `control/skills.py` are starting values; tune them on the real hands and save
   them from the panel.

## Layout

```
configs/r1.yaml                 ports, ids, network, guard limits, gains (TODOs = measure on the robot)
src/r1hands/common/             joint layouts, message types, URDF limits
src/r1hands/hal/                base interfaces; mock, mujoco_sim, revo2 (SDK + DDS bridge), unitree_r1, remote
src/r1hands/safety/guard.py     HandGuard / BodyGuard
src/r1hands/panel/              hand_server.py, app.py (Mac panel), runner.py, static/index.html
src/r1hands/control/            skills.py (level-gated skill API), kinematics.py (numpy URDF FK/IK)
src/r1hands/retarget/hand.py    21 MANO/MediaPipe keypoints -> 6 Revo2 commands (closed form, per-user calibration)
src/r1hands/policies/cap/       code-as-policy with Claude (sandboxed, level-gated, operator confirm option)
src/r1hands/sim/build_model.py  R1 + 2x Revo2 URDF/MJCF (BrainCo's Apache-2.0 description, mimic joints)
src/r1hands/data/               L1/L2/L3 LeRobot v3 layouts; G1_Brainco -> R1 converter (L2, not yet run)
scripts/                        l1_hand_check.py, revo2_scan.py, deploy_jetson.sh
```

## Verified vs. not

- Verified:
  - 12 tests pass (mock + sim): guard rate limit and stall latch, IK, retargeting, code-as-policy loop
    (with a fake client), panel commands/teleop/recording, and the hand server ↔ remote panel path.
  - `bc-stark-sdk` 2.0.7 API names were checked against the Linux wheel. The driver's error path was run on
    Linux.
- **Not yet run on hardware:**
  - `Revo2Hand` against a real hand.
  - Achievable polling rate. Unitree's bridge polls at 100 Hz, but the real rate is unmeasured.
  - Stall current threshold.
  - The adapter clocking (`mount_roll_deg`) and offset in `sim/build_model.py`.
  - Everything in levels 2–3.
- The Revo2 Basic reports no temperature, so the thermal limit in `HandGuard` only applies to backends that
  report it.
- A live code-as-policy run needs `ANTHROPIC_API_KEY`: `r1h-cap "pinch with the right hand" --backend sim --level 1`.

## Live webcam hand demo

The right-hand demo now includes worker-based MediaPipe tracking, checked open/fist and optional
thumb-pinch calibration, adaptive smoothing, explicit arm/disarm, tracking-loss holds, and expiring
network commands. Start locally with `r1h-panel --backend mock --sides right`.
See [the demo guide](docs/live-hand-demo.md) for operation, timing semantics, hardware validation,
and acceptance targets. Camera accuracy and physical response latency still need bench measurement.

The panel also includes a [hardware-free mapping lab](docs/offline-hand-demo.md): a rotatable
3D Revo2 preview beside the webcam, raw landmark clip recording/import/download, and preview-only
replay with recorded or current calibration. Start with **Load example → Replay in preview**;
no camera or hardware is needed for the example.
