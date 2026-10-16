# Lab deployment: iceberg

The server inventory identifies `iceberg` as `192.168.34.39`. Access requires the lab VPN.
The deployment uses the `lihanlong` account, repository `~/hands`, and an isolated `.venv`.
Credentials are not stored in this repository.

## Open the server panel from your MacBook

```bash
ssh -o PubkeyAuthentication=no -N -L 8099:127.0.0.1:8081 lihanlong@192.168.34.39
```

Keep that terminal open, then visit **http://localhost:8099**. Select **Task Learning**
and click **Start task simulation**. The camera tracker runs in your Mac browser;
physics, rendered task images and episode storage run on iceberg. Task recordings are
saved to `~/hands/logs/tasks/` on the server and downloadable in the browser.

Local simulation remains available independently:

```bash
scripts/run_task_panel.sh local
```

That opens the MacBook's own panel on port 8081. The different tunnel port (8099) lets
you keep local and server panels open together.

## Server process

The panel runs in tmux session `r1hands-panel`, with output at
`~/hands/logs/task-panel-server.log`. It listens on server loopback port 8081 and uses the
mock robot backend; Task Learning starts its separate MuJoCo process on demand.

Inspect it:

```bash
ssh lihanlong@192.168.34.39
cd ~/hands
tmux attach -t r1hands-panel
# Detach without stopping: Ctrl-B, then D
```

To start it again when that session is absent:

```bash
cd ~/hands
mkdir -p logs
tmux new-session -d -s r1hands-panel -c "$HOME/hands" \
  'env MUJOCO_EGL_DEVICE_ID=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 scripts/run_task_panel.sh server > logs/task-panel-server.log 2>&1'
```

To stop the panel cleanly (also saves an active partial episode):

```bash
tmux send-keys -t r1hands-panel C-c
```

GPU 1 was idle at deployment; GPU 0 was busy. `MUJOCO_EGL_DEVICE_ID=1` selects GPU 1
for offscreen EGL rendering. MuJoCo physics runs on the CPU. This does not reserve the
GPU for later training jobs. The machine has two NVIDIA A100 80 GB GPUs, 96 logical CPUs,
and about 251 GiB RAM. Recheck availability before scheduling larger workloads.

## Validation and environment

Verified on 2026-10-16: all 47 Python tests passed on iceberg. The Mac browser accessed
the server through the tunnel, initialized the camera tracker with a synthetic camera,
and completed a simulated grasp, episode download and mobile layout check. EGL produced
240 × 320 RGB frames using the NVIDIA A100 renderer (OpenGL 4.6, NVIDIA 580.178.04).

A state baseline trained on scripted fixtures succeeded on three unseen reset seeds
(101, 102, 103). This checks the simulation pipeline; it is not a result from human
demonstrations or hardware. The report is at `~/hands/logs/deployment-validation.json`.


Run the server-side tests with the headless renderer selected before Python imports MuJoCo:

```bash
cd ~/hands
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q
```

The project was copied as a Git bundle from the committed local repository, then cloned
with `main` checked out. Its origin points to `https://github.com/HanlongLi/hands.git`.
GitHub credentials were not copied to the server; future `git pull` needs that account's
GitHub access, or another bundle transfer. Python dependencies were installed inside
`.venv` with `.[panel,sim,dev]` and MuJoCo pinned to 3.15.0 for parity with the Mac tests.

See [task learning](task-learning.md) for task scope, recording schema, training commands
and the general local/server setup.
