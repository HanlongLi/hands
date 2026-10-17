"""Revo2 hand control panel: a local web app you open in a browser on the Mac.

    r1h-panel                                   # real hands through the hand server (configs/r1.yaml: hand_server)
    r1h-panel --remote ws://<host>:8765/ws      # explicit hand server
    r1h-panel --backend sim                     # MuJoCo R1+Revo2, no hardware
    r1h-panel --backend mock                    # numpy mock (index finger blocked at 0.6 to show the stall latch)
then open http://localhost:8080

Features: per-finger sliders with live position readback, grasp presets (saved to configs/grasps.yaml),
E-stop (Space), mirror L<->R, finger sweep / grasp cycle routines, live motor-current charts, recording
to logs/panel/*.npz, and webcam teleop (MediaPipe in the browser -> HandRetargeter here).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import io
import hashlib
import time
import webbrowser
from pathlib import Path

import numpy as np
import yaml

from r1hands.common.robot_model import REPO
from r1hands.common.types import HAND_JOINTS, N_HAND
from r1hands.control.skills import GRASPS
from r1hands.hal.factory import load_config, make_robot
from r1hands.panel.runner import HandRunner
from r1hands.tasks.service import TaskService
from r1hands.panel.preview import HandPreview, MESH_DIR, example_clip, hand_model, profile
from r1hands.retarget.hand import HandRetargeter
from r1hands.retarget.live import LiveHand, validate_landmarks

STATIC = Path(__file__).parent / "static"
GRASP_FILE = REPO / "configs" / "grasps.yaml"
LOGDIR = REPO / "logs" / "panel"


def _server_sides(ws_url: str) -> list[str] | None:
    import urllib.request
    url = ws_url.replace("ws://", "http://").rsplit("/ws", 1)[0] + "/health"
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            return json.load(r)["sides"]
    except Exception as e:
        raise ConnectionError(f"hand server not reachable at {url}: {e}. Is r1h-hand-server running on the robot?")


class Panel:
    def __init__(self, backend: str, remote: str | None, sides=("left", "right"), mapping_model=None, task_renderer="auto"):
        from r1hands.data.mapping import load_model
        model = load_model(mapping_model) if mapping_model is not None else None
        self.mapping_model_sha256 = hashlib.sha256(Path(mapping_model).read_bytes()).hexdigest() if mapping_model else None
        cfg = load_config()
        self.backend = backend
        if backend == "remote":  # show only the hands the server actually serves
            sides = tuple(_server_sides(remote or cfg.get("hand_server")) or sides)
        self.robot = make_robot(backend, level=1, sides=sides, remote=remote).connect()
        if backend == "mock" and "right" in self.robot.hands:  # a virtual object under the right index finger, to demo contact + stall latch
            self.robot.hands["right"].inner.block_at = np.array([1, 1, 0.6, 1, 1, 1.0])
        self.link = getattr(next(iter(self.robot.hands.values())), "link", None)
        self.url = getattr(self.link, "url", None)
        self.runner = HandRunner(self.robot)
        self.stall_current = float(cfg.get("hand_limits", {}).get("stall_current", 600))
        self.grasps = dict(GRASPS)
        if GRASP_FILE.exists():
            self.grasps.update(yaml.safe_load(GRASP_FILE.read_text()) or {})
        self.mirror = False
        self.retarget = {s: HandRetargeter(smooth=0.6) for s in sides}
        self.calib = {s: {"open": [], "fist": [], "thumb": []} for s in sides}
        self.calibrated = {s: False for s in sides}
        self.live = {s: LiveHand(self.retarget[s]) for s in sides}
        self.teleop_armed = False
        self.teleop_owner = None
        self.teleop_seq = -1
        self.teleop_t = 0.0
        self.teleop_metrics = {}
        self.teleop_side = "right" if "right" in sides else sides[0]  # robot hand the webcam drives
        self.thumb_calibrated = {s: False for s in sides}
        self.preview = HandPreview(model)
        self.task_service = TaskService(REPO / "logs" / "tasks", renderer=task_renderer)
        self.rec: dict | None = None
        self.log: list[str] = []

    def note(self, msg: str):
        self.log = (self.log + [f"{time.strftime('%H:%M:%S')} {msg}"])[-40:]

    # ------------------------------------------------------------ commands from the browser
    def _set(self, side, pos):
        self.runner.set_target(side, pos)
        if self.mirror:
            other = "left" if side == "right" else "right"
            if other in self.runner.sides:
                self.runner.set_target(other, pos)

    def disarm_teleop(self):
        was_armed = self.teleop_armed
        self.teleop_armed = False
        self.teleop_owner = None
        if was_armed and self.teleop_side in self.runner.sides:
            self.runner.hold(self.teleop_side)
        for live in self.live.values():
            live.reset()

    def handle(self, m: dict, owner=None):
        t = m["type"]
        if t == "preview":
            if m.get("source") == "replay" and self.teleop_armed:
                raise ValueError("Disarm live control before preview replay")
            # Replayed clips are right-hand mapping data; the live preview follows the driven side.
            side = "right" if m.get("source") == "replay" else self.teleop_side
            accepted = self.preview.update(m, profile(self.retarget.get(side, HandRetargeter())))
            if accepted and self.task_service.mapping_enabled and m.get("source", "live") == "live":
                observation = m.get("hand")
                self.task_service.set_target(self.preview.target, source="mapping",
                    valid=self.preview.live.status == "tracking", landmarks=observation.get("points") if observation else None)
            return
        if t == "teleop_side":
            if m["side"] not in self.runner.sides:
                raise ValueError(f"No {m['side']} hand is served")
            if owner is not self.teleop_owner and self.teleop_armed:
                raise ValueError("Teleop is controlled by another browser")
            self.disarm_teleop()
            self.teleop_side = m["side"]
            return
        if t == "teleop_arm":
            if m.get("on"):
                if self.task_service.mapping_enabled:
                    raise ValueError("Stop task mapping input before arming robot teleop")
                if self.teleop_armed and owner is not self.teleop_owner:
                    raise ValueError("Teleop is controlled by another browser")
                if not self.calibrated.get(self.teleop_side, False):
                    raise ValueError(f"Calibrate the {self.teleop_side} hand open and fist before driving")
                if self.runner.estopped:
                    raise ValueError("Resume before arming teleop")
                self.disarm_teleop()
                self.teleop_armed, self.teleop_owner = True, owner
                self.teleop_seq = -1
            elif owner is self.teleop_owner:
                self.disarm_teleop()
            return
        if t in {"calib_reset", "calib", "calib_begin"} and self.task_service.snapshot().get("recording"):
            raise ValueError("Finish the task episode before changing calibration")
        if t in {"finger", "grasp", "routine", "estop", "calib_reset", "calib", "calib_begin"}:
            self.disarm_teleop()
        if t == "finger":
            pos = self.runner.target[m["side"]].copy()
            pos[int(m["i"])] = float(m["v"])
            self._set(m["side"], pos)
        elif t == "grasp":
            self._set(m["side"], self.grasps[m["name"]])
        elif t == "speed":
            with self.runner.lock:
                self.runner.speed = float(np.clip(m["v"], 0.05, 1.0))
        elif t == "estop":
            self.runner.estop()
            self.task_service.hold(lock=True)
            if self.link:
                self.link.send({"type": "estop"})
            self.note("E-STOP: hands opening, commands locked")
        elif t == "resume":
            if self.link:
                self.link.send({"type": "resume"})
            self.runner.resume()
            self.task_service.locked = False
            self.note("resumed")
        elif t == "mirror":
            self.mirror = bool(m["on"])
        elif t == "save_grasp":
            name = str(m["name"]).strip().lower().replace(" ", "_")[:24]
            if name:
                self.grasps[name] = np.round(self.runner.target[m["side"]], 3).tolist()
                custom = {k: v for k, v in self.grasps.items() if GRASPS.get(k) != v}
                GRASP_FILE.write_text(yaml.safe_dump(custom, default_flow_style=None))
                self.note(f"saved grasp '{name}' from {m['side']} hand")
        elif t == "routine":
            self._routine(m["name"], m["side"])
        elif t == "record":
            self._record(bool(m["on"]))
        elif t == "teleop":
            if not self.teleop_armed or owner is not self.teleop_owner:
                return
            seq = int(m.get("seq", -1))
            age = float(m.get("age_ms", float("inf")))
            # Echoed panel timestamp provides a conservative transport-age bound without
            # trusting browser and Python wall clocks to be synchronized.
            echoed = float(m.get("panel_time", 0))
            now = time.monotonic()
            if (seq <= self.teleop_seq or not np.isfinite(age) or not 0 <= age <= 200
                    or not np.isfinite(echoed) or not 0 <= now - echoed <= 0.3):
                return
            self.teleop_seq = seq
            self.teleop_metrics = {"frame_age_ms": age, "inference_ms": m.get("inference_ms")}
            accepted = False
            for side, live in self.live.items():
                if side != self.teleop_side:
                    continue
                obs = m.get("hands", {}).get(side)
                if obs is None or not self.calibrated[side]:
                    live.reset()
                    self.runner.hold(side)
                    continue
                q = live.update(obs.get("points"), float(obs.get("score", 0)), now)
                if q is None:
                    self.runner.hold(side)
                else:
                    # Teleop follows anatomical sides only; mirror is a manual-control feature.
                    accepted |= self.runner.set_target(side, q, ttl_s=0.25)
            if accepted:
                self.teleop_t = now
        elif t == "calib_begin":
            side, pose = m["side"], m["pose"]
            self.calib[side][pose] = []
            if pose in ("open", "fist"):
                self.calibrated[side] = False
            self.thumb_calibrated[side] = False
        elif t == "calib":
            side, pose = m["side"], m["pose"]
            if not np.isfinite(float(m.get("score", 1.0))) or float(m.get("score", 1.0)) < 0.8:
                raise ValueError("Calibration requires a clear hand identity")
            kp = validate_landmarks(m["kp"])
            self.calib[side][pose].append(kp)
            self.calib[side][pose] = self.calib[side][pose][-60:]
            c = self.calib[side]
            if pose != "thumb" and len(c["open"]) >= 5 and len(c["fist"]) >= 5:
                valid = self.retarget[side].calibrate(np.array(c["open"][-30:]), np.array(c["fist"][-30:]))
                self.thumb_calibrated[side] = bool(valid[:2].all())
                self.calibrated[side] = True
            if len(c["open"]) >= 5 and len(c["thumb"]) >= 5:
                self.retarget[side].calibrate_thumb(c["open"], c["thumb"])
                self.thumb_calibrated[side] = True
        elif t == "calib_reset":
            self.calib[m["side"]] = {"open": [], "fist": [], "thumb": []}
            self.retarget[m["side"]] = HandRetargeter(smooth=0.6)
            self.calibrated[m["side"]] = False
            self.live[m["side"]] = LiveHand(self.retarget[m["side"]])
            self.thumb_calibrated[m["side"]] = False

    def _routine(self, name, side):
        z = np.zeros(N_HAND)
        if name == "sweep":
            kf = []
            for i in range(N_HAND):
                g = z.copy()
                g[i] = 1.0
                kf += [(side, g, 1.5), (side, z, 1.5)]
        elif name == "cycle":
            kf = [(side, np.array(v), 1.2) for k, v in self.grasps.items() if k != "open"]
            kf = [x for pair in zip(kf, [(side, z, 0.8)] * len(kf)) for x in pair]
        else:
            return
        if self.runner.play(kf):
            self.note(f"routine '{name}' on {side} hand (any manual command cancels it)")

    def _record(self, on: bool):
        if on and self.rec is None:
            self.rec = {"t0": time.monotonic(), "rows": []}
            self.note("recording started")
        elif not on and self.rec is not None:
            rows, self.rec = self.rec["rows"], None
            if rows:
                LOGDIR.mkdir(parents=True, exist_ok=True)
                p = LOGDIR / f"{time.strftime('%Y%m%d_%H%M%S')}_panel.npz"
                arr = {}
                for s in self.runner.sides:
                    arr[f"{s}_target"] = np.array([r[s]["target"] for r in rows])
                    arr[f"{s}_position"] = np.array([r[s]["position"] for r in rows])
                    arr[f"{s}_current"] = np.array([r[s]["current"] or [np.nan] * N_HAND for r in rows], float)
                    arr[f"{s}_open_features"] = self.retarget[s].open_feat.copy()
                    arr[f"{s}_closed_features"] = self.retarget[s].fist_feat.copy()
                for name in ("inference_ms", "frame_age_ms", "robot_rtt_ms"):
                    arr[name] = np.array([r.get(name, np.nan) for r in rows], float)
                arr["teleop_armed"] = np.array([r["teleop_armed"] for r in rows])
                np.savez(p, t=np.array([r["t"] for r in rows]), joints=np.array(HAND_JOINTS), **arr)
                self.note(f"saved {len(rows)} frames -> {p.relative_to(REPO) if p.is_relative_to(REPO) else p}")

    def record_sample(self):
        """Called once per sampling tick, independent of browser count."""
        if self.rec is not None:
            self.rec["rows"].append({"t": time.monotonic() - self.rec["t0"],
                                     **self.runner.snapshot()["hands"], **self.teleop_metrics,
                                     "teleop_armed": self.teleop_armed,
                                     "robot_rtt_ms": self.link.rtt_ms if self.link else np.nan})

    # ------------------------------------------------------------ state to the browser
    def snapshot(self) -> dict:
        if self.link is not None and self.link.state.get("estopped") and not self.runner.estopped:
            # The hand server stays E-stopped across panel restarts; adopt its state so the UI
            # shows the stop and Space sends resume, instead of commands being silently ignored.
            self.runner.estop()
            self.disarm_teleop()
            self.note("hand server is E-stopped: press Space to resume")
        snap = self.runner.snapshot()
        conn = {"backend": self.backend, "url": self.url, "ok": True, "rate": None}
        if self.link is not None:
            snap["error"] = snap["error"] or self.link.state.get("error")
            conn.update(ok=self.link.connected and time.monotonic() - self.link.last_rx < 0.3, rate=self.link.rate_hz, rtt_ms=self.link.rtt_ms)
        return {**snap, "task": self.task_service.snapshot(), "preview": self.preview.snapshot(),
                "retarget_profile": profile(self.retarget.get("right", HandRetargeter())),
                "panel_time": time.monotonic(), "teleop_armed": self.teleop_armed, "teleop_side": self.teleop_side,
                "tracking": {s: (v.status if v.last_t is not None and time.monotonic() - v.last_t < 0.25
                                 else "waiting") for s, v in self.live.items()},
                "teleop_metrics": self.teleop_metrics, "thumb_calibrated": self.thumb_calibrated,
                "conn": conn, "mirror": self.mirror, "grasps": list(self.grasps),
                "recording": self.rec is not None, "stall_current": self.stall_current,
                "calibrated": self.calibrated, "teleop_live": self.teleop_armed and time.monotonic() - self.teleop_t < 0.25,
                "calib_counts": {s: {k: len(v) for k, v in c.items()} for s, c in self.calib.items()},
                "log": self.log[-12:], "joints": list(HAND_JOINTS)}


def make_app(panel: Panel):
    from aiohttp import WSMsgType, web

    async def index(_):
        return web.FileResponse(STATIC / "index.html")

    async def mapping_export(request):
        from r1hands.data.mapping import convert_clip
        try:
            clip = await request.json()
            arrays = await asyncio.to_thread(convert_clip, clip)
            output = io.BytesIO()
            await asyncio.to_thread(np.savez_compressed, output, **arrays)
            return web.Response(body=output.getvalue(), content_type="application/octet-stream",
                                headers={"Content-Disposition": 'attachment; filename="right-hand-mapping.npz"'})
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            raise web.HTTPBadRequest(text=str(error)) from error

    async def task_action(request):
        try:
            data = await request.json()
            action = data.get("action")
            service = panel.task_service
            if action == "start":
                await asyncio.to_thread(service.start)
            elif action == "reset":
                seed = int(data.get("seed", 0))
                if not 0 <= seed < 2**32: raise ValueError("Seed must be between 0 and 2^32-1")
                await asyncio.to_thread(service.call, "reset", seed=seed)
            elif action == "release":
                await asyncio.to_thread(service.call, "release")
            elif action == "begin":
                provenance = {"operator_id": str(data.get("operator_id", ""))[:200],
                    "task_description": "Grasp the block and hold after support withdrawal",
                    "mapping_profile": profile(panel.retarget.get("right", HandRetargeter())),
                    "mapping_mapper": panel.preview.snapshot()["mapper"],
                    "mapping_model_sha256": panel.mapping_model_sha256}
                await asyncio.to_thread(service.call, "begin", provenance=provenance)
            elif action == "finish":
                outcome = data.get("outcome", "unspecified")
                if outcome not in ("unspecified", "success", "failure"): raise ValueError("Unknown operator outcome")
                await asyncio.to_thread(service.call, "finish", outcome=outcome)
            elif action == "example":
                if not service.latest: raise ValueError("Start task simulation first")
                service.mapping_enabled = False
                service.set_target([0.734, 0.382, 0.323, 0.541, 0.78, 0.626], source="scripted")
            elif action == "target":
                if not service.latest: raise ValueError("Start task simulation first")
                service.mapping_enabled = False
                service.set_target(data["position"])
            elif action == "mapping":
                if not service.latest: raise ValueError("Start task simulation first")
                if panel.teleop_armed: raise ValueError("Disarm robot control before using the mapping as task input")
                if service.locked: raise ValueError("Resume the stop lockout first")
                service.hold()
                service.mapping_enabled = bool(data.get("on"))
            elif action == "hold":
                service.hold()
            else: raise ValueError("Unknown task action")
            return web.json_response(service.snapshot())
        except (ValueError, TypeError, KeyError, RuntimeError, TimeoutError) as error:
            raise web.HTTPBadRequest(text=str(error)) from error

    async def task_frame(_):
        if panel.task_service.png is None: raise web.HTTPNotFound()
        return web.Response(body=panel.task_service.png, content_type="image/png", headers={"Cache-Control":"no-store"})

    async def task_download(request):
        import re
        identifier = request.match_info["identifier"]
        if not re.fullmatch(r"[0-9a-f]{32}", identifier): raise web.HTTPNotFound()
        path = panel.task_service.directory / (identifier + ".npz")
        if not path.is_file(): raise web.HTTPNotFound()
        return web.FileResponse(path, headers={"Content-Disposition": f'attachment; filename="task-{identifier}.npz"'})

    async def preview_model(_):
        return web.json_response(hand_model())

    async def preview_mesh(request):
        name = request.match_info["name"]
        allowed = {Path(v["mesh"]).name for link in hand_model()["links"] for v in link["visuals"]}
        if name not in allowed:
            raise web.HTTPNotFound()
        return web.FileResponse(MESH_DIR / name)

    async def example(_):
        return web.json_response(example_clip())

    async def ws_handler(request):
        ws = web.WebSocketResponse(heartbeat=5.0)
        await ws.prepare(request)

        async def pump():
            while not ws.closed:
                await ws.send_str(json.dumps(panel.snapshot()))
                await asyncio.sleep(1 / 30)

        task = asyncio.create_task(pump())
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        if data.get("type") == "ping":
                            await ws.send_json({"type": "pong", "id": data.get("id")})
                        else:
                            panel.handle(data, owner=ws)
                    except Exception as e:
                        panel.note(f"error: {e!r}")
        finally:
            if panel.teleop_owner is ws:
                panel.disarm_teleop()
            task.cancel()
        return ws

    async def recording_lifecycle(app):
        async def sample():
            while True:
                panel.record_sample()
                await asyncio.sleep(1 / 30)
        task = asyncio.create_task(sample())
        try:
            yield
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            panel._record(False)
            await asyncio.to_thread(panel.task_service.close)

    app = web.Application(client_max_size=20 * 1024 * 1024)
    app.cleanup_ctx.append(recording_lifecycle)
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/preview/model", preview_model)
    app.router.add_get("/preview/meshes/{name}", preview_mesh)
    app.router.add_get("/preview/example", example)
    app.router.add_post("/mapping/export", mapping_export)
    app.router.add_post("/tasks/action", task_action)
    app.router.add_get("/tasks/frame.png", task_frame)
    app.router.add_get("/tasks/episodes/{identifier}", task_download)
    app.router.add_static("/static", STATIC)
    return app


def main():
    from aiohttp import web
    ap = argparse.ArgumentParser(description="Revo2 hand control panel")
    ap.add_argument("--backend", default="remote", choices=["remote", "sim", "mock", "real"])
    ap.add_argument("--remote", default=None, help="hand server URL, e.g. ws://192.168.123.164:8765/ws")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--sides", default="left,right")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--task-renderer", choices=["auto", "glfw", "egl", "osmesa"], default="auto",
                    help="Task simulation rendering: auto/local default, egl/headless Linux GPU, osmesa/headless Linux CPU")
    ap.add_argument("--mapping-model", type=Path, help="Learned mapping NPZ for the read-only preview; motor control remains analytic")
    a = ap.parse_args()
    panel = Panel(a.backend, a.remote, tuple(a.sides.split(",")), mapping_model=a.mapping_model, task_renderer=a.task_renderer)
    if not a.no_browser:
        webbrowser.open(f"http://localhost:{a.port}")
    try:
        web.run_app(make_app(panel), host="127.0.0.1", port=a.port, print=lambda *_: print(
            f"panel on http://localhost:{a.port}  (backend={a.backend}{', ' + panel.url if panel.url else ''})"))
    finally:
        panel.runner.close()
        panel.robot.close()


if __name__ == "__main__":
    main()
