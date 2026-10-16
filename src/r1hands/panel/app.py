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
    def __init__(self, backend: str, remote: str | None, sides=("left", "right")):
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
        self.calib = {s: {"open": [], "fist": [], "pinch": []} for s in sides}
        self.calibrated = {s: False for s in sides}
        self.live = {s: LiveHand(self.retarget[s]) for s in sides}
        self.teleop_armed = False
        self.teleop_owner = None
        self.teleop_seq = -1
        self.teleop_t = 0.0
        self.teleop_metrics = {}
        self.thumb_calibrated = {s: False for s in sides}
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
        if was_armed and "right" in self.runner.sides:
            self.runner.hold("right")
        for live in self.live.values():
            live.reset()

    def handle(self, m: dict, owner=None):
        t = m["type"]
        if t == "teleop_arm":
            if m.get("on"):
                if self.teleop_armed and owner is not self.teleop_owner:
                    raise ValueError("Teleop is controlled by another browser")
                if not self.calibrated.get("right", False):
                    raise ValueError("Calibrate the right hand open and fist before driving")
                if self.runner.estopped:
                    raise ValueError("Resume before arming teleop")
                self.disarm_teleop()
                self.teleop_armed, self.teleop_owner = True, owner
                self.teleop_seq = -1
            elif owner is self.teleop_owner:
                self.disarm_teleop()
            return
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
            if self.link:
                self.link.send({"type": "estop"})
            self.note("E-STOP: hands opening, commands locked")
        elif t == "resume":
            if self.link:
                self.link.send({"type": "resume"})
            self.runner.resume()
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
                if side != "right":
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
            if pose != "pinch" and len(c["open"]) >= 5 and len(c["fist"]) >= 5:
                valid = self.retarget[side].calibrate(np.array(c["open"][-30:]), np.array(c["fist"][-30:]))
                self.thumb_calibrated[side] = bool(valid[:2].all())
                self.calibrated[side] = True
            if len(c["open"]) >= 5 and len(c["pinch"]) >= 5:
                self.retarget[side].calibrate_thumb(c["open"], c["pinch"])
                self.thumb_calibrated[side] = True
        elif t == "calib_reset":
            self.calib[m["side"]] = {"open": [], "fist": [], "pinch": []}
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
        snap = self.runner.snapshot()
        conn = {"backend": self.backend, "url": self.url, "ok": True, "rate": None}
        if self.link is not None:
            snap["error"] = snap["error"] or self.link.state.get("error")
            conn.update(ok=self.link.connected and time.monotonic() - self.link.last_rx < 0.3, rate=self.link.rate_hz, rtt_ms=self.link.rtt_ms)
        return {**snap, "panel_time": time.monotonic(), "teleop_armed": self.teleop_armed,
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

    app = web.Application()
    app.cleanup_ctx.append(recording_lifecycle)
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
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
    a = ap.parse_args()
    panel = Panel(a.backend, a.remote, tuple(a.sides.split(",")))
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
