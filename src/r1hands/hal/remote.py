"""Remote hands: use hands served by r1hands.panel.hand_server over the network, as if they were local.

    robot = make_robot("remote", level=1, remote="ws://192.168.123.164:8765/ws")

One WebSocket link carries both hands. command() only sends when the target changed (or as a
0.05 s keepalive); the server re-applies targets at 100 Hz through its own HandGuard.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time

import numpy as np

from r1hands.common.types import N_HAND, HandCommand, HandState, Side, now
from r1hands.hal.base import Hand


class RemoteLink:
    def __init__(self, url: str):
        self.url = url
        self.state: dict = {}
        self.connected = False
        self.rx_times: list[float] = []
        self._out: asyncio.Queue | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop = False
        self._ready = threading.Event()
        self._pending = {}  # one latest target per hand; never replay a movement backlog
        self.last_rx = 0.0
        self.rtt_ms = None

    def start(self, timeout: float = 5.0):
        self._thread = threading.Thread(target=lambda: asyncio.run(self._run()), daemon=True, name="remote-link")
        self._thread.start()
        if not self._ready.wait(timeout):
            raise ConnectionError(f"no hand server at {self.url}")

    async def _run(self):
        import aiohttp
        self._loop = asyncio.get_running_loop()
        self._out = asyncio.Queue(maxsize=8)
        while not self._stop:
            try:
                async with aiohttp.ClientSession() as sess, sess.ws_connect(self.url, heartbeat=2.0) as ws:
                    self.connected = True
                    sender = asyncio.create_task(self._send(ws))
                    async for msg in ws:
                        if self._stop:
                            break
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            continue
                        m = json.loads(msg.data)
                        if m.get("type") == "pong":
                            self.rtt_ms = (time.monotonic() - float(m["id"])) * 1000
                        if m.get("type") == "state":
                            self.state = m
                            t = time.monotonic()
                            self.last_rx = t
                            self.rx_times = [x for x in self.rx_times[-60:] if t - x < 1.0] + [t]
                            self._ready.set()
                    sender.cancel()
                    await asyncio.gather(sender, return_exceptions=True)
            except Exception:
                if 'sender' in locals():
                    sender.cancel()
                    await asyncio.gather(sender, return_exceptions=True)
            self._pending.clear()
            while not self._out.empty():
                self._out.get_nowait()
            self.connected = False
            await asyncio.sleep(0.5)  # reconnect

    async def _send(self, ws):
        last_ping = 0.0
        while not self._stop:
            while not self._out.empty():
                await ws.send_str(json.dumps(self._out.get_nowait()))
            pending, self._pending = self._pending, {}
            for msg in pending.values():
                if msg.get("deadline", 0) >= self.state.get("t", float("inf")):
                    await ws.send_str(json.dumps(msg))
            t = time.monotonic()
            if t - last_ping > 0.5:
                await ws.send_str(json.dumps({"type": "ping", "id": t}))
                last_ping = t
            await asyncio.sleep(0.005)

    def send(self, msg: dict):
        def enqueue():
            if not self.connected or self._stop:
                return
            if msg.get("type") == "set":
                self._pending[msg["side"]] = msg
            else:
                if msg.get("type") == "estop":
                    self._pending.clear()
                    while not self._out.empty():
                        self._out.get_nowait()
                if not self._out.full():
                    self._out.put_nowait(msg)
        if self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(enqueue)

    @property
    def rate_hz(self) -> float:
        return float(sum(time.monotonic() - t < 1 for t in self.rx_times))

    def close(self):
        self._stop = True
        if hasattr(self, "_thread"):
            self._thread.join(1.0)


class RemoteHand(Hand):
    def __init__(self, link: RemoteLink, side: Side):
        self.link, self.side = link, side
        self._last_sent = None
        self._last_t = 0.0
        self.events: list[str] = []
        self._cap = np.full(N_HAND, np.inf)

    def connect(self):
        if not self.link._ready.is_set():
            self.link.start()

    def close(self):
        self.link.close()

    def read(self) -> HandState:
        h = self.link.state.get("hands", {}).get(self.side)
        if h is None:
            return HandState(position=np.zeros(N_HAND), fault=True, stamp=float("nan"))
        self.events = h.get("events", [])
        self._cap = np.where(h.get("capped", [False] * N_HAND), 0.0, np.inf)
        cur = h.get("current")
        return HandState(position=np.array(h["position"]), current=None if cur is None else np.array(cur),
                         fault=bool(h.get("fault")) or not self.link.connected or now() - self.link.last_rx > 0.3,
                         stamp=self.link.last_rx)

    def command(self, cmd: HandCommand):
        t = time.monotonic()
        if not self.link.connected or t - self.link.last_rx > 0.3:
            return
        pos = np.round(cmd.position, 4)
        if self._last_sent is not None and np.array_equal(pos, self._last_sent) and t - self._last_t < 0.05:
            return
        self.link.send({"type": "set", "side": self.side, "position": pos.tolist(), "speed": cmd.speed,
                        "deadline": self.link.state["t"] + 0.3})
        self._last_sent, self._last_t = pos, t
