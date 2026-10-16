"""Hand server: runs on the Linux machine wired to the Revo2 hands (R1 Jetson or a lab PC).

    python -m r1hands.panel.hand_server --backend real --port 8765      # on the robot side
    python -m r1hands.panel.hand_server --backend mock                   # anywhere, for testing

WebSocket protocol on ws://<host>:<port>/ws (JSON text frames):
  server -> client  every 20 ms   {"type": "state", "t": <server time>, "hands": {...}, "estopped": bool, ...}
  client -> server                {"type": "set", "side": "left", "position": [6 floats 0..1], "speed": 0..1}
                                  {"type": "estop"} | {"type": "resume"}
The HandGuard (rate limit, stall latch) runs here, next to the hardware. If a client disconnects, the
motion targets expire after at most 300 ms and hold the measured pose. Only E-stop opens them.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

from r1hands.hal.factory import make_robot
from r1hands.panel.runner import HandRunner


def make_app(runner: HandRunner, period: float = 0.02):
    from aiohttp import WSMsgType, web

    async def ws_handler(request):
        ws = web.WebSocketResponse(heartbeat=2.0)
        await ws.prepare(request)

        async def pump():
            while not ws.closed:
                snap = runner.snapshot()
                await ws.send_str(json.dumps({"type": "state", "t": time.time(), **snap}))
                await asyncio.sleep(period)

        task = asyncio.create_task(pump())
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                m = json.loads(msg.data)
                if m["type"] == "set":
                    # The client echoes a deadline derived from server state time. This
                    # rejects delayed packets without synchronizing Mac/Jetson clocks.
                    remaining = float(m.get("deadline", 0)) - time.time()
                    if 0 < remaining <= 0.35:
                        runner.set_target(m["side"], m["position"], m.get("speed"),
                                          ttl_s=min(remaining, 0.3))
                elif m["type"] == "ping":
                    await ws.send_json({"type": "pong", "id": m.get("id")})
                elif m["type"] == "estop":
                    runner.estop()
                elif m["type"] == "resume":
                    runner.resume()
        finally:
            task.cancel()
        return ws

    async def health(_):
        return web.json_response({"ok": True, "sides": runner.sides, "estopped": runner.estopped})

    app = web.Application()
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/health", health)
    return app


def main():
    from aiohttp import web
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="real", choices=["real", "sim", "mock"])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--sides", default=None, help="default: hands with a port set in configs/r1.yaml")
    a = ap.parse_args()
    if a.sides:
        sides = tuple(a.sides.split(","))
    elif a.backend == "real":
        from r1hands.hal.factory import load_config
        sides = tuple(s for s, c in load_config()["hands"].items() if c.get("port"))
    else:
        sides = ("left", "right")
    print(f"hand server: backend={a.backend} sides={sides} port={a.port}")
    robot = make_robot(a.backend, level=1, sides=sides).connect()
    runner = HandRunner(robot)
    try:
        web.run_app(make_app(runner), host=a.host, port=a.port)
    finally:
        runner.estop()
        time.sleep(1.5)
        runner.close()
        robot.close()


if __name__ == "__main__":
    main()
