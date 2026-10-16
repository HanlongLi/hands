"""HandRunner: owns the latest target per hand and re-applies it at a fixed rate.

The HandGuard's rate limit and stall monitor need a continuous command stream, but UI sliders send
sparse updates. The runner closes that gap, and is used by both the hand server (next to the hardware)
and the Mac panel (when it drives mock/sim locally). Routines (sweep, grasp cycle) play target
sequences through the same path, and any manual command or E-stop cancels them.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from r1hands.common.types import N_HAND, HandCommand, Side
from r1hands.hal.base import Robot


class HandRunner:
    def __init__(self, robot: Robot, rate_hz: float = 100.0):
        self.robot = robot
        self.rate_hz = rate_hz
        self.sides: list[Side] = list(robot.hands)
        # Hold wherever the hands are now. Refuse to start on placeholder state, which would move them.
        self.target = {}
        for s in self.sides:
            st = robot.hands[s].read()
            if not np.isfinite(st.stamp):
                raise RuntimeError(f"{s} hand has no telemetry yet; refusing to start (would command a placeholder pose)")
            self.target[s] = st.position.copy()
        self._expires = {}
        self.last_error = None
        self.speed = 0.5
        self.estopped = False
        self.lock = threading.Lock()
        self._routine: threading.Thread | None = None
        self._routine_gen = 0
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True, name="hand-runner")
        self.thread.start()

    # ------------------------------------------------------------ inputs
    def set_target(self, side: Side, position, speed: float | None = None, from_routine: bool = False, ttl_s: float | None = None):
        position = np.asarray(position, float).reshape(N_HAND)
        if side not in self.sides or not np.isfinite(position).all():
            raise ValueError("Unknown hand or non-finite target")
        if speed is not None and not np.isfinite(speed):
            raise ValueError("Non-finite speed")
        with self.lock:
            if self.estopped:
                return False
            if not from_routine:
                self._routine_gen += 1  # cancels a running routine
            self.target[side] = np.clip(position, 0, 1)
            if ttl_s is None:
                self._expires.pop(side, None)
            else:
                self._expires[side] = time.monotonic() + max(0, ttl_s)
            if speed is not None:
                self.speed = float(np.clip(speed, 0.05, 1.0))
        return True

    def hold(self, side):
        """Cancel remaining motion by holding measured pose, retaining guard monitoring."""
        with self.lock:
            self._routine_gen += 1
            self._expires.pop(side, None)
            if not self.estopped:
                try:
                    self.target[side] = self.robot.hands[side].hold()
                except Exception as e:
                    # Keep retrying hold; never resume dispatching the previous target.
                    self._expires[side] = 0.0
                    self.last_error = f"{side}: hold failed: {e}"

    def estop(self):
        """Open both hands slowly and lock out commands until resume()."""
        with self.lock:
            self._routine_gen += 1
            self._expires.clear()
            for s in self.sides:
                self.target[s] = np.zeros(N_HAND)
            if not self.estopped:
                self._speed_before = self.speed
            self.speed = 0.2
            self.estopped = True

    def resume(self):
        with self.lock:
            if self.estopped:
                self.speed = getattr(self, "_speed_before", self.speed)
            self.estopped = False

    def play(self, keyframes: list[tuple[Side, np.ndarray, float]]):
        """Run (side, target, seconds) keyframes, linearly interpolated, in a thread."""
        with self.lock:
            if self.estopped:
                return False
            self._routine_gen += 1
            gen = self._routine_gen

        def run():
            for side, goal, dur in keyframes:
                start = self.target[side].copy()
                n = max(1, int(dur * 50))
                for i in range(1, n + 1):
                    if self._routine_gen != gen:
                        return
                    self.set_target(side, start + (np.asarray(goal) - start) * i / n, from_routine=True)
                    time.sleep(dur / n)

        self._routine = threading.Thread(target=run, daemon=True)
        self._routine.start()
        return True

    @property
    def routine_running(self) -> bool:
        return self._routine is not None and self._routine.is_alive()

    # ------------------------------------------------------------ loop
    def _loop(self):
        period = 1.0 / self.rate_hz
        while not self._stop.is_set():
            t0 = time.monotonic()
            with self.lock:
                for side, deadline in list(self._expires.items()):
                    if t0 >= deadline:
                        if not self.estopped:
                            try:
                                self.target[side] = self.robot.hands[side].hold()
                            except Exception as e:
                                self.last_error = f"{side}: hold failed: {e}"
                                continue
                        del self._expires[side]
                for s in self.sides:
                    if self._expires.get(s, float("inf")) <= t0:
                        continue
                    try:
                        self.robot.hands[s].command(HandCommand(self.target[s].copy(), speed=self.speed))
                    except Exception as e:  # keep the other hand alive; surface the error in events
                        self.last_error = f"{s}: {e}"
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def close(self):
        self._routine_gen += 1
        self._stop.set()
        self.thread.join(1.0)

    # ------------------------------------------------------------ state
    def snapshot(self) -> dict:
        out = {}
        for s in self.sides:
            h = self.robot.hands[s]
            st = h.read()
            ev = getattr(h, "events", [])
            out[s] = {
                "position": np.round(st.position, 4).tolist(),
                "current": None if st.current is None else np.round(np.abs(st.current), 1).tolist(),  # magnitude
                "fault": bool(st.fault),
                "target": np.round(self.target[s], 4).tolist(),
                "events": ev[-5:],
                "capped": [bool(np.isfinite(c)) for c in getattr(h, "_cap", np.full(N_HAND, np.inf))],
            }
        return {"hands": out, "estopped": self.estopped, "speed": self.speed, "routine": self.routine_running, "error": self.last_error}
