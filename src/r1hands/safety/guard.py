"""Safety wrappers that sit between any controller (teleop, IL, RL, LLM code) and the HAL.

Every controller goes through these. Nothing reaches hardware unfiltered.
- HandGuard: rate limit on finger targets, and a stall/thermal monitor. If a finger holds high current
  for longer than `stall_s`, its target backs off to the current position, so the firmware protection
  never has to trip.
- BodyGuard: joint limits from the URDF, per-step delta limit, a velocity cap on loco commands, and a
  watchdog that releases the upper body when commands stop arriving.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from r1hands.common.types import (
    N_HAND, HandCommand, HandState, JointCommand, LocoCommand, now,
)
from r1hands.hal.base import Body, Hand


@dataclass
class HandLimits:
    max_rate: float = 2.0          # normalized units / s (full close in 0.5 s)
    stall_current: float = 600.0   # mA; set from the level-1 telemetry audit
    stall_s: float = 1.5
    max_temp: float = 65.0         # deg C; soft limit below firmware protection
    backoff: float = 0.05          # how far to open a stalled finger


class HandGuard(Hand):
    def __init__(self, hand: Hand, limits: HandLimits | None = None):
        self.inner = hand
        self.side = hand.side
        self.lim = limits or HandLimits()
        self._last_target: np.ndarray | None = None
        self._last_t = now()
        self._stall_since = np.full(N_HAND, np.nan)
        self._cap = np.full(N_HAND, np.inf)   # latched per-finger closure cap after a stall
        self.last_state: HandState | None = None
        self.events: list[str] = []

    def connect(self):
        self.inner.connect()
        self.last_state = self.inner.read()
        self._last_target = self.last_state.position.copy()

    def close(self):
        self.inner.close()

    def read(self) -> HandState:
        s = self.inner.read()
        self.last_state = s
        return s

    def hold(self):
        # Reset the limiter origin too: otherwise a previously distant command can
        # keep closing the fingers while its target ramps back to the measured pose.
        self._last_target = self.read().position.copy()
        self._last_t = now()
        caps = self._cap.copy()
        self.command(HandCommand(self._last_target.copy()))
        self._cap = np.minimum(self._cap, caps)
        return self._last_target.copy()

    def command(self, cmd: HandCommand):
        t = now()
        dt = max(t - self._last_t, 1e-3)
        self._last_t = t
        s = self.read()
        if (not np.isfinite(s.stamp) or t - s.stamp > 0.5
                or not np.isfinite(s.position).all()):
            raise RuntimeError(f"{self.side}: telemetry missing or stale; motion refused")
        target = cmd.position.copy()

        # Rate limit.
        step = self.lim.max_rate * dt
        target = np.clip(target, self._last_target - step, self._last_target + step)

        # Stall and thermal protection, per finger.
        if s.current is not None:
            hot = np.abs(s.current) > self.lim.stall_current  # Revo2 current is signed (negative when opening)
            self._stall_since = np.where(hot & np.isnan(self._stall_since), t, self._stall_since)
            self._stall_since = np.where(~hot, np.nan, self._stall_since)
            stalled = (t - np.nan_to_num(self._stall_since, nan=t)) > self.lim.stall_s
            if stalled.any():
                # Latch a cap just inside the contact point. It holds the object without squeezing,
                # and clears once the controller asks the finger to open past it.
                self._cap = np.where(stalled, np.maximum(s.position - self.lim.backoff, 0.0), self._cap)
                self._stall_since[stalled] = np.nan
                self.events.append(f"{t:.2f} {self.side} stall backoff fingers={np.flatnonzero(stalled).tolist()}")
        released = cmd.position < self._cap - 2 * self.lim.backoff
        self._cap = np.where(released, np.inf, self._cap)
        target = np.minimum(target, self._cap)
        if s.temperature is not None and (s.temperature > self.lim.max_temp).any():
            over = s.temperature > self.lim.max_temp
            target = np.where(over, np.minimum(target, s.position), target)  # do not squeeze harder
            self.events.append(f"{t:.2f} {self.side} temp limit fingers={np.flatnonzero(over).tolist()}")

        self.events = self.events[-200:]
        self._last_target = target
        self.inner.command(HandCommand(target, cmd.speed, cmd.force))


@dataclass
class BodyLimits:
    q_min: dict[str, float]
    q_max: dict[str, float]
    max_dq_per_s: float = 1.5      # rad/s on any commanded joint
    max_v: tuple[float, float, float] = (0.5, 0.3, 0.8)  # vx, vy, wz
    watchdog_s: float = 0.5


class BodyGuard(Body):
    def __init__(self, body: Body, limits: BodyLimits):
        self.inner = body
        self.lim = limits
        self._last_q: dict[str, float] = {}
        self._last_cmd_t = None
        self._last_t = now()

    def connect(self):
        self.inner.connect()

    def close(self):
        self.inner.close()

    def read(self):
        return self.inner.read()

    def command_joints(self, cmd: JointCommand):
        t = now()
        dt = max(t - self._last_t, 1e-3)
        self._last_t = t
        if not self._last_q:
            s = self.inner.read()
            self._last_q = dict(zip(cmd.names, s.get(cmd.names)))
        q = np.array(cmd.q, float)
        for i, n in enumerate(cmd.names):
            lo, hi = self.lim.q_min.get(n, -np.inf), self.lim.q_max.get(n, np.inf)
            prev = self._last_q.get(n, q[i])
            step = self.lim.max_dq_per_s * dt
            q[i] = np.clip(np.clip(q[i], prev - step, prev + step), lo, hi)
            self._last_q[n] = q[i]
        self._last_cmd_t = t
        self.inner.command_joints(JointCommand(cmd.names, q, cmd.dq, cmd.kp, cmd.kd, cmd.tau_ff))

    def check_watchdog(self):
        """Call from the control loop. Releases the upper body if the controller went silent."""
        if self._last_cmd_t is not None and now() - self._last_cmd_t > self.lim.watchdog_s:
            self.inner.release_joints()
            self._last_cmd_t = None
            self._last_q.clear()
            return True
        return False

    def release_joints(self):
        self._last_q.clear()
        self.inner.release_joints()

    def loco(self, cmd: LocoCommand):
        vx, vy, wz = self.lim.max_v
        self.inner.loco(LocoCommand(float(np.clip(cmd.vx, -vx, vx)), float(np.clip(cmd.vy, -vy, vy)),
                                    float(np.clip(cmd.wz, -wz, wz))))

    def stand(self):
        self.inner.stand()

    def damp(self):
        self.inner.damp()
