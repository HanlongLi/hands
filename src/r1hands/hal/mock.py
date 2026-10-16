"""Numpy-only mock backend: first-order joint lag, plus a fake object that blocks finger closure.

The fake contact makes the safety/telemetry code testable without hardware: a blocked finger keeps
drawing current and heats up, which is the failure mode the Revo2 firmware protection guards against.
"""
from __future__ import annotations

import numpy as np

from r1hands.common.types import (
    BODY_JOINTS, N_BODY, N_HAND, BodyState, HandCommand, HandState, JointCommand, LocoCommand, Side, now,
)
from r1hands.hal.base import Body, Hand


class MockHand(Hand):
    def __init__(self, side: Side, tau: float = 0.15, block_at: np.ndarray | None = None):
        self.side = side
        self.tau = tau
        # Per-finger closure where a virtual object stops the finger (1.0 = never blocked).
        self.block_at = np.ones(N_HAND) if block_at is None else np.asarray(block_at, float)
        self._q = np.zeros(N_HAND)
        self._target = np.zeros(N_HAND)
        self._temp = np.full(N_HAND, 30.0)
        self._t = None
        self.connected = False

    def connect(self):
        self.connected = True
        self._t = now()

    def close(self):
        self.connected = False

    def _step(self):
        t = now()
        dt = max(t - self._t, 0.0)
        self._t = t
        a = 1.0 - np.exp(-dt / self.tau)
        self._q += a * (self._target - self._q)
        self._q = np.minimum(self._q, self.block_at)
        blocked = (self._target > self.block_at + 1e-3) & (self._q >= self.block_at - 1e-3)
        self._current = np.where(blocked, 800.0, 80.0 * np.abs(self._target - self._q) + 20.0)
        self._temp += dt * (self._current / 800.0 * 0.5 - (self._temp - 30.0) * 0.01)

    def read(self) -> HandState:
        self._step()
        return HandState(position=self._q.copy(), current=self._current.copy(), temperature=self._temp.copy())

    def command(self, cmd: HandCommand):
        self._step()
        self._target = cmd.position.copy()


class MockBody(Body):
    def __init__(self, tau: float = 0.2):
        self.tau = tau
        self._q = np.zeros(N_BODY)
        self._target = np.zeros(N_BODY)
        self._held = np.zeros(N_BODY, bool)
        self.base_vel = np.zeros(3)
        self.mode = "idle"
        self._t = now()

    def connect(self):
        self._t = now()
        self.mode = "stand"

    def close(self):
        pass

    def _step(self):
        t = now()
        dt = t - self._t
        self._t = t
        a = 1.0 - np.exp(-dt / self.tau)
        tgt = np.where(self._held, self._target, 0.0)
        self._q += a * (tgt - self._q)

    def read(self) -> BodyState:
        self._step()
        z = np.zeros(3)
        return BodyState(q=self._q.copy(), dq=np.zeros(N_BODY), tau=np.zeros(N_BODY),
                         base_quat_wxyz=np.array([1.0, 0, 0, 0]), base_gyro=z, base_acc=np.array([0, 0, 9.81]))

    def command_joints(self, cmd: JointCommand):
        self._step()
        for n, q in zip(cmd.names, cmd.q):
            i = BODY_JOINTS.index(n)
            self._target[i] = q
            self._held[i] = True

    def release_joints(self):
        self._held[:] = False

    def loco(self, cmd: LocoCommand):
        self.base_vel = np.array([cmd.vx, cmd.vy, cmd.wz])

    def stand(self):
        self.mode = "stand"

    def damp(self):
        self.mode = "damp"
        self.base_vel[:] = 0
