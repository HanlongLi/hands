"""Numpy-only URDF kinematics for R1 arm chains (pelvis -> wrist_roll_link -> Revo2 palm).

The R1 arm has 5 DoF, so a full 6-D hand pose is generally unreachable. `ik` solves palm position
plus (optionally) the palm approach axis with weighted damped least squares. Waist joints can be
added to the chain for extra reach (level 3 uses them).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from r1hands.common.robot_model import R1_URDF
from r1hands.common.types import ARM_JOINTS, WAIST_JOINTS, Side


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(axis, a):
    k = np.asarray(axis, float)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


@dataclass
class _J:
    name: str
    type: str
    parent: str
    child: str
    T: np.ndarray       # origin transform (4x4)
    axis: np.ndarray
    lower: float
    upper: float


@lru_cache
def _joints(urdf=str(R1_URDF)) -> dict[str, _J]:
    out = {}
    for j in ET.parse(urdf).getroot().iter("joint"):
        o = j.find("origin")
        xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
        rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = _rpy(*rpy), xyz
        ax = j.find("axis")
        lim = j.find("limit")
        out[j.get("name")] = _J(j.get("name"), j.get("type"), j.find("parent").get("link"), j.find("child").get("link"),
                                T, np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else np.zeros(3),
                                float(lim.get("lower")) if lim is not None else 0.0,
                                float(lim.get("upper")) if lim is not None else 0.0)
    return out


def chain_to(link: str, root: str = "pelvis_link", urdf: str = str(R1_URDF)) -> list[_J]:
    by_child = {j.child: j for j in _joints(urdf).values()}
    chain = []
    while link != root:
        j = by_child[link]
        chain.append(j)
        link = j.parent
    return chain[::-1]


class ArmKinematics:
    """FK/IK for one arm. `tool` is the palm frame offset in wrist_roll_link coordinates (measure on the robot)."""

    def __init__(self, side: Side, tool_xyz=(0.13, 0.0, 0.0), use_waist: bool = False,
                 urdf: str | None = None, root: str = "pelvis_link", tip: str | None = None,
                 active: list[str] | None = None):
        """Defaults describe the R1 arm. Pass urdf/root/tip/active to use another robot (e.g. G1 for retargeting)."""
        self.side = side
        urdf = str(urdf or R1_URDF)
        self.chain = chain_to(tip or f"{side}_wrist_roll_link", root, urdf)
        self.active = active or (list(ARM_JOINTS[side]) + (list(WAIST_JOINTS) if use_waist else []))
        self.tool = np.eye(4)
        self.tool[:3, 3] = tool_xyz
        js = _joints(urdf)
        self.lo = np.array([js[n].lower for n in self.active])
        self.hi = np.array([js[n].upper for n in self.active])

    def shoulder_position(self) -> np.ndarray:
        """Position of the first active joint (shoulder) in the root frame; constant for an arm chain."""
        return self.fk({})[1][0][1]

    def fk(self, q_named: dict[str, float]) -> tuple[np.ndarray, list[tuple[str, np.ndarray, np.ndarray]]]:
        """Palm pose in pelvis frame, plus (joint, world_pos, world_axis) for each active joint."""
        T = np.eye(4)
        frames = []
        for j in self.chain:
            T = T @ j.T
            if j.type == "revolute":
                if j.name in self.active:
                    frames.append((j.name, T[:3, 3].copy(), T[:3, :3] @ j.axis))
                R = np.eye(4)
                R[:3, :3] = _axis_angle(j.axis, q_named.get(j.name, 0.0))
                T = T @ R
        return T @ self.tool, frames

    def ik(self, target_pos, q0: dict[str, float], approach=None, iters=200, damping=1e-2, tol=2e-3):
        """Solve for palm position; approach (palm x-axis direction) is matched as well as the remaining
        DoF allow, in the null space of the position task. A final position-only pass guarantees the
        orientation preference never costs position accuracy. Returns (q_named, pos_err_m)."""
        q = dict(q0)
        x = np.array([q.get(n, 0.0) for n in self.active])
        d = None if approach is None else np.asarray(approach, float) / np.linalg.norm(approach)
        x0 = x.copy()
        if d is not None:
            x = self._solve(x, q, target_pos, d, iters, damping, tol)
        x = self._solve(x, q, target_pos, None, iters, damping, tol)
        err = self._err(x, q, target_pos)
        if d is not None and err > tol:  # orientation pass ended in a bad basin: position-only from q0
            x_pos = self._solve(x0, q, target_pos, None, iters, damping, tol)
            if self._err(x_pos, q, target_pos) < err:
                x = x_pos
        for n, v in zip(self.active, x):
            q[n] = v
        return q, self._err(x, q, target_pos)

    def _err(self, x, q, target_pos):
        for n, v in zip(self.active, x):
            q[n] = v
        return float(np.linalg.norm(np.asarray(target_pos) - self.fk(q)[0][:3, 3]))

    def _solve(self, x, q, target_pos, d, iters, damping, tol):
        n = len(self.active)
        for _ in range(iters):
            for name, v in zip(self.active, x):
                q[name] = v
            T, frames = self.fk(q)
            p = T[:3, 3]
            ep = np.asarray(target_pos) - p
            Jp = np.stack([np.cross(ax, p - o) for _, o, ax in frames], 1)
            Jp_pinv = Jp.T @ np.linalg.inv(Jp @ Jp.T + damping * np.eye(3))
            dx = Jp_pinv @ ep
            if d is not None:
                a = T[:3, 0]
                Jr = np.stack([np.cross(ax, a) for _, _, ax in frames], 1)
                N = np.eye(n) - Jp_pinv @ Jp
                JrN = Jr @ N
                dx = dx + N @ (JrN.T @ np.linalg.solve(JrN @ JrN.T + damping * np.eye(3), (d - a) - Jr @ dx))
            if np.linalg.norm(ep) < tol and (d is None or np.linalg.norm(dx) < 1e-4):
                break
            dx *= min(1.0, 0.2 / (np.abs(dx).max() + 1e-12))
            x = np.clip(x + dx, self.lo, self.hi)
        return x
