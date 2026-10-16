"""Canonical joint layouts and message types shared by every layer.

Conventions (used everywhere above the HAL; drivers convert to vendor units):
- Hand joints are normalized to [0, 1]: 0 = fully open, 1 = fully closed.
  Order matches Unitree's G1_Brainco LeRobot datasets (Thumb, ThumbAux, Index, Middle, Ring, Pinky),
  so those datasets map onto the Revo2 without reordering.
- Body joints are in radians, named as in R1.urdf (26 actuated DoF).
- Sides are "left" / "right".
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Side = Literal["left", "right"]
SIDES: tuple[Side, Side] = ("left", "right")

HAND_JOINTS = ("thumb", "thumb_aux", "index", "middle", "ring", "pinky")
N_HAND = len(HAND_JOINTS)

LEG_JOINTS = tuple(
    f"{s}_{j}_joint"
    for s in SIDES
    for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
)
WAIST_JOINTS = ("waist_roll_joint", "waist_yaw_joint")
ARM_JOINT_SUFFIXES = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")
ARM_JOINTS = {s: tuple(f"{s}_{j}_joint" for j in ARM_JOINT_SUFFIXES) for s in SIDES}
HEAD_JOINTS = ("head_pitch_joint", "head_yaw_joint")

# Logical order used for state/action vectors in our datasets and policies.
# The hardware motor index order is a separate map in configs/r1.yaml.
BODY_JOINTS = LEG_JOINTS + WAIST_JOINTS + ARM_JOINTS["left"] + ARM_JOINTS["right"] + HEAD_JOINTS
N_BODY = len(BODY_JOINTS)  # 26
UPPER_JOINTS = WAIST_JOINTS + ARM_JOINTS["left"] + ARM_JOINTS["right"] + HEAD_JOINTS


def now() -> float:
    return time.monotonic()


@dataclass
class HandCommand:
    """Target finger positions, normalized [0, 1]. `speed` and `force` are normalized [0, 1] and optional."""
    position: np.ndarray
    speed: float | None = None
    force: float | None = None

    def __post_init__(self):
        self.position = np.clip(np.asarray(self.position, dtype=np.float64).reshape(N_HAND), 0.0, 1.0)


@dataclass
class HandState:
    position: np.ndarray                     # normalized [0, 1]
    velocity: np.ndarray | None = None       # normalized units / s
    current: np.ndarray | None = None        # mA as reported by the hand; None if unavailable
    temperature: np.ndarray | None = None    # deg C; None if unavailable
    fault: bool = False
    stamp: float = field(default_factory=now)


@dataclass
class JointCommand:
    """Joint-space PD target for a subset of body joints (by name)."""
    names: tuple[str, ...]
    q: np.ndarray
    dq: np.ndarray | None = None
    kp: np.ndarray | None = None
    kd: np.ndarray | None = None
    tau_ff: np.ndarray | None = None


@dataclass
class BodyState:
    q: np.ndarray                 # (26,) in BODY_JOINTS order
    dq: np.ndarray
    tau: np.ndarray
    base_quat_wxyz: np.ndarray    # IMU orientation
    base_gyro: np.ndarray
    base_acc: np.ndarray
    motor_temp: np.ndarray | None = None
    stamp: float = field(default_factory=now)

    def get(self, names) -> np.ndarray:
        idx = [BODY_JOINTS.index(n) for n in names]
        return self.q[idx]


@dataclass
class LocoCommand:
    """Velocity command for the locomotion layer (built-in Unitree controller or our RL policy)."""
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
