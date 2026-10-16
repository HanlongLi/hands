"""Skill API: the one surface that teleop scripts, IL/RL policy runners and LLM-written code all call.

Levels gate what is exposed:
    1 = hands only    2 = + arms (reach/IK)    3 = + locomotion, waist, head
Calls block until done (or timeout) and return a short dict that an LLM can read.
"""
from __future__ import annotations

import time

import numpy as np

from r1hands.common.types import (
    ARM_JOINTS, HAND_JOINTS, HandCommand, JointCommand, LocoCommand, Side,
)
from r1hands.control.kinematics import ArmKinematics
from r1hands.hal.base import Robot

# Initial grasp library in normalized Revo2 units [thumb, thumb_aux, index, middle, ring, pinky].
# Starting values only: tune them on the real hand in level 1 and save the result to configs/grasps.yaml.
GRASPS = {
    "open": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "power": [0.75, 0.55, 0.95, 0.95, 0.95, 0.95],
    "pinch": [0.55, 0.85, 0.60, 0.10, 0.10, 0.10],
    "tripod": [0.50, 0.85, 0.55, 0.55, 0.10, 0.10],
    "lateral": [0.75, 0.10, 0.85, 0.85, 0.85, 0.85],
    "hook": [0.0, 0.0, 0.70, 0.70, 0.70, 0.70],
    "point": [0.60, 0.50, 0.0, 1.0, 1.0, 1.0],
    "fist": [0.80, 0.60, 1.0, 1.0, 1.0, 1.0],
}


class Skills:
    def __init__(self, robot: Robot, level: int = 1, rate_hz: float = 50.0, tool_xyz=(0.13, 0.0, 0.0),
                 grasps: dict | None = None):
        self.robot = robot
        self.level = level
        self.dt = 1.0 / rate_hz
        self.grasps = {**GRASPS, **(grasps or {})}
        self.kin = {s: ArmKinematics(s, tool_xyz) for s in ("left", "right")} if level >= 2 else {}
        self.log: list[dict] = []

    # ---------------------------------------------------------------- level 1: hands
    def set_fingers(self, side: Side, position, duration: float = 1.0) -> dict:
        """Move the 6 Revo2 motors to `position` (normalized 0 = open .. 1 = closed) over `duration` s.
        Order: thumb, thumb_aux, index, middle, ring, pinky."""
        hand = self.robot.hands[side]
        start = hand.read().position
        goal = np.clip(np.asarray(position, float), 0, 1)
        n = max(1, int(duration / self.dt))
        for i in range(1, n + 1):
            hand.command(HandCommand(start + (goal - start) * i / n))
            time.sleep(self.dt)
        s = self._settle(hand)
        res = {"side": side, "reached": np.round(s.position, 3).tolist(),
               "blocked_fingers": [HAND_JOINTS[i] for i in np.flatnonzero(goal - s.position > 0.08)]}
        self.log.append({"skill": "set_fingers", **res})
        return res

    def _settle(self, hand, timeout: float = 0.8, eps: float = 0.005):
        """Wait until the fingers stop moving (reached the goal or stopped on contact)."""
        prev = hand.read().position
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            time.sleep(0.05)
            s = hand.read()
            if np.abs(s.position - prev).max() < eps:
                return s
            prev = s.position
        return hand.read()

    def grasp(self, side: Side, kind: str = "power", duration: float = 1.0) -> dict:
        """Close the hand into a named grasp. Kinds: open, power, pinch, tripod, lateral, hook, point, fist.
        `blocked_fingers` in the result means those fingers stopped early, usually on the object (contact)."""
        if kind not in self.grasps:
            raise ValueError(f"unknown grasp {kind!r}; choose from {sorted(self.grasps)}")
        return self.set_fingers(side, self.grasps[kind], duration)

    def open_hand(self, side: Side, duration: float = 0.8) -> dict:
        """Fully open the hand."""
        return self.set_fingers(side, self.grasps["open"], duration)

    def hand_state(self, side: Side) -> dict:
        """Current finger positions, motor currents and temperatures (if the hand reports them)."""
        s = self.robot.hands[side].read()
        out = {"position": np.round(s.position, 3).tolist()}
        if s.current is not None:
            out["current_mA"] = np.round(s.current).tolist()
        if s.temperature is not None:
            out["temp_C"] = np.round(s.temperature, 1).tolist()
        return out

    # ---------------------------------------------------------------- level 2: arms
    def _need(self, level: int, name: str):
        if self.level < level:
            raise PermissionError(f"{name} requires level {level}; running at level {self.level}")

    def palm_position(self, side: Side) -> list[float]:
        """Current palm position (m) in the pelvis frame: x forward, y left, z up."""
        self._need(2, "palm_position")
        q = dict(zip(ARM_JOINTS[side], self.robot.body.read().get(ARM_JOINTS[side])))
        return np.round(self.kin[side].fk(q)[0][:3, 3], 4).tolist()

    def reach(self, side: Side, xyz, approach=None, duration: float = 2.0) -> dict:
        """Move the palm to xyz (m, pelvis frame), optionally pointing the fingers along `approach` (3-vector).
        The 5-DoF arm cannot reach every orientation; position wins. Returns the IK error in metres."""
        self._need(2, "reach")
        body = self.robot.body
        names = ARM_JOINTS[side]
        q0 = dict(zip(names, body.read().get(names)))
        q_goal, err = self.kin[side].ik(np.asarray(xyz, float), q0, approach=approach)
        if err > 0.03:
            return {"ok": False, "reason": f"unreachable: IK error {err:.3f} m", "palm": self.palm_position(side)}
        start = np.array([q0[n] for n in names])
        goal = np.array([q_goal[n] for n in names])
        n = max(1, int(duration / self.dt))
        for i in range(1, n + 1):
            a = 0.5 - 0.5 * np.cos(np.pi * i / n)  # smooth start/stop
            body.command_joints(JointCommand(names, start + (goal - start) * a))
            time.sleep(self.dt)
        res = {"ok": True, "ik_err_m": round(err, 4), "palm": self.palm_position(side)}
        self.log.append({"skill": "reach", "side": side, "target": list(map(float, xyz)), **res})
        return res

    def release_arms(self) -> dict:
        """Return arm control to the default controller (arms go back to its nominal pose)."""
        self._need(2, "release_arms")
        self.robot.body.release_joints()
        return {"ok": True}

    # ---------------------------------------------------------------- level 3: whole body
    def walk(self, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0, duration: float = 1.0) -> dict:
        """Walk with body velocity (vx forward m/s, vy left m/s, wz yaw rad/s) for `duration` s, then stop."""
        self._need(3, "walk")
        t_end = time.monotonic() + duration
        while time.monotonic() < t_end:
            self.robot.body.loco(LocoCommand(vx, vy, wz))
            time.sleep(self.dt)
        self.robot.body.loco(LocoCommand())
        self.log.append({"skill": "walk", "v": [vx, vy, wz], "duration": duration})
        return {"ok": True}

    def stop(self) -> dict:
        """Stop walking (zero velocity). Does not damp motors."""
        self._need(3, "stop")
        self.robot.body.loco(LocoCommand())
        return {"ok": True}

    def set_waist(self, yaw: float = 0.0, duration: float = 1.0) -> dict:
        """Waist yaw (rad, + turns left). Extends the reach sideways without stepping.
        (R1-A5 waist roll is not commandable through the arm_sdk blend.)"""
        self._need(3, "set_waist")
        names = ("waist_yaw_joint",)
        start = self.robot.body.read().get(names)
        goal = np.array([yaw])
        n = max(1, int(duration / self.dt))
        for i in range(1, n + 1):
            self.robot.body.command_joints(JointCommand(names, start + (goal - start) * i / n))
            time.sleep(self.dt)
        return {"ok": True}

    def look(self, pitch: float = 0.0, yaw: float = 0.0) -> dict:
        """Point the head camera (rad)."""
        self._need(3, "look")
        self.robot.body.command_joints(JointCommand(("head_pitch_joint", "head_yaw_joint"), np.array([pitch, yaw])))
        return {"ok": True}

    def wait(self, seconds: float) -> dict:
        """Sleep."""
        time.sleep(seconds)
        return {"ok": True}

    # ---------------------------------------------------------------- introspection for code-as-policy
    def available(self) -> list[str]:
        base = ["set_fingers", "grasp", "open_hand", "hand_state", "wait"]
        if self.level >= 2:
            base += ["palm_position", "reach", "release_arms"]
        if self.level >= 3:
            base += ["walk", "stop", "set_waist", "look"]
        return base
