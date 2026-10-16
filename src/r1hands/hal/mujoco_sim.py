"""MuJoCo backend with the same Hand/Body interface as the real drivers.

The base is fixed (robot on a gantry) by default, which matches how levels 1-2 are run on hardware.
`loco()` slides the fixed base kinematically, so level-3 code-as-policy programs can be dry-run. Real
whole-body balance in sim belongs to the RL stack (policies/rl), not this backend.
Objects can be added with `extra_xml` (bodies inserted into the worldbody).
"""
from __future__ import annotations

import threading
import time
import xml.etree.ElementTree as ET

import numpy as np

from r1hands.common.robot_model import REPO
from r1hands.common.types import (
    BODY_JOINTS, N_HAND, BodyState, HandCommand, HandState, JointCommand, LocoCommand, Side, now,
)
from r1hands.hal.base import Body, Hand
from r1hands.sim.build_model import HAND_EFFORT, HAND_RANGE, hand_joint_names
from r1hands.common.types import HAND_JOINTS

MODEL = REPO / "assets" / "r1_revo2" / "r1_revo2.xml"


class MujocoSim:
    def __init__(self, fixed_base: bool = True, extra_xml: str = "", realtime: bool = True, viewer: bool = False):
        import mujoco
        self.mj = mujoco
        root = ET.parse(MODEL).getroot()
        pelvis = root.find("worldbody").find("body")
        if fixed_base:
            fj = pelvis.find("freejoint")
            if fj is not None:
                pelvis.remove(fj)
            pelvis.set("mocap", "true")
        if extra_xml:
            for el in ET.fromstring(f"<w>{extra_xml}</w>"):
                root.find("worldbody").append(el)
        # meshdir is relative to the model file; load from a string with the right asset dir
        root.find("compiler").set("meshdir", str(MODEL.parent))
        self.m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.d = mujoco.MjData(self.m)
        self.fixed_base = fixed_base
        self.lock = threading.RLock()
        self.realtime = realtime
        self.base_vel = np.zeros(3)
        self.qadr = {n: self.m.joint(n).qposadr[0] for n in BODY_JOINTS}
        self.vadr = {n: self.m.joint(n).dofadr[0] for n in BODY_JOINTS}
        self.act = {self.m.actuator(i).name: i for i in range(self.m.nu)}
        mujoco.mj_forward(self.m, self.d)
        self._stop = threading.Event()
        self._viewer = None
        if viewer:
            import mujoco.viewer
            self._viewer = mujoco.viewer.launch_passive(self.m, self.d)
        if realtime:
            threading.Thread(target=self._loop, daemon=True).start()

    def step(self, n: int = 1):
        with self.lock:
            for _ in range(n):
                if self.fixed_base and self.base_vel.any():
                    dt = self.m.opt.timestep
                    yaw_q = self.d.mocap_quat[0].copy()
                    yaw = 2 * np.arctan2(yaw_q[3], yaw_q[0])
                    c, s = np.cos(yaw), np.sin(yaw)
                    self.d.mocap_pos[0][:2] += dt * np.array([c * self.base_vel[0] - s * self.base_vel[1],
                                                              s * self.base_vel[0] + c * self.base_vel[1]])
                    yaw += dt * self.base_vel[2]
                    self.d.mocap_quat[0] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
                self.mj.mj_step(self.m, self.d)
            if self._viewer is not None:
                self._viewer.sync()

    def _loop(self):
        dt = self.m.opt.timestep
        n = max(1, int(0.004 / dt))
        while not self._stop.is_set():
            t0 = time.monotonic()
            self.step(n)
            time.sleep(max(0.0, n * dt - (time.monotonic() - t0)))

    def close(self):
        self._stop.set()


class MujocoHand(Hand):
    def __init__(self, sim: MujocoSim, side: Side):
        self.sim, self.side = sim, side
        self.joints = hand_joint_names(side)
        self.rng = np.array([HAND_RANGE[j] for j in HAND_JOINTS])
        self.acts = [sim.act[f"{side}_{j}"] for j in HAND_JOINTS]
        self.effort = np.array([HAND_EFFORT[j] for j in HAND_JOINTS])

    def connect(self):
        pass

    def close(self):
        pass

    def read(self) -> HandState:
        m, d = self.sim.m, self.sim.d
        with self.sim.lock:
            q = np.array([d.qpos[m.joint(n).qposadr[0]] for n in self.joints]) / self.rng
            frc = np.array([d.actuator_force[a] for a in self.acts])
        # Pseudo-current: torque as a fraction of the effort limit, scaled so a stalled finger reads ~800 mA
        # like the real hand's stall level (placeholder until the level-1 audit measures it).
        return HandState(position=np.clip(q, 0, 1), current=np.abs(frc) / self.effort * 800.0 + 20.0, stamp=now())

    def command(self, cmd: HandCommand):
        with self.sim.lock:
            for a, v, r in zip(self.acts, cmd.position, self.rng):
                self.sim.d.ctrl[a] = v * r


class MujocoBody(Body):
    def __init__(self, sim: MujocoSim):
        self.sim = sim
        self.home = {n: 0.0 for n in BODY_JOINTS}

    def connect(self):
        with self.sim.lock:
            for n in BODY_JOINTS:
                self.sim.d.ctrl[self.sim.act[n]] = self.home[n]

    def close(self):
        self.sim.close()

    def read(self) -> BodyState:
        s, d = self.sim, self.sim.d
        with s.lock:
            q = np.array([d.qpos[s.qadr[n]] for n in BODY_JOINTS])
            dq = np.array([d.qvel[s.vadr[n]] for n in BODY_JOINTS])
            tau = np.array([d.actuator_force[s.act[n]] for n in BODY_JOINTS])
            quat = d.mocap_quat[0].copy() if s.fixed_base else d.qpos[3:7].copy()
        return BodyState(q, dq, tau, quat, np.zeros(3), np.array([0, 0, 9.81]))

    def command_joints(self, cmd: JointCommand):
        with self.sim.lock:
            for n, q in zip(cmd.names, cmd.q):
                self.sim.d.ctrl[self.sim.act[n]] = q

    def release_joints(self):
        self.connect()

    def loco(self, cmd: LocoCommand):
        self.sim.base_vel = np.array([cmd.vx, cmd.vy, cmd.wz])

    def stand(self):
        pass

    def damp(self):
        self.sim.base_vel[:] = 0
