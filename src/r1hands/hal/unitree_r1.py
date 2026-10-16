"""Unitree R1 EDU body driver (unitree_sdk2_python, unitree_hg LowCmd_/LowState_).

Modes
- "arm_sdk" (levels 2 and 3, default): Unitree's built-in controller keeps balance and walks; we blend
  arm, waist-yaw and head targets in through rt/arm_sdk. On R1 the blend weight is `mode_pr` in 0..100
  (on G1 it is slot 29). Walking goes through LocoClient (service "sport").
- "lowlevel" (our own RL whole-body policy): MotionSwitcher releases the built-in controller and we own
  every motor on rt/lowcmd. Only enter this with the robot on a gantry. It is opt-in via lowlevel=True.

IDL slot map (unitree_sdk2 include/unitree/dds_wrapper/robots/r1/defines.h). R1-A5 arm is assumed (5-DoF,
matching R1.urdf). A7 units have wrist pitch/yaw in slots 20-21 / 27-28; set arm="A7" and extend the joint lists.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from r1hands.common.types import (
    BODY_JOINTS, N_BODY, BodyState, JointCommand, LocoCommand,
)
from r1hands.hal.base import Body

SLOT = {
    "left_hip_pitch_joint": 0, "left_hip_roll_joint": 1, "left_hip_yaw_joint": 2, "left_knee_joint": 3,
    "left_ankle_pitch_joint": 4, "left_ankle_roll_joint": 5,
    "right_hip_pitch_joint": 6, "right_hip_roll_joint": 7, "right_hip_yaw_joint": 8, "right_knee_joint": 9,
    "right_ankle_pitch_joint": 10, "right_ankle_roll_joint": 11,
    "waist_roll_joint": 12, "waist_yaw_joint": 13,
    "left_shoulder_pitch_joint": 15, "left_shoulder_roll_joint": 16, "left_shoulder_yaw_joint": 17,
    "left_elbow_joint": 18, "left_wrist_roll_joint": 19,
    "right_shoulder_pitch_joint": 22, "right_shoulder_roll_joint": 23, "right_shoulder_yaw_joint": 24,
    "right_elbow_joint": 25, "right_wrist_roll_joint": 26,
    "head_pitch_joint": 29, "head_yaw_joint": 30,
}
N_SLOTS = 35
ARM_SDK_JOINTS = (
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "waist_yaw_joint", "head_pitch_joint", "head_yaw_joint",
)


class R1Body(Body):
    def __init__(self, network_interface: str = "eth0", level: int = 2, lowlevel: bool = False,
                 rate_hz: float = 250.0, gains: dict | None = None, release_s: float = 2.0, init_dds: bool = True):
        self.iface = network_interface
        self.level = level
        self.lowlevel = lowlevel
        self.rate_hz = rate_hz
        self.release_s = release_s
        self.init_dds = init_dds
        # PD gains per joint group. Starting values (G1 arm_sdk examples use similar); verify on the robot.
        g = {"arm": (60.0, 1.5), "waist": (150.0, 3.0), "head": (20.0, 0.5), "leg": (100.0, 2.0)} | (gains or {})
        self.kp = {n: g[self._group(n)][0] for n in SLOT}
        self.kd = {n: g[self._group(n)][1] for n in SLOT}
        self._targets: dict[str, float] = {}
        self._weight = 0.0
        self._weight_goal = 0.0
        self._state: BodyState | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

    @staticmethod
    def _group(n):
        return "arm" if ("shoulder" in n or "elbow" in n or "wrist" in n) else \
            "waist" if "waist" in n else "head" if "head" in n else "leg"

    # ------------------------------------------------------------ lifecycle
    def connect(self):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC
        if self.init_dds:
            ChannelFactoryInitialize(0, self.iface)
        self._crc = CRC()
        self._msg = unitree_hg_msg_dds__LowCmd_()
        self._sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._sub.Init(self._on_state, 10)
        t0 = time.monotonic()
        while self._state is None:
            if time.monotonic() - t0 > 5:
                raise TimeoutError(f"no rt/lowstate on {self.iface}; check the cable and interface name")
            time.sleep(0.05)
        if self.lowlevel:
            self._release_builtin()
            topic = "rt/lowcmd"
        else:
            topic = "rt/arm_sdk"
            if self.level >= 3:
                from unitree_sdk2py.r1.loco.r1_loco_client import LocoClient
                self._loco = LocoClient()
                self._loco.SetTimeout(10.0)
                self._loco.Init()
        self._pub = ChannelPublisher(topic, LowCmd_)
        self._pub.Init()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="r1-cmd")
        self._thread.start()

    def _release_builtin(self):
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
        msc = MotionSwitcherClient()
        msc.SetTimeout(5.0)
        msc.Init()
        _, result = msc.CheckMode()
        while result and result.get("name"):
            msc.ReleaseMode()
            time.sleep(1.0)
            _, result = msc.CheckMode()

    def close(self):
        if not self.lowlevel and hasattr(self, "_thread"):
            self.release_joints()
            time.sleep(self.release_s + 0.1)
        self._stop.set()

    # ------------------------------------------------------------ state
    def _on_state(self, m):
        q = np.array([m.motor_state[SLOT[n]].q for n in BODY_JOINTS])
        dq = np.array([m.motor_state[SLOT[n]].dq for n in BODY_JOINTS])
        tau = np.array([m.motor_state[SLOT[n]].tau_est for n in BODY_JOINTS])
        temp = np.array([np.max(m.motor_state[SLOT[n]].temperature) for n in BODY_JOINTS], float)
        imu = m.imu_state
        st = BodyState(q, dq, tau, np.array(imu.quaternion), np.array(imu.gyroscope), np.array(imu.accelerometer),
                       motor_temp=temp)
        with self._lock:
            self._state = st
            self._mode_machine = m.mode_machine

    def read(self) -> BodyState:
        with self._lock:
            return self._state

    # ------------------------------------------------------------ commands
    def command_joints(self, cmd: JointCommand):
        allowed = set(SLOT) if self.lowlevel else set(ARM_SDK_JOINTS)
        bad = [n for n in cmd.names if n not in allowed]
        if bad:
            raise ValueError(f"joints not commandable in {'lowlevel' if self.lowlevel else 'arm_sdk'} mode: {bad}")
        with self._lock:
            if not self._targets:  # first command: start from the measured pose so the blend has no jump
                s = self._state
                self._targets = {n: float(s.q[BODY_JOINTS.index(n)]) for n in allowed}
            for n, q in zip(cmd.names, cmd.q):
                self._targets[n] = float(q)
            self._weight_goal = 1.0

    def release_joints(self):
        with self._lock:
            self._weight_goal = 0.0

    def _loop(self):
        period = 1.0 / self.rate_hz
        dw = period / self.release_s
        while not self._stop.is_set():
            t0 = time.monotonic()
            with self._lock:
                self._weight += float(np.clip(self._weight_goal - self._weight, -dw, dw))
                if self._weight <= 0 and self._weight_goal == 0:
                    self._targets = {}
                targets = dict(self._targets)
                weight = self._weight
                mm = self._mode_machine
            if targets or self.lowlevel:
                m = self._msg
                m.mode_machine = mm
                if not self.lowlevel:
                    m.mode_pr = int(np.clip(weight, 0, 1) * 100)
                for n, q in targets.items():
                    mc = m.motor_cmd[SLOT[n]]
                    mc.mode, mc.q, mc.dq, mc.tau = 1, q, 0.0, 0.0
                    mc.kp, mc.kd = self.kp[n], self.kd[n]
                m.crc = self._crc.Crc(m)
                self._pub.Write(m)
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    # ------------------------------------------------------------ locomotion (built-in controller)
    def _need_loco(self):
        if not hasattr(self, "_loco"):
            raise RuntimeError("locomotion needs level 3 in arm_sdk mode")

    def loco(self, cmd: LocoCommand):
        self._need_loco()
        self._loco.Move(cmd.vx, cmd.vy, cmd.wz)

    def stand(self):
        self._need_loco()
        self._loco.Start()

    def damp(self):
        if hasattr(self, "_loco"):
            self._loco.Damp()
        self.release_joints()
