"""BrainCo Revo2 (Basic) drivers. Two transports, same interface:

- Revo2Hand: direct, via `bc-stark-sdk` (>=2.0.7, Linux x86_64/aarch64 wheels only; no macOS).
  RS485/Modbus at 460800 baud. Default slave ids: left 0x7E, right 0x7F. One serial port per hand.
- Revo2DdsHand: via Unitree's `brainco_hand_service` bridge (C++, polls both hands at 100 Hz) on DDS topics
  rt/brainco/{side}/cmd|state (unitree_go MotorCmds_/MotorStates_). This is what xr_teleoperate uses for
  R1 + BrainCo, so teleop data collection and our stack can share one hand process.

SDK facts (bc_stark_sdk 2.0.7, names checked against the Linux wheel 2026-10-14; import is
`from bc_stark_sdk import main_mod`): asyncio API; positions 0..1000
(0 = open) in FingerUnitMode.Normalized; finger order Thumb, ThumbAux, Index, Middle, Ring, Pinky;
get_motor_status() -> positions/speeds/currents/states (Unknown, Idle, Running, Stall, Turbo). No temperature.
"""
from __future__ import annotations

import asyncio
import threading
import time

import numpy as np

from r1hands.common.types import N_HAND, HandCommand, HandState, Side, now
from r1hands.hal.base import Hand

DEFAULT_IDS = {"left": 0x7E, "right": 0x7F}


class Revo2Hand(Hand):
    """Direct SDK driver. A private asyncio loop in a thread owns the port: it polls status at `poll_hz`
    and sends the most recent command. `command()` / `read()` never block on the bus."""

    def __init__(self, side: Side, port: str = "/dev/ttyUSB0", slave_id: int | None = None,
                 baud: int = 460800, poll_hz: float = 100.0, protected_current_mA: int | None = None):
        self.side = side
        self.port, self.baud, self.poll_hz = port, baud, poll_hz
        self.slave_id = DEFAULT_IDS[side] if slave_id is None else slave_id
        self.protected_current = protected_current_mA
        self._lock = threading.Lock()
        self._cmd: tuple[np.ndarray, np.ndarray] | None = None
        self._state = HandState(position=np.zeros(N_HAND), fault=True, stamp=float("nan"))  # placeholder: no data yet
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._err: BaseException | None = None
        self.info = None

    # -- lifecycle
    def connect(self):
        self._thread = threading.Thread(target=lambda: asyncio.run(self._run()), daemon=True,
                                        name=f"revo2-{self.side}")
        self._thread.start()
        if not self._ready.wait(10.0):
            raise TimeoutError(f"Revo2 {self.side}: no response on {self.port} id={self.slave_id:#x}")
        if self._err:
            raise self._err

    def close(self):
        self._stop.set()
        if hasattr(self, "_thread"):
            self._thread.join(2.0)

    # -- sync API
    def read(self) -> HandState:
        with self._lock:
            return self._state

    def command(self, cmd: HandCommand):
        speed = 1.0 if cmd.speed is None else float(np.clip(cmd.speed, 0.001, 1.0))
        with self._lock:
            self._cmd = (np.round(cmd.position * 1000).astype(int), np.full(N_HAND, int(speed * 1000)))

    # -- bus thread
    @staticmethod
    async def _probe(ctx, slave_id):
        try:
            return await asyncio.wait_for(ctx.get_device_info(slave_id), 1.0)
        except Exception:
            return None

    async def _run(self):
        try:
            from bc_stark_sdk import main_mod as libstark
        except ImportError as e:  # pragma: no cover
            self._err = ImportError("pip install 'bc-stark-sdk>=2.0.7' (Linux only)")
            self._ready.set()
            raise e
        ctx = None
        try:
            baud = getattr(libstark.Baudrate, f"Baud{self.baud}")
            try:
                ctx = await libstark.modbus_open(self.port, baud)
            except Exception as e:
                raise ConnectionError(f"cannot open {self.port}: {e}") from e
            if not ctx:
                raise ConnectionError(f"cannot open {self.port}")
            self.info = await self._probe(ctx, self.slave_id)
            if not self.info:  # ports swapped or ids changed: try the other default id before giving up
                for alt in (DEFAULT_IDS["left"], DEFAULT_IDS["right"]):
                    if alt != self.slave_id and (info := await self._probe(ctx, alt)):
                        print(f"[revo2 {self.side}] no hand at id {self.slave_id:#x} on {self.port}, "
                              f"found id {alt:#x} instead. Check which hand is on this port!")
                        self.slave_id, self.info = alt, info
                        break
            if not self.info:
                raise ConnectionError(f"no Revo2 at ids 0x7e/0x7f on {self.port} ({self.baud} baud)")
            await ctx.set_finger_unit_mode(self.slave_id, libstark.FingerUnitMode.Normalized)
            if self.protected_current is not None:
                await ctx.set_finger_protected_currents(self.slave_id, [self.protected_current] * N_HAND)
            period = 1.0 / self.poll_hz
            last_sent = None
            while not self._stop.is_set():
                t0 = time.monotonic()
                with self._lock:
                    cmd = self._cmd
                if cmd is not None and (last_sent is None or not (np.array_equal(cmd[0], last_sent[0]) and np.array_equal(cmd[1], last_sent[1]))):
                    await ctx.set_finger_positions_and_speeds(self.slave_id, cmd[0].tolist(), cmd[1].tolist())
                    last_sent = cmd
                st = await ctx.get_motor_status(self.slave_id)
                states = [str(s) for s in st.states]
                with self._lock:
                    self._state = HandState(
                        position=np.asarray(st.positions, float) / 1000.0,
                        velocity=np.asarray(st.speeds, float) / 1000.0,
                        current=np.asarray(st.currents, float),
                        fault=any("Stall" in s for s in states),
                        stamp=now(),
                    )
                # connect() returns only after the first real telemetry. Before that, read() returns a
                # placeholder, and anything that holds "the current position" would command it.
                self._ready.set()
                await asyncio.sleep(max(0.0, period - (time.monotonic() - t0)))
        except BaseException as e:
            self._err = e
            self._ready.set()
        finally:
            if ctx:
                libstark.modbus_close(ctx)


class Revo2DdsHand(Hand):
    """Talks to Unitree's brainco_hand_service over DDS. q and dq are 0..1; state tau_est carries current/1000."""

    def __init__(self, side: Side, network_interface: str | None = None, init_dds: bool = True):
        self.side = side
        self.iface = network_interface
        self.init_dds = init_dds
        self._state = HandState(position=np.zeros(N_HAND))
        self._lock = threading.Lock()

    def connect(self):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
        if self.init_dds:
            ChannelFactoryInitialize(0, self.iface) if self.iface else ChannelFactoryInitialize(0)
        self._msg = MotorCmds_()
        self._msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(N_HAND)]
        self._pub = ChannelPublisher(f"rt/brainco/{self.side}/cmd", MotorCmds_)
        self._pub.Init()
        self._sub = ChannelSubscriber(f"rt/brainco/{self.side}/state", MotorStates_)
        self._sub.Init(self._on_state, 10)

    def _on_state(self, msg):
        s = msg.states[:N_HAND]
        with self._lock:
            self._state = HandState(position=np.array([m.q for m in s]), velocity=np.array([m.dq for m in s]),
                                    current=np.array([m.tau_est for m in s]) * 1000.0, stamp=now())

    def close(self):
        pass

    def read(self) -> HandState:
        with self._lock:
            return self._state

    def command(self, cmd: HandCommand):
        speed = 1.0 if cmd.speed is None else float(cmd.speed)
        for i in range(N_HAND):
            self._msg.cmds[i].q = float(cmd.position[i])
            self._msg.cmds[i].dq = speed
        self._pub.Write(self._msg)
