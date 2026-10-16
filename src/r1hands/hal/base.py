"""Hardware abstraction. Everything above this layer talks only to these interfaces.

Backends: `mock` (pure numpy, for tests/CI), `mujoco` (sim), `revo2` + `unitree_r1` (real hardware).
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field

from r1hands.common.types import (
    BodyState, HandCommand, HandState, JointCommand, LocoCommand, Side,
)


class Hand(abc.ABC):
    side: Side

    @abc.abstractmethod
    def connect(self) -> None: ...

    @abc.abstractmethod
    def close(self) -> None: ...

    @abc.abstractmethod
    def read(self) -> HandState: ...

    @abc.abstractmethod
    def command(self, cmd: HandCommand) -> None: ...

    def hold(self):
        """Cancel remaining motion at the measured pose; return the held target."""
        position = self.read().position.copy()
        self.command(HandCommand(position))
        return position

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.close()


class Body(abc.ABC):
    """R1 body. Upper-body joint commands blend over the locomotion controller (built-in or ours)."""

    @abc.abstractmethod
    def connect(self) -> None: ...

    @abc.abstractmethod
    def close(self) -> None: ...

    @abc.abstractmethod
    def read(self) -> BodyState: ...

    @abc.abstractmethod
    def command_joints(self, cmd: JointCommand) -> None:
        """PD targets for arm / waist / head joints. Leg joints only in full low-level mode."""

    @abc.abstractmethod
    def release_joints(self) -> None:
        """Hand control of the upper body back to the locomotion controller (smoothly)."""

    @abc.abstractmethod
    def loco(self, cmd: LocoCommand) -> None: ...

    @abc.abstractmethod
    def stand(self) -> None: ...

    @abc.abstractmethod
    def damp(self) -> None:
        """Emergency: all motors to damping mode."""


@dataclass
class Robot:
    hands: dict[Side, Hand] = field(default_factory=dict)
    body: Body | None = None

    def connect(self):
        for h in self.hands.values():
            h.connect()
        if self.body is not None:
            self.body.connect()
        return self

    def close(self):
        for h in self.hands.values():
            h.close()
        if self.body is not None:
            self.body.close()

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.close()
