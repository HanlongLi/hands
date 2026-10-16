"""Build a guarded Robot from configs/r1.yaml for a given backend and capability level."""
from __future__ import annotations

from pathlib import Path

import yaml

from r1hands.common.robot_model import REPO, body_limits
from r1hands.hal.base import Robot
from r1hands.safety.guard import BodyGuard, HandGuard, HandLimits

CONFIG = REPO / "configs" / "r1.yaml"


def load_config(path: str | Path = CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_robot(backend: str = "mock", level: int = 1, sides=("left", "right"), config: str | Path = CONFIG,
               sim_kwargs: dict | None = None, remote: str | None = None) -> Robot:
    """level 1: hands only. level >= 2: hands + body (arms; level 3 also locomotion)."""
    cfg = load_config(config)
    hl = HandLimits(**cfg.get("hand_limits", {}))
    if backend == "mock":
        from r1hands.hal.mock import MockBody, MockHand
        hands = {s: MockHand(s) for s in sides}
        body = MockBody() if level >= 2 else None
    elif backend == "remote":
        # Hands served by r1hands.panel.hand_server; the guard runs there, next to the hardware.
        from r1hands.hal.remote import RemoteHand, RemoteLink
        link = RemoteLink(remote or cfg.get("hand_server", "ws://192.168.123.164:8765/ws"))
        return Robot(hands={s: RemoteHand(link, s) for s in sides})
    elif backend == "sim":
        from r1hands.hal.mujoco_sim import MujocoBody, MujocoHand, MujocoSim
        sim = MujocoSim(**(sim_kwargs or {}))
        hands = {s: MujocoHand(sim, s) for s in sides}
        body = MujocoBody(sim)  # always present in sim (it owns the sim thread)
    elif backend == "real":
        from r1hands.hal.revo2 import Revo2DdsHand, Revo2Hand
        if cfg.get("hand_transport", "sdk") == "dds":
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            ChannelFactoryInitialize(0, cfg["body"]["network_interface"])
            hands = {s: Revo2DdsHand(s, init_dds=False) for s in sides}
        else:
            hands = {s: Revo2Hand(s, **cfg["hands"][s]) for s in sides}
        body = None
        if level >= 2:
            from r1hands.hal.unitree_r1 import R1Body
            dds_ready = cfg.get("hand_transport", "sdk") == "dds"
            body = R1Body(level=level, init_dds=not dds_ready, **cfg["body"])
    else:
        raise ValueError(backend)
    hands = {s: HandGuard(h, hl) for s, h in hands.items()}
    if body is not None:
        body = BodyGuard(body, body_limits(**cfg.get("body_limits", {})))
    return Robot(hands=hands, body=body)
