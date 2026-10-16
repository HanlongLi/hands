"""R1 model facts read straight from the URDF, so limits are never hand-copied."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
R1_URDF = REPO / "assets" / "r1_description" / "R1.urdf"


@lru_cache
def joint_limits(urdf: str | Path = R1_URDF) -> dict[str, dict[str, float]]:
    """{joint: {lower, upper, effort, velocity}} for every revolute joint."""
    root = ET.parse(urdf).getroot()
    out = {}
    for j in root.iter("joint"):
        if j.get("type") != "revolute":
            continue
        lim = j.find("limit")
        out[j.get("name")] = {k: float(lim.get(k)) for k in ("lower", "upper", "effort", "velocity")}
    return out


def body_limits(margin: float = 0.05, **kw):
    """BodyLimits with a safety margin (rad) inside the URDF range."""
    from r1hands.safety.guard import BodyLimits
    lim = joint_limits()
    return BodyLimits(q_min={n: v["lower"] + margin for n, v in lim.items()},
                      q_max={n: v["upper"] - margin for n, v in lim.items()}, **kw)
