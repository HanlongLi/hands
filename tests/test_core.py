import time

import numpy as np
import pytest

from r1hands.common.types import HandCommand
from r1hands.control.kinematics import ArmKinematics
from r1hands.control.skills import Skills
from r1hands.hal.factory import make_robot
from r1hands.hal.mock import MockHand
from r1hands.policies.cap.agent import UnsafeCode, check_code, run_sandboxed
from r1hands.retarget.hand import HandRetargeter, hand_features
from r1hands.safety.guard import HandGuard, HandLimits


def test_ik_reaches_reachable_points():
    k = ArmKinematics("left")
    p0 = k.fk({"left_elbow_joint": -1.0})[0][:3, 3]
    for d in ([0.05, 0, 0.05], [0.1, 0.05, 0.1], [0, -0.1, 0.05]):
        _, err = k.ik(p0 + d, {})
        assert err < 3e-3


def test_ik_respects_limits():
    k = ArmKinematics("right")
    q, _ = k.ik([0.0, -0.2, -0.6], {})  # far and below: unreachable
    for n, v in q.items():
        i = k.active.index(n)
        assert k.lo[i] - 1e-9 <= v <= k.hi[i] + 1e-9


def test_hand_guard_rate_limit():
    g = HandGuard(MockHand("left"), HandLimits(max_rate=1.0))
    g.connect()
    g.command(HandCommand(np.ones(6)))
    assert g._last_target.max() < 0.2  # one step cannot jump to closed


def test_hand_guard_stall_backoff():
    h = MockHand("left", tau=0.01, block_at=np.array([1, 1, 0.4, 1, 1, 1]))
    g = HandGuard(h, HandLimits(max_rate=100, stall_s=0.2))
    g.connect()
    t_end = time.monotonic() + 0.6
    while time.monotonic() < t_end:
        g.command(HandCommand(np.ones(6)))
        time.sleep(0.01)
    assert any("stall" in e for e in g.events)
    assert g._last_target[2] < 0.4  # index backed off below the contact


def test_skills_level_gating():
    with make_robot("mock", level=1) as r:
        s = Skills(r, level=1)
        s.grasp("left", "pinch", duration=0.1)
        with pytest.raises(PermissionError):
            s.walk(0.1, duration=0.1)


def test_reach_mock():
    with make_robot("mock", level=2) as r:
        s = Skills(r, level=2)
        p = s.palm_position("left")
        res = s.reach("left", [p[0] + 0.05, p[1], p[2] + 0.05], duration=0.2)
        assert res["ok"]


def _synthetic_hand(curl: float):
    """Planar-ish synthetic hand: fingers along +x, curling toward -z by `curl` rad per joint."""
    kp = np.zeros((21, 3))
    bases = {5: 0.02, 9: 0.0, 13: -0.02, 17: -0.04}
    for b, y in bases.items():
        p = np.array([0.09, y, 0.0])
        kp[b] = p
        d = np.array([1.0, 0, 0])
        for j in range(1, 4):
            ang = curl * j
            d = np.array([np.cos(ang), 0, -np.sin(ang)])
            p = p + 0.025 * d
            kp[b + j] = p
    kp[1:5] = [[0.03, 0.03, 0], [0.05, 0.05, 0], [0.07, 0.06, 0], [0.09, 0.065, 0]]
    return kp


def test_retarget_monotone_in_curl():
    r = HandRetargeter(smooth=0)
    q_open = r(_synthetic_hand(0.05))
    q_closed = r(_synthetic_hand(1.0))
    assert (q_closed[2:] > q_open[2:] + 0.3).all()
    assert hand_features(_synthetic_hand(0.5)).shape == (6,)


def test_sandbox():
    calls = []
    out = run_sandboxed("x = f(2)\nprint(x)", {"f": lambda v: calls.append(v) or v * 2})
    assert "4" in out and calls == [2]
    for bad in ("import os", "().__class__", "f.__globals__", "while True: pass"):
        with pytest.raises(UnsafeCode):
            check_code(bad)
    assert "ERROR" in run_sandboxed("open('x')", {})


def test_sim_grasp_and_reach():
    pytest.importorskip("mujoco")
    from r1hands.common.robot_model import REPO
    if not (REPO / "assets" / "r1_revo2" / "r1_revo2.xml").exists():
        pytest.skip("run python -m r1hands.sim.build_model first")
    with make_robot("sim", level=3) as r:
        s = Skills(r, level=3)
        res = s.grasp("right", "power", duration=0.5)
        assert np.mean(res["reached"][2:]) > 0.7, res
        p = s.palm_position("right")
        out = s.reach("right", [p[0] + 0.05, p[1] + 0.05, p[2] + 0.08], duration=1.0)
        time.sleep(0.5)
        assert out["ok"]
        assert np.linalg.norm(np.array(s.palm_position("right")) - [p[0] + 0.05, p[1] + 0.05, p[2] + 0.08]) < 0.03
        s.walk(0.3, duration=0.5)


def test_cap_loop_with_fake_client():
    from types import SimpleNamespace as NS

    from r1hands.policies.cap import agent as cap

    script = [
        [NS(type="tool_use", id="t1", name="run_code",
            input={"code": "r = grasp('right', 'pinch', duration=0.1)\nprint(r['reached'])"})],
        [NS(type="tool_use", id="t2", name="run_code", input={"code": "import os"})],
        [NS(type="tool_use", id="t3", name="done", input={"success": True, "summary": "pinched"})],
    ]
    seen = []

    class FakeMessages:
        def create(self, **kw):
            seen.append(kw["messages"][-1])
            return NS(content=script[len(seen) - 1], stop_reason="tool_use")

    with make_robot("mock", level=1) as r:
        a = cap.CodeAsPolicy.__new__(cap.CodeAsPolicy)
        a.client = NS(beta=NS(messages=FakeMessages()))
        a.skills = Skills(r, level=1)
        a.funcs = {n: getattr(a.skills, n) for n in a.skills.available()}
        a.model, a.effort, a.max_turns, a.confirm, a.transcript = cap.MODEL, "low", 5, None, []
        a.system = cap.SYSTEM.format(level=1, api=cap._api_doc(a.funcs))
        res = a.run("pinch with the right hand")
    assert res["success"] and len(res["transcript"]) == 2
    r1, r2 = seen[1]["content"][0], seen[2]["content"][0]
    assert r1["tool_use_id"] == "t1" and not r1["is_error"] and "[" in r1["content"]
    assert r2["is_error"] and "Rejected" in r2["content"]
    assert "walk" not in a.system and "grasp(" in a.system


def test_hand_guard_stall_negative_current():
    class SignedHand(MockHand):
        def read(self):
            s = super().read()
            s.current = -s.current  # Revo2 reports signed current
            return s
    h = SignedHand("left", tau=0.01, block_at=np.array([1, 1, 0.4, 1, 1, 1]))
    g = HandGuard(h, HandLimits(max_rate=100, stall_s=0.2))
    g.connect()
    t_end = time.monotonic() + 0.6
    while time.monotonic() < t_end:
        g.command(HandCommand(np.ones(6)))
        time.sleep(0.01)
    assert any("stall" in e for e in g.events)
