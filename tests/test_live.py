"""Live demo contracts: observation validity, mapping, expiry, ownership and transport."""
import asyncio
import time

import numpy as np
import pytest

from r1hands.common.types import HandCommand
from r1hands.hal.base import Robot
from r1hands.hal.mock import MockHand
from r1hands.hal.remote import RemoteHand, RemoteLink
from r1hands.panel.hand_server import make_app
from r1hands.panel.runner import HandRunner
from r1hands.retarget.hand import HandRetargeter, hand_features
from r1hands.retarget.live import AdaptiveFilter, LiveHand, validate_landmarks
from tests.test_core import _synthetic_hand
from tests.test_panel import _panel


def calibrate(panel):
    for pose, curl in (("open", 0.05), ("fist", 1.0)):
        for _ in range(6):
            panel.handle({"type": "calib", "side": "right", "pose": pose,
                          "kp": _synthetic_hand(curl).tolist()})


def frame(seq, points=None, **overrides):
    return {"type": "teleop", "seq": seq, "age_ms": 10, "panel_time": time.monotonic(),
            "hands": {"right": {"points": (points if points is not None else _synthetic_hand(1)).tolist(),
                                 "score": 0.99}}, **overrides}


def test_geometry_and_mapping_are_rigid_transform_invariant():
    kp = _synthetic_hand(0.6)
    r = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    transformed = 2.3 * kp @ r.T + [1, 2, 3]
    validate_landmarks(transformed)
    np.testing.assert_allclose(hand_features(kp), hand_features(transformed), atol=1e-5)
    for bad in (np.zeros((21, 3)), np.full((21, 3), np.nan), np.zeros((20, 3))):
        with pytest.raises(ValueError):
            validate_landmarks(bad)


def test_calibration_rejects_identical_poses_and_preserves_unobserved_thumb():
    r = HandRetargeter()
    opened = np.repeat(_synthetic_hand(0.05)[None], 6, axis=0)
    closed = np.repeat(_synthetic_hand(1)[None], 6, axis=0)
    defaults = r.fist_feat[:2].copy()
    with pytest.raises(ValueError):
        r.calibrate(opened, opened)
    valid = r.calibrate(opened, closed)
    assert valid[2:].all() and not valid[:2].any()
    np.testing.assert_equal(r.fist_feat[:2], defaults)
    with pytest.raises(ValueError):
        r.calibrate_thumb(opened, opened)


def test_live_reacquires_after_uncertainty_and_timeout():
    live = LiveHand(HandRetargeter())
    kp = _synthetic_hand(0.5)
    assert live.update(kp, 0.99, 0) is None
    assert live.update(kp, 0.99, 0.03) is None
    assert live.update(kp, 0.99, 0.06) is not None
    assert live.update(kp, 0.2, 0.09) is None
    assert live.update(kp, 0.99, 0.12) is None
    assert live.update(kp, 0.99, 1.0) is None
    assert live.good_frames == 1


def test_adaptive_filter_reduces_jitter_and_follows_a_step():
    filt = AdaptiveFilter()
    raw = np.array([0.5 + 0.03 * (-1)**i for i in range(90)])
    smooth = np.array([filt([q], i / 30)[0] for i, q in enumerate(raw)])
    assert smooth[30:].std() < raw[30:].std() * 0.3
    response = [filt([1.0], (90 + i) / 30)[0] for i in range(6)]
    assert response[-1] > 0.9


def test_panel_requires_arm_and_rejects_stale_and_wrong_owner(tmp_path, monkeypatch):
    p = _panel(tmp_path, monkeypatch)
    owner, stranger = object(), object()
    try:
        with pytest.raises(ValueError):
            p.handle({"type": "teleop_arm", "on": True}, owner)
        calibrate(p)
        p.handle(frame(0), owner)
        assert p.teleop_seq == -1
        p.handle({"type": "teleop_arm", "on": True}, owner)
        with pytest.raises(ValueError):
            p.handle({"type": "teleop_arm", "on": True}, stranger)
        p.handle(frame(0), stranger)
        p.handle(frame(0, age_ms=250), owner)
        p.handle(frame(0, panel_time=time.monotonic() - 1), owner)
        assert p.teleop_seq == -1
        p.mirror = True
        for seq in range(3):
            p.handle(frame(seq), owner)
        assert p.runner.target["right"][2:].min() > 0.95
        assert not p.runner.target["left"].any()
        last = p.runner.target["right"].copy()
        p.handle(frame(2, points=_synthetic_hand(0.05)), owner)
        np.testing.assert_equal(last, p.runner.target["right"])
        p.handle(frame(3, hands={}), owner)
        assert p.live["right"].status == "waiting"
        assert "right" not in p.runner._expires
        p.handle({"type": "estop"}, stranger)
        assert not p.teleop_armed and p.runner.estopped
    finally:
        p.runner.close()
        p.robot.close()


def test_expired_target_holds_measured_pose_and_nonfinite_is_rejected():
    with Robot(hands={"right": MockHand("right", tau=1)}) as robot:
        runner = HandRunner(robot)
        try:
            runner.set_target("right", np.ones(6), ttl_s=0.04)
            time.sleep(0.1)
            assert "right" not in runner._expires
            held = runner.target["right"].copy()
            assert held.max() < 0.2
            time.sleep(0.05)
            np.testing.assert_equal(held, runner.target["right"])
            with pytest.raises(ValueError):
                runner.set_target("right", np.full(6, np.nan))
        finally:
            runner.close()


def test_real_websocket_deadlines_and_remote_path():
    from aiohttp.test_utils import TestClient, TestServer

    async def run():
        with Robot(hands={"right": MockHand("right")}) as robot:
            runner = HandRunner(robot)
            client = TestClient(TestServer(make_app(runner)))
            link = None
            try:
                await client.start_server()
                ws = await client.ws_connect('/ws')
                state = await ws.receive_json()
                await ws.send_json({"type": "set", "side": "right", "position": [1]*6,
                                    "deadline": state['t'] - 1})
                await asyncio.sleep(0.03)
                assert not runner.target['right'].any()
                link = RemoteLink(str(client.make_url('/ws')).replace('http:', 'ws:'))
                remote = RemoteHand(link, 'right')
                await asyncio.to_thread(remote.connect)
                remote.command(HandCommand(np.full(6, 0.7), speed=0.3))
                await asyncio.sleep(0.08)
                np.testing.assert_allclose(runner.target['right'], 0.7)
                assert runner.speed == 0.3 and not remote.read().fault
                await asyncio.sleep(0.35)
                assert runner.target['right'].max() < 0.7
                assert not link._pending
                await ws.close()
            finally:
                if link is not None:
                    await asyncio.to_thread(link.close)
                await client.close()
                runner.close()
    asyncio.run(run())


def test_hold_resets_guard_limiter_and_preserves_stall_cap():
    from r1hands.safety.guard import HandGuard, HandLimits
    h = MockHand('right', tau=10)
    g = HandGuard(h, HandLimits(max_rate=2))
    g.connect()
    g._last_target = np.full(6, 0.95)
    g._cap[2] = 0.2
    held = g.hold()
    assert held.max() < 0.01
    assert g._cap[2] == 0.2


def test_stale_telemetry_cannot_issue_motion():
    from r1hands.safety.guard import HandGuard
    class StaleHand(MockHand):
        def read(self):
            s = super().read()
            s.stamp = time.monotonic() - 1
            return s
    g = HandGuard(StaleHand('right'))
    g.connect()
    with pytest.raises(RuntimeError, match='stale'):
        g.command(HandCommand(np.ones(6)))


def test_failed_hold_does_not_block_stop_or_reissue_old_target():
    class FailingHoldHand(MockHand):
        def hold(self):
            raise RuntimeError('lost telemetry')
    with Robot(hands={'right': FailingHoldHand('right')}) as robot:
        runner = HandRunner(robot)
        try:
            runner.set_target('right', np.ones(6))
            runner.hold('right')
            assert runner._expires['right'] == 0
            runner.estop()
            assert runner.estopped
            assert not runner.target['right'].any()
        finally:
            runner.close()


def test_separate_thumb_calibration_maps_pinch_endpoint():
    opened = _synthetic_hand(0.05)
    pinched = opened.copy()
    pinched[1:5] = [[0.03, 0.03, 0], [0.05, 0.03, -0.02],
                    [0.05, 0.03, -0.04], [0.04, 0.03, -0.05]]
    validate_landmarks(pinched)
    r = HandRetargeter(smooth=0)
    r.calibrate_thumb([opened] * 6, [pinched] * 6)
    np.testing.assert_allclose(r(opened)[:2], 0, atol=1e-5)
    np.testing.assert_allclose(r(pinched)[:2], 1, atol=1e-5)


def test_panel_websocket_disconnect_and_recording_lifecycle(tmp_path, monkeypatch):
    from aiohttp.test_utils import TestClient, TestServer
    from r1hands.panel.app import make_app as panel_make_app

    async def run():
        p = _panel(tmp_path, monkeypatch)
        calibrate(p)
        client = TestClient(TestServer(panel_make_app(p)))
        try:
            await client.start_server()
            response = await client.get('/')
            assert 'Arm right-hand demo' in await response.text()
            response = await client.get('/static/hand-worker.js')
            assert response.status == 200 and 'detectForVideo' in await response.text()
            owner = await client.ws_connect('/ws')
            spectator = await client.ws_connect('/ws')
            await owner.receive_json()
            await spectator.receive_json()
            await owner.send_json({'type': 'teleop_arm', 'on': True})
            for seq in range(3):
                await owner.send_json(frame(seq))
                await asyncio.sleep(0.02)
            assert p.teleop_armed and p.live['right'].status == 'tracking'
            p._record(True)
            await asyncio.sleep(0.12)
            count = len(p.rec['rows'])
            for _ in range(10):
                p.snapshot()
            assert len(p.rec['rows']) == count and count >= 2
            await owner.close()
            await asyncio.sleep(0.03)
            assert not p.teleop_armed
            await spectator.close()
        finally:
            await client.close()
            p.runner.close()
            p.robot.close()
        saved = list((tmp_path / 'logs').glob('*.npz'))
        assert len(saved) == 1
        with np.load(saved[0]) as recording:
            assert 'inference_ms' in recording
            assert recording['right_open_features'].shape == (6,)
    asyncio.run(run())
