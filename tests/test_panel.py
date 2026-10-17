import time

import numpy as np

from r1hands.panel import app as panel_app
from tests.test_core import _synthetic_hand


def _panel(tmp_path, monkeypatch):
    monkeypatch.setattr(panel_app, "GRASP_FILE", tmp_path / "grasps.yaml")
    monkeypatch.setattr(panel_app, "LOGDIR", tmp_path / "logs")
    return panel_app.Panel("mock", None)


def test_panel_commands(tmp_path, monkeypatch):
    p = _panel(tmp_path, monkeypatch)
    try:
        p.handle({"type": "finger", "side": "left", "i": 2, "v": 0.7})
        assert p.runner.target["left"][2] == 0.7
        p.handle({"type": "mirror", "on": True})
        p.handle({"type": "grasp", "side": "left", "name": "pinch"})
        assert np.allclose(p.runner.target["right"], p.grasps["pinch"])
        p.handle({"type": "save_grasp", "side": "left", "name": "My Pinch"})
        assert "my_pinch" in (tmp_path / "grasps.yaml").read_text()
        p.handle({"type": "record", "on": True})
        for _ in range(5):
            p.record_sample()
        p.handle({"type": "record", "on": False})
        assert len(list((tmp_path / "logs").glob("*.npz"))) == 1
        p.handle({"type": "estop"})
        p.handle({"type": "grasp", "side": "left", "name": "fist"})
        assert not p.runner.target["left"].any()          # locked out while E-stopped
        p.handle({"type": "resume"})
        assert p.runner.speed != 0.2
        p.handle({"type": "routine", "name": "sweep", "side": "right"})
        time.sleep(0.3)
        assert p.runner.routine_running
        p.handle({"type": "finger", "side": "right", "i": 0, "v": 0.1})  # manual input cancels routine
        time.sleep(0.1)
        assert not p.runner.routine_running
    finally:
        p.runner.close()


def test_panel_teleop_and_calibration(tmp_path, monkeypatch):
    p = _panel(tmp_path, monkeypatch)
    try:
        for _ in range(6):
            p.handle({"type": "calib", "side": "right", "pose": "open", "kp": _synthetic_hand(0.05).tolist()})
            p.handle({"type": "calib", "side": "right", "pose": "fist", "kp": _synthetic_hand(1.0).tolist()})
        assert p.calibrated["right"]
        p.retarget["right"].smooth = 0
        p.handle({"type": "teleop_arm", "on": True})
        for seq in range(3):
            p.handle({"type": "teleop", "seq": seq, "age_ms": 10,
                      "panel_time": time.monotonic(),
                      "hands": {"right": {"points": _synthetic_hand(1.0).tolist(), "score": 0.99}}})
        assert p.runner.target["right"][2:].min() > 0.95
        # Let the adaptive filter converge on a held open pose.
        for seq in range(3, 25):
            p.handle({"type": "teleop", "seq": seq, "age_ms": 10,
                      "panel_time": time.monotonic(),
                      "hands": {"right": {"points": _synthetic_hand(0.05).tolist(), "score": 0.99}}})
            time.sleep(0.01)
        assert p.runner.target["right"][2:].max() < 0.05
        assert p.snapshot()["teleop_live"]
    finally:
        p.runner.close()


def test_panel_teleop_left_hand(tmp_path, monkeypatch):
    p = _panel(tmp_path, monkeypatch)
    mirror = np.array([-1.0, 1.0, 1.0])  # an anatomical left hand is a reflected right hand
    try:
        p.handle({"type": "teleop_side", "side": "left"})
        for _ in range(6):
            p.handle({"type": "calib", "side": "left", "pose": "open", "kp": (_synthetic_hand(0.05) * mirror).tolist()})
            p.handle({"type": "calib", "side": "left", "pose": "fist", "kp": (_synthetic_hand(1.0) * mirror).tolist()})
        assert p.calibrated["left"] and not p.calibrated["right"]
        p.handle({"type": "teleop_arm", "on": True})
        assert p.teleop_armed and p.snapshot()["teleop_side"] == "left"
        right_before = p.runner.target["right"].copy()
        for seq in range(3):
            p.handle({"type": "teleop", "seq": seq, "age_ms": 10, "panel_time": time.monotonic(),
                      "hands": {"left": {"points": (_synthetic_hand(1.0) * mirror).tolist(), "score": 0.99},
                                "right": {"points": _synthetic_hand(1.0).tolist(), "score": 0.99}}})
        assert p.runner.target["left"][2:].min() > 0.95
        np.testing.assert_equal(p.runner.target["right"], right_before)  # only the selected hand moves
        p.handle({"type": "teleop_side", "side": "right"})  # switching sides disarms
        assert not p.teleop_armed
    finally:
        p.runner.close()


def test_panel_adopts_hand_server_estop(tmp_path, monkeypatch):
    class Link:
        state, connected, last_rx, rate_hz, rtt_ms, url = {"estopped": True}, True, 0.0, None, None, None
        def send(self, msg): pass
    p = _panel(tmp_path, monkeypatch)
    try:
        p.link = Link()
        assert p.snapshot()["estopped"]
        p.link.state = {"estopped": False}
        p.handle({"type": "resume"})
        assert not p.snapshot()["estopped"]
    finally:
        p.runner.close()
