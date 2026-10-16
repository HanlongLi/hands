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
