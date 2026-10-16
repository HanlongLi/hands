"""State/action layouts per capability level, shared by converters, the recorder and policy runners.

All datasets are LeRobot v3.0. Joint angles in radians, hands normalized [0, 1] in Revo2 order
(thumb, thumb_aux, index, middle, ring, pinky), base command in m/s and rad/s.

    L1  hands only                         state = action = hands (12)
    L2  arms + hands (fixed/standing base) state = action = arms (10) + hands (12)
    L3  loco-manipulation                  state  = arms (10) + waist_yaw + head (2) + hands (12) + base ang-vel (3) + gravity (3)
                                           action = arms (10) + waist_yaw + head (2) + hands (12) + base cmd vx, vy, wz (3)
"""
from __future__ import annotations

from r1hands.common.types import ARM_JOINTS, HAND_JOINTS, SIDES

HAND_NAMES = [f"{s}_hand_{j}" for s in SIDES for j in HAND_JOINTS]
ARM_NAMES = [n for s in SIDES for n in ARM_JOINTS[s]]
TORSO_HEAD = ["waist_yaw_joint", "head_pitch_joint", "head_yaw_joint"]
BASE_STATE = ["base_ang_vel_x", "base_ang_vel_y", "base_ang_vel_z", "gravity_x", "gravity_y", "gravity_z"]
BASE_CMD = ["base_cmd_vx", "base_cmd_vy", "base_cmd_wz"]

LAYOUTS = {
    1: {"state": HAND_NAMES, "action": HAND_NAMES},
    2: {"state": ARM_NAMES + HAND_NAMES, "action": ARM_NAMES + HAND_NAMES},
    3: {"state": ARM_NAMES + TORSO_HEAD + HAND_NAMES + BASE_STATE,
        "action": ARM_NAMES + TORSO_HEAD + HAND_NAMES + BASE_CMD},
}
ROBOT_TYPE = "unitree_r1_revo2"
CAMERAS = ("head", "left_wrist", "right_wrist")  # names under observation.images.*


def features(level: int, image_shape=(480, 640, 3), cameras=CAMERAS, extra: dict | None = None) -> dict:
    """LeRobot `features` dict for LeRobotDataset.create()."""
    lay = LAYOUTS[level]
    f = {
        "observation.state": {"dtype": "float32", "shape": (len(lay["state"]),), "names": lay["state"]},
        "action": {"dtype": "float32", "shape": (len(lay["action"]),), "names": lay["action"]},
        "observation.hand_current": {"dtype": "float32", "shape": (12,), "names": HAND_NAMES},
    }
    for c in cameras:
        f[f"observation.images.{c}"] = {"dtype": "video", "shape": image_shape, "names": ["height", "width", "channels"]}
    return f | (extra or {})
