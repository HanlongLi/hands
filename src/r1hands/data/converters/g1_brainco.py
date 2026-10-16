"""Convert Unitree G1_Brainco_* LeRobot v3 datasets (7-DoF arms + Revo2) to the R1 + Revo2 L2 layout.

Arms: joint angles are not transferable (7 vs 5 DoF, different link lengths), so we retarget the palm
trajectory. G1 FK -> palm position relative to the G1 shoulder, scaled by the arm-length ratio -> R1 shoulder
-> R1 IK warm-started from the previous frame, with the G1 palm approach axis as a secondary objective.
Hands: copied as-is. Both robots use the same Revo2 encoding and order.

The output is a new LeRobot v3 dataset: data parquet rewritten, stats recomputed for state/action, and videos
symlinked (if present). Per-frame IK error is stored as `retarget.ik_err` [left, right] in metres so
frames can be filtered.

    python -m r1hands.data.converters.g1_brainco SRC_ROOT DST_ROOT [--workers 32]
"""
from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from r1hands.common.robot_model import REPO
from r1hands.common.types import ARM_JOINTS, SIDES
from r1hands.control.kinematics import ArmKinematics
from r1hands.data.schema import LAYOUTS, ROBOT_TYPE

G1_URDF = REPO / "assets" / "g1_description" / "g1_29dof_lock_waist_rev_1_0.urdf"
G1_ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
# G1_Brainco column layout: 7 left arm, 7 right arm, 6 left hand, 6 right hand.
G1_SLICES = {"left_arm": slice(0, 7), "right_arm": slice(7, 14), "left_hand": slice(14, 20), "right_hand": slice(20, 26)}


class ArmRetargeter:
    def __init__(self, side, g1_tool=(0.10, 0, 0), r1_tool=(0.13, 0, 0)):
        self.g1 = ArmKinematics(side, g1_tool, urdf=str(G1_URDF), root="pelvis", tip=f"{side}_wrist_yaw_link",
                                active=[f"{side}_{j}_joint" for j in G1_ARM])
        self.r1 = ArmKinematics(side, r1_tool)
        self.side = side
        self.sh_g1, self.sh_r1 = self.g1.shoulder_position(), self.r1.shoulder_position()
        reach = lambda k: np.linalg.norm(k.fk({})[0][:3, 3] - k.shoulder_position())
        self.scale = reach(self.r1) / reach(self.g1)
        self.q_prev = {n: 0.0 for n in ARM_JOINTS[side]}

    def __call__(self, q_g1: np.ndarray) -> tuple[np.ndarray, float]:
        T, _ = self.g1.fk(dict(zip(self.g1.active, q_g1)))
        target = self.sh_r1 + self.scale * (T[:3, 3] - self.sh_g1)
        q, err = self.r1.ik(target, self.q_prev, approach=T[:3, 0], iters=60)
        self.q_prev = q
        return np.array([q[n] for n in ARM_JOINTS[self.side]]), err


def convert_episode(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(T,26) G1_Brainco vectors -> (T,22) R1 L2 vectors and (T,2) IK errors."""
    rts = {s: ArmRetargeter(s) for s in SIDES}
    out = np.zeros((len(rows), 22), np.float32)
    errs = np.zeros((len(rows), 2), np.float32)
    for t, r in enumerate(rows):
        for i, s in enumerate(SIDES):
            q, e = rts[s](r[G1_SLICES[f"{s}_arm"]])
            out[t, 5 * i:5 * i + 5] = q
            errs[t, i] = e
        out[t, 10:16] = r[G1_SLICES["left_hand"]]
        out[t, 16:22] = r[G1_SLICES["right_hand"]]
    return out, errs


def _convert_file(args):
    src, dst = args
    import pandas as pd
    df = pd.read_parquet(src)
    new_state, new_action, ik = [], [], []
    for _, ep in df.groupby("episode_index", sort=False):
        s, e1 = convert_episode(np.stack(ep["observation.state"].values))
        a, e2 = convert_episode(np.stack(ep["action"].values))
        new_state.append(s), new_action.append(a), ik.append(np.maximum(e1, e2))
    order = np.concatenate([g.index.values for _, g in df.groupby("episode_index", sort=False)])
    df = df.loc[order].copy()
    df["observation.state"] = list(np.concatenate(new_state))
    df["action"] = list(np.concatenate(new_action))
    df["retarget.ik_err"] = list(np.concatenate(ik))
    df = df.sort_values("index")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst)
    return dst


def _stats(x: np.ndarray) -> dict:
    return {"min": x.min(0).tolist(), "max": x.max(0).tolist(), "mean": x.mean(0).tolist(),
            "std": x.std(0).tolist(), "count": [len(x)]}


def convert(src: Path, dst: Path, workers: int = 8):
    import pandas as pd
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted((src / "data").glob("chunk-*/file-*.parquet"))
    jobs = [(f, dst / f.relative_to(src)) for f in files]
    with ProcessPoolExecutor(workers) as ex:
        for p in ex.map(_convert_file, jobs):
            print("wrote", p)
    if (src / "videos").exists() and not (dst / "videos").exists():
        (dst / "videos").symlink_to((src / "videos").resolve())

    # meta: info.json, stats.json, episodes (per-episode stats for changed features), tasks
    shutil.copytree(src / "meta", dst / "meta", dirs_exist_ok=True)
    info = json.loads((src / "meta" / "info.json").read_text())
    names = LAYOUTS[2]["state"]
    for k in ("observation.state", "action"):
        info["features"][k] = {"dtype": "float32", "shape": [22], "names": names}
    info["features"]["retarget.ik_err"] = {"dtype": "float32", "shape": [2], "names": ["left", "right"]}
    info["robot_type"] = ROBOT_TYPE
    info["source"] = {"dataset": src.name, "converter": "r1hands.data.converters.g1_brainco"}
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    data = pd.concat([pd.read_parquet(d) for _, d in jobs])
    stats = json.loads((src / "meta" / "stats.json").read_text())
    for k in ("observation.state", "action", "retarget.ik_err"):
        stats[k] = _stats(np.stack(data[k].values))
    (dst / "meta" / "stats.json").write_text(json.dumps(stats, indent=4))

    for ep_file in sorted((dst / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        eps = pd.read_parquet(ep_file)
        for k in ("observation.state", "action", "retarget.ik_err"):
            per = {i: _stats(np.stack(g[k].values)) for i, g in data.groupby("episode_index")}
            for stat in ("min", "max", "mean", "std", "count"):
                eps[f"stats/{k}/{stat}"] = [np.asarray(per[i][stat]) for i in eps["episode_index"]]
        eps.to_parquet(ep_file)

    err = np.stack(data["retarget.ik_err"].values)
    print(f"IK error (m): median {np.median(err):.4f}, p95 {np.percentile(err, 95):.4f}, "
          f">3cm {np.mean(err.max(1) > 0.03) * 100:.1f}% of frames")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    convert(Path(a.src), Path(a.dst), a.workers)


if __name__ == "__main__":
    main()
