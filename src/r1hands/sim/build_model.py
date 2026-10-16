"""Build the combined R1 + 2x Revo2 model (URDF for Isaac/pinocchio, MJCF for MuJoCo/mjlab).

    python -m r1hands.sim.build_model            # writes assets/r1_revo2/{r1_revo2.urdf, r1_revo2.xml}

Mount: each Revo2 base link is attached to <side>_wrist_roll_link with a fixed joint. The hand's fingers
point along its +z and the R1 forearm along wrist +x, so the mount pitches the hand +90 deg about y.
`mount_roll_deg` clocks the hand about the forearm axis. Its default puts the palm facing the body midline
with the arm at zero. That has to match how the adapter is printed and mounted, so check it against a photo
of the real robot. `mount_offset_m` is the distance from the wrist_roll joint to the Revo2 wrist interface
along the forearm (stock-neck depth + adapter length). TODO: measure on the robot.

Mimic joints (distal = k * proximal, from BrainCo's URDF) become MuJoCo joint equality constraints, and each
of the 6 motors per hand gets a position actuator on its driving joint.
"""
from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from r1hands.common.robot_model import REPO
from r1hands.common.types import BODY_JOINTS, HAND_JOINTS, SIDES

R1_DIR = REPO / "assets" / "r1_description"
REVO2_DIR = REPO / "assets" / "revo2_description"
OUT = REPO / "assets" / "r1_revo2"

# Motor -> driving URDF joint, in HAND_JOINTS order (BrainCo ROS2 driver mapping).
HAND_DRIVE = {
    "thumb": "thumb_proximal_joint", "thumb_aux": "thumb_metacarpal_joint", "index": "index_proximal_joint",
    "middle": "middle_proximal_joint", "ring": "ring_proximal_joint", "pinky": "pinky_proximal_joint",
}
# Effort limits (N*m) of the driving joints, from BrainCo's URDF.
HAND_EFFORT = {"thumb": 1.1, "thumb_aux": 0.5, "index": 2.0, "middle": 2.0, "ring": 2.0, "pinky": 2.0}
# Joint range of each driving joint, used to map normalized [0,1] <-> radians.
HAND_RANGE = {"thumb": 1.04720, "thumb_aux": 1.55334, "index": 1.41, "middle": 1.41, "ring": 1.41, "pinky": 1.41}


def hand_joint_names(side: str) -> list[str]:
    return [f"{side}_{HAND_DRIVE[j]}" for j in HAND_JOINTS]


def hand_to_rad(q01: np.ndarray) -> np.ndarray:
    return np.asarray(q01) * np.array([HAND_RANGE[j] for j in HAND_JOINTS])


def _rot(axis, a):
    c, s_ = np.cos(a), np.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]]),
            "y": np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]]),
            "z": np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])}[axis]


def _mat_to_rpy(R):
    """Inverse of URDF's R = Rz(yaw) Ry(pitch) Rx(roll)."""
    pitch = np.arcsin(-np.clip(R[2, 0], -1, 1))
    if abs(np.cos(pitch)) < 1e-6:
        return np.array([np.arctan2(-R[1, 2], R[1, 1]), pitch, 0.0])
    return np.array([np.arctan2(R[2, 1], R[2, 2]), pitch, np.arctan2(R[1, 0], R[0, 0])])


def build_urdf(mount_offset_m: float = 0.045, mount_roll_deg: float | None = None) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    root = ET.parse(R1_DIR / "R1.urdf").getroot()
    root.set("name", "r1_revo2")
    for mj in root.findall("mujoco"):
        root.remove(mj)
    for m in root.iter("mesh"):
        m.set("filename", "../r1_description/" + m.get("filename"))
    for side in SIDES:
        h = ET.parse(REVO2_DIR / "urdf" / f"revo2_{side}_hand.urdf").getroot()
        for el in h:
            if el.tag not in ("link", "joint"):
                continue
            el = copy.deepcopy(el)
            for m in el.iter("mesh"):
                m.set("filename", m.get("filename").replace("package://revo2_description/", "../revo2_description/"))
            root.append(el)
        # Default roll: palm toward the midline (right palm +y, left palm -y) with the forearm along +x.
        roll = (90.0 if side == "right" else -90.0) if mount_roll_deg is None else \
            (mount_roll_deg if side == "right" else -mount_roll_deg)
        j = ET.SubElement(root, "joint", name=f"{side}_hand_mount_joint", type="fixed")
        rpy = _mat_to_rpy(_rot("x", np.radians(roll)) @ _rot("y", np.pi / 2))  # pitch fingers onto +x, then clock
        ET.SubElement(j, "origin", xyz=f"{mount_offset_m} 0 0", rpy=" ".join(f"{v:.6f}" for v in rpy))
        ET.SubElement(j, "parent", link=f"{side}_wrist_roll_link")
        ET.SubElement(j, "child", link=f"{side}_base_link")
    ET.indent(root)
    p = OUT / "r1_revo2.urdf"
    ET.ElementTree(root).write(p, xml_declaration=True, encoding="utf-8")
    return p


def build_mjcf(urdf: Path, body_kp: float = 100.0, body_kd: float = 3.0, hand_kp: float = 2.0) -> Path:
    import mujoco
    root = ET.parse(urdf).getroot()
    mimics = [(j.get("name"), j.find("mimic").get("joint"), float(j.find("mimic").get("multiplier", 1)))
              for j in root.iter("joint") if j.find("mimic") is not None]
    mj = ET.SubElement(root, "mujoco")
    ET.SubElement(mj, "compiler", meshdir=".", strippath="false", balanceinertia="true", discardvisual="true",
                  fusestatic="false")
    tmp = OUT / "_tmp_mj.urdf"
    ET.ElementTree(root).write(tmp)
    model = mujoco.MjModel.from_xml_path(str(tmp))
    raw = OUT / "_tmp_raw.xml"
    mujoco.mj_saveLastXML(str(raw), model)
    tmp.unlink()

    x = ET.parse(raw).getroot()
    raw.unlink()
    x.set("model", "r1_revo2")
    x.find("compiler").set("meshdir", ".")
    for m in x.iter("mesh"):  # saved paths are relative to OUT already; keep them
        pass
    wb = x.find("worldbody")
    pelvis = wb.find("body")
    if pelvis.find("freejoint") is None and pelvis.find("joint") is None:
        ET.SubElement(pelvis, "freejoint", name="floating_base")
        pelvis.insert(0, pelvis[-1])
        pelvis.remove(pelvis[-1])
        pelvis.set("pos", "0 0 0.75")
    ET.SubElement(wb, "geom", name="floor", type="plane", size="0 0 0.05", rgba="0.85 0.85 0.85 1")
    ET.SubElement(wb, "light", pos="0 0 3", dir="0 0 -1", directional="true")
    # MuJoCo's URDF importer already turns <mimic> into joint equalities; only add any it missed.
    eq = x.find("equality")
    if eq is None:
        eq = ET.SubElement(x, "equality")
    have = {(e.get("joint1"), e.get("joint2")) for e in eq.iter("joint")}
    for child, parent, k in mimics:
        if (child, parent) not in have:
            ET.SubElement(eq, "joint", joint1=child, joint2=parent, polycoef=f"0 {k} 0 0 0", solref="0.005 1")
    act = x.find("actuator")
    if act is None:
        act = ET.SubElement(x, "actuator")
    lim = {j.get("name"): j for j in x.iter("joint")}
    for n in BODY_JOINTS:
        r = lim[n].get("range")
        ET.SubElement(act, "position", name=n, joint=n, kp=str(body_kp), kv=str(body_kd),
                      **({"ctrlrange": r, "ctrllimited": "true"} if r else {}))
    for side in SIDES:
        for hj, n in zip(HAND_JOINTS, hand_joint_names(side)):
            # Full effort at 0.25 rad error, torque clamped to BrainCo's effort limit.
            e = HAND_EFFORT[hj]
            ET.SubElement(act, "position", name=f"{side}_{hj}", joint=n, kp=f"{e / 0.25:.3f}", kv=f"{e / 20:.3f}",
                          ctrlrange=f"0 {HAND_RANGE[hj]}", ctrllimited="true",
                          forcerange=f"{-e} {e}", forcelimited="true")
    # Hand links don't collide with each other: the convex hull of the palm mesh would block the fingers.
    # They still collide with the robot body, the floor and objects (contype 2 / conaffinity 1).
    hand_links = set()
    for side in SIDES:
        hand_links |= {l.get("name") for l in ET.parse(REVO2_DIR / "urdf" / f"revo2_{side}_hand.urdf").getroot().iter("link")}
    for b in x.iter("body"):
        if b.get("name") in hand_links:
            for g in b.findall("geom"):
                g.set("contype", "2")
                g.set("conaffinity", "1")
    # Geared micro-motors have large reflected rotor inertia. Without armature the finger links' tiny inertia
    # makes the position loop unstable at a 2 ms step (natural frequency ~3e3 rad/s).
    for b in x.iter("body"):
        if b.get("name") in hand_links:
            for j in b.findall("joint"):
                j.set("armature", "0.002")
                j.set("damping", "0.02")
    opt = x.find("option")
    if opt is None:
        opt = ET.Element("option")
        x.insert(1, opt)
    opt.set("integrator", "implicitfast")
    # Unitree's R1 meshes merge the stock hand into <side>_wrist_roll_link, so that link's collision mesh
    # overlaps the Revo2. Exclude hand <-> wrist-link contacts. The forearm still collides with the world.
    contact = x.find("contact")
    if contact is None:
        contact = ET.SubElement(x, "contact")
    for side in SIDES:
        for l in sorted(hand_links):
            if l.startswith(side + "_"):
                ET.SubElement(contact, "exclude", body1=f"{side}_wrist_roll_link", body2=l)
    ET.indent(x)
    p = OUT / "r1_revo2.xml"
    ET.ElementTree(x).write(p)
    mujoco.MjModel.from_xml_path(str(p))  # validate
    return p


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mount-offset", type=float, default=0.045)
    ap.add_argument("--mount-roll", type=float, default=None, help="deg; default palm faces the midline")
    a = ap.parse_args()
    u = build_urdf(a.mount_offset, a.mount_roll)
    m = build_mjcf(u)
    print("wrote", u, "and", m)


if __name__ == "__main__":
    main()
