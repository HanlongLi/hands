"""Human hand keypoints -> Revo2 6-DoF normalized command.

Input: 21 keypoints in the MANO / OpenPose order, which HaMeR, WiLoR, HaWoR, MediaPipe, EgoDex,
HOT3D and ARCTIC (via MANO joints) all produce:
    0 wrist | 1-4 thumb CMC, MCP, IP, tip | 5-8 index MCP, PIP, DIP, tip | 9-12 middle | 13-16 ring | 17-20 pinky

Method: read joint angles off the keypoints, then map them linearly to [0, 1] between per-operator
calibrated "open" and "fist" values. Each Revo2 finger is one actuator driving coupled joints, so
summed flexion is the right quantity. This is closed-form, runs at video rate on CPU, and has no
optimizer to diverge on noisy egocentric detections. When a verified Revo2 URDF is available,
`dex-retargeting` fingertip-vector retargeting can replace it behind the same interface.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from r1hands.common.types import N_HAND

FINGER_CHAINS = {          # keypoint chains whose bend angles are summed
    "index": (0, 5, 6, 7, 8),
    "middle": (0, 9, 10, 11, 12),
    "ring": (0, 13, 14, 15, 16),
    "pinky": (0, 17, 18, 19, 20),
}
THUMB_FLEX_CHAIN = (1, 2, 3, 4)


def _unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def _bend(kp, chain):
    """Sum of angles between consecutive bone vectors along a chain (rad)."""
    pts = kp[list(chain)]
    bones = _unit(np.diff(pts, axis=0))
    cos = np.clip((bones[:-1] * bones[1:]).sum(-1), -1, 1)
    return float(np.arccos(cos).sum())


def hand_features(kp: np.ndarray) -> np.ndarray:
    """(21,3) keypoints -> raw features in HAND_JOINTS order (radians, except thumb_aux in palm widths)."""
    kp = np.asarray(kp, float).reshape(21, 3)
    # Palm frame: x from wrist to middle MCP, normal from the index/pinky MCP spread.
    x = _unit(kp[9] - kp[0])
    n = _unit(np.cross(kp[5] - kp[0], kp[17] - kp[0]))
    thumb_flex = _bend(kp, THUMB_FLEX_CHAIN)
    # Thumb aux (rotation toward the palm): how far the thumb tip has crossed the palm, from the
    # index MCP toward the pinky MCP, in palm widths (~-0.8 abducted, ~1 folded onto the pinky base).
    # It stays in the palm plane, so it does not rely on monocular depth, which is weakest exactly
    # when the thumb crosses in front of a camera-facing palm.
    across = kp[17] - kp[5]
    lateral = _unit(across - (across @ x) * x)
    thumb_aux = (kp[4] - kp[5]) @ lateral / (np.linalg.norm(across) + 1e-9)
    feats = [thumb_flex, thumb_aux] + [_bend(kp, FINGER_CHAINS[f]) for f in ("index", "middle", "ring", "pinky")]
    return np.array(feats)


@dataclass
class HandRetargeter:
    """Linear map from features to [0,1]. Defaults are typical adult values; run `calibrate` per operator."""
    open_feat: np.ndarray = field(default_factory=lambda: np.array([0.35, -0.60, 0.25, 0.25, 0.25, 0.30]))
    fist_feat: np.ndarray = field(default_factory=lambda: np.array([1.30, 0.70, 3.30, 3.40, 3.40, 3.30]))
    smooth: float = 0.5            # EMA factor, 0 = no smoothing
    _prev: np.ndarray | None = None

    def reset(self):
        self._prev = None

    def calibrate(self, open_kps: np.ndarray, fist_kps: np.ndarray):
        """Each argument is (N,21,3) frames of the operator holding the pose."""
        opened = np.median([hand_features(k) for k in open_kps], axis=0)
        closed = np.median([hand_features(k) for k in fist_kps], axis=0)
        # A fist does not reliably excite thumb opposition. Preserve the default map for
        # unobserved channels rather than divide by a near-zero (or reversed) range.
        valid = closed - opened > 0.15
        if not valid[2:].all():
            raise ValueError("Open/fist calibration needs distinct poses for all four fingers")
        self.open_feat = np.where(valid, opened, self.open_feat)
        self.fist_feat = np.where(valid, closed, self.fist_feat)
        self.reset()
        return valid

    def calibrate_thumb(self, open_kps, fold_kps):
        """Calibrate thumb flexion/opposition separately: open palm vs. thumb folded across the palm."""
        opened = np.median([hand_features(k) for k in open_kps], axis=0)[:2]
        folded = np.median([hand_features(k) for k in fold_kps], axis=0)[:2]
        if not (folded - opened > 0.15).all():
            raise ValueError("Thumb fold must visibly bend the thumb and move its tip across the palm")
        self.open_feat[:2], self.fist_feat[:2] = opened, folded
        self.reset()

    def __call__(self, kp: np.ndarray) -> np.ndarray:
        f = hand_features(kp)
        q = np.clip((f - self.open_feat) / (self.fist_feat - self.open_feat + 1e-9), 0, 1)
        if self._prev is not None and self.smooth > 0:
            q = self.smooth * self._prev + (1 - self.smooth) * q
        self._prev = q
        assert q.shape == (N_HAND,)
        return q

    def batch(self, kps: np.ndarray) -> np.ndarray:
        """Offline retargeting of a (T,21,3) sequence (no smoothing state carried over)."""
        self._prev = None
        out = np.stack([self(k) for k in kps])
        self._prev = None
        return out
