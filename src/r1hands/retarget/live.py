"""Validated, time-aware mapping for live camera observations. No hardware or UI dependencies."""
from __future__ import annotations

import math
import numpy as np

from r1hands.retarget.hand import HandRetargeter


def validate_landmarks(points) -> np.ndarray:
    kp = np.asarray(points, dtype=float)
    if kp.shape != (21, 3) or not np.isfinite(kp).all():
        raise ValueError("Expected 21 finite 3D landmarks")
    palm = np.linalg.norm(kp[9] - kp[0])
    area = np.linalg.norm(np.cross(kp[5] - kp[0], kp[17] - kp[0]))
    if palm < 1e-5 or area < palm * palm * 0.015:
        raise ValueError("Palm geometry is collapsed")
    for chain in ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12),
                  (13, 14, 15, 16), (17, 18, 19, 20)):
        lengths = np.linalg.norm(np.diff(kp[list(chain)], axis=0), axis=1) / palm
        if (lengths < 0.015).any() or (lengths > 1.5).any():
            raise ValueError("Finger geometry is implausible")
    return kp


class AdaptiveFilter:
    """One Euro filter: stable at rest, faster during deliberate finger movement."""
    def __init__(self, min_cutoff=1.5, beta=0.7):
        self.min_cutoff, self.beta = min_cutoff, beta
        self.reset()

    def reset(self):
        self.value = self.raw = self.derivative = None
        self.t = None

    def __call__(self, value, t):
        value = np.asarray(value, float)
        if self.t is None or t - self.t > 0.25:
            self.value = value.copy()
            self.raw = value.copy()
            self.derivative = np.zeros_like(value)
        else:
            dt = max(t - self.t, 0.001)
            alpha_d = 1 / (1 + 1 / (2 * math.pi * dt))
            self.derivative += alpha_d * ((value - self.raw) / dt - self.derivative)
            cutoff = self.min_cutoff + self.beta * np.abs(self.derivative)
            alpha = 1 / (1 + 1 / (2 * math.pi * cutoff * dt))
            self.value += alpha * (value - self.value)
            self.raw = value.copy()
        self.t = t
        return self.value.copy()


class LiveHand:
    """Reject uncertain observations; require consecutive good frames after any dropout."""
    def __init__(self, retarget: HandRetargeter, min_score=0.8, acquire_frames=3):
        self.retarget = retarget
        self.min_score, self.acquire_frames = min_score, acquire_frames
        self.filter = AdaptiveFilter()
        self.reset()

    def reset(self):
        self.good_frames = 0
        self.last_t = None
        self.status = "waiting"
        self.filter.reset()

    def update(self, points, score, t):
        try:
            if not np.isfinite(score) or score < self.min_score:
                raise ValueError("Uncertain hand identity")
            kp = validate_landmarks(points)
        except (ValueError, TypeError):
            self.reset()
            self.status = "rejected"
            return None
        if self.last_t is not None and t - self.last_t > 0.25:
            self.reset()
        self.last_t = t
        self.good_frames += 1
        if self.good_frames < self.acquire_frames:
            self.status = "acquiring"
            return None
        # Live smoothing is time-aware; avoid applying the legacy EMA as well.
        previous_smooth = self.retarget.smooth
        self.retarget.smooth = 0
        try:
            q = self.retarget(kp)
        finally:
            self.retarget.smooth = previous_smooth
        self.status = "tracking"
        return self.filter(q, t)
