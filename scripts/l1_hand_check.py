"""Level 1: bring-up and characterization of the Revo2 hands (no arm/body motion).

Run on the Linux machine the hands are wired to (bc-stark-sdk has no macOS wheel). `--backend mock` works anywhere.

  python scripts/l1_hand_check.py probe                      # connect, print device + status
  python scripts/l1_hand_check.py sweep --side right         # each finger alone, slowly 0 -> 1 -> 0
  python scripts/l1_hand_check.py grasps --side right        # cycle the grasp library
  python scripts/l1_hand_check.py audit --side right --hold 20
        # place a firm foam block in the hand first: closes a power grasp WITHOUT the stall guard and logs
        # current/position/state at full rate until firmware protection trips or --hold expires.
        # It sets hand_limits.stall_current and answers whether current is usable as a contact signal.
  python scripts/l1_hand_check.py rate                       # achievable status polling rate
  python scripts/l1_hand_check.py webcam --side right        # MediaPipe hand tracking -> Revo2 (pip install mediapipe)
Logs go to logs/l1/<timestamp>_<mode>.npz
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from r1hands.common.types import HAND_JOINTS, HandCommand
from r1hands.control.skills import GRASPS, Skills
from r1hands.hal.factory import make_robot

LOGDIR = Path(__file__).resolve().parents[1] / "logs" / "l1"


def save(mode, **arrays):
    LOGDIR.mkdir(parents=True, exist_ok=True)
    p = LOGDIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{mode}.npz"
    np.savez(p, **arrays)
    print("saved", p)


def record(hand, seconds, rate=100.0):
    t, pos, cur, fault = [], [], [], []
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        s = hand.read()
        t.append(time.monotonic() - t0)
        pos.append(s.position)
        cur.append(s.current if s.current is not None else np.full(6, np.nan))
        fault.append(s.fault)
        time.sleep(1 / rate)
    return np.array(t), np.array(pos), np.array(cur), np.array(fault)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["probe", "sweep", "grasps", "audit", "rate", "webcam"])
    ap.add_argument("--side", default="right", choices=["left", "right"])
    ap.add_argument("--backend", default="real", choices=["real", "sim", "mock"])
    ap.add_argument("--hold", type=float, default=15.0)
    a = ap.parse_args()

    with make_robot(a.backend, level=1, sides=(a.side,)) as robot:
        guarded = robot.hands[a.side]
        raw = guarded.inner
        sk = Skills(robot, level=1)

        if a.mode == "probe":
            print("device:", getattr(raw, "info", None))
            print(sk.hand_state(a.side))

        elif a.mode == "sweep":
            logs = {}
            for i, name in enumerate(HAND_JOINTS):
                goal = np.zeros(6)
                goal[i] = 1.0
                print(f"{name}: close")
                sk.set_fingers(a.side, goal, duration=2.0)
                t, p, c, _ = record(guarded, 0.5)
                print(f"   reached {p[-1, i]:.3f}, current {np.nanmean(c[:, i]):.0f} mA")
                sk.open_hand(a.side, duration=2.0)
                logs[name] = np.concatenate([p, c], 1)
            save("sweep", **logs)

        elif a.mode == "grasps":
            for g in GRASPS:
                print(g, sk.grasp(a.side, g, duration=1.2))
                time.sleep(0.8)
            sk.open_hand(a.side)

        elif a.mode == "audit":
            input("Place a firm foam block in the hand, then press Enter (Ctrl-C aborts)...")
            raw.command(HandCommand(np.array(GRASPS["power"]), speed=0.3))   # bypasses HandGuard on purpose
            try:
                t, p, c, f = record(raw, a.hold, rate=200)
            finally:
                raw.command(HandCommand(np.zeros(6), speed=0.3))
            save("audit", t=t, pos=p, cur=c, fault=f)
            first_fault = t[f.argmax()] if f.any() else None
            print("steady current per finger (mA, last 2 s):", np.round(np.nanmean(c[t > t[-1] - 2], 0)))
            print("firmware stall flagged at:", first_fault, "s")
            print("suggested hand_limits.stall_current:", round(0.8 * float(np.nanmax(np.nanmean(c[t > t[-1] - 2], 0)))))

        elif a.mode == "rate":
            n, t0, last = 0, time.monotonic(), None
            while time.monotonic() - t0 < 3:
                s = raw.read()
                if s.stamp != last:
                    n, last = n + 1, s.stamp
                time.sleep(0.0005)
            print(f"fresh status samples: {n / 3:.0f} Hz")

        elif a.mode == "webcam":
            run_webcam(sk, a.side)


def run_webcam(sk: Skills, side: str):
    import cv2
    import mediapipe as mp

    from r1hands.retarget.hand import HandRetargeter
    rt = HandRetargeter()
    hands = mp.solutions.hands.Hands(max_num_hands=1, model_complexity=1)
    cap = cv2.VideoCapture(0)
    print("show your hand OPEN for calibration, press 'o'; then a FIST, press 'f'; 'q' quits")
    open_kp, fist_kp = [], []
    hand = sk.robot.hands[side]
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        kp = None
        if res.multi_hand_world_landmarks:
            kp = np.array([[l.x, l.y, l.z] for l in res.multi_hand_world_landmarks[0].landmark])
            hand.command(HandCommand(rt(kp)))
        cv2.imshow("webcam", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if kp is not None and k == ord("o"):
            open_kp.append(kp)
        if kp is not None and k == ord("f"):
            fist_kp.append(kp)
        if open_kp and fist_kp and k in (ord("o"), ord("f")):
            rt.calibrate(np.array(open_kp), np.array(fist_kp))
            print("calibrated", rt.open_feat.round(2), rt.fist_feat.round(2))
    sk.open_hand(side)


if __name__ == "__main__":
    main()
