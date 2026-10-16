"""Find the Revo2 hands on the USB-RS485 adapters and pin them to stable device names.

    python scripts/revo2_scan.py                 # report: port -> slave id -> hand (0x7E left, 0x7F right)
    sudo python scripts/revo2_scan.py --write    # also write /etc/udev/rules.d/99-revo2.rules (by USB port path)

The rules use the physical USB port path (ID_PATH), because cheap RS485 adapters often have no unique
serial number. Keep each adapter in the same USB socket after running this. Afterwards the hands are
/dev/revo2_left and /dev/revo2_right; put those in configs/r1.yaml.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import subprocess

IDS = {0x7E: "left", 0x7F: "right"}


async def probe(port: str):
    from bc_stark_sdk import main_mod as libstark
    try:
        ctx = await libstark.modbus_open(port, libstark.Baudrate.Baud460800)
    except Exception as e:
        return None, f"open failed: {e}"
    try:
        for sid, side in IDS.items():
            try:
                info = await asyncio.wait_for(ctx.get_device_info(sid), 1.0)
            except Exception:
                info = None
            if info:
                return side, f"id {sid:#x}: {getattr(info, 'description', info)}"
        return None, "no Revo2 answered at 0x7e/0x7f (460800 baud)"
    finally:
        libstark.modbus_close(ctx)


def id_path(port: str) -> str | None:
    out = subprocess.run(["udevadm", "info", "-q", "property", "-n", port], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("ID_PATH="):
            return line.split("=", 1)[1]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write /etc/udev/rules.d/99-revo2.rules (needs sudo)")
    a = ap.parse_args()
    ports = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
    if not ports:
        print("no /dev/ttyUSB* or /dev/ttyACM* found: check the adapters and the hand power")
        return
    async def probe_all():  # one event loop for all ports (the SDK's tasks are bound to the loop)
        return [await probe(p) for p in ports]

    rules = []
    for p, (side, msg) in zip(ports, asyncio.run(probe_all())):
        print(f"{p:16s} {side or '-':6s} {msg}")
        if side and (ip := id_path(p)):
            rules.append(f'SUBSYSTEM=="tty", ENV{{ID_PATH}}=="{ip}", SYMLINK+="revo2_{side}", MODE="0666"')
    if rules:
        print("\nudev rules:\n" + "\n".join(rules))
    if a.write and rules:
        with open("/etc/udev/rules.d/99-revo2.rules", "w") as f:
            f.write("# written by r1hands scripts/revo2_scan.py\n" + "\n".join(rules) + "\n")
        subprocess.run(["udevadm", "control", "--reload-rules"])
        subprocess.run(["udevadm", "trigger"])
        print("written; devices: ", glob.glob("/dev/revo2_*"))


if __name__ == "__main__":
    main()
