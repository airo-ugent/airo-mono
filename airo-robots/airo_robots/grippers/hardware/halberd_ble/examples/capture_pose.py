"""Capture the current UR TCP pose and save it as a .npy file in the poses folder.

Run this script once per pose you want to record. Each run saves one file.

    python airo-robots/airo_robots/grippers/hardware/halberd_ble/examples/capture_pose.py
"""

from pathlib import Path

import numpy as np
from airo_robots.manipulators.hardware.ur_rtde import URrtde

np.set_printoptions(precision=3, suppress=True)

UR_IP_ADDRESS = "10.42.0.162"
POSES_DIR = Path(__file__).resolve().parent / "poses"

if __name__ == "__main__":
    POSES_DIR.mkdir(parents=True, exist_ok=True)

    existing = sorted(POSES_DIR.glob("*.npy"))
    if existing:
        print("Existing poses:")
        for f in existing:
            print(f"  {f.name}")
    else:
        print("No poses saved yet.")

    name = input("\nEnter a name for this pose (e.g. pose7_above_table): ").strip()
    if not name:
        raise ValueError("Pose name cannot be empty.")

    filename = POSES_DIR / (name if name.endswith(".npy") else name + ".npy")
    if filename.exists():
        overwrite = input(f"{filename.name} already exists. Overwrite? [y/N] ").strip().lower()
        if overwrite != "y":
            print("Aborted.")
            raise SystemExit(0)

    print(f"\nConnecting to UR robot at {UR_IP_ADDRESS}...")
    robot = URrtde(UR_IP_ADDRESS, URrtde.UR3E_CONFIG)

    tcp_pose = robot.get_tcp_pose()
    print(f"Current TCP pose:\n{tcp_pose}")

    np.save(filename, tcp_pose)
    print(f"\nPose saved to {filename}")
