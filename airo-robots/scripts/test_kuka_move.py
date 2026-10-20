"""Manual hardware test script for the KUKA PositionManipulator implementation."""

import click
import numpy as np
from airo_robots.manipulators.hardware.kuka import DEFAULT_OPCUA_PORT, Kuka


@click.command()
@click.option("--ip-address", default="192.168.1.2", show_default=True, help="IP address of the KUKA controller.")
@click.option("--port", default=DEFAULT_OPCUA_PORT, show_default=True, help="KUKA OPC-UA port.")
@click.option("--target-a1-deg", default=90.0, show_default=True, help="Target A1 angle in degrees (toggles with 0).")
@click.option("--speed-deg-s", default=50.0, show_default=True, help="Leading-axis joint speed in degrees/second.")
def main(ip_address: str, port: int, target_a1_deg: float, speed_deg_s: float) -> None:
    """Connect to a KUKA KRC4 controller, wait for PY_SLOW to be active, and execute a test joint move."""
    with Kuka(ip_address, port=port) as robot:
        robot.wait_until_ready(timeout=120.0)

        q_start = robot.get_joint_configuration()
        tcp_start = robot.get_tcp_pose()
        print("Connected to:", robot.model)
        print("Start joints (deg):", np.round(np.degrees(q_start), 3))
        print("Start TCP position (m):", np.round(tcp_start[:3, 3], 4))

        target = q_start.copy()
        if abs(np.degrees(q_start[0]) - target_a1_deg) > 5.0:
            target[0] = np.radians(target_a1_deg)
        else:
            target[0] = 0.0

        print(f"Commanding A1 -> {np.degrees(target[0]):.1f} deg at {speed_deg_s:.1f} deg/s...")
        action = robot.move_to_joint_configuration(target, joint_speed=np.radians(speed_deg_s))
        action.wait(timeout=60.0)

        q_end = robot.get_joint_configuration()
        print("Final joints (deg):", np.round(np.degrees(q_end), 3))


if __name__ == "__main__":
    main()
