"""Minimal working example for controlling a Fairino FR3 arm with airo-robots.

The robot reads its state, rotates the last joint back and forth and moves the TCP
5 cm up and back down along a straight line.

Prerequisites:
- The Fairino Python SDK is importable (see the ImportError message in ``fairino.py``).
- The controller is reachable over the network and the workspace around the robot is clear.

Run with:
    python fairino_minimal_example.py --ip-address 192.168.58.2
"""

import click
import numpy as np
from airo_robots.manipulators.hardware.fairino import Fairino
from loguru import logger


@click.command()
@click.option("--ip-address", default="192.168.58.2", show_default=True, help="IP address of the Fairino controller.")
def main(ip_address: str) -> None:
    with Fairino(ip_address) as robot:
        start_joints = robot.get_joint_configuration()
        start_pose = robot.get_tcp_pose()
        logger.info(f"Joint configuration [rad]: {np.round(start_joints, 3)}")
        logger.info(f"TCP position [m]: {np.round(start_pose[:3, 3], 3)}")

        # Joint-space moves: rotate joint 6 by +/- 15 degrees and return.
        for offset in (np.radians(15.0), -np.radians(15.0), 0.0):
            target_joints = start_joints.copy()
            target_joints[5] += offset
            robot.move_to_joint_configuration(target_joints, joint_speed=0.3).wait()

        # Cartesian linear moves: 5 cm up and back down.
        up_pose = start_pose.copy()
        up_pose[2, 3] += 0.05
        robot.move_linear_to_tcp_pose(up_pose, linear_speed=0.05).wait()
        robot.move_linear_to_tcp_pose(start_pose, linear_speed=0.05).wait()

        logger.info("Done.")


if __name__ == "__main__":
    main()
