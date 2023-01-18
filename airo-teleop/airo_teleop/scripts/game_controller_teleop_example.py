"""Script to
1. show how to use a game controller to teleoperate your robot
2. quickly test the teleoperation code
"""

import click
from airo_robots.grippers.hardware.robotiq_2f85_tcp import Robotiq2F85
from airo_robots.manipulators.hardware.ur_rtde import UR_RTDE
from airo_teleop.controller_teleop import GameControllerTeleop
from airo_teleop.game_controller_mapping import LogitechF310Layout, XBox360Layout  # noqa


@click.command()
@click.option("--ip_address", help="IP address of the UR robot")
@click.option("--no-gripper", is_flag=True, default=False, help="do not control gripper")
@click.option(
    "--controller_layout",
    default="LogitechF310Layout",
    help="Layout to use, must exactly match the variable name name",
)
def test_teleop(ip_address: str, no_gripper: bool, controller_layout: str):
    robot = UR_RTDE(ip_address, UR_RTDE.UR3E_CONFIG)
    if not no_gripper:
        gripper = Robotiq2F85(ip_address)
        robot.gripper = gripper
    try:
        # python magic to get the global variable with this name
        layout = globals()[controller_layout]
    except Exception:
        raise ValueError("Could not find layout, make sure it exactly matches the variable name")

    joystick_teleop = GameControllerTeleop(robot, 10, layout)
    joystick_teleop.teleoperate()


if __name__ == "__main__":
    test_teleop()
