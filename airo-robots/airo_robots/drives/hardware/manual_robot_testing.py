"""code for manual testing of mobile robot base class implementations.
"""
from airo_robots.drives.mobile_robot import MobileRobot


def manually_test_robot_implementation(robot: MobileRobot) -> None:
    input("robot will now move forward 10cm")
    robot.set_platform_velocity_target(0.1, 0.0, 0.0, 1.0)

    input("robot will now move left 10cm")
    robot.set_platform_velocity_target(0.0, 0.1, 0.0, 1.0)

    input("robot will now make two short rotations")
    robot.set_platform_velocity_target(0.0, 0.0, 0.1, 1.0)
    robot.set_platform_velocity_target(0.0, 0.0, -0.1, 1.0)

    input("robot will now return to original position")
    robot.set_platform_velocity_target(-0.1, -0.1, 0.0, 1.0)
