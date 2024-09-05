"""code for manual testing of mobile robot base class implementations.
"""
from airo_robots.drives.hardware.kelo_robile import KELORobile
from airo_robots.drives.mobile_robot import MobileRobot


def manually_test_robot_implementation(robot: MobileRobot) -> None:
    if isinstance(robot, KELORobile):
        input("robot will rotate drives")
        robot.align_drives(0.1, 0.0, 0.0, 1.0).wait()

    input("robot will now move forward 10cm")
    robot.set_platform_velocity_target(0.1, 0.0, 0.0, 1.0).wait()

    input("robot will now move left 10cm")
    robot.set_platform_velocity_target(0.0, 0.1, 0.0, 1.0).wait()

    input("robot will now make two short rotations")
    robot.set_platform_velocity_target(0.0, 0.0, 0.1, 1.0).wait()
    robot.set_platform_velocity_target(0.0, 0.0, -0.1, 1.0).wait()

    input("robot will now return to original position")
    robot.set_platform_velocity_target(-0.1, -0.1, 0.0, 1.0).wait()

    input("robot will now move to the pose (30 cm, 60 cm, pi/4 rad)")
    robot.move_platform_to_pose(0.3, 0.6, 3.14 / 4, 5.0).wait()
