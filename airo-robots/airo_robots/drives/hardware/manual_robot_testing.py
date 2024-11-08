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

    input("robot will now move to 10 cm along +X relative from its starting position with position control")
    robot.move_platform_to_pose(0.1, 0.0, 0.0, 10.0).wait()


if __name__ == "__main__":
    mobi = KELORobile("10.10.129.21")
    manually_test_robot_implementation(mobi)
