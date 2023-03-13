import numpy as np
from airo_robots.manipulators.position_manipulator import PositionManipulator
from airo_spatial_algebra import SE3Container

# make numpy prints more readable
np.set_printoptions(precision=3)


def manual_test_servo(robot: PositionManipulator, control_freq: int = 500, linear_speed: float = 0.2) -> None:
    """test servo functionality by having robot move to pose and then do zig-zag motion for periods of 1 sec with the specified linear speed
    while sending servo commands at the specified control freq.

    Make sure that the robot is clear of any obstactles!

    Args:
        robot (PositionManipulator): _description_
        control_freq (int, optional): _description_. Defaults to 500.
    """
    start_pose = SE3Container.from_euler_angles_and_translation(
        np.array([0, np.pi, 0.0001]), np.array([0.2, -0.3, 0.02])
    ).homogeneous_matrix
    action = robot.move_linear_to_tcp_pose(start_pose)
    action.wait()
    pose = np.copy(start_pose)
    for i in range(8 * control_freq):
        direction = np.array([1.0, 0.0, -1.0])
        direction /= np.linalg.norm(direction)
        if (i // control_freq) % 2:
            pose[:3, 3] += direction * linear_speed / control_freq
        else:
            pose[:3, 3] -= direction * linear_speed / control_freq

        robot.servo_to_tcp_pose(pose, 1 / control_freq).wait()


def manual_test_ik_fk(robot: PositionManipulator) -> None:

    pose = SE3Container.from_euler_angles_and_translation(
        np.array([0, np.pi, 0.0001]), np.array([0, -0.3, 0.2])
    ).homogeneous_matrix
    joint_config = robot.inverse_kinematics(pose)
    if not joint_config:
        print("IK failed unexpectedly")
        return

    print(f"original pose: \n {pose}")
    print(f"ik joint config = {joint_config}")
    try:
        fk_pose = robot.forward_kinematics(joint_config)
        print(f"FK(IK(pose)): \n {fk_pose}")
        input("FK(IK(pose)) should match original pose press key to continue.")

    except NotImplementedError:
        # catch for UR3e forward kinematics issue
        print("FK not implemented")

    input(
        "when moving to the IK Joint config, the resulting pose should match the desired pose. Press key to start moving"
    )
    robot.move_to_joint_configuration(joint_config).wait()


def manual_test_move(robot: PositionManipulator) -> None:
    start_pose = SE3Container.from_euler_angles_and_translation(
        np.array([0, np.pi / 4 * 3, 0.0001]), np.array([0.1, -0.2, 0.2])
    ).homogeneous_matrix
    end_pose = SE3Container.from_euler_angles_and_translation(
        np.array([0, np.pi, 0.0001]), np.array([0.25, -0.3, 0.1])
    ).homogeneous_matrix
    input(
        "robot should move in a straight line to the start pose and then in joint space to the end pose, press key to start"
    )
    action = robot.move_linear_to_tcp_pose(start_pose)
    print("method returned, will now wait for action to finish")
    action.wait()
    print("action finished, robot will now move back to start pose")
    robot.move_to_tcp_pose(end_pose).wait()
    print("robot movement finished")


def manual_test_robot(robot: PositionManipulator) -> None:
    input(
        "these tests will make the robot move in the +X, -Y quadrant, make sure it is clear of any obstacles! Press key to start"
    )

    print(robot.get_joint_configuration())
    print(robot.get_tcp_pose())
    input(
        "robot printed its pose and joint config, which you should check (on intuition or using the polyscope), press key if this is the case"
    )

    input("IK/FK will now be tested, press key to start")
    manual_test_ik_fk(robot)

    input("move functions will now be tested, press key to start")
    manual_test_move(robot)

    input(
        "servo will now be tested, robot should move in zig-zag patterns of 1 second with smooth motions, press key to start"
    )
    manual_test_servo(robot, 500, 0.15)
