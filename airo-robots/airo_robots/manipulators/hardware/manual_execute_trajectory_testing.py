import numpy as np
from airo_robots.grippers import Robotiq2F85
from airo_robots.manipulators.position_manipulator import PositionManipulator
from airo_typing import JointPathContainer, SingleArmTrajectory

# make numpy prints more readable
np.set_printoptions(precision=3)


def _create_scene():
    from airo_drake import add_floor, add_manipulator, add_meshcat, finish_build
    from pydrake.planning import RobotDiagramBuilder

    robot_diagram_builder = RobotDiagramBuilder()

    meshcat = add_meshcat(robot_diagram_builder)
    arm_index, gripper_index = add_manipulator(robot_diagram_builder, "ur5e", "robotiq_2f_85", static_gripper=True)
    add_floor(robot_diagram_builder)

    robot_diagram, context = finish_build(robot_diagram_builder, meshcat)
    del robot_diagram_builder  # no longer needed

    return robot_diagram, context, meshcat, arm_index


def manual_test_trajectory(robot: PositionManipulator) -> None:
    """Test the trajectory execution of the robot by moving to a start pose and then executing a simple trajectory."""
    from airo_drake import animate_joint_trajectory, discretize_drake_joint_trajectory, time_parametrize_toppra

    robot_diagram, context, meshcat, arm_index = _create_scene()

    q_start = np.array([1.6067, -2.3319, 2.0195, 0.99888, 1.5795, 3.2378])
    print(
        f"The robot will move to the start configuration:\n{q_start}\nMake sure the robot is clear of any obstacles!"
    )
    input("Press any key to continue.")
    robot.move_to_joint_configuration(q_start).wait()

    print(
        "The robot will now execute a simple trajectory, moving the arm forward and back. Make sure the robot is clear of any obstacles!"
    )
    print("Check the meshcat window to see the robot's current pose and the computed trajectory.")
    q_goal1 = np.array([1.6067, -1.4846, 1.734, -0.29404, 1.5756, 3.2427])
    path = np.stack([q_start, q_goal1, q_start])

    trajectory = time_parametrize_toppra(robot_diagram.plant(), path, 0.5, 1.0)
    trajectory = discretize_drake_joint_trajectory(trajectory)
    animate_joint_trajectory(meshcat, robot_diagram, arm_index, trajectory)

    input("Press any key to continue.")
    robot.execute_trajectory(trajectory)

    print(
        "The robot will now execute a longer trajectory, moving the arm in a circular motion. Make sure the robot is clear of any obstacles!"
    )
    print("Check the meshcat window to see the robot's current pose and the computed trajectory.")
    q_goal2 = np.array([0.96853, -1.7022, 2.0413, -0.41677, 1.6038, 3.2449])
    q_goal3 = np.array([0.82242, -1.7172, 1.248, 0.44915, 1.4598, 1.3976])
    q_goal4 = np.array([1.4904, -1.2562, 1.2875, 0.82876, 0.99272, 2.8237])
    q_goal5 = q_goal1.copy()
    path = np.stack([q_start, q_goal1, q_goal2, q_goal3, q_goal4, q_goal5, q_start])

    trajectory = time_parametrize_toppra(robot_diagram.plant(), path)
    trajectory = discretize_drake_joint_trajectory(trajectory)
    animate_joint_trajectory(meshcat, robot_diagram, arm_index, trajectory)

    input("Press any key to continue.")
    robot.execute_trajectory(trajectory)


def manual_test_gripper_trajectory(robot: PositionManipulator) -> None:
    """Test the gripper trajectory execution of the robot by moving to a start pose and then executing a simple trajectory."""
    print("The robot will now open and close the gripper.")
    input("Press any key to continue.")
    robot.gripper.open().wait()
    robot.gripper.close().wait()

    print(
        "Now, we will execute a gripper trajectory that opens the gripper. Make sure the gripper is clear of any obstacles!"
    )
    q_current = robot.get_joint_configuration().copy()
    q_path = np.array([q_current, q_current, q_current])
    positions = np.array([0.05, 0.5, 1.0])
    times = np.array([0, 0.5, 1.0])
    trajectory = SingleArmTrajectory(times, JointPathContainer(q_path), JointPathContainer(positions))

    input("Press any key to continue.")
    robot.execute_trajectory(trajectory)

    print(
        "Now, we will execute a gripper trajectory that opens and closes the gripper. Make sure the gripper is clear of any obstacles!"
    )
    q_current = robot.get_joint_configuration().copy()
    q_path = np.array([q_current, q_current, q_current, q_current, q_current, q_current])
    positions = np.array([0.05, 0.5, 0.25, 0.75, 0.5, 1.0])
    times = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    trajectory = SingleArmTrajectory(times, JointPathContainer(q_path), JointPathContainer(positions))

    input("Press any key to continue.")
    robot.execute_trajectory(trajectory)


if __name__ == "__main__":
    print(
        "This script will execute some trajectories on the robot arm. Please make sure you have a safe working environment. "
        "Remember to remove any additional hardware from the robot arm that could cause collisions."
    )
    print("This script requires some additional dependencies, which can be installed with the following command:")
    print("pip install airo-drake==0.0.5 ur_analytic_ik")

    import click
    from airo_robots.manipulators import URrtde

    @click.command()
    @click.option("--ip_address", help="IP address of the UR robot")
    def test_ur_trajectory(ip_address: str) -> None:
        print(f"{ip_address=}")
        gripper = Robotiq2F85(ip_address)
        robot = URrtde(ip_address, URrtde.UR3E_CONFIG, gripper)
        manual_test_trajectory(robot)
        manual_test_gripper_trajectory(robot)

    test_ur_trajectory()
