import unittest
from typing import List, Optional

import numpy as np
from airo_robots.awaitable_action import AwaitableAction
from airo_robots.exceptions import (
    InvalidTrajectoryException,
    RobotSafetyViolationException,
    TrajectoryConstraintViolationException,
)
from airo_robots.manipulators import PositionManipulator
from airo_robots.manipulators.bimanual_position_manipulator import DualArmPositionManipulator
from airo_robots.manipulators.position_manipulator import ManipulatorSpecs
from airo_typing import (
    DualArmTrajectory,
    HomogeneousMatrixType,
    JointConfigurationType,
    JointPathConstraintType,
    JointPathContainer,
    SingleArmTrajectory,
)


class DummyDualArmPositionManipulator(DualArmPositionManipulator):
    def __init__(self):
        left_manipulator = DummyPositionManipulator()
        right_manipulator = DummyPositionManipulator()
        left_manipulator_pose_in_base = np.eye(4)
        right_manipulator_pose_in_base = np.eye(4)
        left_manipulator_pose_in_base[0, 3] = 1.0

        super().__init__(
            left_manipulator, left_manipulator_pose_in_base, right_manipulator, right_manipulator_pose_in_base
        )


class DummyPositionManipulator(PositionManipulator):
    """Dummy implementation of a position manipulator for testing purposes.

    It only implements the functions required for execute_trajectory testing."""

    def __init__(self):
        super().__init__(ManipulatorSpecs([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 1.0))
        self.q = np.zeros(6)

    def get_tcp_pose(self) -> HomogeneousMatrixType:
        pass

    def get_joint_configuration(self) -> JointConfigurationType:
        return self.q

    def move_to_tcp_pose(
        self, tcp_pose: HomogeneousMatrixType, joint_speed: Optional[float] = None
    ) -> AwaitableAction:
        pass

    def move_linear_to_tcp_pose(
        self, tcp_pose: HomogeneousMatrixType, linear_speed: Optional[float] = None
    ) -> AwaitableAction:
        pass

    def move_to_joint_configuration(
        self, joint_configuration: JointConfigurationType, joint_speed: Optional[float] = None
    ) -> AwaitableAction:
        self.q = joint_configuration
        return AwaitableAction(lambda: True)

    def servo_to_tcp_pose(self, tcp_pose: HomogeneousMatrixType, time: float) -> AwaitableAction:
        pass

    def servo_to_joint_configuration(
        self, joint_configuration: JointConfigurationType, time: float
    ) -> AwaitableAction:
        return AwaitableAction(lambda: True)

    def inverse_kinematics(
        self, tcp_pose: HomogeneousMatrixType, joint_configuration_near: Optional[JointConfigurationType] = None
    ) -> Optional[JointConfigurationType]:
        pass

    def forward_kinematics(self, joint_configuration: JointConfigurationType) -> HomogeneousMatrixType:
        pass

    def _is_joint_configuration_reachable(self, joint_configuration: JointConfigurationType) -> bool:
        pass


def create_trajectory(
    qs: List[JointConfigurationType], constraint: Optional[JointPathConstraintType]
) -> SingleArmTrajectory:
    """Creates a trajectory from a list of joint configurations."""
    times = np.arange(0, len(qs), 1.0)
    return SingleArmTrajectory(times, JointPathContainer(np.stack(qs), constraint=constraint))


def create_dual_arm_trajectory(
    qs_left: List[JointConfigurationType],
    qs_right: List[JointConfigurationType],
    constraint_left: Optional[JointPathConstraintType] = None,
    constraint_right: Optional[JointPathConstraintType] = None,
) -> DualArmTrajectory:
    """Creates a trajectory from a list of joint configurations."""
    times = np.arange(0, len(qs_left), 1.0)
    return DualArmTrajectory(
        times,
        JointPathContainer(np.stack(qs_left), constraint=constraint_left),
        JointPathContainer(np.stack(qs_right), constraint=constraint_right),
    )


class TestTrajectoryValidationSingleArm(unittest.TestCase):
    def _get_manipulator(self) -> PositionManipulator:
        return DummyPositionManipulator()

    def test_execution_fails_with_far_start_configuration(self):
        dummy_robot = self._get_manipulator()

        # Define a trajectory with a start configuration that is far from the robot's current configuration
        q_start = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])

        with self.assertRaises(InvalidTrajectoryException):
            dummy_robot.execute_trajectory(create_trajectory([q_start], None))

    def test_execution_fails_without_trajectory_positions(self):
        dummy_robot = self._get_manipulator()

        # Define a trajectory with no positions
        trajectory = SingleArmTrajectory(np.array([0]), JointPathContainer())

        with self.assertRaises(InvalidTrajectoryException):
            dummy_robot.execute_trajectory(trajectory)

    def test_execution_fails_when_start_time_is_not_zero(self):
        dummy_robot = self._get_manipulator()

        # Define a trajectory with a start time that is not zero
        q_start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        trajectory = create_trajectory([q_start], None)
        trajectory.times[0] = 1.0

        with self.assertRaises(InvalidTrajectoryException):
            dummy_robot.execute_trajectory(trajectory)

    def test_execution_fails_when_leading_axis_velocity_is_too_high(self):
        dummy_robot = self._get_manipulator()

        # Define a trajectory with a leading axis velocity that is too high
        q_start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        q_end = np.array([1.0, 1.0, 1.0, 5.0, 1.0, 1.0])
        trajectory = create_trajectory([q_start, q_end], None)
        trajectory.path.velocities = np.stack([q_start - q_start, q_end - q_start])

        with self.assertRaises(RobotSafetyViolationException):
            dummy_robot.execute_trajectory(trajectory)

    def test_execution_succeeds(self):
        dummy_robot = self._get_manipulator()

        # Define a trajectory with a valid start configuration
        q_start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        q_end = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        trajectory = create_trajectory([q_start, q_end], None)

        # Execute the trajectory. If no exceptions are raised, the test will pass.
        dummy_robot.execute_trajectory(trajectory)

    def test_execution_succeeds_with_constraint(self):
        dummy_robot = self._get_manipulator()

        # Add a constraint that checks that joints remain within the [0,1] range.
        class MyConstraint:
            def __call__(self, joint_configuration: JointConfigurationType) -> float:
                return np.sum(np.maximum(joint_configuration - 1.0, 0)) + np.sum(np.maximum(-joint_configuration, 0))

        # Define a trajectory with a valid start configuration
        q_start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        q_end = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        trajectory = create_trajectory([q_start, q_end], (MyConstraint(), 0.0))

        # Execute the trajectory with a constraint
        dummy_robot.execute_trajectory(trajectory)

    def test_invalid_constraint_causes_error(self):
        dummy_robot = self._get_manipulator()

        class InvalidConstraint:
            def __call__(self, joint_configuration: JointConfigurationType) -> float:
                return 1.0

        # Define a trajectory with a valid start configuration
        q_start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        q_end = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        trajectory = create_trajectory([q_start, q_end], (InvalidConstraint(), 0.0))

        with self.assertRaises(TrajectoryConstraintViolationException):
            dummy_robot.execute_trajectory(trajectory)


class TestTrajectoryValidationDualArm(unittest.TestCase):
    def _get_manipulator(self) -> DualArmPositionManipulator:
        return DummyDualArmPositionManipulator()

    def test_execution_fails_with_far_start_configuration(self):
        dummy_robot = self._get_manipulator()

        # Define a trajectory with a start configuration that is far from the robot's current configuration
        q_start_left = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        q_start_right = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        with self.assertRaises(InvalidTrajectoryException):
            dummy_robot.execute_trajectory(create_dual_arm_trajectory([q_start_left], [q_start_right]))

        # Now do the same but with left and right swapped.
        with self.assertRaises(InvalidTrajectoryException):
            dummy_robot.execute_trajectory(create_dual_arm_trajectory([q_start_right], [q_start_left]))

        # Now do the same with two invalid configurations.
        with self.assertRaises(InvalidTrajectoryException):
            dummy_robot.execute_trajectory(create_dual_arm_trajectory([q_start_left], [q_start_left]))

    def test_execution_fails_without_trajectory_positions(self):
        dummy_robot = self._get_manipulator()

        # Define a trajectory with no positions
        trajectory = DualArmTrajectory(np.array([0.0]), JointPathContainer(), JointPathContainer())

        with self.assertRaises(InvalidTrajectoryException):
            dummy_robot.execute_trajectory(trajectory)

    def test_execution_fails_when_start_time_is_not_zero(self):
        dummy_robot = self._get_manipulator()

        # Define a trajectory with a start time that is not zero
        q_start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        trajectory = create_dual_arm_trajectory([q_start], [q_start])
        trajectory.times[0] = 1.0

        with self.assertRaises(InvalidTrajectoryException):
            dummy_robot.execute_trajectory(trajectory)

    def test_execution_fails_when_leading_axis_velocity_is_too_high(self):
        dummy_robot = self._get_manipulator()

        # Define a trajectory with a leading axis velocity that is too high
        q_start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        q_end = np.array([1.0, 1.0, 1.0, 5.0, 1.0, 1.0])
        trajectory = create_dual_arm_trajectory([q_start, q_end], [q_start, q_start])
        trajectory.path_left.velocities = np.stack([q_start - q_start, q_end - q_start])
        trajectory.path_right.velocities = np.stack([q_start - q_start, q_start - q_start])

        with self.assertRaises(RobotSafetyViolationException):
            dummy_robot.execute_trajectory(trajectory)

        # Now do the same with the right arm.
        trajectory = create_dual_arm_trajectory([q_start, q_start], [q_start, q_end])
        trajectory.path_left.velocities = np.stack([q_start - q_start, q_start - q_start])
        trajectory.path_right.velocities = np.stack([q_start - q_start, q_end - q_start])

        with self.assertRaises(RobotSafetyViolationException):
            dummy_robot.execute_trajectory(trajectory)

        # And now with both arms.
        trajectory = create_dual_arm_trajectory([q_start, q_end], [q_start, q_end])
        trajectory.path_left.velocities = np.stack([q_start - q_start, q_start - q_start])
        trajectory.path_right.velocities = np.stack([q_start - q_start, q_end - q_start])
        with self.assertRaises(RobotSafetyViolationException):
            dummy_robot.execute_trajectory(trajectory)

    def test_execution_succeeds(self):
        dummy_robot = self._get_manipulator()

        # Define a trajectory with a valid start configuration
        q_start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        q_end = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        trajectory = create_dual_arm_trajectory([q_start, q_end], [q_start, q_end])

        # Execute the trajectory. If no exceptions are raised, the test will pass.
        dummy_robot.execute_trajectory(trajectory)

    def test_execution_succeeds_with_constraint(self):
        dummy_robot = self._get_manipulator()

        # Add a constraint that checks that joints remain within the [0,1] range.
        class MyConstraint:
            def __call__(self, joint_configuration: JointConfigurationType) -> float:
                return np.sum(np.maximum(joint_configuration - 1.0, 0)) + np.sum(np.maximum(-joint_configuration, 0))

        # Define a trajectory with a valid start configuration
        q_start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        q_end = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        trajectory = create_dual_arm_trajectory(
            [q_start, q_end],
            [q_start, q_end],
            constraint_left=(MyConstraint(), 0.0),
            constraint_right=(MyConstraint(), 0.0),
        )

        # Execute the trajectory with a constraint
        dummy_robot.execute_trajectory(trajectory)

    def test_invalid_constraint_causes_error(self):
        dummy_robot = self._get_manipulator()

        class InvalidConstraint:
            def __call__(self, joint_configuration: JointConfigurationType) -> float:
                return 1.0

        # Define a trajectory with a valid start configuration
        q_start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        q_end = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        trajectory = create_dual_arm_trajectory(
            [q_start, q_end], [q_start, q_end], constraint_left=(InvalidConstraint(), 0.0)
        )

        with self.assertRaises(TrajectoryConstraintViolationException):
            dummy_robot.execute_trajectory(trajectory)

        trajectory = create_dual_arm_trajectory(
            [q_start, q_end], [q_start, q_end], constraint_right=(InvalidConstraint(), 0.0)
        )

        # Now do the same with the right arm.
        with self.assertRaises(TrajectoryConstraintViolationException):
            dummy_robot.execute_trajectory(trajectory)

        trajectory = create_dual_arm_trajectory(
            [q_start, q_end],
            [q_start, q_end],
            constraint_left=(InvalidConstraint(), 0.0),
            constraint_right=(InvalidConstraint(), 0.0),
        )

        # Now with both arms.
        with self.assertRaises(TrajectoryConstraintViolationException):
            dummy_robot.execute_trajectory(trajectory)
