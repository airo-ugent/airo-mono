import time
from abc import ABC
from typing import List, Optional, Tuple, Union

import numpy as np
from airo_robots.awaitable_action import AwaitableAction
from airo_robots.exceptions import (
    InvalidTrajectoryException,
    RobotConfigurationException,
    TrajectoryConstraintViolationException,
)
from airo_robots.manipulators.position_manipulator import PositionManipulator, evaluate_constraint, lerp_positions
from airo_typing import DualArmTrajectory, HomogeneousMatrixType, JointConfigurationType


class BimanualPositionManipulator(ABC):
    """
    base class for bimanual position-controlled manipulators. This could be a bimanual robot or a combination of 2 unimanual arms
    """

    @property
    def left_manipulator_pose_in_base(self) -> HomogeneousMatrixType:
        raise NotImplementedError

    @property
    def right_manipulator_pose_in_base(self) -> HomogeneousMatrixType:
        raise NotImplementedError


class DualArmPositionManipulator(BimanualPositionManipulator):
    """Class to use two single arm position-controlled manipulators as a bimanual manipulator.
    This is mostly a convenience wrapper to convert the target poses/configurations back from the base frame of this dual-arm setup to the individual base frames of the arms.

    Note that the robots are not guaranteed to move simultaneously, even if the relative poses are the same. This is due to small delays between sending the command to the first robot
    and then sending it to the second robot. If you want them to move 'in sync', you have to use the servo function with a predefined trajectory so that they can self-correct this small difference.
    """

    # TODO: better name for this class. should reflect that this is for using
    # two single-arm robots as a bimanual robot.
    def __init__(
        self,
        left_manipulator: PositionManipulator,
        left_manipulator_pose_in_base: HomogeneousMatrixType,
        right_manipulator: PositionManipulator,
        right_manipulator_pose_in_base: HomogeneousMatrixType,
    ) -> None:
        super().__init__()
        self._left_manipulator = left_manipulator
        self._left_manipulator_pose_in_base = left_manipulator_pose_in_base
        self._right_manipulator = right_manipulator
        self._right_manipulator_pose_in_base = right_manipulator_pose_in_base

    @property
    def left_manipulator(self) -> PositionManipulator:
        return self._left_manipulator

    @property
    def right_manipulator(self) -> PositionManipulator:
        return self._right_manipulator

    @property
    def left_manipulator_pose_in_base(self) -> HomogeneousMatrixType:
        return self._left_manipulator_pose_in_base

    @property
    def right_manipulator_pose_in_base(self) -> HomogeneousMatrixType:
        return self._right_manipulator_pose_in_base

    def _move_to_tcp_pose_shared(
        self,
        move_linear: bool,
        left_tcp_pose_in_base: Union[HomogeneousMatrixType, None],
        right_tcp_pose_in_base: Union[HomogeneousMatrixType, None],
        speed: Optional[float] = None,
    ) -> AwaitableAction:
        """Shared implementation for move_to_tcp_pose and move_linear_to_tcp_pose. Do not call directly."""
        assert (
            left_tcp_pose_in_base is not None or right_tcp_pose_in_base is not None
        ), "At least one of the TCP poses should be specified"
        awaitables: List[AwaitableAction] = []
        if left_tcp_pose_in_base is not None:
            left_tcp_pose_left_base = self.transform_pose_to_left_arm_base(left_tcp_pose_in_base)

            if move_linear:
                left_awaitable = self._left_manipulator.move_linear_to_tcp_pose(left_tcp_pose_left_base, speed)
            else:
                left_awaitable = self._left_manipulator.move_to_tcp_pose(left_tcp_pose_left_base, speed)
            awaitables.append(left_awaitable)

        if right_tcp_pose_in_base is not None:
            right_tcp_pose_right_base = self.transform_pose_to_right_arm_base(right_tcp_pose_in_base)
            if move_linear:
                right_awaitable = self._right_manipulator.move_linear_to_tcp_pose(right_tcp_pose_right_base, speed)
            else:
                right_awaitable = self._right_manipulator.move_to_tcp_pose(right_tcp_pose_right_base, speed)

            awaitables.append(right_awaitable)

        # compose the awaitable actions
        def done_condition() -> bool:
            return all([awaitable.is_action_done() for awaitable in awaitables])

        return AwaitableAction(
            done_condition,
            awaitables[0]._default_timeout,
            awaitables[0]._default_sleep_resolution,
        )

    def move_to_tcp_pose(
        self,
        left_tcp_pose_in_base: Union[HomogeneousMatrixType, None],
        right_tcp_pose_in_base: Union[HomogeneousMatrixType, None],
        joint_speed: Optional[float] = None,
    ) -> AwaitableAction:
        """Move both arms to a given TCP pose in the base frame. You can specify None to not move one of the arms.
        Args:
            left_tcp_pose_in_base: The TCP pose of the left arm in the base frame. If None is specified, the left arm will not move.
            right_tcp_pose_in_base: The TCP pose of the right arm in the base frame. If None is specified, the right arm will not move.
            joint_speed: Speed for the joint movements in rad/s. If not specified, the default speed of the manipulator is used.

        Returns:
            awaitable with termination condition that both robots have reached their target pose or have timed out.
        """
        return self._move_to_tcp_pose_shared(False, left_tcp_pose_in_base, right_tcp_pose_in_base, joint_speed)

    def move_linear_to_tcp_pose(
        self,
        left_tcp_pose_in_base: Union[HomogeneousMatrixType, None],
        right_tcp_pose_in_base: Union[HomogeneousMatrixType, None],
        linear_speed: Optional[float] = None,
    ) -> AwaitableAction:
        """Move both arms to a given TCP pose in the base frame. You can specify None to not move one of the arms.
        Args:
            left_tcp_pose_in_base: The TCP pose of the left arm in the base frame. If None is specified, the left arm will not move.
            right_tcp_pose_in_base: The TCP pose of the right arm in the base frame. If None is specified, the right arm will not move.
            linear_speed: The linear speed of the end effector in m/s. If not specified, the default speed of the manipulator is used.

        Returns:
            awaitable with termination condition that both robots have reached their target pose or have timed out.
        """
        return self._move_to_tcp_pose_shared(True, left_tcp_pose_in_base, right_tcp_pose_in_base, linear_speed)

    def servo_to_tcp_pose(
        self,
        left_tcp_pose_in_base: Union[HomogeneousMatrixType, None],
        right_tcp_pose_in_base: Union[HomogeneousMatrixType, None],
        time: float,
    ) -> AwaitableAction:
        """
        Servo both arms to a given TCP pose in the base frame. You can specify None to not move one of the arms.
        Args:
            left_tcp_pose_in_base: The TCP pose of the left arm in the base frame. If None is specified, the left arm will not move.
            right_tcp_pose_in_base: The TCP pose of the right arm in the base frame. If None is specified, the right arm will not move.
            time: The time in seconds to reach the target pose.

        Returns:
            awaitable with termination condition that the time has passed. Waiting on this action has limited accuracy on non real-time OS, cf the airo-robots Readme.
        """
        if left_tcp_pose_in_base is None and right_tcp_pose_in_base is None:
            raise RobotConfigurationException("At least one of the TCP poses should be specified")

        awaitables: List[AwaitableAction] = []
        if left_tcp_pose_in_base is not None:
            left_tcp_pose_left_base = self.transform_pose_to_left_arm_base(left_tcp_pose_in_base)
            left_awaitable = self._left_manipulator.servo_to_tcp_pose(left_tcp_pose_left_base, time)
            awaitables.append(left_awaitable)
        if right_tcp_pose_in_base is not None:
            right_tcp_pose_right_base = self.transform_pose_to_right_arm_base(right_tcp_pose_in_base)
            right_awaitable = self._right_manipulator.servo_to_tcp_pose(right_tcp_pose_right_base, time)
            awaitables.append(right_awaitable)

        # compose the awaitable actions
        def done_condition() -> bool:
            return all([awaitable.is_action_done() for awaitable in awaitables])

        return AwaitableAction(
            done_condition,
            awaitables[0]._default_timeout,
            awaitables[0]._default_sleep_resolution,
        )

    def servo_to_joint_configuration(
        self,
        left_joint_configuration: JointConfigurationType,
        right_joint_configuration: JointConfigurationType,
        time: float,
    ) -> AwaitableAction:
        """Servo to the desired joint configuration for the specified time (the function blocks for this time). Servoing implies 'best-effort' movements towards the target pose instead of
        open-loop trajectories with a velocity profile that brings the robot to zero. So this function can be used for 'closed-loop'/higher-frequency control.

        See PositionManipulator.servo_to_joint_configuration for more information.

        Args:
            left_joint_configuration (JointConfigurationType): desired joint configuration for the left manipulator
            right_joint_configuration (JointConfigurationType): desired joint configuration for the right manipulator
            time (float): time to reach the desired joint configuration

        Returns:
            AwaitableAction: with termination condition that the time has passed. Waiting on this action has limited accuracy on non real-time OS, cf the airo-robots Readme.
        """
        left_awaitable = self._left_manipulator.servo_to_joint_configuration(left_joint_configuration, time)
        right_awaitable = self._right_manipulator.servo_to_joint_configuration(right_joint_configuration, time)
        awaitables: Tuple[AwaitableAction, AwaitableAction] = (left_awaitable, right_awaitable)

        # compose the awaitable actions
        def done_condition() -> bool:
            return all([awaitable.is_action_done() for awaitable in awaitables])

        return AwaitableAction(
            done_condition,
            awaitables[0]._default_timeout,
            awaitables[0]._default_sleep_resolution,
        )

    def execute_trajectory(
        self,
        joint_trajectory: DualArmTrajectory,
        sampling_frequency: float = 100.0,
    ) -> None:
        """Execute a joint trajectory. This function will interpolate the trajectory and send the commands to the robot.
        The gripper trajectory (if any) will be ignored.

        This function is implemented according to the notes of https://github.com/airo-ugent/airo-mono/issues/150.
        Please refer to this issue for design decisions.

        To ensure that the trajectories of both manipulators in the DualArmPositionManipulator are executed in a synchronized manner,
        the implementation of execute_trajectory in this class does not delegate to the _left_manipulator (resp. _right)'s implementation,
        but is its own implementation with some code duplication.

        Args:
            joint_trajectory: the joint trajectory to execute.
            sampling_frequency: The frequency at which the trajectory is sampled and commands are sent to the robot. This is a best-effort parameter, and the actual frequency may be lower due to the time it takes to send the commands to the robot or other computations. The default is 100 Hz."""
        self._assert_joint_trajectory_is_executable(joint_trajectory, sampling_frequency)

        period = (
            1 / sampling_frequency
        )  # Time per servo, approximately. This may be slightly changed because of rounding errors.
        # The period determines the times at which we sample the trajectory that was time-parameterized.
        duration = (joint_trajectory.times[-1] - joint_trajectory.times[0]).item()

        n_servos = int(np.ceil(duration / period))
        period_adjusted = duration / n_servos  # can be slightly different from period due to rounding

        logged_lag_warning = False
        loop_start_time_ns = time.time_ns()
        for servo_index in range(n_servos):
            iteration_start_time_ns = time.time_ns()
            t_ns = iteration_start_time_ns - loop_start_time_ns
            t = t_ns / 1e9
            if t > duration:
                logger.warning(
                    f"Time exceeded trajectory duration at servo index {servo_index} / {n_servos}. This means we are lagging, but we should have reached the final configuration. Stopping trajectory execution."
                )
                break

            # Find the two joint configurations that are closest to time t.
            i0 = np.searchsorted(joint_trajectory.times, t, side="left") - 1  # - 1: i0 is always >= 1 otherwise.
            i1 = i0 + 1

            if i1 == len(joint_trajectory.times):
                break

            # Interpolate between the two joint configurations.
            q_interp_left = lerp_positions(i0, i1, joint_trajectory.path_left.positions, joint_trajectory.times, t)
            q_interp_right = lerp_positions(i0, i1, joint_trajectory.path_right.positions, joint_trajectory.times, t)
            self.servo_to_joint_configuration(q_interp_left, q_interp_right, period_adjusted)
            # We do not wait for the servo to finish, because we want to sample the trajectory at a fixed rate and avoid lagging.

            iter_duration_ns = time.time_ns() - iteration_start_time_ns
            period_adjusted_ns = int(period_adjusted * 1e9)
            # We want to wait for the period, but we also want to avoid waiting too long if the iteration took too long.
            # Sleeping is not very accurate (see airo_robots/scripts/measure_sleep_accuracy.py), so we busy-wait for the period.
            if iter_duration_ns < period_adjusted_ns:
                current_time = time.time_ns()
                while time.time_ns() < current_time + (period_adjusted_ns - iter_duration_ns):
                    pass
            else:
                if not logged_lag_warning:
                    logger.warning(
                        "Trajectory execution is lagging behind! This can cause large jumps with ServoJ, and should be avoided."
                    )
                    logged_lag_warning = True

        # This avoids the abrupt stop and "thunk" sounds at the end of paths that end with non-zero velocity
        # However, I believe these functions are blocking, so right only stops after left has stopped.
        # Specifically for UR robots.
        self.servo_stop()

        # Servo can overshoot. Do a final move to the last configuration.
        if joint_trajectory.path_left is not None:
            left_finished = self._left_manipulator.move_to_joint_configuration(
                joint_trajectory.path_left.positions[-1]
            )
        else:
            left_finished = AwaitableAction(lambda: True, 0.0, 0.0)

        if joint_trajectory.path_right is not None:
            right_finished = self._right_manipulator.move_to_joint_configuration(
                joint_trajectory.path_right.positions[-1]
            )
        else:
            right_finished = AwaitableAction(lambda: True, 0.0, 0.0)

        left_finished.wait()
        right_finished.wait()

    def servo_stop(self):
        if hasattr(self._left_manipulator, "rtde_control"):
            self._left_manipulator.rtde_control.servoStop(2.0)
        else:
            logger.warning("Left manipulator does not support servo stop.")
        if hasattr(self._right_manipulator, "rtde_control"):
            self._right_manipulator.rtde_control.servoStop(2.0)
        else:
            logger.warning("Right manipulator does not support servo stop.")

    def transform_pose_to_left_arm_base(self, pose_in_base: HomogeneousMatrixType) -> HomogeneousMatrixType:
        """Transform a pose in the base frame to the left arm base frame"""
        return np.linalg.inv(self._left_manipulator_pose_in_base) @ pose_in_base

    def transform_pose_to_right_arm_base(self, pose_in_base: HomogeneousMatrixType) -> HomogeneousMatrixType:
        """Transform a pose in the base frame to the right arm base frame"""
        return np.linalg.inv(self._right_manipulator_pose_in_base) @ pose_in_base

    def is_tcp_pose_reachable_for_left(self, tcp_pose_in_base: HomogeneousMatrixType) -> bool:
        tcp_pose_left_base = self.transform_pose_to_left_arm_base(tcp_pose_in_base)
        return self._left_manipulator.is_tcp_pose_reachable(tcp_pose_left_base)

    def is_tcp_pose_reachable_for_right(self, tcp_pose_in_base: HomogeneousMatrixType) -> bool:
        tcp_pose_right_base = self.transform_pose_to_right_arm_base(tcp_pose_in_base)
        return self._right_manipulator.is_tcp_pose_reachable(tcp_pose_right_base)

    def are_tcp_poses_reachable(
        self,
        left_tcp_pose_in_base: HomogeneousMatrixType,
        right_tcp_pose_in_base: HomogeneousMatrixType,
    ) -> bool:
        left_reachable = self.is_tcp_pose_reachable_for_left(left_tcp_pose_in_base)
        right_reachable = self.is_tcp_pose_reachable_for_right(right_tcp_pose_in_base)
        return left_reachable and right_reachable

    def _assert_joint_trajectory_is_executable(
        self,
        joint_trajectory: DualArmTrajectory,
        sampling_frequency: float,
    ) -> None:
        if joint_trajectory.times[0] != 0.0:
            raise InvalidTrajectoryException("joint trajectory should start at time 0.0")

        if joint_trajectory.path_left.positions is None:
            raise InvalidTrajectoryException("left joint trajectory should contain joint positions")

        if joint_trajectory.path_right.positions is None:
            raise InvalidTrajectoryException("right joint trajectory should contain joint positions")

        if not self._left_manipulator._is_joint_configuration_nearby(joint_trajectory.path_left.positions[0]):
            raise InvalidTrajectoryException(
                f"joint trajectory should start at the current configuration {self._left_manipulator.get_joint_configuration()}, "
                f"but starts at {joint_trajectory.path_left.positions[0]}"
            )
        if not self._right_manipulator._is_joint_configuration_nearby(joint_trajectory.path_right.positions[0]):
            raise InvalidTrajectoryException(
                f"joint trajectory should start at the current configuration {self._right_manipulator.get_joint_configuration()}, "
                f"but starts at {joint_trajectory.path_right.positions[0]}"
            )

        self._assert_trajectory_constraints_satisfied(joint_trajectory, sampling_frequency)

        if joint_trajectory.path_left.velocities is not None:
            leading_axis_velocity = np.max(np.abs(joint_trajectory.path_left.velocities), axis=1)
            for velocity in leading_axis_velocity:
                self._left_manipulator._assert_joint_speed_is_valid(velocity)

        if joint_trajectory.path_right.velocities is not None:
            leading_axis_velocity = np.max(np.abs(joint_trajectory.path_right.velocities), axis=1)
            for velocity in leading_axis_velocity:
                self._right_manipulator._assert_joint_speed_is_valid(velocity)

    def _assert_trajectory_constraints_satisfied(self, joint_trajectory: DualArmTrajectory, sampling_frequency: float):
        if joint_trajectory.path_left.constraint is not None:
            constraint_satisfied = evaluate_constraint(
                joint_trajectory.path_left.positions,
                joint_trajectory.times,
                joint_trajectory.path_left.constraint,
                sampling_frequency,
            )
            if not constraint_satisfied:
                raise TrajectoryConstraintViolationException(
                    "left joint trajectory does not satisfy the trajectory constraint."
                )
            constraint_satisfied = evaluate_constraint(
                joint_trajectory.path_right.positions,
                joint_trajectory.times,
                joint_trajectory.path_right.constraint,
                sampling_frequency,
            )
            if not constraint_satisfied:
                raise TrajectoryConstraintViolationException(
                    "right joint trajectory does not satisfy the trajectory constraint."
                )


if __name__ == "__main__":
    """quick test/demo script for a 2-UR5e setup where the robots are approx 0.9cm apart on their shared x-axis"""
    # TODO: should be moved to manual_manipualator_testing but have to to make tests reusable for other setups as well.
    # or properly document what is expected of the robot setup before running this script.
    from airo_robots.manipulators.hardware.ur_rtde import URrtde
    from airo_spatial_algebra import SE3Container
    from loguru import logger

    logger.info("DualArmPositionManipulator test started.")

    np.set_printoptions(precision=3, suppress=True)

    left = URrtde("10.42.0.162", URrtde.UR3E_CONFIG)
    right = URrtde("10.42.0.163", URrtde.UR3E_CONFIG)
    left_arm_pose_in_base = np.eye(4)
    right_arm_pose_in_base = np.eye(4)
    right_arm_pose_in_base[0, 3] = -0.9
    dual_arm = DualArmPositionManipulator(left, left_arm_pose_in_base, right, right_arm_pose_in_base)

    # Move to start joint configurations so we are sure the TCP poses are reachable without passing through singularities
    joints_start = np.deg2rad([-90, -90, -90, -90, 90, 0])

    logger.info("Moving to start joint configurations")
    dual_arm.left_manipulator.move_to_joint_configuration(joints_start).wait()
    dual_arm.right_manipulator.move_to_joint_configuration(joints_start).wait()

    left_target_pose = SE3Container.from_euler_angles_and_translation(
        np.array([np.pi, 0, 0]), np.array([-0.2, -0.3, 0.3])
    ).homogeneous_matrix

    right_target_pose = SE3Container.from_euler_angles_and_translation(
        np.array([np.pi, 0, 0]), np.array([-0.6, -0.3, 0.3])
    ).homogeneous_matrix

    logger.info("Checking if target poses are reachable")
    left_reachable = dual_arm.is_tcp_pose_reachable_for_left(left_target_pose)
    right_reachable = dual_arm.is_tcp_pose_reachable_for_right(right_target_pose)
    both_reachable = dual_arm.are_tcp_poses_reachable(left_target_pose, right_target_pose)
    print("Left target pose:")
    print(left_target_pose)
    print("Left reachable: ", left_reachable)
    print("Left arm current pose in base:")
    print(dual_arm.left_manipulator.get_tcp_pose())
    print("Right target pose:")
    print(right_target_pose)
    print("Right reachable: ", right_reachable)
    print("Right arm current pose in base:")
    print(dual_arm.right_manipulator.get_tcp_pose())
    print("Both poses reachable: ", both_reachable)

    logger.info("Moving to target poses")
    dual_arm.move_to_tcp_pose(left_target_pose, right_target_pose, joint_speed=0.5).wait(timeout=10)

    time.sleep(1.0)

    logger.info("Starting servo movements.")
    for _ in range(200):
        left_target_pose[0, 3] += 0.001
        right_target_pose[0, 3] += 0.001
        dual_arm.servo_to_tcp_pose(left_target_pose, right_target_pose, 0.02).wait(timeout=1)

    logger.info("Moving linearly")
    left_target_pose[1, 3] -= 0.2
    right_target_pose[1, 3] -= 0.2
    time.sleep(0.5)
    dual_arm.move_linear_to_tcp_pose(left_target_pose, right_target_pose, 0.1).wait(timeout=10)

    logger.info("DualArmPositionManipulator test finished.")
