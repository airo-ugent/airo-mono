import time
from abc import ABC
from typing import List, Optional, Union

import numpy as np
from airo_robots.awaitable_action import AwaitableAction
from airo_robots.manipulators.position_manipulator import PositionManipulator
from airo_typing import HomogeneousMatrixType


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
        assert (
            left_tcp_pose_in_base is not None or right_tcp_pose_in_base is not None
        ), "At least one of the TCP poses should be specified"

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
