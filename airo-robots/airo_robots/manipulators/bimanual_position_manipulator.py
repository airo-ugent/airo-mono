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

        assert (
            left_tcp_pose_in_base is not None or right_tcp_pose_in_base is not None
        ), "At least one of the TCP poses should be specified"
        awaitables: List[AwaitableAction] = []
        if left_tcp_pose_in_base is not None:
            left_tcp_pose_left_base = self.transform_pose_to_left_arm_base(left_tcp_pose_in_base)
            left_awaitable = self._left_manipulator.move_linear_to_tcp_pose(left_tcp_pose_left_base, linear_speed)
            awaitables.append(left_awaitable)
        if right_tcp_pose_in_base is not None:
            right_tcp_pose_right_base = self.transform_pose_to_right_arm_base(right_tcp_pose_in_base)
            right_awaitable = self._right_manipulator.move_linear_to_tcp_pose(right_tcp_pose_right_base, linear_speed)
            awaitables.append(right_awaitable)

        # compose the awaitable actions
        def done_condition() -> bool:
            return all([awaitable.is_action_done() for awaitable in awaitables])

        return AwaitableAction(done_condition, awaitables[0]._default_timeout, awaitables[0]._default_sleep_resolution)

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

        return AwaitableAction(done_condition, awaitables[0]._default_timeout, awaitables[0]._default_sleep_resolution)

    def transform_pose_to_left_arm_base(self, pose_in_base: HomogeneousMatrixType) -> HomogeneousMatrixType:
        """Transform a pose in the base frame to the left arm base frame"""
        return np.linalg.inv(self._left_manipulator_pose_in_base) @ pose_in_base

    def transform_pose_to_right_arm_base(self, pose_in_base: HomogeneousMatrixType) -> HomogeneousMatrixType:
        """Transform a pose in the base frame to the right arm base frame"""
        return np.linalg.inv(self._right_manipulator_pose_in_base) @ pose_in_base


if __name__ == "__main__":
    """quick test/demo script for a 2-UR5e setup where the robots are approx 0.9cm apart on their shared x-axis"""
    # TODO: should be moved to manual_manipualator_testing but have to to make tests reusable for other setups as well.
    # or properly document what is expected of the robot setup before running this script.
    from airo_robots.manipulators.hardware.ur_rtde import URrtde
    from airo_spatial_algebra import SE3Container

    left = URrtde("10.42.0.162", URrtde.UR3E_CONFIG)
    right = URrtde("10.42.0.163", URrtde.UR3E_CONFIG)
    left_arm_pose_in_base = np.eye(4)
    right_arm_pose_in_base = np.eye(4)
    right_arm_pose_in_base[0, 3] = -0.9
    dual_arm = DualArmPositionManipulator(left, left_arm_pose_in_base, right, right_arm_pose_in_base)

    left_target_pose = SE3Container.from_euler_angles_and_translation(
        np.array([np.pi, 0, 0]), np.array([-0.2, -0.3, 0.3])
    ).homogeneous_matrix

    right_target_pose = SE3Container.from_euler_angles_and_translation(
        np.array([np.pi, 0, 0]), np.array([-0.6, -0.3, 0.3])
    ).homogeneous_matrix
    print(left_target_pose)
    print(dual_arm.left_manipulator.get_tcp_pose())
    dual_arm.move_linear_to_tcp_pose(left_target_pose, right_target_pose, 0.1).wait(timeout=100)

    for _ in range(20):
        left_target_pose[0, 3] += 0.01
        right_target_pose[0, 3] += 0.01
        dual_arm.servo_to_tcp_pose(left_target_pose, right_target_pose, 0.2).wait(timeout=1)
