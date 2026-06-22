"""Position-controlled manipulator implementation for RealMan robots."""

import importlib
import threading
import time
from types import ModuleType
from typing import Any, Optional

import numpy as np
from airo_robots.awaitable_action import AwaitableAction
from airo_robots.grippers import ParallelPositionGripper
from airo_robots.manipulators.position_manipulator import ManipulatorSpecs, PositionManipulator
from airo_spatial_algebra import SE3Container
from airo_typing import HomogeneousMatrixType, JointConfigurationType
from loguru import logger

RealmanPoseType = np.ndarray
"""A RealMan pose ``[x, y, z, rx, ry, rz]`` in metres and radians."""


def _import_realman_api() -> ModuleType:
    """Import the optional RealMan SDK with an actionable error message."""
    try:
        return importlib.import_module("Robotic_Arm.rm_robot_interface")
    except ImportError as exception:
        raise ImportError(
            'RealmanControl requires the RealMan Python SDK. Install it with `pip install "airo-robots[realman]"`.'
        ) from exception


class RealmanControl(PositionManipulator):
    """Control a RealMan robot through the official ``Robotic_Arm`` Python SDK.

    AIRO represents joint angles in radians, while the RealMan API represents
    joint angles in degrees. This class performs that conversion at the API
    boundary. Cartesian positions and Euler angles use metres and radians in
    both APIs.

    Motion commands are sent in non-blocking mode. Their returned
    :class:`AwaitableAction` completes after the RealMan trajectory event has
    reported success and the requested target is within tolerance.

    No collision checking or obstacle avoidance is performed beyond the
    checks configured on the robot controller.

    Args:
        ip_address: IP address of the robot controller.
        port: Robot controller port. RealMan controllers use 8080 by default.
        manipulator_specs: Optional specification override. By default, maximum
            joint and TCP speeds are queried from the controller.
        gripper: Optional gripper associated with this manipulator.
    """

    def __init__(
        self,
        ip_address: str,
        port: int = 8080,
        manipulator_specs: Optional[ManipulatorSpecs] = None,
        gripper: Optional[ParallelPositionGripper] = None,
    ) -> None:
        self.ip_address = ip_address
        self.port = port
        self._api = _import_realman_api()
        self.robot: Any = self._api.RoboticArm(self._api.rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.handle: Any = self.robot.rm_create_robot_arm(ip_address, port)
        if self.handle.id == -1:
            raise RuntimeError(
                f"Could not connect to the RealMan robot at {ip_address}:{port}. "
                "Check the IP address, controller state, and network connection."
            )

        robot_info = self._get_api_value("rm_get_robot_info")
        self.dof = int(robot_info["arm_dof"])
        # The SDK's Python wrapper converts the model and force-sensor enums to names.
        self.model = str(robot_info["arm_model"])
        self.force_type = str(robot_info["force_type"])

        if manipulator_specs is None:
            max_joint_speeds_degrees = np.asarray(self._get_api_value("rm_get_joint_max_speed"), dtype=float)
            max_linear_speed = float(self._get_api_value("rm_get_arm_max_line_speed"))
            manipulator_specs = ManipulatorSpecs(
                np.radians(max_joint_speeds_degrees).tolist(),
                max_linear_speed,
            )
        if manipulator_specs.dof != self.dof:
            raise ValueError(
                f"Manipulator specs have {manipulator_specs.dof} joints, but the connected robot has {self.dof}."
            )

        super().__init__(manipulator_specs, gripper)

        self._joint_lower_limits = np.radians(np.asarray(self._get_api_value("rm_get_joint_min_pos"), dtype=float))
        self._joint_upper_limits = np.radians(np.asarray(self._get_api_value("rm_get_joint_max_pos"), dtype=float))
        self._validate_controller_array(self._joint_lower_limits, "minimum joint limits")
        self._validate_controller_array(self._joint_upper_limits, "maximum joint limits")

        self._pose_reached_l2_threshold = 0.01
        self._joint_configuration_reached_l2_threshold = 0.01
        self._motion_completed = threading.Event()
        self._last_motion_succeeded = False
        # Keep a reference: ctypes callbacks are invalid after garbage collection.
        self._event_callback = self._api.rm_event_callback_ptr(self._handle_robot_event)
        self.robot.rm_get_arm_event_call_back(self._event_callback)
        self._closed = False

        logger.info(
            f"Connected to RealMan robot model {self.model} at {self.ip_address}:{self.port} ({self.dof} DoF)."
        )
        self.robot.rm_set_avoid_singularity_mode(1)

    def close(self) -> None:
        """Close the connection to the robot controller."""
        if self._closed:
            return
        self._raise_for_error(self.robot.rm_delete_robot_arm(), "rm_delete_robot_arm")
        self._closed = True

    def __enter__(self) -> "RealmanControl":
        return self

    def __exit__(self, exception_type: Any, exception: Any, traceback: Any) -> None:
        self.close()

    def get_joint_configuration(self) -> JointConfigurationType:
        """Get the current joint configuration in radians."""
        joint_configuration_degrees = np.asarray(self._get_api_value("rm_get_joint_degree"), dtype=float)
        self._validate_controller_array(joint_configuration_degrees, "joint configuration")
        return np.radians(joint_configuration_degrees)

    def get_tcp_pose(self) -> HomogeneousMatrixType:
        """Get the current TCP pose as a homogeneous matrix."""
        state = self._get_api_value("rm_get_current_arm_state")
        return self._convert_realman_pose_to_homogeneous_pose(np.asarray(state["pose"], dtype=float))

    def move_linear_to_tcp_pose(
        self, tcp_pose: HomogeneousMatrixType, linear_speed: Optional[float] = None
    ) -> AwaitableAction:
        """Move the TCP to a pose along a Cartesian straight line."""
        self._assert_pose_is_valid(tcp_pose)
        speed = self.default_linear_speed if linear_speed is None else linear_speed
        self._assert_positive_speed(speed, "linear_speed")
        self._assert_linear_speed_is_valid(speed)
        speed_percentage = self._speed_to_percentage(speed, self.manipulator_specs.max_linear_speed)
        realman_pose = self._convert_homogeneous_pose_to_realman_pose(tcp_pose)

        self._prepare_motion()
        self._raise_for_error(
            self.robot.rm_movel(realman_pose.tolist(), speed_percentage, 0, 0, 0),
            "rm_movel",
        )
        return self._pose_motion_action(tcp_pose)

    def move_to_tcp_pose(
        self, tcp_pose: HomogeneousMatrixType, joint_speed: Optional[float] = None
    ) -> AwaitableAction:
        """Move the TCP to a pose using joint-space planning."""
        self._assert_pose_is_valid(tcp_pose)
        speed = self.default_joint_speed if joint_speed is None else joint_speed
        self._assert_positive_speed(speed, "joint_speed")
        self._assert_joint_speed_is_valid(speed)
        speed_percentage = self._speed_to_percentage(speed, min(self.manipulator_specs.max_joint_speeds))
        realman_pose = self._convert_homogeneous_pose_to_realman_pose(tcp_pose)

        self._prepare_motion()
        self._raise_for_error(
            self.robot.rm_movej_p(realman_pose.tolist(), speed_percentage, 0, 0, 0),
            "rm_movej_p",
        )
        return self._pose_motion_action(tcp_pose)

    def move_to_joint_configuration(
        self,
        joint_configuration: JointConfigurationType,
        joint_speed: Optional[float] = None,
    ) -> AwaitableAction:
        """Move to a joint configuration using controller trajectory planning."""
        self._assert_joint_configuration_is_valid(joint_configuration)
        speed = self.default_joint_speed if joint_speed is None else joint_speed
        self._assert_positive_speed(speed, "joint_speed")
        self._assert_joint_speed_is_valid(speed)
        speed_percentage = self._speed_to_percentage(speed, min(self.manipulator_specs.max_joint_speeds))

        target = np.asarray(joint_configuration, dtype=float)
        self._prepare_motion()
        self._raise_for_error(
            self.robot.rm_movej(np.degrees(target).tolist(), speed_percentage, 0, 0, 0),
            "rm_movej",
        )
        return AwaitableAction(
            lambda: self._motion_succeeded()
            and bool(
                np.linalg.norm(self.get_joint_configuration() - target)
                < self._joint_configuration_reached_l2_threshold
            )
        )

    def servo_to_tcp_pose(self, tcp_pose: HomogeneousMatrixType, duration: float) -> AwaitableAction:
        """Send a Cartesian CANFD pass-through setpoint."""
        self._assert_positive_duration(duration)
        pose = self._convert_homogeneous_pose_to_realman_pose(tcp_pose)
        high_follow = duration <= 0.01
        self._raise_for_error(self.robot.rm_movep_canfd(pose.tolist(), high_follow), "rm_movep_canfd")
        return self._duration_action(duration)

    def servo_to_joint_configuration(
        self, joint_configuration: JointConfigurationType, duration: float
    ) -> AwaitableAction:
        """Send a joint CANFD pass-through setpoint."""
        self._assert_positive_duration(duration)
        joint_configuration_array = np.asarray(joint_configuration, dtype=float)
        if joint_configuration_array.shape != (self.dof,):
            raise ValueError(f"Expected a joint configuration with shape ({self.dof},).")
        high_follow = duration <= 0.01
        self._raise_for_error(
            self.robot.rm_movej_canfd(np.degrees(joint_configuration_array).tolist(), high_follow),
            "rm_movej_canfd",
        )
        return self._duration_action(duration)

    def inverse_kinematics(
        self,
        tcp_pose: HomogeneousMatrixType,
        joint_configuration_near: Optional[JointConfigurationType] = None,
    ) -> Optional[JointConfigurationType]:
        """Solve inverse kinematics, returning radians or ``None`` on failure."""
        if joint_configuration_near is None:
            joint_configuration_near = self.get_joint_configuration()
        near = np.asarray(joint_configuration_near, dtype=float)
        if near.shape != (self.dof,):
            raise ValueError(f"Expected a joint configuration with shape ({self.dof},).")

        pose = self._convert_homogeneous_pose_to_realman_pose(tcp_pose)
        params = self._api.rm_inverse_kinematics_params_t(np.degrees(near).tolist(), pose.tolist(), 1)
        error_code, solution_degrees = self.robot.rm_algo_inverse_kinematics(params)
        if error_code != 0:
            return None
        return np.radians(np.asarray(solution_degrees, dtype=float))

    def forward_kinematics(self, joint_configuration: JointConfigurationType) -> HomogeneousMatrixType:
        """Calculate the TCP pose for a joint configuration."""
        joint_configuration_array = np.asarray(joint_configuration, dtype=float)
        if joint_configuration_array.shape != (self.dof,):
            raise ValueError(f"Expected a joint configuration with shape ({self.dof},).")
        pose = self.robot.rm_algo_forward_kinematics(np.degrees(joint_configuration_array).tolist(), 1)
        return self._convert_realman_pose_to_homogeneous_pose(np.asarray(pose, dtype=float))

    def _is_joint_configuration_reachable(self, joint_configuration: JointConfigurationType) -> bool:
        joint_configuration_array = np.asarray(joint_configuration, dtype=float)
        return bool(
            joint_configuration_array.shape == (self.dof,)
            and np.all(joint_configuration_array >= self._joint_lower_limits)
            and np.all(joint_configuration_array <= self._joint_upper_limits)
        )

    def _get_api_value(self, method_name: str) -> Any:
        error_code, value = getattr(self.robot, method_name)()
        self._raise_for_error(error_code, method_name)
        return value

    @staticmethod
    def _raise_for_error(error_code: int, method_name: str) -> None:
        if error_code != 0:
            raise RuntimeError(f"RealMan API call {method_name} failed with error code {error_code}.")

    def _validate_controller_array(self, values: np.ndarray, name: str) -> None:
        if values.shape != (self.dof,):
            raise RuntimeError(
                f"RealMan controller returned {name} with shape {values.shape}; expected ({self.dof},)."
            )

    def _prepare_motion(self) -> None:
        self._last_motion_succeeded = False
        self._motion_completed.clear()

    def _handle_robot_event(self, event: Any) -> None:
        if (
            event.handle_id == self.handle.id
            and event.event_type == 1
            and event.device == 0
            and event.trajectory_connect == 0
        ):
            self._last_motion_succeeded = bool(event.trajectory_state)
            self._motion_completed.set()

    def _motion_succeeded(self) -> bool:
        return self._motion_completed.is_set() and self._last_motion_succeeded

    def _pose_motion_action(self, target_pose: HomogeneousMatrixType) -> AwaitableAction:
        return AwaitableAction(
            lambda: self._motion_succeeded()
            and bool(np.linalg.norm(self.get_tcp_pose() - target_pose) < self._pose_reached_l2_threshold)
        )

    @staticmethod
    def _speed_to_percentage(speed: float, maximum_speed: float) -> int:
        return int(np.clip(np.ceil(100 * speed / maximum_speed), 1, 100))

    @staticmethod
    def _assert_positive_speed(speed: float, parameter_name: str) -> None:
        if speed <= 0:
            raise ValueError(f"{parameter_name} must be greater than zero.")

    @staticmethod
    def _assert_positive_duration(duration: float) -> None:
        if duration <= 0:
            raise ValueError("duration must be greater than zero.")

    @staticmethod
    def _duration_action(duration: float) -> AwaitableAction:
        action_sent_time = time.perf_counter_ns()
        return AwaitableAction(
            lambda: time.perf_counter_ns() - action_sent_time > duration * 1e9,
            default_timeout=2 * duration,
            default_sleep_resolution=0.002,
        )

    @staticmethod
    def _convert_realman_pose_to_homogeneous_pose(realman_pose: RealmanPoseType) -> HomogeneousMatrixType:
        if realman_pose.shape != (6,):
            raise ValueError(f"Expected a RealMan pose with shape (6,), received {realman_pose.shape}.")
        return SE3Container.from_euler_angles_and_translation(realman_pose[3:], realman_pose[:3]).homogeneous_matrix

    @staticmethod
    def _convert_homogeneous_pose_to_realman_pose(
        homogeneous_pose: HomogeneousMatrixType,
    ) -> RealmanPoseType:
        pose = SE3Container.from_homogeneous_matrix(homogeneous_pose)
        return np.concatenate([pose.translation, pose.orientation_as_euler_angles])


if __name__ == "__main__":
    import click
    from airo_robots.manipulators.hardware.manual_manipulator_testing import manual_test_robot

    @click.command()
    @click.option("--ip-address", required=True, help="IP address of the RealMan robot.")
    @click.option("--port", default=8080, show_default=True, help="RealMan controller port.")
    def test_realman(ip_address: str, port: int) -> None:
        """Run the manual manipulator tests against a RealMan robot."""
        with RealmanControl(ip_address, port) as robot:
            manual_test_robot(robot)

    test_realman()
