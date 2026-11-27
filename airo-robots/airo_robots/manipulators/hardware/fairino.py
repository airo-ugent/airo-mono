"""Position-controlled manipulator implementation for Fairino FR3 robots."""

import importlib
import time
from types import ModuleType
from typing import Any, Optional

import numpy as np
from airo_robots.awaitable_action import AwaitableAction
from airo_robots.grippers import ParallelPositionGripper
from airo_robots.manipulators.position_manipulator import ManipulatorSpecs, PositionManipulator
from airo_spatial_algebra import SE3Container
from airo_typing import HomogeneousMatrixType, JointConfigurationType, WrenchType
from loguru import logger
from numpy.typing import NDArray

FairinoPoseType = np.ndarray
"""A Fairino pose ``[x, y, z, rx, ry, rz]`` in millimetres and degrees."""

# FR3 hardware specifications (from datasheet/empirics)
# Max joint speeds: ~180°/s per joint (verified at 100% speed setting)
# Payload capacity: 3 kg
# Reach: 1700 mm
# Rated TCP speed: ~2 m/s at 100% speed setting
FR3_SPECS = ManipulatorSpecs(
    max_joint_speeds=[np.radians(180.0)] * 6,  # 180°/s → rad/s for all 6 joints
    max_linear_speed=2.0,  # m/s at 100% speed; Fairino API uses percentage 0-100
    max_joint_torques=None,
)

# Range for the servo command period (cmdT) recommended by the Fairino SDK documentation.
_SERVO_PERIOD_MIN = 0.001
_SERVO_PERIOD_MAX = 0.016
_NO_EXTERNAL_AXES = [0.0, 0.0, 0.0, 0.0]  # Fairino controllers support up to 4 external axes


def _import_fairino_api() -> ModuleType:
    """Import the optional Fairino SDK with an actionable error message."""
    try:
        fairino_module = importlib.import_module("fairino")
    except ImportError as exception:
        raise ImportError(
            "Fairino requires the Fairino Python SDK, which is not available on PyPI. "
            "Download it from https://fairino-doc-en.readthedocs.io/latest/download.html#python-sdk, "
            "unzip it (e.g. to ~/bin/fairino_sdk) and make it importable by adding a .pth file "
            "(e.g. fairino_sdk.pth) containing that path to the site-packages folder of your environment."
        ) from exception

    if hasattr(fairino_module, "Robot"):
        return fairino_module

    try:
        robot_module = importlib.import_module("fairino.Robot")
    except ImportError as exception:
        raise ImportError(
            "Fairino SDK was found but the expected `fairino.Robot` module is missing. "
            "Ensure the vendor SDK is unpacked correctly and that the SDK root is on sys.path."
        ) from exception

    setattr(fairino_module, "Robot", robot_module)
    return fairino_module


class Fairino(PositionManipulator):
    """Control a Fairino FR3 robot through the official Fairino Python SDK.

    AIRO represents joint angles in radians, while the Fairino API represents
    joint angles in degrees. This class performs that conversion at the API
    boundary. Cartesian positions use metres and radians in AIRO, while Fairino
    uses millimetres and degrees.

    Speed commands use percentages [0-100] of the controller's maximum velocities.
    This class converts from physical units (rad/s, m/s) to percentages using
    the values defined in ManipulatorSpecs.

    Motion commands are sent in non-blocking mode (blendT/blendR >= 0). Their
    returned :class:`AwaitableAction` completes after GetRobotMotionDone returns
    true and the requested target is within tolerance.

    Servo commands stream setpoints within a single controller servo session: the first
    servo call sends ServoMoveStart, subsequent calls only send ServoJ/ServoCart, and the
    session is ended with ServoMoveEnd by :meth:`servo_stop`, by any move or freedrive
    command, by switching between joint and Cartesian servoing, or by :meth:`close`.
    The servo ``duration`` is passed to the controller as the command period (cmdT),
    clipped to the SDK's recommended range of 1-16 ms.

    No collision checking or obstacle avoidance is performed beyond the
    checks configured on the Fairino controller.

    Args:
        ip_address: IP address of the Fairino robot controller.
        manipulator_specs: Optional specification override. Defaults to FR3_SPECS.
        gripper: Optional gripper associated with this manipulator.
    """

    def __init__(
        self,
        ip_address: str,
        manipulator_specs: Optional[ManipulatorSpecs] = None,
        gripper: Optional[ParallelPositionGripper] = None,
    ) -> None:
        self.ip_address = ip_address
        self._fairino_api = _import_fairino_api()

        # Connect to the robot
        try:
            self.robot: Any = self._fairino_api.Robot.RPC(ip_address)
        except Exception as exception:
            raise RuntimeError(
                f"Could not connect to Fairino robot at {ip_address}. "
                "Check the IP address, controller state, and network connection. "
                f"Error: {exception}"
            ) from exception

        # Verify connection via SDK version query
        try:
            error_code, sdk_version = self.robot.GetSDKVersion()
            if error_code != 0:
                raise RuntimeError(f"GetSDKVersion returned error code {error_code}")
            logger.info(f"Connected to Fairino robot at {ip_address}, SDK version: {sdk_version}")
        except Exception as exception:
            raise RuntimeError(f"Failed to verify Fairino connection: {exception}") from exception

        # Use provided specs or default to FR3
        if manipulator_specs is None:
            manipulator_specs = FR3_SPECS

        super().__init__(manipulator_specs, gripper)

        # Conservative joint limits (in radians), not queried from the controller
        self._joint_lower_limits = np.radians(np.array([-180.0] * 6, dtype=float))
        self._joint_upper_limits = np.radians(np.array([180.0] * 6, dtype=float))

        # Thresholds for motion completion detection
        self._pose_reached_l2_threshold = 0.01  # meters
        self._joint_configuration_reached_l2_threshold = 0.01  # radians

        # Active tool offset (0 = default TCP, 1 = custom tool with SetToolCoord offset)
        self._active_tool_id = 0

        # Active servo session: None, "joint" (ServoJ) or "cartesian" (ServoCart)
        self._servo_session: Optional[str] = None

        # Payload configuration state
        self._payload_mass = 0.0  # kg
        self._payload_cog: NDArray[np.float64] = np.zeros(3, dtype=float)  # m

        self._closed = False

        logger.info(f"Fairino manipulator initialized with specs: {manipulator_specs}.")

    def close(self) -> None:
        """Close the connection to the robot controller."""
        if self._closed:
            return
        try:
            self.servo_stop()
        except RuntimeError as exception:
            logger.warning(f"Error while ending Fairino servo session: {exception}")
        try:
            self.robot.CloseRPC()
            self._closed = True
            logger.info(f"Fairino connection to {self.ip_address} closed.")
        except Exception as exception:
            logger.warning(f"Error while closing Fairino connection: {exception}")
            self._closed = True

    def __enter__(self) -> "Fairino":
        return self

    def __exit__(self, exception_type: Any, exception: Any, traceback: Any) -> None:
        self.close()

    ###########
    # Getters #
    ###########

    def get_joint_configuration(self) -> JointConfigurationType:
        """Get the current joint configuration in radians.

        Queries GetActualJointPosRadian(flag=0).
        """
        try:
            error_code, joint_pos_rad = self.robot.GetActualJointPosRadian(0)
            if error_code != 0:
                raise RuntimeError(f"GetActualJointPosRadian returned error code {error_code}")
            return np.asarray(joint_pos_rad, dtype=float)
        except Exception as exception:
            raise RuntimeError(f"Failed to get joint configuration: {exception}") from exception

    def get_tcp_pose(self) -> HomogeneousMatrixType:
        """Get the current TCP pose as a homogeneous matrix.

        Queries GetActualTCPPose(flag=0) and converts from [x,y,z,rx,ry,rz] mm/deg
        to a homogeneous matrix in m/rad.
        """
        try:
            response = self.robot.GetActualTCPPose(0)
            if isinstance(response, (tuple, list)) and len(response) == 2:
                error_code, pose_mm_deg = response
            elif isinstance(response, int):
                raise RuntimeError(f"GetActualTCPPose returned error code {response} without pose payload")
            else:
                raise RuntimeError(f"GetActualTCPPose returned unexpected response: {response!r}")

            if error_code != 0:
                raise RuntimeError(f"GetActualTCPPose returned error code {error_code}")
            pose_array = np.asarray(pose_mm_deg, dtype=float)
            return self._convert_fairino_pose_to_homogeneous_pose(pose_array)
        except Exception as exception:
            raise RuntimeError(f"Failed to get TCP pose: {exception}") from exception

    def get_wrench(self, compensated: bool = True) -> WrenchType:
        """Get the end-effector six-axis force/torque reading.

        Raises:
            NotImplementedError: Wrench measurement is not supported for Fairino robots.
        """
        raise NotImplementedError("Wrench measurement is not available on Fairino robots.")

    def get_point_record_button_state(self) -> int:
        """Get the state of the end-point record button.

        Returns the raw SDK state:
        - 0: pressed
        - 1: released
        """
        try:
            error_code, button_state = self.robot.GetAxlePointRecordBtnState()
            if error_code != 0:
                raise RuntimeError(f"GetAxlePointRecordBtnState returned error code {error_code}")
            return int(button_state)
        except Exception as exception:
            raise RuntimeError(f"Failed to get point record button state: {exception}") from exception

    def get_smarttool_button_state(self) -> int:
        """Get the SmartTool handle button bitfield.

        Returns the raw SDK bitfield from GetSmarttoolBtnState().
        """
        try:
            error_code, button_state = self.robot.GetSmarttoolBtnState()
            if error_code != 0:
                raise RuntimeError(f"GetSmarttoolBtnState returned error code {error_code}")
            return int(button_state)
        except Exception as exception:
            raise RuntimeError(f"Failed to get SmartTool button state: {exception}") from exception

    ##################
    # Motion Methods #
    ##################

    def move_to_joint_configuration(
        self,
        joint_configuration: JointConfigurationType,
        joint_speed: Optional[float] = None,
    ) -> AwaitableAction:
        """Move to a joint configuration using controller trajectory planning.

        Uses MoveJ with non-blocking mode (blendT >= 0) and returns an AwaitableAction
        that completes when motion is done and target is reached.
        """
        self.servo_stop()
        self._assert_joint_configuration_is_valid(joint_configuration)
        speed = self.default_joint_speed if joint_speed is None else joint_speed
        self._assert_positive_speed(speed, "joint_speed")
        self._assert_joint_speed_is_valid(speed)

        speed_percentage = self._speed_to_percentage(speed, min(self.manipulator_specs.max_joint_speeds))
        target_deg = np.degrees(np.asarray(joint_configuration, dtype=float))

        try:
            error_code = self.robot.MoveJ(
                target_deg.tolist(),
                self._active_tool_id,
                0,
                desc_pos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                vel=speed_percentage,
                acc=0.0,
                ovl=100.0,
                blendT=0.0,
            )
            if error_code != 0:
                raise RuntimeError(f"MoveJ returned error code {error_code}")
        except Exception as exception:
            raise RuntimeError(f"Failed to execute move_to_joint_configuration: {exception}") from exception

        return AwaitableAction(
            lambda: self._is_motion_done()
            and bool(
                np.linalg.norm(self.get_joint_configuration() - np.asarray(joint_configuration, dtype=float))
                < self._joint_configuration_reached_l2_threshold
            )
        )

    def move_linear_to_tcp_pose(
        self, tcp_pose: HomogeneousMatrixType, linear_speed: Optional[float] = None
    ) -> AwaitableAction:
        """Move the TCP to a pose along a Cartesian straight line.

        Uses MoveL with non-blocking mode.
        """
        self.servo_stop()
        self._assert_pose_is_valid(tcp_pose)
        speed = self.default_linear_speed if linear_speed is None else linear_speed
        self._assert_positive_speed(speed, "linear_speed")
        self._assert_linear_speed_is_valid(speed)

        speed_percentage = self._speed_to_percentage(speed, self.manipulator_specs.max_linear_speed)
        fairino_pose = self._convert_homogeneous_pose_to_fairino_pose(tcp_pose)

        try:
            error_code = self.robot.MoveL(
                fairino_pose.tolist(),
                self._active_tool_id,
                0,
                vel=speed_percentage,
                acc=0.0,
                ovl=100.0,
                blendR=0.0,
                blendMode=0,
                exaxis_pos=[0.0, 0.0, 0.0, 0.0],
                search=0,
                offset_flag=0,
                offset_pos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            )
            if error_code != 0:
                raise RuntimeError(f"MoveL returned error code {error_code}")
        except Exception as exception:
            raise RuntimeError(f"Failed to execute move_linear_to_tcp_pose: {exception}") from exception

        return AwaitableAction(
            lambda: self._is_motion_done()
            and bool(np.linalg.norm(self.get_tcp_pose() - tcp_pose) < self._pose_reached_l2_threshold)
        )

    def move_to_tcp_pose(
        self, tcp_pose: HomogeneousMatrixType, joint_speed: Optional[float] = None
    ) -> AwaitableAction:
        """Move the TCP to a pose using joint-space planning.

        Uses MoveCart (PTP Cartesian move) with non-blocking mode.
        """
        self.servo_stop()
        self._assert_pose_is_valid(tcp_pose)
        speed = self.default_joint_speed if joint_speed is None else joint_speed
        self._assert_positive_speed(speed, "joint_speed")
        self._assert_joint_speed_is_valid(speed)

        speed_percentage = self._speed_to_percentage(speed, min(self.manipulator_specs.max_joint_speeds))
        fairino_pose = self._convert_homogeneous_pose_to_fairino_pose(tcp_pose)

        try:
            error_code = self.robot.MoveCart(
                fairino_pose.tolist(),
                self._active_tool_id,
                0,
                vel=speed_percentage,
                acc=0.0,
                ovl=100.0,
                blendT=0.0,
                config=-1,
            )
            if error_code != 0:
                raise RuntimeError(f"MoveCart returned error code {error_code}")
        except Exception as exception:
            raise RuntimeError(f"Failed to execute move_to_tcp_pose: {exception}") from exception

        return AwaitableAction(
            lambda: self._is_motion_done()
            and bool(np.linalg.norm(self.get_tcp_pose() - tcp_pose) < self._pose_reached_l2_threshold)
        )

    #################
    # Servo Methods #
    #################

    def servo_to_joint_configuration(
        self, joint_configuration: JointConfigurationType, duration: float
    ) -> AwaitableAction:
        """Send a joint servo setpoint.

        Starts a joint servo session on the first call and then only sends ServoJ, using
        ``duration`` as the command period. Call :meth:`servo_stop` to end the session.
        """
        self._assert_positive_duration(duration)
        target_rad = np.asarray(joint_configuration, dtype=float)
        target_deg = np.degrees(target_rad)

        try:
            self._ensure_servo_session("joint")
            # ServoJ(joint_pos, axisPos, ..., cmdT): no external axes, cmdT is the servo period
            error_code = self.robot.ServoJ(target_deg.tolist(), _NO_EXTERNAL_AXES, cmdT=self._servo_period(duration))
            if error_code != 0:
                raise RuntimeError(f"ServoJ returned error code {error_code}")
        except Exception as exception:
            raise RuntimeError(f"Failed to execute servo_to_joint_configuration: {exception}") from exception

        action_sent_time = time.perf_counter_ns()
        return AwaitableAction(
            lambda: time.perf_counter_ns() - action_sent_time > duration * 1e9,
            default_timeout=2 * duration,
            default_sleep_resolution=0.002,
        )

    def servo_to_tcp_pose(self, tcp_pose: HomogeneousMatrixType, duration: float) -> AwaitableAction:
        """Send a Cartesian servo setpoint.

        Starts a Cartesian servo session on the first call and then only sends ServoCart, using
        ``duration`` as the command period. Call :meth:`servo_stop` to end the session.
        """
        self._assert_positive_duration(duration)
        fairino_pose = self._convert_homogeneous_pose_to_fairino_pose(tcp_pose)

        try:
            self._ensure_servo_session("cartesian")
            # ServoCart(mode, desc_pos, exaxis, ..., cmdT): mode=0 for absolute positioning in the base frame
            error_code = self.robot.ServoCart(
                0, fairino_pose.tolist(), _NO_EXTERNAL_AXES, cmdT=self._servo_period(duration)
            )
            if error_code != 0:
                raise RuntimeError(f"ServoCart returned error code {error_code}")
        except Exception as exception:
            raise RuntimeError(f"Failed to execute servo_to_tcp_pose: {exception}") from exception

        action_sent_time = time.perf_counter_ns()
        return AwaitableAction(
            lambda: time.perf_counter_ns() - action_sent_time > duration * 1e9,
            default_timeout=2 * duration,
            default_sleep_resolution=0.002,
        )

    def servo_stop(self) -> None:
        """End the active servo session (ServoMoveEnd), if any."""
        if self._servo_session is None:
            return
        # Clear the state first: if ServoMoveEnd fails, retrying it on every command would not help.
        self._servo_session = None
        error_code = self.robot.ServoMoveEnd()
        if error_code != 0:
            raise RuntimeError(f"ServoMoveEnd returned error code {error_code}")

    def _ensure_servo_session(self, session_type: str) -> None:
        """Start a servo session of the given type, ending a session of another type first."""
        if self._servo_session == session_type:
            return
        self.servo_stop()
        error_code = self.robot.ServoMoveStart()
        if error_code != 0:
            raise RuntimeError(f"ServoMoveStart returned error code {error_code}")
        self._servo_session = session_type

    @staticmethod
    def _servo_period(duration: float) -> float:
        """Convert a servo duration to a command period (cmdT) within the SDK's recommended range."""
        return float(np.clip(duration, _SERVO_PERIOD_MIN, _SERVO_PERIOD_MAX))

    ##############
    # Kinematics #
    ##############

    def inverse_kinematics(
        self,
        tcp_pose: HomogeneousMatrixType,
        joint_configuration_near: Optional[JointConfigurationType] = None,
    ) -> Optional[JointConfigurationType]:
        """Solve inverse kinematics for a TCP pose using the controller's solver.

        Uses GetInverseKinRef with the current configuration as reference if
        joint_configuration_near is not provided.
        """
        if joint_configuration_near is None:
            joint_configuration_near = self.get_joint_configuration()

        near_deg = np.degrees(np.asarray(joint_configuration_near, dtype=float))
        fairino_pose = self._convert_homogeneous_pose_to_fairino_pose(tcp_pose)

        try:
            error_code, solution_deg = self.robot.GetInverseKinRef(0, fairino_pose.tolist(), near_deg.tolist())
            if error_code != 0:
                logger.warning(
                    f"Inverse kinematics failed (error code {error_code}) for pose "
                    f"position {fairino_pose[:3].round(1).tolist()} mm."
                )
                return None
            return np.radians(np.asarray(solution_deg, dtype=float))
        except Exception as exception:
            logger.warning(f"Failed to compute inverse kinematics: {exception}")
            return None

    def forward_kinematics(self, joint_configuration: JointConfigurationType) -> HomogeneousMatrixType:
        """Calculate the TCP pose for a joint configuration using the controller's FK.

        Uses GetForwardKin.
        """
        joint_configuration_array = np.asarray(joint_configuration, dtype=float)
        joint_configuration_deg = np.degrees(joint_configuration_array)

        try:
            error_code, fairino_pose = self.robot.GetForwardKin(joint_configuration_deg.tolist())
            if error_code != 0:
                raise RuntimeError(f"GetForwardKin returned error code {error_code}")
            pose_array = np.asarray(fairino_pose, dtype=float)
            return self._convert_fairino_pose_to_homogeneous_pose(pose_array)
        except Exception as exception:
            raise RuntimeError(f"Failed to compute forward kinematics: {exception}") from exception

    def _is_joint_configuration_reachable(self, joint_configuration: JointConfigurationType) -> bool:
        """Check if a joint configuration is within controller limits."""
        joint_configuration_array = np.asarray(joint_configuration, dtype=float)
        return bool(
            joint_configuration_array.shape == (self.manipulator_specs.dof,)
            and np.all(joint_configuration_array >= self._joint_lower_limits)
            and np.all(joint_configuration_array <= self._joint_upper_limits)
        )

    #############
    # Freedrive #
    #############

    def start_freedrive(self) -> None:
        """Enable freedrive (drag-teach) mode.

        Switches the robot to manual mode and enables drag-teach.
        """
        self.servo_stop()
        try:
            error_code = self.robot.Mode(1)  # 1 = manual mode
            if error_code != 0:
                raise RuntimeError(f"Mode(1) returned error code {error_code}")

            error_code = self.robot.DragTeachSwitch(1)  # 1 = enable drag-teach
            if error_code != 0:
                raise RuntimeError(f"DragTeachSwitch(1) returned error code {error_code}")

            logger.info("Freedrive mode enabled.")
        except Exception as exception:
            raise RuntimeError(f"Failed to start freedrive: {exception}") from exception

    def stop_freedrive(self) -> None:
        """Disable freedrive (drag-teach) mode.

        Disables drag-teach and returns to auto mode.
        """
        try:
            error_code = self.robot.DragTeachSwitch(0)  # 0 = disable drag-teach
            if error_code != 0:
                raise RuntimeError(f"DragTeachSwitch(0) returned error code {error_code}")

            error_code = self.robot.Mode(0)  # 0 = auto mode
            if error_code != 0:
                raise RuntimeError(f"Mode(0) returned error code {error_code}")

            logger.info("Freedrive mode disabled.")
        except Exception as exception:
            raise RuntimeError(f"Failed to stop freedrive: {exception}") from exception

    ##################
    # Payload Config #
    ##################

    def set_payload(self, mass: float, cog: np.ndarray) -> None:
        """Configure the payload mass and center-of-gravity offset.

        Args:
            mass: Payload mass in kilograms.
            cog: Center-of-gravity offset in metres, relative to the tool frame.
                 Shape: (3,) for [x, y, z].
        """
        if not isinstance(mass, (int, float)) or mass < 0:
            raise ValueError(f"Payload mass must be a non-negative number, got {mass}")
        if not isinstance(cog, np.ndarray) or cog.shape != (3,):
            raise ValueError(f"CoG must be a numpy array of shape (3,), got shape {cog.shape}")

        self._payload_mass = float(mass)
        self._payload_cog = np.asarray(cog, dtype=float)

        try:
            # SetLoadWeight(loadNum, mass) — mass in kg
            error_code = self.robot.SetLoadWeight(0, self._payload_mass)
            if error_code != 0:
                logger.warning(f"SetLoadWeight returned error code {error_code}")

            # SetLoadCoord(x, y, z, loadNum) — coordinates in mm (convert from m)
            cog_mm = self._payload_cog * 1000.0
            error_code = self.robot.SetLoadCoord(cog_mm[0], cog_mm[1], cog_mm[2], 0)
            if error_code != 0:
                logger.warning(f"SetLoadCoord returned error code {error_code}")

            logger.info(f"Payload configured: mass={mass} kg, CoG={cog.tolist()} m")
        except Exception as exception:
            logger.warning(f"Failed to configure payload: {exception}")

    def set_tcp_offset(self, tcp_pose: HomogeneousMatrixType) -> None:
        """Configure the tool center point offset from the flange.

        This sets tool frame 1 (toolID=1) and switches active commanding to tool 1.
        Subsequent moves will use this offset as the TCP.

        Args:
            tcp_pose: TCP pose offset as a homogeneous matrix (m, rad).
        """
        fairino_pose = self._convert_homogeneous_pose_to_fairino_pose(tcp_pose)

        try:
            # SetToolCoord(toolID, coord[6], type, install, toolID, loadNum)
            # type=0 = TCP (tool center point)
            # install=1 = apply immediately
            error_code = self.robot.SetToolCoord(1, fairino_pose.tolist(), 0, 1, 0, 0)
            if error_code != 0:
                logger.warning(f"SetToolCoord returned error code {error_code}")
                return

            self._active_tool_id = 1
            logger.info(f"TCP offset configured: {fairino_pose.tolist()}")
        except Exception as exception:
            logger.warning(f"Failed to configure TCP offset: {exception}")

    ##################
    # Helper Methods #
    ##################

    def _is_motion_done(self) -> bool:
        """Check if the current motion is complete.

        The vendor SDK exposes two compatible variants of this API:
        - `GetRobotMotionDone()` in the generated Python wrapper used in this environment
        - `GetRobotMotionDone(int[] state)` in the official C++/API documentation

        Support both for compatibility across SDK versions.
        """
        try:
            error_code, motion_done = self.robot.GetRobotMotionDone()
            if error_code != 0:
                logger.warning(f"GetRobotMotionDone returned error code {error_code}")
                return False
            return bool(motion_done)
        except TypeError:
            state = [-1]
            try:
                error_code = self.robot.GetRobotMotionDone(state)
                if error_code != 0:
                    logger.warning(f"GetRobotMotionDone returned error code {error_code}")
                    return False
                if not state:
                    return False
                return bool(state[0])
            except Exception as exception:
                logger.warning(f"Failed to check motion status with array-style API: {exception}")
                return False
        except Exception as exception:
            logger.warning(f"Failed to check motion status: {exception}")
            return False

    @staticmethod
    def _speed_to_percentage(speed: float, maximum_speed: float) -> int:
        """Convert physical speed (rad/s or m/s) to percentage [1, 100]."""
        return int(np.clip(np.ceil(100.0 * speed / maximum_speed), 1, 100))

    @staticmethod
    def _assert_positive_speed(speed: float, parameter_name: str) -> None:
        if speed <= 0:
            raise ValueError(f"{parameter_name} must be greater than zero.")

    @staticmethod
    def _assert_positive_duration(duration: float) -> None:
        if duration <= 0:
            raise ValueError("duration must be greater than zero.")

    @staticmethod
    def _convert_fairino_pose_to_homogeneous_pose(fairino_pose: FairinoPoseType) -> HomogeneousMatrixType:
        """Convert [x, y, z, rx, ry, rz] mm/deg to homogeneous matrix m/rad."""
        if fairino_pose.shape != (6,):
            raise ValueError(f"Expected Fairino pose shape (6,), got {fairino_pose.shape}.")
        # Convert mm to m, deg to rad; assume Euler angles (rx, ry, rz) in fixed XYZ order
        translation = fairino_pose[:3] / 1000.0  # mm → m
        euler_rad = np.radians(fairino_pose[3:])  # deg → rad
        return SE3Container.from_euler_angles_and_translation(euler_rad, translation).homogeneous_matrix

    @staticmethod
    def _convert_homogeneous_pose_to_fairino_pose(
        homogeneous_pose: HomogeneousMatrixType,
    ) -> FairinoPoseType:
        """Convert homogeneous matrix m/rad to [x, y, z, rx, ry, rz] mm/deg."""
        se3 = SE3Container.from_homogeneous_matrix(homogeneous_pose)
        translation = se3.translation * 1000.0  # m → mm
        euler_deg = np.degrees(se3.orientation_as_euler_angles)  # rad → deg
        return np.concatenate([translation, euler_deg])


if __name__ == "__main__":
    import click
    from airo_robots.manipulators.hardware.manual_manipulator_testing import manual_test_robot

    @click.command()
    @click.option("--ip-address", required=True, help="IP address of the Fairino robot.")
    def test_fairino(ip_address: str) -> None:
        """Run the manual manipulator tests against a Fairino robot."""
        with Fairino(ip_address) as robot:
            manual_test_robot(robot)

    test_fairino()
