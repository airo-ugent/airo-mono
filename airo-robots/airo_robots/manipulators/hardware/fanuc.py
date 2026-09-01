"""Position-controlled manipulator implementation for FANUC robots, on top of the airo-fanuc driver."""

import importlib
import time
from types import ModuleType
from typing import Any, Callable, NoReturn, Optional

import numpy as np
from airo_robots.awaitable_action import AwaitableAction
from airo_robots.exceptions import InvalidTrajectoryException
from airo_robots.grippers import ParallelPositionGripper
from airo_robots.grippers.hardware.robotiq_2f85_fanuc import Robotiq2F85Fanuc
from airo_robots.manipulators.position_manipulator import ManipulatorSpecs, PositionManipulator
from airo_spatial_algebra import SE3Container
from airo_typing import (
    HomogeneousMatrixType,
    JointConfigurationType,
    JointPathContainer,
    SingleArmTrajectory,
    WrenchType,
)
from loguru import logger

FanucPoseType = np.ndarray
"""A FANUC pose ``[X, Y, Z, W, P, R]`` in millimeters and degrees, as the controller reports it."""

FANUC_DOF = 6
"""The joint count of the arms this driver supports; a compile-time constant of its real-time core."""

MAX_LINEAR_SPEED = 1.0
"""Placeholder maximum TCP speed [m/s], since `ManipulatorSpecs` requires one. The driver has no
Cartesian path, so nothing here reads it; it is not a datasheet number for any particular arm."""

_KINEMATICS_MESSAGE = (
    "The airo-fanuc driver deliberately does no kinematics: no URDF, no forward kinematics and no "
    "inverse kinematics. Solve it outside the driver with a numerical solver on a FANUC URDF (the "
    "CRX-10iA/L URDF lives in https://github.com/airo-ugent/airo-models), e.g. curobo or drake, and "
    "command the resulting joint configurations. See fanuc_setup.md next to this module."
)

_NO_STATUS_PACKET_MESSAGE = "No Stream Motion status packet has been received from the FANUC controller yet"


def _import_airo_fanuc() -> ModuleType:
    """Import the optional airo-fanuc driver with an actionable error message."""
    try:
        return importlib.import_module("airo_fanuc")
    except ImportError as exception:
        raise ImportError(
            'Fanuc requires the airo-fanuc driver. Install it with `pip install "airo-robots[fanuc]"`.'
        ) from exception


def _raise_no_kinematics(method_name: str, addition: str = "") -> NoReturn:
    """Refuse a method that would need kinematics, pointing at an external solver instead."""
    raise NotImplementedError(f"Fanuc.{method_name} needs kinematics. {_KINEMATICS_MESSAGE} {addition}".strip())


class Fanuc(PositionManipulator):
    """Control a FANUC robot through the airo-fanuc driver (https://github.com/airo-ugent/airo-fanuc).

    The driver speaks Stream Motion (UDP, 125 Hz) for motion and RMI (TCP) for everything else, and
    owns the controller exclusively while it is up. Its constructor is construct-and-go: it blocks
    until the robot is streaming and commandable, or raises with a reason. So is this class'.

    **This class does no kinematics.** The driver ships no URDF, no FK and no IK, so every Cartesian
    command and both kinematics methods raise `NotImplementedError`; plan in joint space with an
    external solver instead. `get_tcp_pose` is available because the *controller* computes it, with
    its own active UTOOL applied. See `fanuc_setup.md` next to this module for the full support table.

    No collision checking or obstacle avoidance is performed beyond what is configured on the
    controller; in particular, nothing checks the servo path.

    The driver exposes more than this interface does (recovery, the ARM gate, timing statistics, the
    RMI session, register access). It stays reachable as `driver`.

    Args:
        ip_address: IP address of the robot controller.
        policy: the driver's ``DriverPolicy``, which carries the ``DriverConfig`` and the arm's
            ``RobotProfile``. There is no default: the profile holds the limits the real-time core
            clamps against, and the driver cannot ask the controller which robot is attached. Build
            one for your arm with ``python -m airo_fanuc.controller_probe --ip <ip> --emit-profile``.
        manipulator_specs: Optional specification override. By default the maximum joint speeds are
            taken from the profile's velocity limits and the maximum linear speed is a placeholder
            (see `MAX_LINEAR_SPEED`).
        gripper: Optional gripper associated with this manipulator. When it is `None` and the driver
            brought up a Robotiq 2F-85 gripper worker, a `Robotiq2F85Fanuc` is attached automatically.
    """

    def __init__(
        self,
        ip_address: str,
        policy: Any,
        manipulator_specs: Optional[ManipulatorSpecs] = None,
        gripper: Optional[ParallelPositionGripper] = None,
    ) -> None:
        self.ip_address = ip_address
        self._fanuc = _import_airo_fanuc()

        profile = policy.config.profile
        if profile.ndof != FANUC_DOF:
            raise ValueError(f"The airo-fanuc driver only supports {FANUC_DOF}-joint arms, got {profile.ndof}.")

        self._joint_lower_limits = np.asarray(profile.position_limits_lower, dtype=float)
        self._joint_upper_limits = np.asarray(profile.position_limits_upper, dtype=float)

        if manipulator_specs is None:
            manipulator_specs = ManipulatorSpecs(
                np.asarray(profile.velocity_limits, dtype=float).tolist(),
                MAX_LINEAR_SPEED,
            )
        if manipulator_specs.dof != FANUC_DOF:
            raise ValueError(
                f"Manipulator specs have {manipulator_specs.dof} joints, but the connected robot has {FANUC_DOF}."
            )

        # Construct-and-go: this blocks until the robot is commandable, or raises with a real reason.
        self.driver: Any = self._fanuc.FanucDriver(ip_address, policy)
        self._closed = False

        super().__init__(manipulator_specs, gripper if gripper is not None else self._build_default_gripper())

        # How long move_to_joint_configuration waits for the arm to come to rest before it plans.
        self._brake_timeout = 5.0
        # Extra time on top of a trajectory's own duration before execute_trajectory gives up on it.
        self._trajectory_timeout_margin = 5.0
        # Last submitted motion, so a caller can read the driver's own MotionResult after waiting.
        self.last_motion_handle: Any = None

        logger.info(
            f"Connected to FANUC robot at {self.ip_address} with profile {profile.describe()}. "
            f"Gripper: {type(self.gripper).__name__ if self.gripper is not None else 'none'}."
        )

    def _build_default_gripper(self) -> Optional[ParallelPositionGripper]:
        """Wrap the driver's gripper worker when it is driving the Robotiq 2F-85 preset.

        The driver builds a register worker for whatever ``gripper_protocol`` the policy names; only
        the shipped Robotiq 2F-85 preset has a matching airo-robots implementation. Any other
        protocol is left to the caller, since its buckets mean something this class cannot guess.
        """
        if self.driver.gripper is None:
            return None
        try:
            return Robotiq2F85Fanuc(self.driver)
        except ValueError as exception:
            logger.warning(
                f"The driver brought up a gripper worker, but it is not driving the Robotiq 2F-85 preset "
                f"({exception}). Use `robot.driver.gripper` directly, or pass your own ParallelPositionGripper."
            )
            return None

    def close(self) -> None:
        """Shut the driver down: quiesce, stop streaming, disconnect and release the ownership lock."""
        if self._closed:
            return
        self._closed = True
        self.driver.close()

    def __enter__(self) -> "Fanuc":
        return self

    def __exit__(self, exception_type: Any, exception: Any, traceback: Any) -> None:
        self.close()

    ###########
    # getters #
    ###########

    def get_joint_configuration(self) -> JointConfigurationType:
        """Get the measured joint configuration in radians.

        The last position the controller reported over Stream Motion, which is published at 125 Hz. It
        trails `get_commanded_joint_configuration` by the controller's servo lag.

        Raises:
            RuntimeError: If no status packet has arrived yet.
        """
        joint_configuration = self.driver.get_joint_configuration()
        if joint_configuration is None:
            raise RuntimeError(f"{_NO_STATUS_PACKET_MESSAGE}, so there is no joint configuration to report.")
        return np.asarray(joint_configuration, dtype=float)

    def get_commanded_joint_configuration(self) -> JointConfigurationType:
        """Get the last *commanded* joint configuration in radians (not part of the interface).

        What the driver put on the wire, as opposed to what the controller reported back. It is the
        pose the driver anchors a new motion at, so it is the configuration a trajectory should start
        from: the measured configuration trails it by the servo lag.

        Raises:
            RuntimeError: If no status packet has arrived yet.
        """
        state = self.driver.get_state()
        # The driver gates its own getters on this, but only exposes q_cmd through the state dict.
        # The snapshot is zero-initialized, and an all-zero configuration is a pose this arm can hold
        # rather than a 'no data' marker, so it must not be handed out as one.
        if int(state.get("rx_mono_ns", 0)) <= 0:
            raise RuntimeError(f"{_NO_STATUS_PACKET_MESSAGE}, so there is no commanded joint configuration.")
        return np.asarray(state["q_cmd"], dtype=float)

    def get_tcp_pose(self) -> HomogeneousMatrixType:
        """Get the current TCP pose as a homogeneous matrix.

        The *controller* computes this pose, with the UTOOL that is active on the pendant applied, so
        configure the UTOOL there before relying on it. This costs an RMI round trip (tens of
        milliseconds) and is not on the 125 Hz timeline, so it does not belong in a control loop;
        `get_flange_pose` is the streamed alternative.

        Raises:
            RuntimeError: If the controller could not be asked. No stale or substituted pose is ever
                returned in its place.
        """
        pose = self.driver.get_tcp_pose()
        if pose is None:
            raise RuntimeError(
                "Could not read the TCP pose from the FANUC controller (the RMI read failed). "
                "Check `robot.driver.get_state()` for the lifecycle state and fault reason."
            )
        return self._convert_fanuc_pose_to_homogeneous_pose(np.asarray(pose, dtype=float))

    def get_flange_pose(self) -> HomogeneousMatrixType:
        """Get the streamed faceplate pose as a homogeneous matrix (not part of the interface).

        The controller's own forward kinematics, read out of the same Stream Motion packet as the
        measured joints, so pose and joints are the same instant. Unlike `get_tcp_pose` this does not
        block, but it is the **faceplate** and not the tool tip: apply your own tool transform.

        Raises:
            RuntimeError: If no status packet has arrived yet.
        """
        pose = self.driver.get_flange_pose()
        if pose is None:
            raise RuntimeError(f"{_NO_STATUS_PACKET_MESSAGE}, so there is no flange pose.")
        return self._convert_fanuc_pose_to_homogeneous_pose(np.asarray(pose, dtype=float))

    def get_wrench(self) -> WrenchType:
        """Get the end-effector six-axis force/torque reading as ``[Fx, Fy, Fz, Mx, My, Mz]`` (N, Nm).

        Raises:
            RuntimeError: If the controller streams no force block. That is the case on Stream Motion
                v3 (type-202) controllers, which is what an R-30iB negotiates by default — on those,
                contact detection is the controller's own collaborative stop and nothing else.
        """
        wrench = self.driver.get_wrench()
        if wrench is None:
            raise RuntimeError(
                "This FANUC controller provides no force telemetry (no force block on the Stream Motion "
                "wire, i.e. a v3 / type-202 status packet), so there is no wrench to read."
            )
        return np.asarray(wrench, dtype=float)

    #############
    # movements #
    #############

    def move_to_joint_configuration(
        self,
        joint_configuration: JointConfigurationType,
        joint_speed: Optional[float] = None,
    ) -> AwaitableAction:
        """Move to a joint configuration with a jerk-limited point-to-point profile.

        The driver plans the profile (offline Ruckig, under the arm's own limits) and hands the whole
        timeline to its real-time core, which owns playback from there.

        ``joint_speed`` is the leading-axis speed in rad/s: it caps every joint and the profile is
        time-synchronised, so the joint with the furthest to travel runs at ``joint_speed`` and the
        others scale down to arrive with it.

        The driver refuses to plan a profile from a moving anchor, since such a plan depends on how
        long the submission itself took and is therefore not repeatable. This brakes first when the
        arm is not at rest, which **preempts any motion that was still running**.
        """
        self._assert_joint_configuration_is_valid(joint_configuration)
        speed = self.default_joint_speed if joint_speed is None else joint_speed
        self._assert_positive_speed(speed, "joint_speed")
        self._assert_joint_speed_is_valid(speed)

        self._brake_to_rest()
        target = np.asarray(joint_configuration, dtype=float)
        handle = self._submit(lambda: self.driver.move_j(target, joint_speed=speed))
        return self._motion_action(handle, f"move to joint configuration {np.degrees(target).round(3).tolist()} deg")

    def servo_to_joint_configuration(
        self, joint_configuration: JointConfigurationType, duration: float
    ) -> AwaitableAction:
        """Send a best-effort servo setpoint, to be reached in ``duration`` seconds.

        Servo targets replace rather than queue: each supersedes the last, and the driver's real-time
        core plans a fresh profile to it under the servo limits and follows it best-effort. A target
        is never refused for being far away, and **no collision check runs on this path**.

        ``duration`` is the spacing between successive calls (``1/f`` in the ``servo(q, 1/f)``
        pattern), which stretches the profile so the command glides between targets instead of
        arriving early and dwelling. The controller interpolates at 125 Hz, so there is nothing to
        gain above that rate: extra setpoints are coalesced away rather than queued.

        A servo stream has no terminal condition — call `hold` to end it. Until you do, the driver
        holds the last target and the arm never counts as being at rest.
        """
        self._assert_positive_duration(duration)
        target = np.asarray(joint_configuration, dtype=float)
        if target.shape != (FANUC_DOF,):
            raise ValueError(f"Expected a joint configuration with shape ({FANUC_DOF},).")
        # Reachability is deliberately not checked here: it costs time this loop does not have, and
        # the driver's real-time core clamps a servo target into the arm's position limits anyway.
        self._submit(lambda: self.driver.servo_j(target, duration))
        return self._duration_action(duration)

    def execute_trajectory(self, joint_trajectory: SingleArmTrajectory, sampling_frequency: float = 100) -> None:
        """Execute a time-parameterized joint trajectory. Blocks until it finishes.

        This overrides the base-class implementation, which streams a trajectory servo command by
        servo command from Python. The FANUC driver takes a whole timeline in one submission and its
        C++ core owns playback from there, so a late Python thread costs nothing and there is no
        reason to keep the host in the loop. The gripper trajectory (if any) is ignored, as in the
        base class.

        The trajectory has to start where the arm is and stay inside the driver's capture envelope,
        and its joint positions are silently clamped into the arm's limits rather than validated.
        ``fanuc_setup.md`` next to this module lists what the driver requires of one. Velocities are
        optional: the driver derives the playback tangents itself when the path carries none.

        Args:
            joint_trajectory: the joint trajectory to execute.
            sampling_frequency: only used to evaluate the trajectory's constraint (if it has one),
                *not* to resample the path. The driver plays the knots back on the controller's own
                8 ms interpolation grid.

        Raises:
            InvalidTrajectoryException: if the trajectory is not executable, either by the
                base-class checks or by the driver's own validation.
            RuntimeError: if the motion did not complete, e.g. because it was preempted or the robot
                faulted. The message carries the driver's fault reason and its operator instruction.
        """
        self._assert_joint_trajectory_is_executable(joint_trajectory, sampling_frequency)

        times = np.asarray(joint_trajectory.times, dtype=float)
        positions = np.asarray(joint_trajectory.path.positions, dtype=float)
        velocities = joint_trajectory.path.velocities
        if velocities is not None:
            velocities = np.asarray(velocities, dtype=float)

        duration = float(times[-1] - times[0])
        times_ns = np.rint((times - times[0]) * 1e9).astype(np.int64)

        try:
            handle = self._submit(lambda: self.driver.move_trajectory(times_ns, positions, velocities))
        except (self._fanuc.TrajectoryValidationError, self._fanuc.RejectedStartMismatch) as exception:
            raise InvalidTrajectoryException(f"The FANUC driver refused the trajectory: {exception}") from exception

        # Playback can legitimately outlast the timeline, by however long the driver allows the
        # measured joints to confirm arrival.
        timeout = duration + float(self.driver.policy.settle.timeout_s) + self._trajectory_timeout_margin
        try:
            result = handle.wait(timeout=timeout)
        except TimeoutError as exception:
            self.driver.stop_j()
            raise RuntimeError(
                f"The trajectory did not finish within {timeout:.1f}s ({duration:.1f}s of trajectory plus margin); "
                f"the robot has been stopped. {self._fault_description()}"
            ) from exception
        self._assert_motion_result_is_done(result, f"trajectory of {duration:.1f}s")

    def move_to_tcp_pose(
        self, tcp_pose: HomogeneousMatrixType, joint_speed: Optional[float] = None
    ) -> AwaitableAction:
        """Not supported: this would need inverse kinematics."""
        _raise_no_kinematics("move_to_tcp_pose")

    def move_linear_to_tcp_pose(
        self, tcp_pose: HomogeneousMatrixType, linear_speed: Optional[float] = None
    ) -> AwaitableAction:
        """Not supported: the driver has no Cartesian motion mode either."""
        _raise_no_kinematics(
            "move_linear_to_tcp_pose",
            "The driver has no Cartesian motion mode, so a straight line in TCP space is a joint "
            "trajectory you plan and hand to execute_trajectory().",
        )

    def servo_to_tcp_pose(self, tcp_pose: HomogeneousMatrixType, duration: float) -> AwaitableAction:
        """Not supported: this would need inverse kinematics."""
        _raise_no_kinematics(
            "servo_to_tcp_pose",
            "Solve the pose in your control loop and stream the joint configurations with "
            "servo_to_joint_configuration().",
        )

    ##############
    # kinematics #
    ##############

    def inverse_kinematics(
        self,
        tcp_pose: HomogeneousMatrixType,
        joint_configuration_near: Optional[JointConfigurationType] = None,
    ) -> Optional[JointConfigurationType]:
        """Not supported."""
        _raise_no_kinematics("inverse_kinematics")

    def forward_kinematics(self, joint_configuration: JointConfigurationType) -> HomogeneousMatrixType:
        """Not supported for arbitrary joint configurations."""
        _raise_no_kinematics(
            "forward_kinematics",
            "For the configuration the arm is currently in, get_tcp_pose() and get_flange_pose() "
            "return the controller's own answer; it cannot be asked about a hypothetical one.",
        )

    def is_tcp_pose_reachable(self, tcp_pose: HomogeneousMatrixType) -> bool:
        """Not supported: the base-class implementation answers this with inverse kinematics."""
        _raise_no_kinematics(
            "is_tcp_pose_reachable",
            "_is_joint_configuration_reachable() does work: it is a joint-limit check, not a kinematic one.",
        )

    def _is_joint_configuration_reachable(self, joint_configuration: JointConfigurationType) -> bool:
        """Is the configuration within the joint position limits of the arm's profile?

        These are the same limits the driver's real-time core clamps every commanded position
        against, so a configuration this rejects is one the driver would not execute as given.
        """
        configuration = np.asarray(joint_configuration, dtype=float)
        return bool(
            configuration.shape == (FANUC_DOF,)
            and np.all(configuration >= self._joint_lower_limits)
            and np.all(configuration <= self._joint_upper_limits)
        )

    def start_freedrive(self) -> None:
        """Not supported while the driver is connected."""
        raise NotImplementedError(
            "A FANUC is hand-guided with the buttons on the arm, but only while nothing holds the motion "
            "group, and this driver holds it for as long as it is connected. Close the driver and use "
            "`airo_fanuc.FanucReceiveInterface` to read joint angles while a human moves the arm: it polls "
            "over RMI and never takes the motion group."
        )

    def stop_freedrive(self) -> None:
        """Not supported, see `start_freedrive`."""
        self.start_freedrive()

    #################
    # driver access #
    #################

    def stop(self) -> None:
        """Preempt whatever is running with a limit-respecting deceleration (not part of the interface).

        Takes effect within one 8 ms tick and never raises. This is the *driver's* brake, not the
        controller's own backstop and not an emergency stop. It preempts everything submitted before
        it, so a servo loop that keeps feeding setpoints restarts the arm a tick later: stop the loop
        as well.
        """
        self.driver.stop_j()

    def hold(self) -> None:
        """Brake to rest and hold the commanded pose (not part of the interface).

        This is the only way a servo stream ends, and it resolves the active motion as preempted.
        """
        self.driver.hold()

    def is_steady(self) -> bool:
        """Is the robot at rest and holding position? (not part of the interface)"""
        return self.driver.is_steady()

    def wait_until_steady(self, timeout: float = 5.0) -> bool:
        """Block until the robot is at rest; return ``False`` on timeout (not part of the interface)."""
        return self.driver.wait_until_steady(timeout)

    ####################
    # non-api methods  #
    ####################

    def _submit(self, command: Callable[[], Any]) -> Any:
        """Run a driver motion command, converting its refusals into airo-robots exceptions."""
        try:
            handle = command()
        except self._fanuc.RobotFaultedError as exception:
            raise RuntimeError(
                f"The FANUC robot is not commandable: {exception}. "
                f"Recover it with `robot.driver.recover()` and, after an e-stop or an operator-required "
                f"fault, re-arm it with `robot.driver.arm()` — deliberately, never in a retry loop."
            ) from exception
        self.last_motion_handle = handle
        return handle

    def _brake_to_rest(self) -> None:
        """Bring the arm to rest so the driver will plan a point-to-point move from it.

        The driver does not brake on the caller's behalf, so that it never silently preempts a motion
        the caller did not know about. This class does brake, to match the other airo-robots
        manipulators, and says so.
        """
        if self.driver.is_steady():
            return
        logger.warning(
            "The FANUC arm is not at rest, so the previous motion (or servo stream) is being preempted "
            "before this point-to-point move is planned."
        )
        self.driver.hold()
        if not self.driver.wait_until_steady(self._brake_timeout):
            raise RuntimeError(
                f"The FANUC arm did not come to rest within {self._brake_timeout}s, so a point-to-point move "
                f"cannot be planned from it. {self._fault_description()}"
            )

    def _motion_action(self, handle: Any, description: str) -> AwaitableAction:
        """An AwaitableAction that completes when the driver's motion handle reaches a terminal result.

        The driver never raises on a motion *outcome* — a faulted, stopped or rejected motion resolves
        to a result instead — so an outcome other than 'done' is logged here rather than raised: the
        action has finished either way. Read ``robot.last_motion_handle.result()`` for the verdict.
        """

        def is_motion_done() -> bool:
            result = handle.result()
            if result is None:
                return False
            if result != self._fanuc.MotionResult.DONE:
                logger.error(f"The FANUC {description} ended as {result}. {self._fault_description()}")
            return True

        # The condition reads the driver's own lock-free motion status, so it can be polled fast.
        return AwaitableAction(is_motion_done, default_sleep_resolution=0.005)

    def _assert_motion_result_is_done(self, result: Any, description: str) -> None:
        if result == self._fanuc.MotionResult.DONE:
            return
        if result == self._fanuc.MotionResult.SETTLE_TIMEOUT:
            logger.warning(
                f"The FANUC {description} was commanded in full, but the measured joints did not confirm "
                f"arrival within the settle window. The arm may be stalled against something, or the settle "
                f"tolerance may be tight for this speed."
            )
            return
        raise RuntimeError(f"The FANUC {description} ended as {result}. {self._fault_description()}")

    def _fault_description(self) -> str:
        """The driver's current fault reason and the operator instruction that goes with it."""
        state = self.driver.get_state()
        fault = state.get("fault_reason", "unknown")
        hint = state.get("operator_hint")
        description = f"Driver state: {state.get('lifecycle_state')}, fault: {fault}."
        if hint:
            description += f" Operator instruction: {hint}"
        return description

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
    def _convert_fanuc_pose_to_homogeneous_pose(fanuc_pose: FanucPoseType) -> HomogeneousMatrixType:
        """Convert a FANUC ``[X, Y, Z, W, P, R]`` pose (mm, degrees) to a homogeneous matrix (m, rad).

        FANUC's W/P/R are fixed-axis (extrinsic) XYZ angles, i.e. ``R = Rz(R) @ Ry(P) @ Rx(W)``, which
        is the convention `SE3Container.from_euler_angles_and_translation` expects.
        """
        if fanuc_pose.shape != (6,):
            raise ValueError(f"Expected a FANUC pose with shape (6,), received {fanuc_pose.shape}.")
        return SE3Container.from_euler_angles_and_translation(
            np.radians(fanuc_pose[3:]), fanuc_pose[:3] / 1000.0
        ).homogeneous_matrix


##########################
# manual hardware tests  #
##########################
# Joint-space only: the Cartesian half of the PositionManipulator interface is not available on this
# robot. Every motion below is relative to the configuration the arm is already in, so these run on
# any FANUC without knowing where it is parked.


MANUAL_TEST_JOINT_SPEED = np.radians(15.0)
"""Leading-axis speed [rad/s] the manual tests move at: slow, for a first run on real hardware."""


def _swing_direction(robot: Fanuc, joint_index: int, amplitude: float) -> float:
    """Which way the joint has room to swing, so a joint parked near one of its stops swings away."""
    configuration = robot.get_joint_configuration()
    for direction in (1.0, -1.0):
        target = configuration.copy()
        target[joint_index] += direction * amplitude
        if robot._is_joint_configuration_reachable(target):
            return direction
    raise RuntimeError(
        f"J{joint_index + 1} is at {np.degrees(configuration[joint_index]):.1f} deg, which leaves no room for a "
        f"{np.degrees(amplitude):.1f} deg swing in either direction within its limits. Move the arm away from "
        f"its stop first, or run the tests on another joint."
    )


def manual_test_fanuc_move(
    robot: Fanuc,
    joint_index: int = 5,
    amplitude: float = np.radians(15.0),
    joint_speed: float = MANUAL_TEST_JOINT_SPEED,
) -> None:
    """Move one joint out and back with point-to-point moves."""
    start_configuration = robot.get_joint_configuration()
    delta = _swing_direction(robot, joint_index, amplitude) * amplitude
    target = start_configuration.copy()
    target[joint_index] += delta

    print(f"current joint configuration (deg): {np.degrees(start_configuration).round(2)}")
    input(
        f"robot will move J{joint_index + 1} by {np.degrees(delta):+.1f} deg at "
        f"{np.degrees(joint_speed):.0f} deg/s, press key to start"
    )
    action = robot.move_to_joint_configuration(target, joint_speed)
    print("method returned, will now wait for the action to finish")
    action.wait()
    print(f"reached (deg): {np.degrees(robot.get_joint_configuration()).round(2)}")

    input("robot will move back to the start configuration, press key to start")
    robot.move_to_joint_configuration(start_configuration, joint_speed).wait()
    print(f"back at (deg): {np.degrees(robot.get_joint_configuration()).round(2)}")


def manual_test_fanuc_servo(
    robot: Fanuc,
    joint_index: int = 5,
    amplitude: float = np.radians(5.0),
    period: float = 4.0,
    control_frequency: int = 50,
) -> None:
    """Servo one joint through a raised cosine, then end the stream with `hold`.

    A raised cosine rather than a plain sine because a sine demands its peak velocity at ``t=0``,
    which the driver's capture splice cannot deliver.
    """
    swing = _swing_direction(robot, joint_index, amplitude) * amplitude
    start_configuration = robot.get_joint_configuration()
    input(
        f"robot will servo J{joint_index + 1} through a {np.degrees(swing):+.1f} deg raised cosine of "
        f"{period:.1f}s at {control_frequency} Hz, press key to start"
    )
    target = start_configuration.copy()
    try:
        for step in range(int(period * control_frequency) + 1):
            t = step / control_frequency
            target[joint_index] = start_configuration[joint_index] + swing * (1 - np.cos(2 * np.pi * t / period)) / 2
            robot.servo_to_joint_configuration(target, 1 / control_frequency).wait()
    finally:
        # Without this the driver keeps holding the last setpoint, so stopping the loop (or Ctrl-C-ing
        # out of it) would not stop the arm.
        robot.hold()
        robot.wait_until_steady()
    print(f"servo finished, back at (deg): {np.degrees(robot.get_joint_configuration()).round(2)}")


def manual_test_fanuc_trajectory(
    robot: Fanuc, joint_index: int = 5, amplitude: float = np.radians(15.0), duration: float = 6.0
) -> None:
    """Execute a raised-cosine joint trajectory that returns to where it started.

    A raised cosine again: it starts and ends at rest, so its first knot is inside the driver's
    capture envelope and its last one does not leave the arm moving. The path is anchored at the
    *commanded* configuration, which is what the driver splices the first knot to.
    """
    swing = _swing_direction(robot, joint_index, amplitude) * amplitude
    start_configuration = robot.get_commanded_joint_configuration()
    times = np.linspace(0.0, duration, int(duration * 20) + 1)
    positions = np.tile(start_configuration, (times.size, 1))
    velocities = np.zeros_like(positions)
    omega = 2 * np.pi / duration
    positions[:, joint_index] += swing * (1 - np.cos(omega * times)) / 2
    velocities[:, joint_index] = swing * omega * np.sin(omega * times) / 2

    trajectory = SingleArmTrajectory(times, JointPathContainer(positions=positions, velocities=velocities))
    input(
        f"robot will execute a {duration:.1f}s trajectory that swings J{joint_index + 1} by "
        f"{np.degrees(swing):+.1f} deg and back, press key to start"
    )
    robot.execute_trajectory(trajectory)
    print(f"trajectory finished ({robot.last_motion_handle.result()}), the robot should be back where it started")
    print(f"joint configuration (deg): {np.degrees(robot.get_joint_configuration()).round(2)}")


def report_realtime_health(robot: Fanuc) -> None:
    """Print how well the host held the controller's interpolation deadline during the tests.

    The driver's ``examples/`` scripts report all of this in much more detail, and its
    troubleshooting page says what a bad number means.
    """
    statistics = robot.driver.timing_stats()
    if not statistics:
        print("no timing statistics available")
        return
    print(
        f"tx interval (ms): p50 {statistics['tx_interval_p50_ms']:.3f}  p99 {statistics['tx_interval_p99_ms']:.3f}  "
        f"max {statistics['tx_interval_max_ms']:.3f}   (target {robot.driver.policy.config.itp_s * 1000:.3f})"
    )
    print(
        f"slew clips: {robot.driver.get_state().get('total_slew_clips')} (non-zero means a commanded step was "
        f"trimmed, so the executed path was not the planned one)"
    )


def manual_test_fanuc(robot: Fanuc, joint_index: int = 5) -> None:
    """Run the joint-space manual tests against a FANUC robot.

    ``joint_index`` is zero-based; J6 (the wrist roll) is the default because a mistake there is the
    cheapest.
    """
    input("these tests will move the robot around its current configuration, make sure it is clear! Press key")

    print(f"joint configuration (deg): {np.degrees(robot.get_joint_configuration()).round(2)}")
    print(f"TCP pose (controller's own, with its active UTOOL):\n{robot.get_tcp_pose().round(4)}")
    print(f"flange pose (streamed):\n{robot.get_flange_pose().round(4)}")
    input("check that the printed pose and joint configuration match the pendant, press key if they do")

    input("point-to-point moves will now be tested, press key to start")
    manual_test_fanuc_move(robot, joint_index)

    input("servoing will now be tested, the robot should move smoothly, press key to start")
    manual_test_fanuc_servo(robot, joint_index)

    input("trajectory execution will now be tested, press key to start")
    manual_test_fanuc_trajectory(robot, joint_index)

    print("all joint-space tests finished")
    report_realtime_health(robot)


def create_crx10ial_profile() -> Any:
    """The ``RobotProfile`` of the CRX-10iA/L at AIRO, for the manual tests below.

    **Copy this, do not import it into your own code:** the profile carries the limits the driver's
    real-time core clamps against, and they are the *active configuration of one controller* rather
    than a property of the model. Read your own arm's off its controller with
    ``python -m airo_fanuc.controller_probe --ip <controller ip> --emit-profile``, see
    ``fanuc_setup.md``.
    """
    fanuc = _import_airo_fanuc()
    return fanuc.RobotProfile.from_degrees(
        name="crx10ial",
        model="FANUC CRX-10iA/L",
        velocity_limits_deg_s=[120.0, 120.0, 180.0, 180.0, 180.0, 180.0],
        acceleration_limits_deg_s2=[240.0, 240.0, 360.0, 360.0, 360.0, 360.0],
        jerk_limits_deg_s3=[1920.0, 1920.0, 2880.0, 2880.0, 2880.0, 2880.0],
        position_limits_lower_deg=[-179.999, -179.999, -270.0, -190.0, -179.999, -225.0],
        position_limits_upper_deg=[179.999, 179.999, 270.0, 190.0, 179.999, 225.0],
        max_payload_kg=10.0,
        source=(
            "velocity + position limits read from the controller's $PARAM_GROUP "
            "(controller_probe --emit-profile); acceleration = 2x velocity and "
            "jerk = 8x acceleration derived, not measured"
        ),
    )


if __name__ == "__main__":
    """test script for the FANUC implementation.
    e.g. python airo-robots/airo_robots/manipulators/hardware/fanuc.py --ip_address 192.168.1.100
    """
    import click
    from airo_robots.grippers.hardware.robotiq_2f85_fanuc import manually_test_fanuc_gripper

    @click.command()
    @click.option("--ip_address", default="192.168.1.100", show_default=True, help="IP address of the FANUC robot.")
    @click.option("--gripper", is_flag=True, help="Bring up the Robotiq 2F-85 gripper and test it too.")
    @click.option("--joint", default=6, show_default=True, type=click.IntRange(1, 6), help="Joint to move (1-6).")
    def test_fanuc(ip_address: str, gripper: bool, joint: int) -> None:
        """Run the manual manipulator tests against a FANUC robot."""
        fanuc = _import_airo_fanuc()

        print("Preconditions: controller in AUTO, drives powered, no active alarm, override at 100%, nothing")
        print("else talking to the controller, and an operator at the robot with the E-stop in hand.")
        if gripper:
            print("The gripper will OPEN during bring-up: the driver probes the dispatcher with a benign open.")
        input(f"about to connect to the FANUC at {ip_address}, press key to start")

        # The profile is the one thing the driver will not guess: it holds the limits its real-time
        # core clamps against. See create_crx10ial_profile() for how to get your own arm's.
        policy = fanuc.DriverPolicy(
            config=fanuc.DriverConfig(profile=create_crx10ial_profile()),
            enable_gripper=gripper,
        )
        # Constructing the robot runs the whole bring-up ladder and blocks until the arm is
        # commandable, or raises with a reason. Closing it (here, by leaving the `with` block) stops
        # the arm, tears the session down and releases the controller.
        with Fanuc(ip_address, policy) as robot:
            manual_test_fanuc(robot, joint - 1)
            if isinstance(robot.gripper, Robotiq2F85Fanuc):
                manually_test_fanuc_gripper(robot.gripper)

    test_fanuc()
