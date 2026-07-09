"""Joint-space torque control for UR e-series robots.

This module extends the position-controlled `URrtde` class with a torque control mode.
When torque control is enabled, a dedicated process runs a PD control loop at 500Hz
that computes joint torques to track a target joint configuration, and streams them to the robot with
`RTDEControlInterface.directTorque`. The target joint configuration can be updated at any time from the main process.

Example usage:

    robot = URrtdeTorque("10.42.0.162")
    robot.move_to_joint_configuration(start_configuration).wait()  # regular position control
    robot.enable_torque_control()
    robot.target_joint_configuration = target_configuration  # tracked by the PD controller
    ...
    robot.disable_torque_control()  # position control is available again

WARNING: the default PD gains were tuned on a UR3e. Before using torque control on another robot model or with a
different payload, validate and tune the gains (`kp` and `kd` constructor arguments) at low torque limits first.

See universal_robots_torque_control.md (in this directory) for a full guide with safety notes and tuning tips.
"""
import atexit
import time
from dataclasses import dataclass
from multiprocessing import Array, Lock, Process, Value
from multiprocessing.sharedctypes import Synchronized, SynchronizedArray
from multiprocessing.synchronize import Lock as LockType
from typing import Optional

import numpy as np
from airo_robots.awaitable_action import AwaitableAction
from airo_robots.grippers import ParallelPositionGripper
from airo_robots.manipulators.hardware.ur_rtde import URrtde
from airo_robots.manipulators.position_manipulator import ManipulatorSpecs
from airo_typing import HomogeneousMatrixType, JointConfigurationType, SingleArmTrajectory
from loguru import logger
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

TORQUE_LIMIT_SAFETY_FACTOR = 0.8
"""The controller is never allowed to command more than this fraction of the robot's maximum joint torques."""

ZERO_TORQUE_RAMP_CYCLES = 20
"""Number of control cycles during which zero torque is commanded before the control script is stopped."""

TORQUE_CONTROL_FREQUENCY = 500.0
"""Frequency of the torque control loop [Hz]: the native RTDE rate of UR e-series robots."""


@dataclass
class JointSpacePDController:
    """A joint-space PD controller with a second-order reference trajectory generator.

    Instead of applying the PD gains directly on the (possibly large) error between the target and the measured
    joint positions, the target is first smoothed by a critically damped second-order system. This limits jerk when
    the target makes a step, at the cost of a small tracking delay. The measured joint velocities are low-pass
    filtered before use in the D-term to reduce noise amplification.

    This class is hardware-agnostic and holds no robot connection, so it can be unit-tested in isolation.

    Attributes:
        kp: proportional gains per joint [Nm/rad].
        kd: derivative gains per joint [Nm/(rad/s)].
        joint_torque_limits: maximum torque magnitude per joint [Nm]; the output is clipped to this.
        control_period: time between control cycles [s].
        reference_natural_frequency: natural frequency [rad/s] of the reference trajectory generator.
        reference_damping_ratio: damping ratio of the reference trajectory generator (>= 1 avoids overshoot).
        velocity_filter_alpha: exponential smoothing factor for the measured joint velocities (1.0 disables filtering).
    """

    kp: np.ndarray
    kd: np.ndarray
    joint_torque_limits: np.ndarray
    control_period: float
    reference_natural_frequency: float = 40.0
    reference_damping_ratio: float = 1.05
    velocity_filter_alpha: float = 0.1

    def __post_init__(self) -> None:
        self.kp = np.asarray(self.kp, dtype=float)
        self.kd = np.asarray(self.kd, dtype=float)
        self.joint_torque_limits = np.asarray(self.joint_torque_limits, dtype=float)
        if not (self.kp.shape == self.kd.shape == self.joint_torque_limits.shape):
            raise ValueError(
                f"kp, kd and joint_torque_limits must have the same shape, "
                f"got {self.kp.shape}, {self.kd.shape} and {self.joint_torque_limits.shape}."
            )
        if np.any(self.kp < 0.0) or np.any(self.kd < 0.0):
            raise ValueError("kp and kd must be non-negative.")
        if np.any(self.joint_torque_limits <= 0.0):
            raise ValueError(f"joint_torque_limits must be strictly positive, got {self.joint_torque_limits}.")
        if self.control_period <= 0.0:
            raise ValueError(f"control_period must be strictly positive, got {self.control_period}.")
        self.reset(np.zeros(self.dof))

    @property
    def dof(self) -> int:
        return len(self.kp)

    def reset(self, joint_positions: JointConfigurationType) -> None:
        """(Re)initialize the reference trajectory at the given joint positions, with zero velocity.

        Call this with the measured joint positions right before starting a control loop, so that the controller
        does not command a jump."""
        self._reference_position: np.ndarray = np.asarray(joint_positions, dtype=float).copy()
        self._reference_velocity: np.ndarray = np.zeros(self.dof)
        self._filtered_velocity: np.ndarray = np.zeros(self.dof)

    def compute_torques(
        self,
        target_joint_positions: JointConfigurationType,
        joint_positions: JointConfigurationType,
        joint_velocities: np.ndarray,
    ) -> np.ndarray:
        """Advance the reference trajectory by one control period and compute the joint torques to command.

        Args:
            target_joint_positions: the joint positions the controller should track [rad].
            joint_positions: the measured joint positions [rad].
            joint_velocities: the measured joint velocities [rad/s].

        Returns:
            The joint torques to command [Nm], clipped to the joint torque limits.
        """
        target = np.asarray(target_joint_positions, dtype=float)
        alpha = self.velocity_filter_alpha
        self._filtered_velocity = (
            alpha * np.asarray(joint_velocities, dtype=float) + (1 - alpha) * self._filtered_velocity
        )

        # Second-order reference trajectory: ref_acc = w^2 * (target - ref_pos) - 2 * zeta * w * ref_vel
        omega = self.reference_natural_frequency
        reference_acceleration = omega * omega * (target - self._reference_position) - (
            2.0 * self.reference_damping_ratio * omega * self._reference_velocity
        )
        self._reference_velocity = self._reference_velocity + reference_acceleration * self.control_period
        self._reference_position = self._reference_position + self._reference_velocity * self.control_period

        torque_p = self.kp * (self._reference_position - np.asarray(joint_positions, dtype=float))
        torque_d = self.kd * (self._reference_velocity - self._filtered_velocity)
        # The D-term alone should not be able to saturate the output, to preserve some P-action near the limits.
        torque_d = np.clip(torque_d, -0.8 * self.joint_torque_limits, 0.8 * self.joint_torque_limits)
        return np.clip(torque_p + torque_d, -self.joint_torque_limits, self.joint_torque_limits)


def _torque_control_worker(
    ip_address: str,
    controller: JointSpacePDController,
    target_shared: SynchronizedArray,
    running_flag: Synchronized,
    target_lock: LockType,
) -> None:
    """Real-time torque control loop, meant to run in a dedicated process.

    Creates its own RTDE interfaces (only one RTDEControlInterface can be connected to a robot at a time, so the
    parent process must disconnect its control interface before starting this worker). Tracks the target joint
    configuration in `target_shared` with the given PD controller until `running_flag` is cleared, then ramps down
    to zero torque and stops the control script.

    Args:
        ip_address: IP address of the UR robot.
        controller: the PD controller to use. Its reference trajectory is reset to the measured joint positions
            before the loop starts.
        target_shared: shared array with the target joint configuration, written by the parent process.
        running_flag: shared boolean; the loop runs for as long as it is set.
        target_lock: lock guarding `target_shared`.
    """
    rtde_control = RTDEControlInterface(ip_address, TORQUE_CONTROL_FREQUENCY)
    rtde_receive = RTDEReceiveInterface(ip_address, TORQUE_CONTROL_FREQUENCY)
    dof = controller.dof
    controller.reset(np.array(rtde_receive.getActualQ()))
    try:
        while running_flag.value:
            cycle_start = rtde_control.initPeriod()
            joint_positions = np.array(rtde_receive.getActualQ())
            joint_velocities = np.array(rtde_receive.getActualQd())
            with target_lock:
                target = np.array(target_shared[:dof])
            torques = controller.compute_torques(target, joint_positions, joint_velocities)
            rtde_control.directTorque(torques.tolist())
            rtde_control.waitPeriod(cycle_start)
    except KeyboardInterrupt:
        logger.info("Torque control loop interrupted by user.")
    except Exception as e:
        logger.error(f"Torque control loop stopped because of an unexpected error: {e}")
    finally:
        try:
            zero_torque = [0.0] * dof
            for _ in range(ZERO_TORQUE_RAMP_CYCLES):
                cycle_start = rtde_control.initPeriod()
                rtde_control.directTorque(zero_torque)
                rtde_control.waitPeriod(cycle_start)
            rtde_control.stopScript()
            rtde_control.disconnect()
            rtde_receive.disconnect()
        except Exception as e:
            logger.warning(f"Failed to shut down the torque control loop cleanly: {e}")


class URrtdeTorque(URrtde):
    """URrtde with an additional joint-space torque control mode.

    This class behaves exactly like `URrtde` until `enable_torque_control()` is called. That call hands the robot's
    control script over to a dedicated process running a PD torque control loop, after which the robot tracks the
    `target_joint_configuration` property. While torque control is active, all position-control commands (move/servo
    methods, trajectory execution, and rtde_control-based queries such as inverse kinematics) raise a `RuntimeError`;
    state queries through the receive interface (`get_joint_configuration`, `get_tcp_pose`, `get_tcp_force`) keep
    working. Call `disable_torque_control()` to stop the torque loop and restore regular position control. Both
    modes can be alternated freely.

    Torque control requires a UR e-series robot (500Hz control rate) and a ur-rtde version that exposes
    `directTorque` (>= 1.6).

    WARNING: the default gains were tuned on a UR3e. Validate them on your robot and payload before relying on them.
    """

    DEFAULT_KP = np.array([120.0, 120.0, 100.0, 30.0, 30.0, 30.0])
    """Default proportional gains [Nm/rad], tuned on a UR3e."""
    DEFAULT_KD = np.array([12.0, 12.0, 10.0, 2.4, 2.0, 1.0])
    """Default derivative gains [Nm/(rad/s)], tuned on a UR3e."""

    def __init__(
        self,
        ip_address: str,
        manipulator_specs: Optional[ManipulatorSpecs] = None,
        gripper: Optional[ParallelPositionGripper] = None,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
    ) -> None:
        """
        Args:
            ip_address: IP address of the UR robot.
            manipulator_specs: specs of the robot. If None, they are auto-detected from the robot model.
                `max_joint_torques` must be set on the specs to use torque control.
            gripper: optional gripper attached to the robot.
            kp: proportional gains per joint [Nm/rad]. Defaults to `DEFAULT_KP` (tuned on a UR3e).
            kd: derivative gains per joint [Nm/(rad/s)]. Defaults to `DEFAULT_KD` (tuned on a UR3e).
        """
        super().__init__(ip_address, manipulator_specs, gripper)
        if not self.model.value.endswith("e"):
            raise RuntimeError(f"Torque control requires a UR e-series robot, but a {self.model.value} was detected.")
        if self.manipulator_specs.max_joint_torques is None:
            raise ValueError("Torque control requires manipulator_specs.max_joint_torques to be set.")
        if not hasattr(self.rtde_control, "directTorque"):
            raise RuntimeError(
                "Your ur-rtde version does not support direct torque control. Upgrade to ur-rtde >= 1.6."
            )
        dof = self.manipulator_specs.dof
        self._kp = np.asarray(kp if kp is not None else self.DEFAULT_KP, dtype=float)
        self._kd = np.asarray(kd if kd is not None else self.DEFAULT_KD, dtype=float)
        if self._kp.shape != (dof,) or self._kd.shape != (dof,):
            raise ValueError(f"kp and kd must have shape ({dof},), got {self._kp.shape} and {self._kd.shape}.")
        self._target_shared = Array("d", dof)
        self._running_flag = Value("b", False)
        self._target_lock = Lock()
        self._torque_process: Optional[Process] = None

    @property
    def is_torque_control_active(self) -> bool:
        """True if the torque control process is running and healthy."""
        return self._torque_process is not None and self._torque_process.is_alive()

    def enable_torque_control(self) -> None:
        """Start the torque control loop in a dedicated process.

        The robot will actively hold its current joint configuration until `target_joint_configuration` is set.
        While torque control is active, position-control commands raise a `RuntimeError`.
        """
        if self.is_torque_control_active:
            logger.warning("Torque control is already active.")
            return
        controller = JointSpacePDController(
            kp=self._kp,
            kd=self._kd,
            joint_torque_limits=np.asarray(self.manipulator_specs.max_joint_torques) * TORQUE_LIMIT_SAFETY_FACTOR,
            control_period=1.0 / TORQUE_CONTROL_FREQUENCY,
        )
        joint_configuration = self.get_joint_configuration()
        with self._target_lock:
            for i, position in enumerate(joint_configuration):
                self._target_shared[i] = float(position)
        # Only one RTDEControlInterface can be connected to the robot at a time, so hand it over to the worker.
        self.rtde_control.disconnect()
        self._running_flag.value = True
        self._torque_process = Process(
            target=_torque_control_worker,
            args=(
                self.ip_address,
                controller,
                self._target_shared,
                self._running_flag,
                self._target_lock,
            ),
            daemon=True,
        )
        self._torque_process.start()
        # Make sure the robot is ramped down to zero torque even if the user forgets to disable torque control.
        atexit.register(self.disable_torque_control)
        logger.info(f"Torque control enabled ({TORQUE_CONTROL_FREQUENCY:.0f}Hz control loop).")

    def disable_torque_control(self) -> None:
        """Stop the torque control loop (with a zero-torque ramp-down) and restore position control."""
        if self._torque_process is None:
            return
        self._running_flag.value = False
        self._torque_process.join(timeout=5.0)
        if self._torque_process.is_alive():
            logger.warning("The torque control process did not stop in time, terminating it.")
            self._torque_process.terminate()
            self._torque_process.join()
        self._torque_process = None
        atexit.unregister(self.disable_torque_control)
        self.rtde_control.reconnect()
        logger.info("Torque control disabled, position control restored.")

    @property
    def target_joint_configuration(self) -> JointConfigurationType:
        """The joint configuration that the torque controller is tracking. Only available while torque control is
        active. Large jumps in the target are smoothed by the controller's reference trajectory generator, but you
        should still prefer sending targets close to the current configuration."""
        self._assert_torque_control_active()
        with self._target_lock:
            return np.array(self._target_shared[:], dtype=float)

    @target_joint_configuration.setter
    def target_joint_configuration(self, joint_configuration: JointConfigurationType) -> None:
        self._assert_torque_control_active()
        joint_configuration = np.asarray(joint_configuration, dtype=float)
        if joint_configuration.shape != (self.manipulator_specs.dof,):
            raise ValueError(
                f"joint configuration must have shape ({self.manipulator_specs.dof},), "
                f"got {joint_configuration.shape}."
            )
        with self._target_lock:
            for i, position in enumerate(joint_configuration):
                self._target_shared[i] = float(position)

    def get_tcp_force(self) -> np.ndarray:
        """The TCP force/torque [Fx, Fy, Fz, Tx, Ty, Tz] in N and Nm, as estimated by the robot.
        Available in both position and torque control mode."""
        return np.array(self.rtde_receive.getActualTCPForce())

    def _assert_torque_control_active(self) -> None:
        if self._torque_process is None:
            raise RuntimeError("Torque control is not enabled. Call enable_torque_control() first.")
        if not self._torque_process.is_alive():
            raise RuntimeError(
                "The torque control process has died unexpectedly (check the logs for errors). "
                "Call disable_torque_control() to restore position control."
            )

    def _assert_torque_control_inactive(self, method_name: str) -> None:
        if self._torque_process is not None:
            raise RuntimeError(
                f"{method_name} is not available while torque control is active. Call disable_torque_control() first."
            )

    # While torque control is active, the worker process owns the robot's control script, so all methods that rely
    # on the parent's rtde_control interface are unavailable and raise a RuntimeError.

    def move_to_tcp_pose(
        self, tcp_pose: HomogeneousMatrixType, joint_speed: Optional[float] = None
    ) -> AwaitableAction:
        self._assert_torque_control_inactive("move_to_tcp_pose")
        return super().move_to_tcp_pose(tcp_pose, joint_speed)

    def move_linear_to_tcp_pose(
        self, tcp_pose: HomogeneousMatrixType, linear_speed: Optional[float] = None
    ) -> AwaitableAction:
        self._assert_torque_control_inactive("move_linear_to_tcp_pose")
        return super().move_linear_to_tcp_pose(tcp_pose, linear_speed)

    def move_to_joint_configuration(
        self, joint_configuration: JointConfigurationType, joint_speed: Optional[float] = None
    ) -> AwaitableAction:
        self._assert_torque_control_inactive("move_to_joint_configuration")
        return super().move_to_joint_configuration(joint_configuration, joint_speed)

    def servo_to_tcp_pose(self, tcp_pose: HomogeneousMatrixType, duration: float) -> AwaitableAction:
        self._assert_torque_control_inactive("servo_to_tcp_pose")
        return super().servo_to_tcp_pose(tcp_pose, duration)

    def servo_to_joint_configuration(
        self, joint_configuration: JointConfigurationType, duration: float
    ) -> AwaitableAction:
        self._assert_torque_control_inactive("servo_to_joint_configuration")
        return super().servo_to_joint_configuration(joint_configuration, duration)

    def execute_trajectory(self, joint_trajectory: SingleArmTrajectory, sampling_frequency: float = 100) -> None:
        self._assert_torque_control_inactive("execute_trajectory")
        super().execute_trajectory(joint_trajectory, sampling_frequency)

    def inverse_kinematics(
        self, tcp_pose: HomogeneousMatrixType, joint_configuration_guess: Optional[JointConfigurationType] = None
    ) -> JointConfigurationType:
        self._assert_torque_control_inactive("inverse_kinematics")
        return super().inverse_kinematics(tcp_pose, joint_configuration_guess)

    def is_tcp_pose_reachable(self, tcp_pose: HomogeneousMatrixType) -> bool:
        self._assert_torque_control_inactive("is_tcp_pose_reachable")
        return super().is_tcp_pose_reachable(tcp_pose)


if __name__ == "__main__":
    """Manual test script for torque control. The robot will hold its current configuration under torque control,
    then track a slow sine wave on the wrist joint, and finally switch back to position control.
    e.g. python airo-robots/airo_robots/manipulators/hardware/ur_rtde_torque.py --ip_address 10.42.0.162
    """
    import click

    @click.command()
    @click.option("--ip_address", help="IP address of the UR robot")
    def test_ur_rtde_torque(ip_address: str) -> None:
        robot = URrtdeTorque(ip_address)
        start_configuration = robot.get_joint_configuration()
        input("The robot will hold its current configuration and then move its wrist joint. Press Enter to start.")

        robot.enable_torque_control()
        time.sleep(2.0)  # hold the current configuration

        t0 = time.time()
        while time.time() - t0 < 10.0:
            target = start_configuration.copy()
            target[5] += 0.3 * np.sin(2 * np.pi * 0.2 * (time.time() - t0))
            robot.target_joint_configuration = target
            time.sleep(0.01)

        robot.disable_torque_control()
        robot.move_to_joint_configuration(start_configuration).wait()

    test_ur_rtde_torque()
