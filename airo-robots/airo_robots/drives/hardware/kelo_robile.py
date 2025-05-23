import math
import time
from threading import Thread

import numpy as np
from airo_robots.awaitable_action import AwaitableAction
from airo_robots.drives.mobile_robot import CompliantLevel, MobileRobot
from airo_tulip.api.client import KELORobile as KELORobileClient  # type: ignore
from airo_tulip.hardware.platform_driver import PlatformDriverType  # type: ignore
from airo_typing import Vector3DType


class KELORobile(MobileRobot):
    """KELO Robile platform implementation of the MobileRobot interface.

    The KELO Robile platform consists of several drives with castor wheels which are connected via EtherCAT and can
    be controlled individually. The airo-tulip API, which is used here, provides a higher level interface which
    controls the entire platform."""

    def __init__(self, robot_ip: str, robot_port: int = 49789):
        """Connect to the KELO robot.

        The KELO robot should already be running the airo-tulip server. If this is not the case, any messages
        sent from the client may be queued and executed once the server is started, resulting in unexpected
        and/or sudden movements.

        Args:
            robot_ip: IP address of the KELO CPU brick.
            robot_port: Port to connect on (default: 49789)."""

        self._kelo_robile = KELORobileClient(robot_ip, robot_port)

        # Position control.
        self._control_loop_done = True

    def align_drives(self, x: float, y: float, a: float, timeout: float = 1.0) -> AwaitableAction:
        """Align all drives for driving in a direction given by the linear and angular velocities.

        Beware that sending any other velocity commands before the awaitable is done may cause unexpected behaviour,
        as this affects the "aligned" condition.

        Args:
            x: The x velocity (linear, m/s).
            y: The y velocity (linear, m/s)
            a: The angular velocity (rad/s).
            timeout: The awaitable will finish after this time at the latest.

        Returns:
            An AwaitableAction which will check if the drives are aligned, or if the timeout has expired."""
        self._kelo_robile.align_drives(x, y, a, timeout=timeout)

        action_sent_time = time.time_ns()
        return AwaitableAction(
            lambda: self._kelo_robile.are_drives_aligned() or time.time_ns() - action_sent_time > timeout * 1e9,
            default_timeout=2 * timeout,
            default_sleep_resolution=0.002,
        )

    def set_platform_velocity_target(self, x: float, y: float, a: float, timeout: float) -> AwaitableAction:
        self._kelo_robile.set_platform_velocity_target(x, y, a, timeout=timeout)

        action_sent_time = time.time_ns()
        return AwaitableAction(
            lambda: time.time_ns() - action_sent_time > timeout * 1e9,
            default_timeout=2 * timeout,
            default_sleep_resolution=0.002,
        )

    def _move_platform_to_pose_control_loop(
        self, target_pose: Vector3DType, action_start_time: float, action_timeout_time: float, timeout: float
    ) -> None:
        stop = False
        while not stop:
            current_pose = self._kelo_robile.get_odometry()
            delta_pose = target_pose - current_pose
            # Fix issues around multiples of 2PI
            while delta_pose[2] > np.pi:
                delta_pose[2] -= 2 * np.pi
            while delta_pose[2] < -np.pi:
                delta_pose[2] += 2 * np.pi

            vel_vec_angle = np.arctan2(delta_pose[1], delta_pose[0]) - current_pose[2]
            vel_vec_norm = min(np.linalg.norm(delta_pose[:2]), 0.5)
            vel_x = vel_vec_norm * np.cos(vel_vec_angle)
            vel_y = vel_vec_norm * np.sin(vel_vec_angle)

            delta_angle = np.arctan2(np.sin(delta_pose[2]), np.cos(delta_pose[2]))
            P_angle = 1.5
            vel_a = max(min(P_angle * delta_angle, math.pi / 4), -math.pi / 4)

            command_timeout = (action_timeout_time - time.time_ns()) * 1e-9
            if command_timeout >= 0.0:
                self._kelo_robile.set_platform_velocity_target(vel_x, vel_y, vel_a, timeout=command_timeout)

            at_target_pose = np.linalg.norm(delta_pose[:2]) < 0.01 and abs(delta_pose[2]) < np.deg2rad(1.5)
            stop = at_target_pose or time.time_ns() - action_start_time > timeout * 1e9

        self._kelo_robile.set_platform_velocity_target(0.0, 0.0, 0.0)
        self._control_loop_done = True

    def move_platform_to_pose(self, x: float, y: float, a: float, timeout: float) -> AwaitableAction:
        target_pose = np.array([x, y, a])
        action_start_time = time.time_ns()
        action_timeout_time = action_start_time + timeout * 1e9

        self._control_loop_done = False  # Will be set to True by the below thread once it's finished.
        thread = Thread(
            target=self._move_platform_to_pose_control_loop,
            args=(target_pose, action_start_time, action_timeout_time, timeout),
        )
        thread.start()

        return AwaitableAction(
            lambda: self._control_loop_done,
            default_timeout=2 * timeout,
            default_sleep_resolution=0.002,
        )

    def enable_compliant_mode(
        self, enabled: bool, compliant_level: CompliantLevel = CompliantLevel.COMPLIANT_WEAK
    ) -> None:
        if enabled:
            if compliant_level == CompliantLevel.COMPLIANT_WEAK:
                self._kelo_robile.set_driver_type(PlatformDriverType.COMPLIANT_WEAK)
            elif compliant_level == CompliantLevel.COMPLIANT_MODERATE:
                self._kelo_robile.set_driver_type(PlatformDriverType.COMPLIANT_MODERATE)
            else:
                self._kelo_robile.set_driver_type(PlatformDriverType.COMPLIANT_STRONG)
        else:
            self._kelo_robile.set_driver_type(PlatformDriverType.VELOCITY)

    def get_odometry(self) -> Vector3DType:
        return self._kelo_robile.get_odometry()

    def get_velocity(self) -> Vector3DType:
        return self._kelo_robile.get_velocity()

    def reset_odometry(self) -> None:
        self._kelo_robile.reset_odometry()
