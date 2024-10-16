from abc import ABC, abstractmethod
from enum import Enum

from airo_robots.awaitable_action import AwaitableAction
from airo_typing import Vector3DType


class CompliantLevel(Enum):
    """The level of compliance expected from the mobile robot.

    Values may not correspond to identical behaviour on different mobile platforms, but are merely an indication.
    A value of weak means a very compliant robot, whereas a value of strong means a slightly compliant robot."""

    COMPLIANT_WEAK = 1
    COMPLIANT_MODERATE = 2
    COMPLIANT_STRONG = 3


class MobileRobot(ABC):
    """
    Base class for a mobile robot.

    Mobile robots typically allow to set a target velocity for the entire platform and apply torques to their wheels
    to achieve the desired velocity.

    All values are in metric units:
    - Linear velocities are expressed in meters/second
    - Angular velocities are expressed in radians/second
    """

    @abstractmethod
    def set_platform_velocity_target(self, x: float, y: float, a: float, timeout: float) -> AwaitableAction:
        """Set the desired platform velocity.

        Args:
            x: Linear velocity along the X axis.
            y: Linear velocity along the Y axis.
            a: Angular velocity.
            timeout: After this time, the platform will automatically stop.

        Returns:
            An awaitable action."""

    @abstractmethod
    def move_platform_to_pose(self, x: float, y: float, a: float, timeout: float) -> AwaitableAction:
        """Move the platform to the given pose, without guarantees about the followed path.

        Args:
            x: Position along the startup pose's X axis.
            y: Position along the startup pose's Y axis.
            a: Orientation around the startup pose's Z axis.
            timeout: After this time, the platform will automatically stop.

        Returns:
            An awaitable action."""

    @abstractmethod
    def enable_compliant_mode(self, enabled: bool, compliant_level: CompliantLevel) -> None:
        """Enable compliant mode on the robot.

        Args:
            enabled: If true, will enable compliant mode. Else, will disable compliant mode.
            compliant_level: The level of compliance to be expected from the robot. Ignored if `enabled` is `False`."""

    @abstractmethod
    def get_odometry(self) -> Vector3DType:
        """Get the estimated robot pose as a 3D vector comprising the `x`, `y`, and `theta` values relative to the
        robot's starting pose.

        Returns: A 3D vector with a value for the `x` position, `y` position, and `theta` angle of the robot."""

    @abstractmethod
    def reset_odometry(self) -> None:
        """Reset the robot's odometry to `(0, 0, 0)`."""

    @abstractmethod
    def get_velocity(self) -> Vector3DType:
        """Get the estimated robot's velocity as a 3D vector comprising the `x`, `y` and `theta` values relative to the
        robot's starting pose.

        Returns:
            A 3D vector with a value for the `x` velocity, `y` velocity, and `theta` angular velocity of the robot."""
