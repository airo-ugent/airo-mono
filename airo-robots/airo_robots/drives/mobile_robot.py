from abc import ABC, abstractmethod

from airo_robots.awaitable_action import AwaitableAction


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
    def set_platform_velocity_target(self, x: float, y: float, a: float, timeout: float,
                                     align_drives_first: bool = False) -> AwaitableAction:
        """Set the desired platform velocity.

        Args:
            x: Linear velocity along the X axis.
            y: Linear velocity along the Y axis.
            a: Angular velocity.
            timeout: After this time, the platform will automatically stop.
            align_drives_first: Align drives before moving (default: False, will start driving instantly).

        Returns:
            An awaitable action."""

    @abstractmethod
    def enable_compliant_mode(self, enabled: bool):
        """Enable compliant mode on the robot.

        Args:
            enabled: If true, will enable compliant mode. Else, will disable compliant mode."""
