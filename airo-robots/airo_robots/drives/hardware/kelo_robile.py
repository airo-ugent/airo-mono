import time
from functools import partial

from airo_tulip.platform_driver import PlatformDriverType
from airo_tulip.server.kelo_robile import KELORobile as KELORobileClient

from airo_robots.awaitable_action import AwaitableAction
from airo_robots.drives.mobile_robot import MobileRobot


class KELORobile(MobileRobot):
    """KELO Robile platform implementation of the MobileRobot interface.

    The KELO Robile platform consists of several drives with castor wheels which are connected via EtherCAT and can
    be controlled individually. The airo-tulip API, which is used here, provides a higher level interface which
    controls the entire platform."""

    def __init__(self, robot_ip: str, robot_port: int):
        """Connect to the KELO robot.

        The KELO robot should already be running the airo-tulip server.

        Args:
            robot_ip: IP address of the KELO CPU brick.
            robot_port: Port to connect on."""
        self._kelo_robile = KELORobileClient(robot_ip, robot_port)

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

        def aligned_awaitable(start_time: float) -> bool:
            return self._kelo_robile.are_drives_aligned() or time.time() > start_time + timeout

        return AwaitableAction(partial(aligned_awaitable, time.time()))

    def set_platform_velocity_target(self, x: float, y: float, a: float, timeout: float) -> AwaitableAction:
        self._kelo_robile.set_platform_velocity_target(x, y, a, timeout=timeout, instantaneous=True)

        def timeout_awaitable() -> bool:
            time.sleep(timeout)
            return True

        return AwaitableAction(timeout_awaitable)

    def enable_compliant_mode(self, enabled: bool, compliant_level: int = 1):  # TODO: Use enum variants, document.
        if enabled:
            if compliant_level == 1:
                self._kelo_robile.set_driver_type(PlatformDriverType.COMPLIANT_WEAK)
            elif compliant_level == 2:
                self._kelo_robile.set_driver_type(PlatformDriverType.COMPLIANT_MODERATE)
            else:
                self._kelo_robile.set_driver_type(PlatformDriverType.COMPLIANT_STRONG)
        else:
            self._kelo_robile.set_driver_type(PlatformDriverType.VELOCITY)
