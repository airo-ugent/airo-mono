import time

from airo_robots.awaitable_action import AwaitableAction
from airo_robots.drives.mobile_robot import MobileRobot
from airo_tulip.server.kelo_robile import KELORobile as KELORobileClient


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

    def set_platform_velocity_target(self, x: float, y: float, a: float, timeout: float) -> AwaitableAction:
        self._kelo_robile.set_platform_velocity_target(x, y, a, timeout)

        def timeout_awaitable() -> bool:
            time.sleep(timeout)
            return True

        return AwaitableAction(timeout_awaitable)
