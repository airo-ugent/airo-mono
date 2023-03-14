from abc import ABC, abstractmethod

from airo_typing import HomogeneousMatrixType, WrenchType


class ForceTorqueSensor(ABC):
    """Interface for a Force-Torque (FT) sensor that is attached to the flange of a Robot manipulator.
    This can be an internal FT sensor (such as with the UR e-series) or an external sensor."""

    @abstractmethod
    def get_wrench(self) -> WrenchType:
        """Returns the wrench on the FT frame expressed in the base frame of the robot it is attached to, W^FT_base.
        All required wrench-on frame or expressed-in frame conversions should be done internally."""

    @property
    def sensor_pose_in_tcp(self) -> HomogeneousMatrixType:
        """Returns the (fixed) transform from the TCP frame to the FT frame, T^FT_tcp.

        Raises:
            NotImplementedError: This function does not need to be implemented, as you sometimes don't know (or need) this transform.
        """
        raise NotImplementedError
