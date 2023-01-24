from abc import ABC, abstractmethod

from airo_typing import HomogeneousMatrixType, WrenchType


class ForceTorqueSensor(ABC):
    """Interface for a Force-Torque (FT) sensor, this can be an internal FT sensor (such as with the UR e-series) or an external sensor."""

    @abstractmethod
    def get_wrench(self) -> WrenchType:
        """Returns the wrench on the TCP frame.
        Any expressed-in frame conversions should be done internally.

        Check if gravity-compensation is applied and to what elements of the total payload, as no convention is enforced in the interface.

        """

    @property
    def sensor_pose_in_tcp_frame(self) -> HomogeneousMatrixType:
        """Returns the (fixed) transform between the FT sensor frame and the TCP frame
        which can be used to express the wrench in other frames or to apply gravity compensation.
        Raises:
            NotImplementedError: This function does not need to be implemented, as you sometimes don't know (nor need) this transform.
        """
        raise NotImplementedError
