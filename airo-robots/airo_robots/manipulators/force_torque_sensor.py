from abc import abstractmethod

from airo_typing import HomogeneousMatrixType, WrenchType


class ForceTorqueSensor:
    """Interface for a Force-Torque (FT) sensor, this can be an internal FT sensor (such as with the UR e-series) or an external sensor."""

    @abstractmethod
    def get_wrench(self) -> WrenchType:
        """Returns the wrench on the TCP frame, so any frame conversions should be done internally."""

    @property
    def wrench_in_tcp_pose(self) -> HomogeneousMatrixType:
        """Returns the (fixed) transform between the FT sensor frame and the TCP frame

        Raises:
            NotImplementedError: This function does not need to be implemented, as you sometimes don't know (nor need) this transform.
        """
        raise NotImplementedError
