from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParallelPositionGripperSpecs:
    """
    all values are in metric units:
    - the position of the gripper is expressed as the width between the fingers in meters
    - the speed in meters/second
    - the force in Newton
    """

    max_width: float
    min_width: float
    max_force: float
    min_force: float
    max_speed: float
    min_speed: float


class ParallelPositionGripper:
    """
    Interface for a position-controlled, 2 finger parallel gripper.

    These grippers typically allow to set a speed and maximum applied force before moving,
    and attempt to move to specified positions under these constraints.

    all values are in metric units:
    - the position of the gripper is expressed as the width between the fingers in meters
    - the speed in meters/second
    - the force in Newton
    """

    def __init__(self, gripper_specs: ParallelPositionGripperSpecs) -> None:
        self.gripper_specs = gripper_specs

    @property
    @abstractmethod
    def speed(self) -> float:
        """speed with which the fingers will move in m/s"""

    @speed.setter
    @abstractmethod
    def speed(self, new_speed: float):
        """sets the moving speed [m/s] synchronously."""

    @property
    @abstractmethod
    def max_grasp_force(self) -> float:
        """max force the fingers will apply in Newton"""

    @max_grasp_force.setter
    @abstractmethod
    def max_grasp_force(self) -> float:
        """sets the max grasping force [N] synchronously."""

    @abstractmethod
    def get_current_width(self) -> float:
        """the current opening of the fingers in meters"""

    @abstractmethod
    def move(self, width: float, speed: Optional[float] = None, force: Optional[float] = None):
        """
        synchronously move the fingers to the desired width between the fingers[m].
        Optionally provide a speed and/or force, that will be used from then on for all move commands."""

    def open(self):
        self.move(self.gripper_specs.max_width)

    def close(self):
        self.move(0.0)

    def is_an_object_grasped(self) -> bool:
        """
        Some grippers have heuristics to check if an object is grasped, usually by looking at motor currents.
        This function returns this heuristic, if it exists.
        """
        raise NotImplementedError
