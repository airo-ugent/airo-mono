from abc import abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
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
    Base class for a position-controlled, 2 finger parallel gripper.

    These grippers typically allow to set a speed and maximum applied force before moving,
    and attempt to move to specified positions under these constraints.

    all values are in metric units:
    - the position of the gripper is expressed as the width between the fingers in meters
    - the speed in meters/second
    - the force in Newton
    """

    def __init__(self, gripper_specs: ParallelPositionGripperSpecs) -> None:
        self._gripper_specs = gripper_specs

    @property
    def gripper_specs(self) -> ParallelPositionGripperSpecs:
        return self._gripper_specs

    @gripper_specs.setter
    def gripper_specs(self, spec: ParallelPositionGripperSpecs):
        self._gripper_specs = spec

    @property
    @abstractmethod
    def speed(self) -> float:
        """speed with which the fingers will move in m/s"""

    @speed.setter
    @abstractmethod
    def speed(self, new_speed: float) -> None:
        """sets the moving speed [m/s] synchronously."""

    @property
    @abstractmethod
    def max_grasp_force(self) -> float:
        """max force the fingers will apply in Newton"""

    @max_grasp_force.setter
    @abstractmethod
    def max_grasp_force(self, new_force: float):
        """sets the max grasping force [N] synchronously."""

    @abstractmethod
    def get_current_width(self) -> float:
        """the current opening of the fingers in meters"""

    @abstractmethod
    def move(self, width: float, speed: Optional[float] = None, force: Optional[float] = None) -> None:
        """
        synchronously move the fingers to the desired width between the fingers[m].
        Optionally provide a speed and/or force, that will be used from then on for all move commands."""

    def open(self) -> None:
        self.move(self.gripper_specs.max_width)

    def close(self) -> None:
        self.move(0.0)

    def is_an_object_grasped(self) -> bool:
        """
        Some grippers have heuristics to check if an object is grasped, usually by looking at motor currents.
        This function returns this heuristic, if it exists.
        """
        raise NotImplementedError


class ParallelGripperDecorator(ParallelPositionGripper):
    """Decorator base class for the Parallel Gripper class"""

    def __init__(self, gripper: ParallelPositionGripper) -> None:
        self.wrapped_gripper = gripper

    @property
    def _gripper_specs(self) -> ParallelPositionGripperSpecs:
        return self.wrapped_gripper._gripper_specs

    @property
    def speed(self) -> float:
        return self.wrapped_gripper.speed

    @speed.setter
    def speed(self, new_speed: float):
        self.wrapped_gripper.speed = new_speed

    @property
    def max_grasp_force(self) -> float:
        return self.wrapped_gripper.max_grasp_force

    @max_grasp_force.setter
    def max_grasp_force(self, force: float):
        self.wrapped_gripper.max_grasp_force = force

    def get_current_width(self) -> float:
        return self.wrapped_gripper.get_current_width()

    def move(self, width: float, speed: Optional[float] = None, force: Optional[float] = None):
        return self.wrapped_gripper.move(width, speed, force)

    def is_an_object_grasped(self) -> bool:
        return self.wrapped_gripper.is_an_object_grasped()


class AsyncParallelGripper(ParallelGripperDecorator):
    # TODO: refactor this async out as a base class that can be used across all HW in airo-robots.
    def __init__(self, gripper: ParallelPositionGripper) -> None:
        super().__init__(gripper)
        self._thread_pool = ThreadPoolExecutor(max_workers=1)

    def _threadpool_execution(self, func, *args, **kwargs) -> Future:
        """helper function to execute a function call asynchronously in the threadpool.

        returns a future which can be waited for (cf. join on a thread) or can be polled to see if the function has finished
        see https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Future
        """
        future = self._thread_pool.submit(func, *args, **kwargs)
        return future

    def move(self, width: float, speed: Optional[float] = None, force: Optional[float] = None) -> Future:
        """synchronously move the fingers to the desired width between the fingers[m].
        Optionally provide a speed and/or force, that will be used from then on for all move commands.
        Returns a Future object"""

        return self._threadpool_execution(self.wrapped_gripper.move, width, speed, force)
