import time
from typing import Optional

from airo_robots.awaitable_action import AwaitableAction
from airo_robots.grippers.parallel_position_gripper import ParallelPositionGripper, ParallelPositionGripperSpecs
from airo_robots.hardware_interaction_utils import AsyncExecutor


class DummyParallelPositionGripper(ParallelPositionGripper):
    """'Idealised' implementation of a parallel position gripper for testing purposes."""

    def __init__(self, gripper_specs: ParallelPositionGripperSpecs) -> None:
        super().__init__(gripper_specs)
        self.gripper_pos = 0
        self.gripper_speed = 0
        self.gripper_force = 0

        self.async_executor = AsyncExecutor()

    def move(self, width: float, speed: Optional[float] = None, force: Optional[float] = None) -> AwaitableAction:
        if speed:
            self.gripper_speed = speed
        if force:
            self.gripper_force = force
        # simulate time to reach HW
        # will make test slower though..

        def simulate_gripper_move():
            time.sleep(1)
            self.gripper_pos = width

        self.async_executor(simulate_gripper_move)
        return AwaitableAction(lambda: self.gripper_pos == width)

    @property
    def speed(self) -> float:
        return self.gripper_speed

    @speed.setter
    def speed(self, value: float):
        self.gripper_speed = value

    @property
    def max_grasp_force(self) -> float:
        return self.gripper_force

    @max_grasp_force.setter
    def max_grasp_force(self, value: float) -> float:
        self.gripper_force = value

    def get_current_width(self) -> float:
        return self.gripper_pos


def test_move_awaitable():
    gripper = DummyParallelPositionGripper(None)
    target_pos = 0.01
    res = gripper.move(target_pos)
    assert isinstance(res, AwaitableAction)
    assert res.is_done() is False
    res.wait()
    assert gripper.get_current_width() == target_pos


def test_instantiation():
    # This tests that the class can be instantiated and hence that the abstract base class can be inherited from as expected.
    DummyParallelPositionGripper(None)


def test_properties():
    gripper = DummyParallelPositionGripper(None)
    gripper.speed = 0.1
    assert gripper.speed == 0.1
    gripper.max_grasp_force = 0.2
    assert gripper.max_grasp_force == 0.2
