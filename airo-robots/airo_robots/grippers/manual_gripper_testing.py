"""code for manual testing of gripper base class implementations.
"""
from airo_robots.grippers.base import ParallelPositionGripper, ParallelPositionGripperSpecs


def manual_test_gripper(gripper: ParallelPositionGripper, specs: ParallelPositionGripperSpecs):
    input("gripper will now open")
    gripper.move(specs.max_width)
    assert abs(gripper.get_current_width() - specs.max_width) < 0.003
    print(f"gripper width {gripper.get_current_width()} should be close to the requested {specs.max_width}")

    input("set speed and check that this happens synchronously")
    gripper.speed = 0.02
    print(f"{gripper.speed=}")
    assert abs(gripper.speed - 0.02) < 0.005

    input("gripper wil now move (sync) slowly to 2cm opening, and will print once the function has returned ")
    gripper.move(0.02)
    print("move completed, gripper should have reached 2cm already")
    assert abs(gripper.get_current_width() - 0.02) < 0.003

    input("reopen gripper fast")
    gripper.move(0.08, 0.15)
    input("close with low force, you can put an object between the fingers to test the force and the grasp detection")
    gripper.max_grasp_force = 200
    gripper.speed = 0.02
    gripper.move(0.02)
    print(f"{gripper.is_an_object_grasped()=}")
