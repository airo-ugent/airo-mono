"""code for manual testing of gripper base class implementations.
"""
from airo_robots.grippers.parallel_position_gripper import ParallelPositionGripper, ParallelPositionGripperSpecs


def manually_test_gripper_implementation(
    gripper: ParallelPositionGripper, specs: ParallelPositionGripperSpecs
) -> None:
    input("gripper will now open")
    gripper.move(specs.max_width).wait()
    assert abs(gripper.get_current_width() - specs.max_width) < 0.003
    print(f"gripper width {gripper.get_current_width()} should be close to the requested {specs.max_width}")

    input("set speed and check that this happens synchronously")
    gripper.speed = 0.02
    print(f"{gripper.speed=}")
    assert abs(gripper.speed - 0.02) < 0.005

    input("gripper wil now move slowly to 2cm opening ")
    res = gripper.move(0.02)
    print("move started and returned awaitable")
    res.wait()
    print("move completed, gripper should have reached 2cm")
    assert abs(gripper.get_current_width() - 0.02) < 0.003

    input("reopen gripper fast")
    gripper.move(0.08, 0.15)
    input("close with high force, you can put an object between the fingers to test the force and the grasp detection")
    gripper.max_grasp_force = 200
    gripper.speed = 0.02
    gripper.move(0.02).wait()
    print(f"{gripper.is_an_object_grasped()=}")
