"""code for manual testing of gripper base class implementations.
"""
from airo_robots.grippers.parallel_position_gripper import ParallelPositionGripper, ParallelPositionGripperSpecs


def manual_test_gripper(gripper: ParallelPositionGripper, specs: ParallelPositionGripperSpecs) -> None:
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
    input("close with high force, you can put an object between the fingers to test the force and the grasp detection")
    gripper.max_grasp_force = 200
    gripper.speed = 0.02
    gripper.move(0.02)
    print(f"{gripper.is_an_object_grasped()=}")

    # print("testing async wrapper:")
    # async_gripper = AsyncParallelGripper(gripper)
    # print("gripper will now open")
    # future = async_gripper.open()
    # print(future.done())
    # print("now a 'blocking' print of the move return value will happen, this is similar to calling thread.join()")
    # print(future.result(timeout=20))
    # print("gripper should have been opened before this line has been printed")
    # print(f"{future.done()=}")
    # print("done should now be true")

    # input(
    #     "we will now show a race condition if you use the async and sync interfaces interleaved: \n  press any key to start"
    # )
    # future = async_gripper.close()
    # # before this has finished, call open on the sync interface
    # gripper.open()
    # print(future.result(timeout=10))
