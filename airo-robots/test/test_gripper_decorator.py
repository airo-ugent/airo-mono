from airo_robots.grippers.parallel_position_gripper import ParallelGripperWrapper


def test_base_decorator_implements_all_abstract_methods():
    # if not all abstract methods have been implemented in the base decorator
    # this construction will raise an error.
    ParallelGripperWrapper(None)
    assert True
