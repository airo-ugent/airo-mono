from airo_robots.manipulators.position_manipulator import PositionManipulatorDecorator


def test_base_decorator_implements_all_abstract_methods():
    # if not all abstract methods have been implemented in the base decorator
    # this construction will raise an error.
    PositionManipulatorDecorator(None)
    assert True
