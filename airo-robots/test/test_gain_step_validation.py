import numpy as np
import pytest
from airo_robots.manipulators.hardware.ur_rtde_torque import URrtdeTorque

DOF = 6
DEFAULT_GAIN = np.full(DOF, 10.0)


def robot() -> URrtdeTorque:
    # _assert_gain_step_is_small only reads MAX_GAIN_STEP_FACTOR, so it can be tested without connecting to a robot.
    return object.__new__(URrtdeTorque)


def test_small_gain_step_is_accepted() -> None:
    current = DEFAULT_GAIN.copy()
    new = current + 0.4 * DEFAULT_GAIN  # within MAX_GAIN_STEP_FACTOR (0.5) of the default gain
    robot()._assert_gain_step_is_small("kp", current, new, DEFAULT_GAIN)


def test_large_gain_step_is_rejected() -> None:
    current = DEFAULT_GAIN.copy()
    new = current + 0.6 * DEFAULT_GAIN  # exceeds MAX_GAIN_STEP_FACTOR (0.5) of the default gain
    with pytest.raises(ValueError):
        robot()._assert_gain_step_is_small("kp", current, new, DEFAULT_GAIN)


def test_large_gain_decrease_is_rejected() -> None:
    current = DEFAULT_GAIN.copy()
    new = current - 0.6 * DEFAULT_GAIN
    with pytest.raises(ValueError):
        robot()._assert_gain_step_is_small("kd", current, new, DEFAULT_GAIN)


def test_large_step_from_zero_gain_is_rejected() -> None:
    current = np.zeros(DOF)
    new = DEFAULT_GAIN.copy()  # a full jump from zero to the default gain exceeds the allowed step
    with pytest.raises(ValueError):
        robot()._assert_gain_step_is_small("kp", current, new, DEFAULT_GAIN)


def test_only_exceeding_joints_are_reported() -> None:
    current = DEFAULT_GAIN.copy()
    new = current.copy()
    new[2] += 0.6 * DEFAULT_GAIN[2]
    with pytest.raises(ValueError, match=r"joint\(s\) \[2\]"):
        robot()._assert_gain_step_is_small("kp", current, new, DEFAULT_GAIN)
