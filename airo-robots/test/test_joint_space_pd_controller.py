import numpy as np
import pytest
from airo_robots.manipulators.hardware.ur_rtde_torque import JointSpacePDController

DOF = 6
CONTROL_PERIOD = 1.0 / 500.0


def make_controller() -> JointSpacePDController:
    return JointSpacePDController(
        kp=np.full(DOF, 100.0),
        kd=np.full(DOF, 10.0),
        joint_torque_limits=np.full(DOF, 50.0),
        control_period=CONTROL_PERIOD,
    )


def test_zero_error_gives_zero_torque() -> None:
    controller = make_controller()
    joint_positions = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
    controller.reset(joint_positions)
    torques = controller.compute_torques(joint_positions, joint_positions, np.zeros(DOF))
    assert np.allclose(torques, 0.0)


def test_torque_has_sign_of_position_error() -> None:
    controller = make_controller()
    controller.reset(np.zeros(DOF))
    target = np.array([0.1, -0.1, 0.1, -0.1, 0.1, -0.1])
    # Hold the measured state constant so the reference trajectory moves towards the target.
    for _ in range(10):
        torques = controller.compute_torques(target, np.zeros(DOF), np.zeros(DOF))
    assert np.all(np.sign(torques) == np.sign(target))


def test_torques_are_clipped_to_limits() -> None:
    controller = make_controller()
    controller.reset(np.zeros(DOF))
    target = np.full(DOF, 10.0)  # far away target: unclipped P-term would be ~1000 Nm
    for _ in range(2000):
        torques = controller.compute_torques(target, np.zeros(DOF), np.zeros(DOF))
        assert np.all(np.abs(torques) <= controller.joint_torque_limits + 1e-9)
    # After convergence of the reference trajectory, the output should sit at the limits.
    assert np.allclose(torques, controller.joint_torque_limits)


def test_reference_trajectory_converges_to_target() -> None:
    controller = make_controller()
    controller.reset(np.zeros(DOF))
    target = np.array([0.2, -0.2, 0.2, -0.2, 0.2, -0.2])
    # Simulate one second of perfect tracking: the measured state follows the reference exactly,
    # so the commanded torque reflects only the reference error, which should vanish.
    joint_positions = np.zeros(DOF)
    joint_velocities = np.zeros(DOF)
    for _ in range(500):
        controller.compute_torques(target, joint_positions, joint_velocities)
        joint_positions = controller._reference_position.copy()
        joint_velocities = controller._reference_velocity.copy()
    assert np.allclose(joint_positions, target, atol=1e-3)
    torques = controller.compute_torques(target, joint_positions, joint_velocities)
    assert np.allclose(torques, 0.0, atol=0.5)


def test_reset_reinitializes_reference() -> None:
    controller = make_controller()
    controller.reset(np.zeros(DOF))
    for _ in range(100):
        controller.compute_torques(np.full(DOF, 1.0), np.zeros(DOF), np.zeros(DOF))
    new_joint_positions = np.full(DOF, -0.5)
    controller.reset(new_joint_positions)
    torques = controller.compute_torques(new_joint_positions, new_joint_positions, np.zeros(DOF))
    assert np.allclose(torques, 0.0)


def test_mismatched_gain_shapes_raise() -> None:
    with pytest.raises(ValueError):
        JointSpacePDController(
            kp=np.full(5, 100.0),
            kd=np.full(DOF, 10.0),
            joint_torque_limits=np.full(DOF, 50.0),
            control_period=CONTROL_PERIOD,
        )


def test_nonpositive_torque_limits_raise() -> None:
    with pytest.raises(ValueError):
        JointSpacePDController(
            kp=np.full(DOF, 100.0),
            kd=np.full(DOF, 10.0),
            joint_torque_limits=np.zeros(DOF),
            control_period=CONTROL_PERIOD,
        )


def test_negative_gains_raise() -> None:
    with pytest.raises(ValueError):
        JointSpacePDController(
            kp=np.full(DOF, -1.0),
            kd=np.full(DOF, 10.0),
            joint_torque_limits=np.full(DOF, 50.0),
            control_period=CONTROL_PERIOD,
        )


def test_nonpositive_control_period_raises() -> None:
    with pytest.raises(ValueError):
        JointSpacePDController(
            kp=np.full(DOF, 100.0),
            kd=np.full(DOF, 10.0),
            joint_torque_limits=np.full(DOF, 50.0),
            control_period=0.0,
        )
