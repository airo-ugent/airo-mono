"""Integration tests for the FANUC implementation against airo-fanuc's in-process fake controller.

These run the real driver — its validation, its capture gate, its real-time core and its register
gripper handshake — against a fake FANUC controller instead of a real one, so they cover what the
mocked tests in ``test_fanuc.py`` cannot. They are skipped unless the optional driver is installed
(``pip install "airo-robots[fanuc]"``).

They need no hardware, but they do run a 125 Hz real-time loop and a few seconds of simulated
motion, so they are slower than the rest of the suite.
"""

import tempfile
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import pytest

pytest.importorskip("airo_fanuc", reason="the FANUC implementation needs the optional airo-fanuc driver")

from airo_fanuc import DriverConfig, DriverPolicy  # noqa: E402
from airo_fanuc.testing import FakeCRXConfig, FakeCRXController  # noqa: E402
from airo_robots.exceptions import InvalidTrajectoryException  # noqa: E402
from airo_robots.grippers.hardware.robotiq_2f85_fanuc import (  # noqa: E402
    FORCE_HARD,
    OPEN_MID,
    OPEN_WIDTHS,
    Robotiq2F85Fanuc,
)
from airo_robots.manipulators.hardware.fanuc import Fanuc, create_crx10ial_profile  # noqa: E402
from airo_typing import JointPathContainer, SingleArmTrajectory  # noqa: E402

JOINT = 5  # J6, the wrist roll: the cheapest joint to be wrong about, on a fake or on a real arm.


@pytest.fixture(scope="module")
def robot() -> Iterator[Fanuc]:
    controller = FakeCRXController(FakeCRXConfig(available_version=3, itp_s=0.008))
    controller.start()
    controller.start_realtime(speed=1.0)
    policy = DriverPolicy(
        config=DriverConfig(
            profile=create_crx10ial_profile(),
            sm_port=controller.sm_port,
            rmi_port=controller.rmi_port,
            sm_version=3,
        ),
        enable_gripper=True,
        lock_path=str(Path(tempfile.gettempdir()) / "airo-fanuc-airo-robots-test.lock"),
    )
    try:
        with Fanuc("127.0.0.1", policy) as fanuc_robot:
            yield fanuc_robot
    finally:
        controller.stop_realtime()
        controller.close()


def _raised_cosine(
    start_configuration: np.ndarray, amplitude: float, duration: float, knots: int = 121
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A path that starts and ends at rest, so its first knot is inside the capture envelope."""
    times = np.linspace(0.0, duration, knots)
    omega = 2 * np.pi / duration
    positions = np.tile(start_configuration, (times.size, 1))
    velocities = np.zeros_like(positions)
    positions[:, JOINT] += amplitude * (1 - np.cos(omega * times)) / 2
    velocities[:, JOINT] = amplitude * omega * np.sin(omega * times) / 2
    return times, positions, velocities


def test_bringup_reports_a_streaming_robot(robot: Fanuc) -> None:
    state = robot.driver.get_state()
    assert state["lifecycle_state"] == "streaming"
    assert robot.is_steady()
    assert robot.get_joint_configuration().shape == (6,)
    assert robot.get_tcp_pose().shape == (4, 4)
    # The gripper worker the driver brought up is wrapped automatically.
    assert isinstance(robot.gripper, Robotiq2F85Fanuc)


def test_move_to_joint_configuration_arrives(robot: Fanuc) -> None:
    start_configuration = robot.get_commanded_joint_configuration()
    target = start_configuration.copy()
    target[JOINT] += np.radians(10.0)

    robot.move_to_joint_configuration(target, np.radians(15.0)).wait()
    assert robot.get_commanded_joint_configuration() == pytest.approx(target, abs=np.radians(0.5))

    robot.move_to_joint_configuration(start_configuration, np.radians(15.0)).wait()


def test_a_move_after_a_servo_stream_brakes_first(robot: Fanuc) -> None:
    start_configuration = robot.get_commanded_joint_configuration()
    target = start_configuration.copy()
    target[JOINT] += np.radians(2.0)

    robot.servo_to_joint_configuration(target, 0.05).wait()
    # A servo stream has no terminal condition, so the arm is not at rest afterwards and the driver
    # would refuse to plan a point-to-point move from it.
    assert not robot.is_steady()

    robot.move_to_joint_configuration(start_configuration, np.radians(15.0)).wait()
    assert robot.is_steady()


def test_execute_trajectory_runs_the_whole_path(robot: Fanuc) -> None:
    times, positions, velocities = _raised_cosine(robot.get_commanded_joint_configuration(), np.radians(10.0), 4.0)
    robot.execute_trajectory(SingleArmTrajectory(times, JointPathContainer(positions, velocities)))
    assert robot.last_motion_handle.result().name == "DONE"
    assert robot.get_commanded_joint_configuration() == pytest.approx(positions[-1], abs=np.radians(0.5))


def test_execute_trajectory_without_velocities_is_accepted_by_the_driver(robot: Fanuc) -> None:
    # The estimated first-knot velocity has to stay inside the driver's capture envelope, or the
    # submission is refused; this is the check that the estimate is good enough to be usable.
    times, positions, _ = _raised_cosine(robot.get_commanded_joint_configuration(), np.radians(10.0), 4.0)
    robot.execute_trajectory(SingleArmTrajectory(times, JointPathContainer(positions)))
    assert robot.last_motion_handle.result().name == "DONE"


def test_a_trajectory_the_driver_refuses_raises_an_airo_robots_exception(robot: Fanuc) -> None:
    # A plain sine demands its peak velocity at t=0, which the capture splice cannot deliver. The
    # driver's typed refusal has to reach the caller as an airo-robots exception.
    start_configuration = robot.get_commanded_joint_configuration()
    times = np.linspace(0.0, 4.0, 121)
    omega = 2 * np.pi / 4.0
    positions = np.tile(start_configuration, (times.size, 1))
    velocities = np.zeros_like(positions)
    positions[:, JOINT] += np.radians(20.0) * np.sin(omega * times)
    velocities[:, JOINT] = np.radians(20.0) * omega * np.cos(omega * times)

    with pytest.raises(InvalidTrajectoryException):
        robot.execute_trajectory(SingleArmTrajectory(times, JointPathContainer(positions, velocities)))


def test_a_preempted_trajectory_is_reported_and_leaves_the_arm_usable(robot: Fanuc) -> None:
    import threading

    start_configuration = robot.get_commanded_joint_configuration()
    times, positions, velocities = _raised_cosine(start_configuration, np.radians(20.0), 6.0)

    stop_timer = threading.Timer(1.5, robot.stop)
    stop_timer.start()
    try:
        with pytest.raises(RuntimeError, match="STOPPED"):
            robot.execute_trajectory(SingleArmTrajectory(times, JointPathContainer(positions, velocities)))
    finally:
        stop_timer.cancel()

    robot.hold()
    assert robot.wait_until_steady()
    robot.move_to_joint_configuration(start_configuration, np.radians(15.0)).wait()
    assert robot.get_commanded_joint_configuration() == pytest.approx(start_configuration, abs=np.radians(0.5))


def test_the_gripper_dispatcher_handshake_completes(robot: Fanuc) -> None:
    gripper = robot.gripper
    assert isinstance(gripper, Robotiq2F85Fanuc)

    gripper.open().wait()
    assert gripper.last_result is not None and gripper.last_result["success"]

    # An in-between width snaps to the nearest bucket the dispatcher implements.
    gripper.move(OPEN_WIDTHS[OPEN_MID] - 0.002).wait()
    assert gripper.last_commanded_width == pytest.approx(OPEN_WIDTHS[OPEN_MID])
    assert gripper.last_result is not None and gripper.last_result["success"]

    gripper.max_grasp_force = gripper.gripper_specs.max_force
    assert gripper.close_force_class == FORCE_HARD
    gripper.close().wait()
    assert gripper.last_result is not None and gripper.last_result["success"]

    gripper.open().wait()
    assert gripper.last_result is not None and gripper.last_result["success"]
