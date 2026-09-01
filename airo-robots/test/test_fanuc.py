from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, List, Optional, Tuple

import numpy as np
import pytest
from airo_robots.exceptions import InvalidTrajectoryException, RobotConfigurationException
from airo_robots.grippers.hardware import robotiq_2f85_fanuc
from airo_robots.grippers.hardware.robotiq_2f85_fanuc import (
    CLOSE_FORCES,
    FORCE_HARD,
    FORCE_LIGHT,
    FORCE_MEDIUM,
    OPEN_FULL,
    OPEN_MID,
    OPEN_NARROW,
    Robotiq2F85Fanuc,
)
from airo_robots.manipulators.hardware import fanuc
from airo_spatial_algebra import SE3Container
from airo_typing import JointPathContainer, SingleArmTrajectory

# The fake driver below mirrors the parts of the airo-fanuc API that airo_robots.manipulators.
# hardware.fanuc uses: radians everywhere except the Cartesian poses (millimeters and degrees),
# getters that return None rather than raising when there is no data, and motion commands that return
# a handle resolving to a non-raising MotionResult.


class FakeMotionResult:
    DONE = "done"
    SETTLE_TIMEOUT = "settle_timeout"
    STOPPED = "stopped"
    FAULTED = "faulted"


class FakeTrajectoryValidationError(Exception):
    pass


class FakeRejectedStartMismatch(Exception):
    pass


class FakeRobotFaultedError(Exception):
    pass


class FakeMotionHandle:
    def __init__(self, result: str) -> None:
        self._result = result

    def result(self) -> str:
        return self._result

    def wait(self, timeout: Optional[float] = None) -> str:
        return self._result


@dataclass
class FakeGripperProtocol:
    name: str = "Robotiq 2F-85 via GRIPDISP"
    open_modifiers: Tuple[int, ...] = (0, 1, 2)
    close_modifiers: Tuple[int, ...] = (0, 1, 2)


@dataclass
class FakeProfile:
    ndof: int = 6
    velocity_limits: np.ndarray = field(default_factory=lambda: np.radians(np.full(6, 120.0)))
    position_limits_lower: np.ndarray = field(default_factory=lambda: np.radians(np.full(6, -180.0)))
    position_limits_upper: np.ndarray = field(default_factory=lambda: np.radians(np.full(6, 180.0)))

    def describe(self) -> str:
        return "fake FANUC"


class FakeGripperWorker:
    def __init__(self, protocol: FakeGripperProtocol) -> None:
        self.protocol = protocol
        self.calls: List[Tuple[str, int]] = []
        self.last_result: dict = {"success": True, "message": "ok"}

    def open_gripper(self, open_state: Optional[int] = None) -> None:
        self.calls.append(("open", 0 if open_state is None else open_state))

    def close_gripper(self, close_force: Optional[int] = None) -> None:
        self.calls.append(("close", 1 if close_force is None else close_force))


class FakeFanucDriver:
    def __init__(self, ip: str, policy: Any) -> None:
        self.ip = ip
        self.policy = policy
        self.gripper = FakeGripperWorker(policy.gripper_protocol) if policy.enable_gripper else None
        self.q_measured = np.zeros(6)
        self.q_commanded = np.zeros(6)
        self.tcp_pose: Optional[np.ndarray] = np.array([500.0, 0.0, 300.0, 0.0, 0.0, 90.0])
        self.flange_pose: Optional[np.ndarray] = np.array([325.0, 0.0, 300.0, 0.0, 0.0, 90.0])
        self.wrench: Optional[np.ndarray] = None
        self.steady = True
        self.motion_result = FakeMotionResult.DONE
        self.raise_on_submit: Optional[Exception] = None
        self.calls: List[Tuple[Any, ...]] = []
        self.closed = False

    # -- state ---------------------------------------------------------
    def get_joint_configuration(self) -> Optional[np.ndarray]:
        return self.q_measured

    def get_state(self) -> dict:
        return {
            "q_meas": self.q_measured.tolist(),
            "q_cmd": self.q_commanded.tolist(),
            "rx_mono_ns": 1,
            "lifecycle_state": "streaming",
            "fault_reason": "none",
            "operator_hint": None,
        }

    def get_tcp_pose(self) -> Optional[np.ndarray]:
        return self.tcp_pose

    def get_flange_pose(self) -> Optional[np.ndarray]:
        return self.flange_pose

    def get_wrench(self) -> Optional[np.ndarray]:
        return self.wrench

    # -- motion --------------------------------------------------------
    def move_j(self, q: np.ndarray, joint_speed: Optional[float] = None) -> FakeMotionHandle:
        if self.raise_on_submit is not None:
            raise self.raise_on_submit
        self.calls.append(("move_j", np.asarray(q), joint_speed))
        self.q_commanded = np.asarray(q, dtype=float)
        self.q_measured = np.asarray(q, dtype=float)
        return FakeMotionHandle(self.motion_result)

    def move_trajectory(self, times: Any, q: Any, qd: Any = None) -> FakeMotionHandle:
        if self.raise_on_submit is not None:
            raise self.raise_on_submit
        self.calls.append(
            ("move_trajectory", np.asarray(times), np.asarray(q), None if qd is None else np.asarray(qd))
        )
        return FakeMotionHandle(self.motion_result)

    def servo_j(self, q: np.ndarray, duration: float) -> FakeMotionHandle:
        self.calls.append(("servo_j", np.asarray(q), duration))
        self.steady = False
        return FakeMotionHandle(self.motion_result)

    def stop_j(self) -> None:
        self.calls.append(("stop_j",))

    def hold(self) -> None:
        self.calls.append(("hold",))
        self.steady = True

    def is_steady(self) -> bool:
        return self.steady

    def wait_until_steady(self, timeout: float = 5.0) -> bool:
        return self.steady

    def close(self) -> None:
        self.closed = True


def _fake_policy(enable_gripper: bool = False) -> Any:
    return SimpleNamespace(
        config=SimpleNamespace(profile=FakeProfile()),
        settle=SimpleNamespace(timeout_s=2.0),
        enable_gripper=enable_gripper,
        gripper_protocol=FakeGripperProtocol(),
    )


def _fake_module() -> Any:
    return SimpleNamespace(
        FanucDriver=FakeFanucDriver,
        MotionResult=FakeMotionResult,
        TrajectoryValidationError=FakeTrajectoryValidationError,
        RejectedStartMismatch=FakeRejectedStartMismatch,
        RobotFaultedError=FakeRobotFaultedError,
    )


def _make_robot(monkeypatch: pytest.MonkeyPatch, enable_gripper: bool = False) -> fanuc.Fanuc:
    policy = _fake_policy(enable_gripper)
    monkeypatch.setattr(fanuc, "_import_airo_fanuc", _fake_module)
    monkeypatch.setattr(
        robotiq_2f85_fanuc,
        "_import_airo_fanuc_gripper",
        lambda: SimpleNamespace(ROBOTIQ_2F85=policy.gripper_protocol),
    )
    return fanuc.Fanuc("192.168.1.100", policy)


@pytest.fixture
def robot(monkeypatch: pytest.MonkeyPatch) -> fanuc.Fanuc:
    return _make_robot(monkeypatch)


@pytest.fixture
def robot_with_gripper(monkeypatch: pytest.MonkeyPatch) -> fanuc.Fanuc:
    return _make_robot(monkeypatch, enable_gripper=True)


@pytest.fixture
def gripper(robot_with_gripper: fanuc.Fanuc) -> Robotiq2F85Fanuc:
    """The gripper the driver brought up, narrowed from ``Optional`` once for every gripper test."""
    assert isinstance(robot_with_gripper.gripper, Robotiq2F85Fanuc)
    return robot_with_gripper.gripper


def test_specs_and_state_use_airo_units(robot: fanuc.Fanuc) -> None:
    assert robot.manipulator_specs.dof == 6
    assert robot.manipulator_specs.max_joint_speeds == pytest.approx(np.radians(np.full(6, 120.0)))

    robot.driver.q_measured = np.radians(np.asarray([0.0, 30.0, -90.0, 0.0, 45.0, 180.0]))
    assert robot.get_joint_configuration() == pytest.approx(robot.driver.q_measured)

    # The FANUC controller reports millimeters and degrees, with W/P/R as fixed-axis XYZ angles.
    expected_pose = SE3Container.from_euler_angles_and_translation(
        np.radians(np.asarray([0.0, 0.0, 90.0])), np.asarray([0.5, 0.0, 0.3])
    ).homogeneous_matrix
    assert robot.get_tcp_pose() == pytest.approx(expected_pose)
    assert robot.get_flange_pose()[:3, 3] == pytest.approx([0.325, 0.0, 0.3])


def test_state_getters_raise_rather_than_returning_a_substitute(robot: fanuc.Fanuc) -> None:
    robot.driver.tcp_pose = None
    with pytest.raises(RuntimeError, match="TCP pose"):
        robot.get_tcp_pose()

    # No force block on the wire (a v3 controller) is a raise, not a zero wrench.
    with pytest.raises(RuntimeError, match="force telemetry"):
        robot.get_wrench()

    robot.driver.wrench = np.arange(6, dtype=float)
    assert robot.get_wrench() == pytest.approx(np.arange(6))


def test_move_to_joint_configuration_plans_with_move_j(robot: fanuc.Fanuc) -> None:
    target = np.radians(np.asarray([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]))
    action = robot.move_to_joint_configuration(target, joint_speed=0.5)

    method, joints, speed = robot.driver.calls[-1]
    assert method == "move_j"
    assert joints == pytest.approx(target)
    assert speed == 0.5
    assert action.wait().name == "SUCCEEDED"


def test_move_to_joint_configuration_brakes_a_moving_arm_first(robot: fanuc.Fanuc) -> None:
    # The driver refuses to plan a point-to-point move from a moving anchor, so a servo stream (which
    # never ends by itself) has to be ended before the next move can be planned.
    robot.servo_to_joint_configuration(np.zeros(6), 0.02)
    assert not robot.driver.is_steady()

    robot.move_to_joint_configuration(np.zeros(6))
    assert [call[0] for call in robot.driver.calls] == ["servo_j", "hold", "move_j"]


def test_out_of_limit_configurations_are_refused(robot: fanuc.Fanuc) -> None:
    assert robot._is_joint_configuration_reachable(np.zeros(6))
    assert not robot._is_joint_configuration_reachable(np.radians(np.full(6, 200.0)))
    assert not robot._is_joint_configuration_reachable(np.zeros(7))

    with pytest.raises(RobotConfigurationException):
        robot.move_to_joint_configuration(np.radians(np.full(6, 200.0)))
    assert robot.driver.calls == []


def test_servo_passes_the_setpoint_and_duration_through(robot: fanuc.Fanuc) -> None:
    target = np.radians(np.ones(6))
    robot.servo_to_joint_configuration(target, 0.02)

    method, joints, duration = robot.driver.calls[-1]
    assert method == "servo_j"
    assert joints == pytest.approx(target)
    assert duration == 0.02

    with pytest.raises(ValueError):
        robot.servo_to_joint_configuration(np.zeros(5), 0.02)
    with pytest.raises(ValueError):
        robot.servo_to_joint_configuration(target, 0.0)


def test_cartesian_and_kinematics_methods_point_at_a_numerical_solver(robot: fanuc.Fanuc) -> None:
    pose = np.eye(4)
    for call in (
        lambda: robot.move_to_tcp_pose(pose),
        lambda: robot.move_linear_to_tcp_pose(pose),
        lambda: robot.servo_to_tcp_pose(pose, 0.02),
        lambda: robot.inverse_kinematics(pose),
        lambda: robot.forward_kinematics(np.zeros(6)),
        lambda: robot.is_tcp_pose_reachable(pose),
    ):
        with pytest.raises(NotImplementedError, match="airo-models"):
            call()

    with pytest.raises(NotImplementedError, match="FanucReceiveInterface"):
        robot.start_freedrive()
    with pytest.raises(NotImplementedError, match="FanucReceiveInterface"):
        robot.stop_freedrive()


def _trajectory(with_velocities: bool = True) -> SingleArmTrajectory:
    times = np.linspace(0.0, 2.0, 21)
    positions = np.zeros((times.size, 6))
    positions[:, 5] = np.radians(10.0) * (1 - np.cos(np.pi * times / 2.0)) / 2
    velocities = np.zeros_like(positions)
    velocities[:, 5] = np.radians(10.0) * (np.pi / 2.0) * np.sin(np.pi * times / 2.0) / 2
    path = JointPathContainer(positions=positions, velocities=velocities if with_velocities else None)
    return SingleArmTrajectory(times, path)


def test_execute_trajectory_submits_the_whole_timeline_at_once(robot: fanuc.Fanuc) -> None:
    trajectory = _trajectory()
    robot.execute_trajectory(trajectory)

    # One submission, not a servo command per interpolation step.
    assert [call[0] for call in robot.driver.calls] == ["move_trajectory"]
    _, times_ns, positions, velocities = robot.driver.calls[-1]
    assert times_ns.dtype == np.int64
    assert times_ns[0] == 0 and times_ns[-1] == 2_000_000_000
    assert positions == pytest.approx(trajectory.path.positions)
    assert velocities == pytest.approx(trajectory.path.velocities)


def test_execute_trajectory_leaves_missing_velocities_to_the_driver(robot: fanuc.Fanuc) -> None:
    # airo-mono's velocities are optional and the driver derives the playback tangents itself, so a
    # path without them is handed over as-is rather than differentiated here: one derivation, and the
    # one that is guaranteed to land inside the driver's first-knot capture envelope.
    robot.execute_trajectory(_trajectory(with_velocities=False))

    _, _, _, velocities = robot.driver.calls[-1]
    assert velocities is None


def test_execute_trajectory_reports_a_refusal_and_a_failed_motion(robot: fanuc.Fanuc) -> None:
    robot.driver.raise_on_submit = FakeRejectedStartMismatch("first knot outside the capture window")
    with pytest.raises(InvalidTrajectoryException, match="capture window"):
        robot.execute_trajectory(_trajectory())

    robot.driver.raise_on_submit = None
    robot.driver.motion_result = FakeMotionResult.FAULTED
    with pytest.raises(RuntimeError, match="faulted"):
        robot.execute_trajectory(_trajectory())

    # A trajectory that was commanded in full but did not confirm arrival is a warning, not a failure.
    robot.driver.motion_result = FakeMotionResult.SETTLE_TIMEOUT
    robot.execute_trajectory(_trajectory())


def test_a_faulted_robot_is_reported_with_its_operator_instruction(robot: fanuc.Fanuc) -> None:
    robot.driver.raise_on_submit = FakeRobotFaultedError("e_stop")
    with pytest.raises(RuntimeError, match="not commandable"):
        robot.move_to_joint_configuration(np.zeros(6))


def test_the_driver_gripper_is_wrapped_automatically(robot_with_gripper: fanuc.Fanuc) -> None:
    assert isinstance(robot_with_gripper.gripper, Robotiq2F85Fanuc)


def test_gripper_snaps_widths_to_the_reachable_buckets(
    robot_with_gripper: fanuc.Fanuc, gripper: Robotiq2F85Fanuc
) -> None:
    worker = robot_with_gripper.driver.gripper

    gripper.open().wait()
    assert worker.calls[-1] == ("open", OPEN_FULL)

    gripper.move(0.058).wait()  # closest to the ~60 mm bucket
    assert worker.calls[-1] == ("open", OPEN_MID)
    assert gripper.last_commanded_width == pytest.approx(0.060)

    gripper.move(0.030).wait()  # closest to the ~35 mm bucket
    assert worker.calls[-1] == ("open", OPEN_NARROW)

    gripper.close().wait()
    assert worker.calls[-1] == ("close", FORCE_MEDIUM)
    assert gripper.last_result == {"success": True, "message": "ok"}


def test_gripper_force_is_rounded_to_a_force_class(robot_with_gripper: fanuc.Fanuc, gripper: Robotiq2F85Fanuc) -> None:
    worker = robot_with_gripper.driver.gripper

    gripper.move(0.0, force=30.0)
    assert worker.calls[-1] == ("close", FORCE_LIGHT)
    assert gripper.max_grasp_force == CLOSE_FORCES[FORCE_LIGHT]

    gripper.max_grasp_force = 1000.0  # clipped to the gripper's maximum, then to the nearest class
    assert gripper.close_force_class == FORCE_HARD
    gripper.close()
    assert worker.calls[-1] == ("close", FORCE_HARD)

    with pytest.raises(ValueError):
        gripper.close_force_class = 7


def test_gripper_refuses_what_a_fanuc_cannot_do(gripper: Robotiq2F85Fanuc) -> None:
    def set_speed() -> None:
        gripper.speed = 0.1

    for call in (
        lambda: gripper.move(0.05, speed=0.1),
        gripper.get_current_width,
        gripper.is_an_object_grasped,
        lambda: gripper.speed,
        set_speed,
    ):
        with pytest.raises(NotImplementedError):
            call()


def test_gripper_requires_a_driver_that_brought_one_up(robot: fanuc.Fanuc) -> None:
    assert robot.gripper is None
    with pytest.raises(ValueError, match="enable_gripper"):
        Robotiq2F85Fanuc(robot.driver)


def test_close_is_idempotent(robot: fanuc.Fanuc) -> None:
    robot.close()
    assert robot.driver.closed
    robot.close()
