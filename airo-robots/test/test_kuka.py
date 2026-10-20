from typing import Any

import numpy as np
import pytest
from airo_robots.exceptions import RobotConfigurationException, RobotSafetyViolationException
from airo_robots.manipulators.hardware import kuka
from airo_robots.manipulators.hardware.kuka import (
    KR3_R540_DEFAULT_LOWER_LIMITS_DEG,
    KR3_R540_DEFAULT_UPPER_LIMITS_DEG,
    KR3_R540_MAX_JOINT_SPEEDS_DEG_S,
    KR3_R540_MAX_LINEAR_SPEED_M_S,
    Kuka,
    KukaKinematicGeometry,
    kuka_flange_forward_kinematics,
    kuka_flange_inverse_kinematics_all,
)
from airo_robots.manipulators.position_manipulator import ManipulatorSpecs
from airo_spatial_algebra import SE3Container


class FakeKukaOpcUaBridge:
    """In-memory fake of ``_KukaOpcUaBridge`` for unit testing ``Kuka`` without hardware."""

    def __init__(
        self,
        ip_address: str,
        port: int = 4840,
        username: str = "OpcUaSecurityAdmin",
        password: str = "kuka",
        program_name: str = "PY_SLOW",
        timeout: float = 10.0,
    ) -> None:
        self.ip_address = ip_address
        self.port = port
        self.username = username
        self.password = password
        self.program_name = program_name
        self.timeout = timeout

        self.model = "KR3R540 C4SR"
        self.lower_limits_deg = np.asarray(KR3_R540_DEFAULT_LOWER_LIMITS_DEG, dtype=float)
        self.upper_limits_deg = np.asarray(KR3_R540_DEFAULT_UPPER_LIMITS_DEG, dtype=float)
        self.base_pose = np.zeros(6, dtype=float)
        self.tool_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -90.0], dtype=float)
        self.geometry = KukaKinematicGeometry()

        self.joints_degrees = np.array([0.0, -90.0, 90.0, 80.0, 0.0, 0.0], dtype=float)
        self.command_id = 0
        self.completed = True
        self.uploaded = False
        self.closed = False
        self.sent_commands: list[tuple[np.ndarray, int, int]] = []

    def read_controller_metadata(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "lower_limits_deg": self.lower_limits_deg.copy(),
            "upper_limits_deg": self.upper_limits_deg.copy(),
            "base_pose": self.base_pose.copy(),
            "tool_pose": self.tool_pose.copy(),
            "geometry": self.geometry,
        }

    def get_joint_degrees(self) -> np.ndarray:
        return self.joints_degrees.copy()

    def get_kuka_tcp_pose(self) -> np.ndarray:
        base_tf = Kuka._convert_kuka_pose_to_homogeneous_pose(self.base_pose)
        tool_tf = Kuka._convert_kuka_pose_to_homogeneous_pose(self.tool_pose)
        flange_tf = kuka_flange_forward_kinematics(np.radians(self.joints_degrees), self.geometry)
        tcp_tf = np.linalg.inv(base_tf) @ flange_tf @ tool_tf
        return Kuka._convert_homogeneous_pose_to_kuka_pose(tcp_tf)

    def upload_krl_program(self) -> None:
        self.uploaded = True

    def send_joint_command(self, joints_degrees: np.ndarray, velocity_percent: int) -> int:
        self.command_id += 1
        self.joints_degrees = np.asarray(joints_degrees, dtype=float).copy()
        self.sent_commands.append((self.joints_degrees.copy(), int(velocity_percent), self.command_id))
        return self.command_id

    def is_command_completed(self, cmd_id: int) -> bool:
        return self.completed and cmd_id <= self.command_id

    def wait_until_krl_ready_and_idle(self, timeout: float = 120.0) -> None:
        self.waited_timeout = timeout

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def robot(monkeypatch: pytest.MonkeyPatch) -> Kuka:
    monkeypatch.setattr(kuka, "_create_opcua_bridge", FakeKukaOpcUaBridge)
    return Kuka("192.168.1.2")


def test_controller_metadata_and_state_use_airo_units(robot: Kuka) -> None:
    assert robot.dof == 6
    assert robot.model == "KR3R540 C4SR"
    assert robot.manipulator_specs.max_joint_speeds == pytest.approx(np.radians(KR3_R540_MAX_JOINT_SPEEDS_DEG_S))
    assert robot.manipulator_specs.max_linear_speed == KR3_R540_MAX_LINEAR_SPEED_M_S

    robot._bridge.joints_degrees = np.array([8.7726555, -90.0, 90.0, 80.0, 0.0, 0.0])
    assert robot.get_joint_configuration() == pytest.approx(np.radians(robot._bridge.joints_degrees))

    # Compare against the live KR3 R540 controller's $POS_ACT reading at this configuration:
    # X=350.8505 mm, Y=-54.1428 mm, Z=624.9980 mm, A=-98.7727 deg, B=10.0 deg, C=-90.0 deg.
    tcp_pose = robot.get_tcp_pose()
    assert tcp_pose[:3, 3] == pytest.approx([0.3508505, -0.0541428, 0.6249980], abs=1e-5)
    assert robot.forward_kinematics(robot.get_joint_configuration()) == pytest.approx(tcp_pose, abs=1e-6)


def test_pose_conversions_round_trip() -> None:
    kuka_pose = np.array([350.85, -54.14, 625.0, -98.77, 10.0, -90.0], dtype=float)
    homogeneous = Kuka._convert_kuka_pose_to_homogeneous_pose(kuka_pose)
    recovered = Kuka._convert_homogeneous_pose_to_kuka_pose(homogeneous)
    assert Kuka._convert_kuka_pose_to_homogeneous_pose(recovered) == pytest.approx(homogeneous, abs=1e-9)

    with pytest.raises(ValueError):
        Kuka._convert_kuka_pose_to_homogeneous_pose(np.zeros(5))


def test_analytical_kinematics_round_trip(robot: Kuka) -> None:
    rng = np.random.default_rng(42)
    for _ in range(25):
        q_rand = np.radians(
            [
                rng.uniform(-150.0, 150.0),
                rng.uniform(-140.0, 30.0),
                rng.uniform(-90.0, 130.0),
                rng.uniform(-150.0, 150.0),
                rng.uniform(-100.0, 100.0),
                rng.uniform(-300.0, 300.0),
            ]
        )
        tcp_pose = robot.forward_kinematics(q_rand)
        # Seed near the true configuration so branch + 2*pi turn selection picks q_rand.
        q_seed = q_rand + rng.uniform(-0.05, 0.05, size=6)
        q_sol = robot.inverse_kinematics(tcp_pose, q_seed)
        assert q_sol is not None
        assert robot._is_joint_configuration_reachable(q_sol)
        assert robot.forward_kinematics(q_sol) == pytest.approx(tcp_pose, abs=1e-6)
        assert q_sol == pytest.approx(q_rand, abs=1e-5)


def test_inverse_kinematics_rejects_unreachable_and_over_limit_poses(robot: Kuka) -> None:
    # Out-of-reach position (5 metres away).
    far_pose = np.eye(4)
    far_pose[:3, 3] = [5.0, 0.0, 0.5]
    assert robot.inverse_kinematics(far_pose) is None
    assert not robot.is_tcp_pose_reachable(far_pose)

    # Reachable position in flange space, but tighten joint limits so all branches violate A1.
    valid_pose = robot.get_tcp_pose()
    robot._joint_lower_limits[0] = np.radians(165.0)
    robot._joint_upper_limits[0] = np.radians(170.0)
    assert robot.inverse_kinematics(valid_pose) is None


def test_planned_moves_convert_units_and_complete(robot: Kuka) -> None:
    target_joints = np.radians([10.0, -85.0, 85.0, 40.0, 15.0, -20.0])
    max_joint_speed = min(robot.manipulator_specs.max_joint_speeds)
    action = robot.move_to_joint_configuration(target_joints, joint_speed=max_joint_speed * 0.25)
    sent_deg, vel_pct, cmd_id = robot._bridge.sent_commands[-1]
    assert sent_deg == pytest.approx(np.degrees(target_joints))
    assert vel_pct == 25
    assert cmd_id == 1
    assert action.is_action_done()

    target_pose = robot.forward_kinematics(np.radians([5.0, -80.0, 80.0, 30.0, 10.0, 0.0]))
    action_tcp = robot.move_to_tcp_pose(target_pose, joint_speed=max_joint_speed * 0.5)
    _, vel_pct_tcp, cmd_id_tcp = robot._bridge.sent_commands[-1]
    assert vel_pct_tcp == 50
    assert cmd_id_tcp == 2
    assert action_tcp.is_action_done()

    target_linear_pose = SE3Container.from_homogeneous_matrix(target_pose).homogeneous_matrix.copy()
    target_linear_pose[:3, 3] += np.array([0.02, -0.01, 0.01])
    action_lin = robot.move_linear_to_tcp_pose(target_linear_pose, linear_speed=0.2)
    _, vel_pct_lin, cmd_id_lin = robot._bridge.sent_commands[-1]
    assert vel_pct_lin == 10
    assert cmd_id_lin == 3
    assert action_lin.is_action_done()


def test_servo_moves_and_validation(robot: Kuka) -> None:
    target_joints = np.radians([2.0, -90.0, 90.0, 80.0, 0.0, 0.0])
    action = robot.servo_to_joint_configuration(target_joints, 0.01)
    sent_deg, _, _ = robot._bridge.sent_commands[-1]
    assert sent_deg == pytest.approx(np.degrees(target_joints))
    action.wait(timeout=0.5)

    target_pose = robot.forward_kinematics(np.radians([4.0, -90.0, 90.0, 80.0, 0.0, 0.0]))
    action_pose = robot.servo_to_tcp_pose(target_pose, 0.01)
    action_pose.wait(timeout=0.5)

    with pytest.raises(ValueError):
        robot.servo_to_joint_configuration(target_joints, 0.0)
    with pytest.raises(ValueError):
        robot.servo_to_joint_configuration(np.zeros(5), 0.05)


def test_invalid_configurations_and_speeds_raise(robot: Kuka) -> None:
    unreachable_joints = np.radians([200.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(RobotConfigurationException):
        robot.move_to_joint_configuration(unreachable_joints)

    far_pose = np.eye(4)
    far_pose[:3, 3] = [3.0, 0.0, 0.0]
    with pytest.raises(RobotConfigurationException):
        robot.move_to_tcp_pose(far_pose)
    with pytest.raises(RobotConfigurationException):
        robot.move_linear_to_tcp_pose(far_pose)
    with pytest.raises(RobotConfigurationException):
        robot.servo_to_tcp_pose(far_pose, 0.05)

    with pytest.raises(RobotSafetyViolationException):
        robot.move_to_joint_configuration(robot.get_joint_configuration(), joint_speed=100.0)
    with pytest.raises(RobotSafetyViolationException):
        robot.move_linear_to_tcp_pose(robot.get_tcp_pose(), linear_speed=100.0)
    with pytest.raises(ValueError):
        robot.move_to_joint_configuration(robot.get_joint_configuration(), joint_speed=0.0)


def test_upload_context_manager_and_specs_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kuka, "_create_opcua_bridge", FakeKukaOpcUaBridge)
    with Kuka("192.168.1.2", upload_krl_program=True) as inst:
        assert inst._bridge.uploaded
        inst.wait_until_ready(timeout=5.0)
        assert inst._bridge.waited_timeout == 5.0
    assert inst._bridge.closed

    with pytest.raises(ValueError):
        Kuka("192.168.1.2", manipulator_specs=ManipulatorSpecs([1.0] * 7, 1.0))


def test_wrist_singularity_branch_in_ik() -> None:
    # Exercise the q5 == 0 wrist singularity branch of kuka_flange_inverse_kinematics_all.
    q_singular = np.radians([15.0, -70.0, 80.0, 0.0, 0.0, 25.0])
    flange_pose = kuka_flange_forward_kinematics(q_singular)
    solutions = kuka_flange_inverse_kinematics_all(flange_pose)
    assert len(solutions) > 0
    assert any(np.allclose(kuka_flange_forward_kinematics(sol), flange_pose, atol=1e-6) for sol in solutions)
