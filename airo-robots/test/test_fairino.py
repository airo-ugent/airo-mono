"""Mock tests for the Fairino FR3 manipulator driver."""

from typing import Any, List, Tuple

import numpy as np
import pytest
from airo_robots.manipulators.hardware import fairino
from airo_spatial_algebra import SE3Container


class FakeFairinoRPC:
    """Mock Robot.RPC object mimicking the Fairino SDK API."""

    def __init__(self, ip: str) -> None:
        self.ip = ip
        self.q_measured = np.zeros(6)  # radians (will be converted to/from deg by SDK)
        self.tcp_pose = np.array([500.0, 0.0, 300.0, 0.0, 0.0, 90.0])  # mm, deg
        self.motion_done = True
        self.point_record_button_state = 1
        self.smarttool_button_state = 0
        self.calls: List[Tuple[Any, ...]] = []
        self.closed = False
        self.tool_id = 0
        self.in_freedrive = False

    def GetSDKVersion(self) -> Tuple[int, str]:
        """Return SDK version."""
        return (0, "1.0.0")

    def GetActualJointPosRadian(self, flag: int) -> Tuple[int, List[float]]:
        """Get current joint position in radians."""
        self.calls.append(("GetActualJointPosRadian", flag))
        return (0, self.q_measured.tolist())

    def GetActualTCPPose(self, flag: int) -> Tuple[int, List[float]]:
        """Get current TCP pose [x, y, z, rx, ry, rz] in mm and degrees."""
        self.calls.append(("GetActualTCPPose", flag))
        return (0, self.tcp_pose.tolist())

    def GetRobotMotionDone(self) -> Tuple[int, int]:
        """Check if motion is complete."""
        self.calls.append(("GetRobotMotionDone",))
        return (0, 1 if self.motion_done else 0)

    def GetAxlePointRecordBtnState(self) -> Tuple[int, int]:
        """Get point record button state."""
        self.calls.append(("GetAxlePointRecordBtnState",))
        return (0, self.point_record_button_state)

    def GetSmarttoolBtnState(self) -> Tuple[int, int]:
        """Get SmartTool button state bitfield."""
        self.calls.append(("GetSmarttoolBtnState",))
        return (0, self.smarttool_button_state)

    def MoveJ(
        self, joint_pos: List[float], tool: int, user: int, vel: float = 20.0, blendT: float = -1.0, **_: Any
    ) -> int:
        """Move to joint position."""
        self.calls.append(("MoveJ", joint_pos, tool, user, vel, blendT))
        self.q_measured = np.radians(np.asarray(joint_pos, dtype=float))
        self.motion_done = True
        return 0

    def MoveL(self, pose: List[float], tool: int, user: int, vel: float = 20.0, blendR: float = -1.0, **_: Any) -> int:
        """Move linearly to pose."""
        self.calls.append(("MoveL", pose, tool, user, vel, blendR))
        self.tcp_pose = np.asarray(pose, dtype=float)
        self.motion_done = True
        return 0

    def MoveCart(
        self,
        pose: List[float],
        tool: int,
        user: int,
        vel: float = 20.0,
        blendT: float = -1.0,
        config: int = -1,
        **_: Any
    ) -> int:
        """Move to Cartesian pose."""
        self.calls.append(("MoveCart", pose, tool, user, vel, blendT, config))
        self.tcp_pose = np.asarray(pose, dtype=float)
        self.motion_done = True
        return 0

    def ServoMoveStart(self) -> int:
        """Start servo mode."""
        self.calls.append(("ServoMoveStart",))
        self.motion_done = False
        return 0

    def ServoJ(self, joint_pos: List[float], axisPos: List[float], cmdT: float = 0.008, **_: Any) -> int:
        """Send servo joint command."""
        self.calls.append(("ServoJ", joint_pos, axisPos, cmdT))
        self.q_measured = np.radians(np.asarray(joint_pos, dtype=float))
        return 0

    def ServoCart(self, mode: int, pose: List[float], exaxis: List[float], cmdT: float = 0.008, **_: Any) -> int:
        """Send servo Cartesian command."""
        self.calls.append(("ServoCart", mode, pose, exaxis, cmdT))
        self.tcp_pose = np.asarray(pose, dtype=float)
        return 0

    def ServoMoveEnd(self) -> int:
        """End servo mode."""
        self.calls.append(("ServoMoveEnd",))
        self.motion_done = True
        return 0

    def GetForwardKin(self, joint_pos: List[float]) -> Tuple[int, List[float]]:
        """Compute forward kinematics."""
        self.calls.append(("GetForwardKin", joint_pos))
        # Simple identity FK for testing
        return (0, self.tcp_pose.tolist())

    def GetInverseKinRef(self, ik_type: int, pose: List[float], joint_ref: List[float]) -> Tuple[int, List[float]]:
        """Compute inverse kinematics."""
        self.calls.append(("GetInverseKinRef", ik_type, pose, joint_ref))
        # Simple identity IK for testing
        return (0, np.degrees(self.q_measured).tolist())

    def Mode(self, mode: int) -> int:
        """Set robot mode (0=auto, 1=manual)."""
        self.calls.append(("Mode", mode))
        return 0

    def DragTeachSwitch(self, state: int) -> int:
        """Enable/disable drag-teach (freedrive)."""
        self.calls.append(("DragTeachSwitch", state))
        self.in_freedrive = bool(state)
        return 0

    def SetLoadWeight(self, load_num: int, weight: float) -> int:
        """Set payload mass."""
        self.calls.append(("SetLoadWeight", load_num, weight))
        return 0

    def SetLoadCoord(self, x: float, y: float, z: float, load_num: int = 0) -> int:
        """Set payload center-of-gravity."""
        self.calls.append(("SetLoadCoord", x, y, z, load_num))
        return 0

    def SetToolCoord(
        self, tool_id: int, coord: List[float], type_: int, install: int, tool_type: int, load_num: int
    ) -> int:
        """Set tool frame offset."""
        self.calls.append(("SetToolCoord", tool_id, coord, type_, install, tool_type, load_num))
        return 0

    def CloseRPC(self) -> None:
        """Close connection."""
        self.calls.append(("CloseRPC",))
        self.closed = True


class FakeFairinoModule:
    """Mock fairino module."""

    class Robot:
        """Mock Robot class with RPC method."""

        @staticmethod
        def RPC(ip: str) -> FakeFairinoRPC:
            """Create RPC connection."""
            return FakeFairinoRPC(ip)


def _fake_fairino_module() -> Any:
    """Return a mock Fairino module."""
    return FakeFairinoModule()


def _make_robot(monkeypatch: pytest.MonkeyPatch) -> fairino.Fairino:
    """Create a Fairino robot with mocked SDK."""
    monkeypatch.setattr(fairino, "_import_fairino_api", _fake_fairino_module)
    return fairino.Fairino("192.168.58.2")


@pytest.fixture
def robot(monkeypatch: pytest.MonkeyPatch) -> fairino.Fairino:
    """Fairino robot fixture."""
    return _make_robot(monkeypatch)


@pytest.fixture
def robot_rpc(robot: fairino.Fairino) -> FakeFairinoRPC:
    """Access to the underlying mock RPC object."""
    assert isinstance(robot.robot, FakeFairinoRPC)
    return robot.robot


def test_connection_and_specs(robot: fairino.Fairino) -> None:
    """Test basic connection and specs."""
    assert robot.ip_address == "192.168.58.2"
    assert robot.manipulator_specs.dof == 6
    assert robot.manipulator_specs.max_joint_speeds == pytest.approx(np.full(6, np.radians(180.0)))
    assert robot.manipulator_specs.max_linear_speed == pytest.approx(2.0)


def test_state_getters_convert_units(robot: fairino.Fairino, robot_rpc: FakeFairinoRPC) -> None:
    """Test that state getters convert between SDK units (deg, mm) and AIRO units (rad, m)."""
    # Set mock state: 30° per joint, TCP at [500 mm, 0, 300 mm] with [0°, 0°, 90°] rotation
    robot_rpc.q_measured = np.radians(np.full(6, 30.0))
    robot_rpc.tcp_pose = np.array([500.0, 0.0, 300.0, 0.0, 0.0, 90.0])

    # get_joint_configuration should return radians
    q = robot.get_joint_configuration()
    assert q == pytest.approx(np.full(6, np.radians(30.0)))

    # get_tcp_pose should return homogeneous matrix with m and rad
    pose = robot.get_tcp_pose()
    expected = SE3Container.from_euler_angles_and_translation(
        np.radians([0.0, 0.0, 90.0]), np.array([0.5, 0.0, 0.3])
    ).homogeneous_matrix
    assert pose == pytest.approx(expected)


def test_move_to_joint_configuration(robot: fairino.Fairino, robot_rpc: FakeFairinoRPC) -> None:
    """Test joint configuration move."""
    target = np.radians(np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]))
    action = robot.move_to_joint_configuration(target, joint_speed=0.5)

    # Check that MoveJ was called with correct parameters
    moves = [call for call in robot_rpc.calls if call[0] == "MoveJ"]
    assert len(moves) == 1
    move_call = moves[0]
    assert np.allclose(move_call[1], np.degrees(target))  # API receives degrees
    assert move_call[4] == 16  # velocity 0.5 rad/s -> ceil(100 * 0.5 / pi) = 16%

    # Check motion completion
    assert action.wait()


def test_move_linear_to_tcp_pose(robot: fairino.Fairino, robot_rpc: FakeFairinoRPC) -> None:
    """Test linear Cartesian move."""
    target_mm = np.array([500.0, 100.0, 300.0, 0.0, 0.0, 90.0])
    target_pose = SE3Container.from_euler_angles_and_translation(
        np.radians([0.0, 0.0, 90.0]), np.array([0.5, 0.1, 0.3])
    ).homogeneous_matrix

    action = robot.move_linear_to_tcp_pose(target_pose, linear_speed=0.2)

    moves = [call for call in robot_rpc.calls if call[0] == "MoveL"]
    assert len(moves) == 1
    move_call = moves[0]
    assert np.allclose(move_call[1], target_mm, atol=1e-6)  # pose in mm
    assert move_call[4] == 10  # speed 0.2 m/s → 10% of max 2 m/s

    assert action.wait()


def test_move_to_tcp_pose(robot: fairino.Fairino, robot_rpc: FakeFairinoRPC) -> None:
    """Test Cartesian PTP move."""
    target_pose = SE3Container.from_euler_angles_and_translation(
        np.radians([10.0, 20.0, 30.0]), np.array([0.5, 0.0, 0.3])
    ).homogeneous_matrix

    action = robot.move_to_tcp_pose(target_pose, joint_speed=0.3)

    moves = [call for call in robot_rpc.calls if call[0] == "MoveCart"]
    assert len(moves) == 1
    move_call = moves[0]
    assert move_call[4] == 10  # speed 0.3 rad/s -> ceil(100 * 0.3 / pi) = 10%

    assert action.wait()


def _servo_calls(robot_rpc: FakeFairinoRPC) -> List[Tuple[Any, ...]]:
    servo_names = ("ServoMoveStart", "ServoJ", "ServoCart", "ServoMoveEnd")
    return [call for call in robot_rpc.calls if call[0] in servo_names]


def test_servo_to_joint_configuration(robot: fairino.Fairino, robot_rpc: FakeFairinoRPC) -> None:
    """Test joint servo command."""
    target = np.radians(np.full(6, 45.0))
    action = robot.servo_to_joint_configuration(target, duration=0.01)

    servo_calls = _servo_calls(robot_rpc)
    assert [call[0] for call in servo_calls] == ["ServoMoveStart", "ServoJ"]
    assert np.allclose(servo_calls[1][1], np.degrees(target))
    assert len(servo_calls[1][2]) == 4  # no external axes
    assert servo_calls[1][3] == pytest.approx(0.01)  # cmdT equals the servo duration

    # Action completes after duration
    assert action.wait(timeout=0.1)


def test_servo_to_tcp_pose(robot: fairino.Fairino, robot_rpc: FakeFairinoRPC) -> None:
    """Test Cartesian servo command."""
    target_pose = SE3Container.from_euler_angles_and_translation(
        np.radians([5.0, 10.0, 15.0]), np.array([0.5, 0.1, 0.3])
    ).homogeneous_matrix

    action = robot.servo_to_tcp_pose(target_pose, duration=0.01)

    servo_calls = _servo_calls(robot_rpc)
    assert [call[0] for call in servo_calls] == ["ServoMoveStart", "ServoCart"]
    assert len(servo_calls[1][3]) == 4  # no external axes
    assert servo_calls[1][4] == pytest.approx(0.01)  # cmdT equals the servo duration

    assert action.wait(timeout=0.1)


def test_servo_session_is_kept_open_between_setpoints(robot: fairino.Fairino, robot_rpc: FakeFairinoRPC) -> None:
    """Consecutive servo calls stream setpoints within a single servo session."""
    for angle in (10.0, 11.0, 12.0):
        robot.servo_to_joint_configuration(np.radians(np.full(6, angle)), duration=0.008)

    assert [call[0] for call in _servo_calls(robot_rpc)] == ["ServoMoveStart", "ServoJ", "ServoJ", "ServoJ"]

    robot.servo_stop()
    robot.servo_stop()  # no-op without an active session
    assert [call[0] for call in _servo_calls(robot_rpc)][-1] == "ServoMoveEnd"
    assert sum(call[0] == "ServoMoveEnd" for call in robot_rpc.calls) == 1


def test_servo_period_is_clipped(robot: fairino.Fairino, robot_rpc: FakeFairinoRPC) -> None:
    """cmdT is clipped to the SDK's recommended range."""
    robot.servo_to_joint_configuration(np.zeros(6), duration=0.0001)
    robot.servo_to_joint_configuration(np.zeros(6), duration=0.5)

    servo_j_calls = [call for call in robot_rpc.calls if call[0] == "ServoJ"]
    assert servo_j_calls[0][3] == pytest.approx(0.001)
    assert servo_j_calls[1][3] == pytest.approx(0.016)


def test_switching_servo_type_restarts_session(robot: fairino.Fairino, robot_rpc: FakeFairinoRPC) -> None:
    """Switching between joint and Cartesian servoing ends the old session and starts a new one."""
    robot.servo_to_joint_configuration(np.zeros(6), duration=0.008)
    robot.servo_to_tcp_pose(np.eye(4), duration=0.008)

    assert [call[0] for call in _servo_calls(robot_rpc)] == [
        "ServoMoveStart",
        "ServoJ",
        "ServoMoveEnd",
        "ServoMoveStart",
        "ServoCart",
    ]


def test_move_and_close_end_servo_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """Moves and close end an active servo session before doing anything else."""
    robot = _make_robot(monkeypatch)
    robot_rpc = robot.robot

    robot.servo_to_joint_configuration(np.zeros(6), duration=0.008)
    robot.move_to_joint_configuration(np.zeros(6))
    names = [call[0] for call in robot_rpc.calls]
    assert names.index("ServoMoveEnd") < names.index("MoveJ")

    robot.servo_to_joint_configuration(np.zeros(6), duration=0.008)
    robot.close()
    names = [call[0] for call in robot_rpc.calls]
    assert names[-2:] == ["ServoMoveEnd", "CloseRPC"]


def test_forward_kinematics(robot: fairino.Fairino, robot_rpc: FakeFairinoRPC) -> None:
    """Test forward kinematics."""
    joint_config = np.radians(np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]))
    pose = robot.forward_kinematics(joint_config)

    fk_calls = [call for call in robot_rpc.calls if call[0] == "GetForwardKin"]
    assert len(fk_calls) == 1
    assert np.allclose(fk_calls[0][1], np.degrees(joint_config))

    assert pose.shape == (4, 4)


def test_inverse_kinematics(robot: fairino.Fairino, robot_rpc: FakeFairinoRPC) -> None:
    """Test inverse kinematics."""
    target_pose = SE3Container.from_euler_angles_and_translation(
        np.radians([0.0, 0.0, 90.0]), np.array([0.5, 0.0, 0.3])
    ).homogeneous_matrix

    result = robot.inverse_kinematics(target_pose)

    ik_calls = [call for call in robot_rpc.calls if call[0] == "GetInverseKinRef"]
    assert len(ik_calls) == 1

    assert result is not None
    assert result.shape == (6,)


def test_inverse_kinematics_with_seed(robot: fairino.Fairino, robot_rpc: FakeFairinoRPC) -> None:
    """Test inverse kinematics with reference configuration."""
    target_pose = SE3Container.from_euler_angles_and_translation(
        np.radians([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])
    ).homogeneous_matrix
    seed = np.radians(np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]))

    result = robot.inverse_kinematics(target_pose, joint_configuration_near=seed)

    ik_calls = [call for call in robot_rpc.calls if call[0] == "GetInverseKinRef"]
    assert len(ik_calls) == 1
    # Check that seed was passed in degrees
    assert np.allclose(ik_calls[0][3], np.degrees(seed))

    assert result is not None


def test_joint_configuration_reachability(robot: fairino.Fairino) -> None:
    """Test joint configuration reachability check."""
    # Valid configuration (within [-180°, 180°] = [-π, π])
    valid = np.radians(np.array([0.0, 45.0, -90.0, 180.0, -180.0, 90.0]))
    assert robot._is_joint_configuration_reachable(valid)

    # Out of range
    out_of_range = np.radians(np.array([200.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert not robot._is_joint_configuration_reachable(out_of_range)

    # Wrong DOF
    wrong_dof = np.zeros(7)
    assert not robot._is_joint_configuration_reachable(wrong_dof)


def test_freedrive(robot: fairino.Fairino, robot_rpc: FakeFairinoRPC) -> None:
    """Test freedrive (drag-teach mode)."""
    assert not robot_rpc.in_freedrive

    robot.start_freedrive()
    mode_calls = [call for call in robot_rpc.calls if call[0] == "Mode"]
    drag_calls = [call for call in robot_rpc.calls if call[0] == "DragTeachSwitch"]
    assert len(mode_calls) >= 1 and mode_calls[-1][1] == 1  # Manual mode
    assert len(drag_calls) >= 1 and drag_calls[-1][1] == 1  # Drag-teach on

    robot_rpc.calls.clear()
    robot.stop_freedrive()
    mode_calls = [call for call in robot_rpc.calls if call[0] == "Mode"]
    drag_calls = [call for call in robot_rpc.calls if call[0] == "DragTeachSwitch"]
    assert len(drag_calls) >= 1 and drag_calls[-1][1] == 0  # Drag-teach off
    assert len(mode_calls) >= 1 and mode_calls[-1][1] == 0  # Auto mode


def test_button_state_polling_methods(robot: fairino.Fairino, robot_rpc: FakeFairinoRPC) -> None:
    """Test button state polling methods."""
    robot_rpc.point_record_button_state = 0
    robot_rpc.smarttool_button_state = 0b10101

    assert robot.get_point_record_button_state() == 0
    assert robot.get_smarttool_button_state() == 0b10101

    point_calls = [call for call in robot_rpc.calls if call[0] == "GetAxlePointRecordBtnState"]
    smarttool_calls = [call for call in robot_rpc.calls if call[0] == "GetSmarttoolBtnState"]

    assert len(point_calls) == 1
    assert len(smarttool_calls) == 1


def test_set_payload(robot: fairino.Fairino, robot_rpc: FakeFairinoRPC) -> None:
    """Test payload configuration."""
    mass = 0.5
    cog = np.array([0.01, 0.02, 0.15])

    robot.set_payload(mass, cog)

    weight_calls = [call for call in robot_rpc.calls if call[0] == "SetLoadWeight"]
    coord_calls = [call for call in robot_rpc.calls if call[0] == "SetLoadCoord"]

    assert len(weight_calls) == 1
    assert weight_calls[0][2] == mass

    assert len(coord_calls) == 1
    # CoG in mm (cog * 1000)
    assert np.allclose([coord_calls[0][1], coord_calls[0][2], coord_calls[0][3]], cog * 1000.0)


def test_set_tcp_offset(robot: fairino.Fairino, robot_rpc: FakeFairinoRPC) -> None:
    """Test TCP offset configuration."""
    tcp_offset = SE3Container.from_translation(np.array([0.0, 0.0, 0.18])).homogeneous_matrix

    robot.set_tcp_offset(tcp_offset)

    tool_calls = [call for call in robot_rpc.calls if call[0] == "SetToolCoord"]
    assert len(tool_calls) == 1
    assert tool_calls[0][1] == 1  # Tool ID 1
    # Pose should be in mm
    assert np.allclose(tool_calls[0][2][2], 180.0)  # Z offset in mm


def test_speed_conversion(robot: fairino.Fairino) -> None:
    """Test speed conversion from physical units to percentages."""
    # Max joint speed: 180°/s = π rad/s
    # 0.5 rad/s should convert to ceil(100 * 0.5 / pi) = 16%
    assert fairino.Fairino._speed_to_percentage(0.5, np.pi) == 16

    # Max linear speed: 2 m/s
    # 0.1 m/s should convert to 5%
    assert fairino.Fairino._speed_to_percentage(0.1, 2.0) == 5

    # Very small speeds should be clipped to 1%
    assert fairino.Fairino._speed_to_percentage(0.001, 1.0) == 1

    # Very large speeds should be clipped to 100%
    assert fairino.Fairino._speed_to_percentage(10.0, 1.0) == 100


def test_context_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that context manager properly closes connection."""
    monkeypatch.setattr(fairino, "_import_fairino_api", _fake_fairino_module)

    with fairino.Fairino("192.168.58.2") as robot:
        robot_rpc = robot.robot
        assert isinstance(robot_rpc, FakeFairinoRPC)
        assert not robot_rpc.closed

    assert robot_rpc.closed
    close_calls = [call for call in robot_rpc.calls if call[0] == "CloseRPC"]
    assert len(close_calls) == 1


def test_pose_conversion_roundtrip(robot: fairino.Fairino) -> None:
    """Test roundtrip conversion between Fairino and homogeneous matrix formats."""
    # Start with a Fairino pose in mm and degrees
    fairino_pose_mm_deg = np.array([100.0, 200.0, 300.0, 10.0, 20.0, 30.0])

    # Convert to homogeneous matrix
    homo = fairino.Fairino._convert_fairino_pose_to_homogeneous_pose(fairino_pose_mm_deg)

    # Convert back to Fairino pose
    fairino_pose_back = fairino.Fairino._convert_homogeneous_pose_to_fairino_pose(homo)

    assert np.allclose(fairino_pose_back, fairino_pose_mm_deg, atol=1e-6)
