from types import SimpleNamespace
from typing import Any, Callable, Optional

import numpy as np
import pytest
from airo_robots.manipulators.hardware import realman
from airo_spatial_algebra import SE3Container


class FakeInverseKinematicsParameters:
    def __init__(self, near: list[float], pose: list[float], flag: int) -> None:
        self.near = near
        self.pose = pose
        self.flag = flag


class FakeRealmanRobot:
    def __init__(self, mode: int, dof: int = 6, model: str = "RM_65") -> None:
        self.mode = mode
        self.dof = dof
        self.model = model
        self.handle = SimpleNamespace(id=1)
        self.callback: Optional[Callable[[Any], None]] = None
        self.joints_degrees = np.zeros(dof)
        self.pose = np.zeros(6)
        self.last_call: tuple[Any, ...] = ()
        self.closed = False
        # Error code returned by rm_algo_inverse_kinematics; non-zero simulates an
        # unreachable pose (no IK solution).
        self.inverse_kinematics_error_code = 0
        # All-solutions solver fake: the solutions it returns, each a list of joint
        # angles in degrees. The diagnostics check these against the joint limits
        # reported by rm_get_joint_min_pos / rm_get_joint_max_pos ([-180, 180] deg).
        self.inverse_kinematics_all_solutions: list[list[float]] = []
        # Six-axis force sensor fake: raw reading (includes the tool weight) and the
        # gravity-compensated external wrench, each [Fx, Fy, Fz, Mx, My, Mz].
        self.raw_force_data = np.zeros(6)
        self.external_force_data = np.zeros(6)
        self.force_data_error_code = 0

    def rm_create_robot_arm(self, ip_address: str, port: int) -> Any:
        self.last_connection = (ip_address, port)
        return self.handle

    def rm_get_robot_info(self) -> tuple[int, dict[str, Any]]:
        return 0, {"arm_dof": self.dof, "arm_model": self.model, "force_type": "B"}

    def rm_get_joint_max_speed(self) -> tuple[int, list[float]]:
        return 0, [180.0] * self.dof

    def rm_get_arm_max_line_speed(self) -> tuple[int, float]:
        return 0, 0.25

    def rm_get_joint_min_pos(self) -> tuple[int, list[float]]:
        return 0, [-180.0] * self.dof

    def rm_get_joint_max_pos(self) -> tuple[int, list[float]]:
        return 0, [180.0] * self.dof

    def rm_get_arm_event_call_back(self, callback: Callable[[Any], None]) -> None:
        self.callback = callback

    def rm_set_avoid_singularity_mode(self, mode: int) -> int:
        self.avoid_singularity_mode = mode
        return 0

    def rm_get_joint_degree(self) -> tuple[int, list[float]]:
        return 0, self.joints_degrees.tolist()

    def rm_get_current_arm_state(self) -> tuple[int, dict[str, list[float]]]:
        return 0, {"pose": self.pose.tolist()}

    def rm_get_force_data(self) -> tuple[int, dict[str, list[float]]]:
        return self.force_data_error_code, {
            "force_data": self.raw_force_data.tolist(),
            "zero_force_data": self.external_force_data.tolist(),
            "work_zero_force_data": self.external_force_data.tolist(),
            "tool_zero_force_data": self.external_force_data.tolist(),
        }

    def rm_movej(self, joints: list[float], speed: int, radius: int, connect: int, block: int) -> int:
        self.last_call = ("rm_movej", joints, speed, radius, connect, block)
        self.joints_degrees = np.asarray(joints)
        self._complete_motion()
        return 0

    def rm_movej_p(self, pose: list[float], speed: int, radius: int, connect: int, block: int) -> int:
        self.last_call = ("rm_movej_p", pose, speed, radius, connect, block)
        self.pose = np.asarray(pose)
        self._complete_motion()
        return 0

    def rm_movel(self, pose: list[float], speed: int, radius: int, connect: int, block: int) -> int:
        self.last_call = ("rm_movel", pose, speed, radius, connect, block)
        self.pose = np.asarray(pose)
        self._complete_motion()
        return 0

    def rm_movej_canfd(self, joints: list[float], follow: bool) -> int:
        self.last_call = ("rm_movej_canfd", joints, follow)
        return 0

    def rm_movep_canfd(self, pose: list[float], follow: bool) -> int:
        self.last_call = ("rm_movep_canfd", pose, follow)
        return 0

    def rm_algo_inverse_kinematics(self, params: FakeInverseKinematicsParameters) -> tuple[int, list[float]]:
        self.last_inverse_kinematics_parameters = params
        return self.inverse_kinematics_error_code, params.near

    def rm_algo_inverse_kinematics_all(self, params: FakeInverseKinematicsParameters) -> SimpleNamespace:
        self.last_inverse_kinematics_all_parameters = params
        q_solve = [list(solution) + [0.0] * (8 - len(solution)) for solution in self.inverse_kinematics_all_solutions]
        while len(q_solve) < 8:
            q_solve.append([0.0] * 8)
        return SimpleNamespace(
            result=0, num=len(self.inverse_kinematics_all_solutions), q_ref=[0.0] * 8, q_solve=q_solve
        )

    def rm_algo_forward_kinematics(self, joints: list[float], flag: int) -> list[float]:
        self.last_forward_kinematics_call = (joints, flag)
        return [0.1, 0.2, 0.3, 0.0, 0.0, 0.0]

    def rm_delete_robot_arm(self) -> int:
        self.closed = True
        return 0

    def _complete_motion(self) -> None:
        assert self.callback is not None
        self.callback(
            SimpleNamespace(
                handle_id=self.handle.id,
                event_type=1,
                trajectory_state=True,
                device=0,
                trajectory_connect=0,
            )
        )


def _make_robot(monkeypatch: pytest.MonkeyPatch, dof: int = 6, model: str = "RM_65") -> realman.RealmanControl:
    fake_api = SimpleNamespace(
        RoboticArm=lambda mode: FakeRealmanRobot(mode, dof=dof, model=model),
        rm_thread_mode_e=SimpleNamespace(RM_TRIPLE_MODE_E=3),
        rm_event_callback_ptr=lambda callback: callback,
        rm_inverse_kinematics_params_t=FakeInverseKinematicsParameters,
    )
    monkeypatch.setattr(realman, "_import_realman_api", lambda: fake_api)
    return realman.RealmanControl("192.168.1.18")


@pytest.fixture
def robot(monkeypatch: pytest.MonkeyPatch) -> realman.RealmanControl:
    return _make_robot(monkeypatch)


@pytest.fixture
def robot_7dof(monkeypatch: pytest.MonkeyPatch) -> realman.RealmanControl:
    return _make_robot(monkeypatch, dof=7, model="RM_75")


def test_controller_metadata_and_state_use_airo_units(robot: realman.RealmanControl) -> None:
    assert robot.dof == 6
    assert robot.model == "RM_65"
    assert robot.manipulator_specs.max_joint_speeds == pytest.approx([np.pi] * 6)
    assert robot.manipulator_specs.max_linear_speed == 0.25

    robot.robot.joints_degrees = np.asarray([0.0, 30.0, -90.0, 0.0, 45.0, 180.0])
    assert robot.get_joint_configuration() == pytest.approx(np.radians(robot.robot.joints_degrees))

    robot.robot.pose = np.asarray([0.1, 0.2, 0.3, 0.0, 0.0, np.pi / 2])
    expected_pose = SE3Container.from_euler_angles_and_translation(
        robot.robot.pose[3:], robot.robot.pose[:3]
    ).homogeneous_matrix
    assert robot.get_tcp_pose() == pytest.approx(expected_pose)


def test_planned_moves_are_non_blocking_and_convert_units(robot: realman.RealmanControl) -> None:
    joint_target = np.radians(np.asarray([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]))
    action = robot.move_to_joint_configuration(joint_target, joint_speed=np.pi / 2)
    assert robot.robot.last_call == ("rm_movej", pytest.approx(np.degrees(joint_target)), 50, 0, 0, 0)
    assert action.is_action_done()

    tcp_target = SE3Container.from_euler_angles_and_translation(
        np.asarray([0.1, -0.2, 0.3]), np.asarray([0.2, 0.1, 0.4])
    ).homogeneous_matrix
    action = robot.move_linear_to_tcp_pose(tcp_target, linear_speed=0.125)
    method, pose, speed, radius, connect, block = robot.robot.last_call
    assert (method, speed, radius, connect, block) == ("rm_movel", 50, 0, 0, 0)
    assert pose == pytest.approx(realman.RealmanControl._convert_homogeneous_pose_to_realman_pose(tcp_target))
    assert action.is_action_done()


def test_servo_uses_canfd_and_duration_controls_follow_mode(robot: realman.RealmanControl) -> None:
    target = np.radians(np.ones(6))
    robot.servo_to_joint_configuration(target, 0.01)
    assert robot.robot.last_call == ("rm_movej_canfd", pytest.approx(np.degrees(target)), True)

    robot.servo_to_tcp_pose(np.eye(4), 0.02)
    assert robot.robot.last_call == ("rm_movep_canfd", pytest.approx(np.zeros(6)), False)


def test_kinematics_convert_joint_angles(robot: realman.RealmanControl) -> None:
    near = np.radians(np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    solution = robot.inverse_kinematics(np.eye(4), near)
    assert solution == pytest.approx(near)
    params = robot.robot.last_inverse_kinematics_parameters
    assert params.near == pytest.approx(np.degrees(near))
    assert params.pose == pytest.approx(np.zeros(6))
    assert params.flag == 1

    pose = robot.forward_kinematics(near)
    assert robot.robot.last_forward_kinematics_call == (pytest.approx(np.degrees(near)), 1)
    assert pose[:3, 3] == pytest.approx([0.1, 0.2, 0.3])


def test_unreachable_pose_moves_raise(robot: realman.RealmanControl) -> None:
    from airo_robots.exceptions import RobotConfigurationException

    robot.robot.inverse_kinematics_error_code = 1  # simulate IK failure / unreachable pose

    with pytest.raises(RobotConfigurationException):
        robot.move_to_tcp_pose(np.eye(4))
    with pytest.raises(RobotConfigurationException):
        robot.move_linear_to_tcp_pose(np.eye(4))

    # No motion command should have been sent to the controller.
    assert robot.robot.last_call == ()


def test_inverse_kinematics_failure_diagnostics(robot: realman.RealmanControl) -> None:
    # The fake controller reports joint limits of [-180, 180] deg (= [-pi, pi] rad).
    params = FakeInverseKinematicsParameters([0.0] * 6, [0.0] * 6, 1)

    # No analytical solutions -> the target is out of reach.
    robot.robot.inverse_kinematics_all_solutions = []
    assert "out of reach" in robot._diagnose_ik_failure(params)

    # Solutions exist but every one drives a joint past its limit (J1 at 200 deg,
    # J3 at 300 deg) -> reachable position, orientation outside the joint range.
    robot.robot.inverse_kinematics_all_solutions = [
        [200.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 300.0, 0.0, 0.0, 0.0],
    ]
    message = robot._diagnose_ik_failure(params)
    assert "joint limit" in message
    assert "J1" in message and "J3" in message

    # At least one solution within the joint limits -> reachable; seeded-solver issue.
    robot.robot.inverse_kinematics_all_solutions = [
        [200.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    assert "within the joint limits" in robot._diagnose_ik_failure(params)


def test_seven_dof_arm_uses_its_own_dof_everywhere(robot_7dof: realman.RealmanControl) -> None:
    assert robot_7dof.dof == 7
    assert robot_7dof.model == "RM_75"
    assert robot_7dof.manipulator_specs.max_joint_speeds == pytest.approx([np.pi] * 7)

    robot_7dof.robot.joints_degrees = np.asarray([0.0, 30.0, -90.0, 0.0, 45.0, 90.0, 180.0])
    assert robot_7dof.get_joint_configuration() == pytest.approx(np.radians(robot_7dof.robot.joints_degrees))

    joint_target = np.radians(np.asarray([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]))
    action = robot_7dof.move_to_joint_configuration(joint_target, joint_speed=np.pi / 2)
    assert robot_7dof.robot.last_call == ("rm_movej", pytest.approx(np.degrees(joint_target)), 50, 0, 0, 0)
    assert action.is_action_done()

    robot_7dof.servo_to_joint_configuration(joint_target, 0.01)
    assert robot_7dof.robot.last_call == ("rm_movej_canfd", pytest.approx(np.degrees(joint_target)), True)

    with pytest.raises(ValueError):
        robot_7dof.servo_to_joint_configuration(np.zeros(6), 0.01)

    assert robot_7dof._is_joint_configuration_reachable(np.zeros(7))
    assert not robot_7dof._is_joint_configuration_reachable(np.zeros(6))


def test_diagnose_ik_failure_is_a_noop_for_non_six_dof_arms(robot_7dof: realman.RealmanControl) -> None:
    # The all-solutions diagnostics are only implemented for six-DOF arms; a
    # seven-DOF arm should get no extra detail rather than an incorrect one.
    params = FakeInverseKinematicsParameters([0.0] * 7, [0.0] * 6, 1)
    assert robot_7dof._diagnose_ik_failure(params) == ""


def test_joint_limits_and_close(robot: realman.RealmanControl) -> None:
    assert robot._is_joint_configuration_reachable(np.zeros(6))
    assert not robot._is_joint_configuration_reachable(np.full(6, 2 * np.pi))
    assert not robot._is_joint_configuration_reachable(np.zeros(7))

    robot.close()
    assert robot.robot.closed
    robot.close()


def test_get_wrench_returns_compensated_or_raw_reading(robot: realman.RealmanControl) -> None:
    robot.robot.raw_force_data = np.asarray([1.0, 2.0, 9.81, 0.1, 0.2, 0.3])
    robot.robot.external_force_data = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # By default the gravity-compensated external wrench is returned.
    assert robot.get_wrench() == pytest.approx(robot.robot.external_force_data)
    assert robot.get_wrench(compensated=False) == pytest.approx(robot.robot.raw_force_data)


def test_get_wrench_raises_without_force_sensor(robot: realman.RealmanControl) -> None:
    robot.robot.force_data_error_code = 1
    with pytest.raises(RuntimeError, match="rm_get_force_data"):
        robot.get_wrench()
