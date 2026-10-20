"""Position-controlled manipulator implementation for KUKA KRC4 robots over OPC-UA."""

import asyncio
import importlib
import threading
import time
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Self

import numpy as np
from airo_robots.awaitable_action import AwaitableAction
from airo_robots.exceptions import RobotConfigurationException
from airo_robots.grippers import ParallelPositionGripper
from airo_robots.manipulators.position_manipulator import ManipulatorSpecs, PositionManipulator
from airo_spatial_algebra import SE3Container
from airo_typing import HomogeneousMatrixType, JointConfigurationType
from loguru import logger
from scipy.spatial.transform import Rotation, Slerp

KukaPoseType = np.ndarray
"""A KUKA Cartesian pose ``[X, Y, Z, A, B, C]`` in millimetres and degrees (`R = Rz(A) @ Ry(B) @ Rx(C)`)."""

KUKA_DOF = 6
"""Degrees of freedom supported by this manipulator implementation."""

DEFAULT_OPCUA_PORT = 4840
DEFAULT_OPCUA_USERNAME = "OpcUaSecurityAdmin"
DEFAULT_OPCUA_PASSWORD = "kuka"
DEFAULT_KRL_PROGRAM = "PY_SLOW"

# Datasheet values for the KUKA KR 3 R540 (AGILUS small robot series).
KR3_R540_MAX_JOINT_SPEEDS_DEG_S = (530.0, 529.0, 538.0, 600.0, 600.0, 800.0)
KR3_R540_MAX_LINEAR_SPEED_M_S = 2.0
KR3_R540_DEFAULT_LOWER_LIMITS_DEG = (-170.0, -170.0, -110.0, -175.0, -120.0, -350.0)
KR3_R540_DEFAULT_UPPER_LIMITS_DEG = (170.0, 50.0, 155.0, 175.0, 120.0, 350.0)

_SYSVAR_PREFIX = "ns=8;s=ns=7%3Bi=5004??krlvar:#"
_KRLFILE_PROGRAM_DIR = "ns=8;s=ns=7%3Bi=5202??krlfile:KRC:/R1/Program"
_TASK_PROGRAM_NAME_NODE = "ns=8;s=5104.Robot.ParameterSet.TaskProgramName"
_TASK_PROGRAM_STATE_NODE = "ns=8;s=5104.Robot.ParameterSet.ProgramState"

KRL_DAT_TEMPLATE = """&ACCESS RVP
&REL 1
&COMMENT Python Motion Server
&PARAM DISKPATH = KRC:\\R1\\Program
DEFDAT  {program_name} PUBLIC
;FOLD EXTERNAL DECLARATIONS;%{{PE}}%MKUKATPBASIS,%CEXT,%VCOMMON,%P
;FOLD BASISTECH EXT;%{{PE}}%MKUKATPBASIS,%CEXT,%VEXT,%P
EXT  BAS (BAS_COMMAND  :IN,REAL  :IN )
;ENDFOLD (BASISTECH EXT)
;ENDFOLD (EXTERNAL DECLARATIONS)

DECL GLOBAL REAL PY_A1=0.0
DECL GLOBAL REAL PY_A2=-90.0
DECL GLOBAL REAL PY_A3=90.0
DECL GLOBAL REAL PY_A4=80.0
DECL GLOBAL REAL PY_A5=0.0
DECL GLOBAL REAL PY_A6=0.0
DECL GLOBAL INT PY_VEL=20
DECL GLOBAL INT PY_CMD_ID=0
DECL GLOBAL INT PY_ACK_ID=0
DECL GLOBAL BOOL PY_READY=FALSE
DECL GLOBAL BOOL PY_MOVING=FALSE
DECL GLOBAL BOOL PY_QUIT=FALSE

ENDDAT
"""

KRL_SRC_TEMPLATE = """&ACCESS RVP
&REL 1
&COMMENT Python Motion Server
&PARAM DISKPATH = KRC:\\R1\\Program
DEF  {program_name} ( )
  DECL E6AXIS TARGET_AXIS

  ;FOLD INI
  ;FOLD BASISTECH INI
  GLOBAL INTERRUPT DECL 3 WHEN $STOPMESS==TRUE DO IR_STOPM ( )
  INTERRUPT ON 3 
  BAS (#INITMOV,0 )
  ;ENDFOLD (BASISTECH INI)
  ;ENDFOLD (INI)

  $ADVANCE = 0
  $OV_PRO = 100
  BAS (#VEL_PTP, 20 )
  BAS (#ACC_PTP, 50 )

  PY_A1 = $AXIS_ACT.A1
  PY_A2 = $AXIS_ACT.A2
  PY_A3 = $AXIS_ACT.A3
  PY_A4 = $AXIS_ACT.A4
  PY_A5 = $AXIS_ACT.A5
  PY_A6 = $AXIS_ACT.A6
  PY_ACK_ID = PY_CMD_ID
  PY_MOVING = FALSE
  PY_QUIT = FALSE

  PTP $AXIS_ACT

  PY_READY = TRUE

  LOOP
    WAIT FOR (PY_CMD_ID <> PY_ACK_ID) OR PY_QUIT

    IF PY_QUIT THEN
      EXIT
    ENDIF

    PY_MOVING = TRUE

    IF PY_VEL < 1 THEN
      PY_VEL = 1
    ENDIF
    IF PY_VEL > 100 THEN
      PY_VEL = 100
    ENDIF

    BAS (#VEL_PTP, PY_VEL )
    BAS (#ACC_PTP, 50 )

    TARGET_AXIS = $AXIS_ACT
    TARGET_AXIS.A1 = PY_A1
    TARGET_AXIS.A2 = PY_A2
    TARGET_AXIS.A3 = PY_A3
    TARGET_AXIS.A4 = PY_A4
    TARGET_AXIS.A5 = PY_A5
    TARGET_AXIS.A6 = PY_A6

    PTP TARGET_AXIS

    PY_MOVING = FALSE
    PY_ACK_ID = PY_CMD_ID
  ENDLOOP

  PY_READY = FALSE
END
"""


@dataclass(frozen=True)
class KukaKinematicGeometry:
    """Spherical-wrist 6-DOF KUKA arm geometry in millimetres (extracted from ``$machine.dat``).

    Defaults match ``#KR3R540 C4SR`` (`KUKA KR 3 R540`).
    """

    tiroro_z_mm: float = 345.0
    length_a_mm: float = 20.0
    length_b_mm: float = 260.0
    tx3p3_x_mm: float = 260.0
    tx3p3_z_mm: float = 20.0
    tflwp_z_mm: float = 75.0


def _import_asyncua() -> ModuleType:
    """Import the optional ``asyncua`` library with an actionable error message."""
    try:
        return importlib.import_module("asyncua")
    except ImportError as exception:
        raise ImportError(
            'Kuka requires the asyncua library. Install it with `pip install "airo-robots[kuka]"`.'
        ) from exception


def _rot_x(angle_rad: float) -> np.ndarray:
    c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def _rot_y(angle_rad: float) -> np.ndarray:
    c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def _rot_z(angle_rad: float) -> np.ndarray:
    c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _trans_matrix(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, 3] = [x, y, z]
    return matrix


def _rot_matrix_4x4(rotation_3x3: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = rotation_3x3
    return matrix


def kuka_flange_forward_kinematics(
    joint_configuration_rad: np.ndarray,
    geometry: KukaKinematicGeometry = KukaKinematicGeometry(),
) -> HomogeneousMatrixType:
    """Compute exact KUKA 6-DOF flange forward kinematics in the root frame (metres).

    Follows KUKA's ``$TIRORO``, ``$LENGTH_A``, ``$LENGTH_B``, ``$TX3P3``, ``$TFLWP`` chain with
    axis sign conventions ``(-A1, +A2, +A3, -A4, +A5, -A6)``.
    """
    q1, q2, q3, q4, q5, q6 = np.asarray(joint_configuration_rad, dtype=float)
    d1 = geometry.tiroro_z_mm / 1000.0
    a1 = geometry.length_a_mm / 1000.0
    a2 = geometry.length_b_mm / 1000.0
    d4 = geometry.tx3p3_x_mm / 1000.0
    a3 = geometry.tx3p3_z_mm / 1000.0
    d6 = geometry.tflwp_z_mm / 1000.0

    return (
        _trans_matrix(z=d1)
        @ _rot_matrix_4x4(_rot_z(-q1))
        @ _trans_matrix(x=a1)
        @ _rot_matrix_4x4(_rot_y(q2))
        @ _trans_matrix(x=a2)
        @ _rot_matrix_4x4(_rot_y(q3))
        @ _trans_matrix(x=d4, z=a3)
        @ _rot_matrix_4x4(_rot_x(-q4))
        @ _rot_matrix_4x4(_rot_y(q5))
        @ _rot_matrix_4x4(_rot_x(-q6))
        @ _rot_matrix_4x4(_rot_y(np.pi / 2.0))
        @ _trans_matrix(z=d6)
    )


def kuka_flange_inverse_kinematics_all(
    flange_pose_in_root: HomogeneousMatrixType,
    geometry: KukaKinematicGeometry = KukaKinematicGeometry(),
    q4_reference: float = 0.0,
) -> list[np.ndarray]:
    """Compute all analytical 8-branch spherical-wrist IK solutions for a flange pose in root frame.

    Args:
        flange_pose_in_root: 4x4 homogeneous transform of the flange in the robot root frame (metres).
        geometry: Kinematic link dimensions in millimetres.
        q4_reference: Reference angle for A4 (radians) used to resolve the coaxial A4/A6 degree of
            freedom when the wrist is at the ``A5 = 0`` singularity.

    Returns:
        List of 6-element joint configuration arrays in radians (wrapped to ``[-pi, pi]``).
    """
    d1 = geometry.tiroro_z_mm / 1000.0
    a1 = geometry.length_a_mm / 1000.0
    a2 = geometry.length_b_mm / 1000.0
    d4 = geometry.tx3p3_x_mm / 1000.0
    a3 = geometry.tx3p3_z_mm / 1000.0
    d6 = geometry.tflwp_z_mm / 1000.0

    wrist_pose = flange_pose_in_root @ _trans_matrix(z=-d6) @ _rot_matrix_4x4(_rot_y(-np.pi / 2.0))
    pw = wrist_pose[:3, 3]
    rw = wrist_pose[:3, :3]

    solutions: list[np.ndarray] = []
    rho_xy = float(np.hypot(pw[0], pw[1]))
    phi_base = float(np.arctan2(-pw[1], pw[0])) if rho_xy > 1e-9 else 0.0
    l3 = float(np.hypot(d4, a3))
    psi = float(np.arctan2(a3, d4))

    for shoulder_sign in (1.0, -1.0):
        q1 = phi_base if shoulder_sign > 0 else float(np.arctan2(pw[1], -pw[0]))
        c1, s1 = float(np.cos(q1)), float(np.sin(q1))
        r_plane = c1 * pw[0] - s1 * pw[1] - a1
        z_plane = d1 - pw[2]
        dist_sq = r_plane * r_plane + z_plane * z_plane
        cos_gamma = (dist_sq - a2 * a2 - l3 * l3) / (2.0 * a2 * l3)
        if abs(cos_gamma) > 1.0 + 1e-7:
            continue
        cos_gamma = float(np.clip(cos_gamma, -1.0, 1.0))

        for elbow_sign in (1.0, -1.0):
            gamma = elbow_sign * float(np.arccos(cos_gamma))
            q3 = gamma + psi
            alpha = float(np.arctan2(z_plane, r_plane))
            beta = float(np.arctan2(l3 * np.sin(gamma), a2 + l3 * np.cos(gamma)))
            q2 = alpha - beta

            r03 = _rot_z(-q1) @ _rot_y(q2 + q3)
            r36 = r03.T @ rw
            r00 = float(np.clip(r36[0, 0], -1.0, 1.0))

            for wrist_sign in (1.0, -1.0):
                q5 = wrist_sign * float(np.arccos(r00))
                sin_q5 = float(np.sin(q5))
                if abs(sin_q5) > 1e-6:
                    q4 = float(np.arctan2(-r36[1, 0] / sin_q5, -r36[2, 0] / sin_q5))
                    q6 = float(np.arctan2(-r36[0, 1] / sin_q5, r36[0, 2] / sin_q5))
                else:
                    q4 = float(q4_reference)
                    if r00 >= 0.0:
                        q4_plus_q6 = float(np.arctan2(r36[1, 2], r36[1, 1]))
                        q6 = q4_plus_q6 - q4
                    else:
                        q4_minus_q6 = float(np.arctan2(-r36[1, 2], r36[1, 1]))
                        q6 = q4 - q4_minus_q6
                q_sol = np.array([q1, q2, q3, q4, q5, q6], dtype=float)
                q_sol = (q_sol + np.pi) % (2.0 * np.pi) - np.pi
                solutions.append(q_sol)

    return solutions


class _KukaOpcUaBridge:
    """Synchronous facade over an ``asyncua.Client`` running in a dedicated background thread."""

    def __init__(
        self,
        ip_address: str,
        port: int = DEFAULT_OPCUA_PORT,
        username: str = DEFAULT_OPCUA_USERNAME,
        password: str = DEFAULT_OPCUA_PASSWORD,
        program_name: str = DEFAULT_KRL_PROGRAM,
        timeout: float = 10.0,
    ) -> None:
        self.ip_address = ip_address
        self.port = port
        self.username = username
        self.password = password
        self.program_name = program_name.upper()
        self.timeout = timeout

        self._asyncua = _import_asyncua()
        self._ua = self._asyncua.ua
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="kuka-opcua-loop", daemon=True)
        self._thread.start()
        self._client: Any = None
        self._connected = False

        try:
            self._run(self._async_connect())
        except Exception:
            self.close()
            raise

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run(self, coro: Any) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=self.timeout)

    def _prog_var_node_id(self, var_name: str) -> str:
        return f"ns=8;s=ns=7%3Bi=5004??krlvar:/R1/Program/{self.program_name}#{var_name}"

    async def _async_connect(self) -> None:
        url = f"opc.tcp://{self.ip_address}:{self.port}"
        self._client = self._asyncua.Client(url=url, timeout=self.timeout)
        self._client.set_user(self.username)
        self._client.set_password(self.password)
        await self._client.connect()
        self._connected = True

        self._axis_nodes = [self._client.get_node(f"{_SYSVAR_PREFIX}$AXIS_ACT.A{i}") for i in range(1, 7)]
        self._pos_nodes = [self._client.get_node(f"{_SYSVAR_PREFIX}$POS_ACT.{c}") for c in ("X", "Y", "Z", "A", "B", "C")]
        self._py_axis_nodes = [self._client.get_node(self._prog_var_node_id(f"PY_A{i}")) for i in range(1, 7)]
        self._py_vel_node = self._client.get_node(self._prog_var_node_id("PY_VEL"))
        self._py_cmd_id_node = self._client.get_node(self._prog_var_node_id("PY_CMD_ID"))
        self._py_ack_id_node = self._client.get_node(self._prog_var_node_id("PY_ACK_ID"))
        self._py_ready_node = self._client.get_node(self._prog_var_node_id("PY_READY"))
        self._py_moving_node = self._client.get_node(self._prog_var_node_id("PY_MOVING"))
        self._mode_op_node = self._client.get_node(f"{_SYSVAR_PREFIX}$MODE_OP")
        self._ov_pro_node = self._client.get_node(f"{_SYSVAR_PREFIX}$OV_PRO")

    async def _write_node(self, node: Any, value: Any, variant_type: Any) -> None:
        data_value = self._ua.DataValue(self._ua.Variant(value, variant_type))
        await node.write_value(data_value)

    def read_controller_metadata(self) -> dict[str, Any]:
        """Read model name, joint limits, active ``$BASE`` and ``$TOOL``, and kinematic geometry."""
        return self._run(self._async_read_controller_metadata())

    async def _async_read_controller_metadata(self) -> dict[str, Any]:
        try:
            model_raw = await self._client.get_node(f"{_SYSVAR_PREFIX}$TRAFONAME[]").read_value()
            model = str(model_raw).strip().lstrip("#")
        except Exception:
            model = "KR3R540"

        try:
            softn_nodes = [self._client.get_node(f"{_SYSVAR_PREFIX}$SOFTN_END[{i}]") for i in range(1, 7)]
            softp_nodes = [self._client.get_node(f"{_SYSVAR_PREFIX}$SOFTP_END[{i}]") for i in range(1, 7)]
            lower_deg = [float(x) for x in await self._client.read_values(softn_nodes)]
            upper_deg = [float(x) for x in await self._client.read_values(softp_nodes)]
        except Exception:
            lower_deg = list(KR3_R540_DEFAULT_LOWER_LIMITS_DEG)
            upper_deg = list(KR3_R540_DEFAULT_UPPER_LIMITS_DEG)

        base_pose = await self._async_read_frame("$BASE")
        tool_pose = await self._async_read_frame("$TOOL")
        geometry = await self._async_read_geometry()

        return {
            "model": model,
            "lower_limits_deg": np.asarray(lower_deg, dtype=float),
            "upper_limits_deg": np.asarray(upper_deg, dtype=float),
            "base_pose": base_pose,
            "tool_pose": tool_pose,
            "geometry": geometry,
        }

    async def _async_read_frame(self, frame_var: str) -> np.ndarray:
        try:
            nodes = [self._client.get_node(f"{_SYSVAR_PREFIX}{frame_var}.{c}") for c in ("X", "Y", "Z", "A", "B", "C")]
            values = [float(x) for x in await self._client.read_values(nodes)]
            return np.asarray(values, dtype=float)
        except Exception:
            return np.zeros(6, dtype=float)

    async def _async_read_geometry(self) -> KukaKinematicGeometry:
        try:
            nodes = [
                self._client.get_node(f"{_SYSVAR_PREFIX}$TIRORO.Z"),
                self._client.get_node(f"{_SYSVAR_PREFIX}$LENGTH_A"),
                self._client.get_node(f"{_SYSVAR_PREFIX}$LENGTH_B"),
                self._client.get_node(f"{_SYSVAR_PREFIX}$TX3P3.X"),
                self._client.get_node(f"{_SYSVAR_PREFIX}$TX3P3.Z"),
                self._client.get_node(f"{_SYSVAR_PREFIX}$TFLWP.Z"),
            ]
            vals = [float(x) for x in await self._client.read_values(nodes)]
            return KukaKinematicGeometry(
                tiroro_z_mm=vals[0],
                length_a_mm=vals[1],
                length_b_mm=vals[2],
                tx3p3_x_mm=vals[3],
                tx3p3_z_mm=vals[4],
                tflwp_z_mm=vals[5],
            )
        except Exception:
            return KukaKinematicGeometry()

    def get_joint_degrees(self) -> np.ndarray:
        """Read ``$AXIS_ACT.A1..A6`` in degrees."""
        values = self._run(self._client.read_values(self._axis_nodes))
        return np.asarray([float(v) for v in values], dtype=float)

    def get_kuka_tcp_pose(self) -> np.ndarray:
        """Read ``$POS_ACT.X, Y, Z, A, B, C`` in millimetres and degrees."""
        values = self._run(self._client.read_values(self._pos_nodes))
        return np.asarray([float(v) for v in values], dtype=float)

    def upload_krl_program(self) -> None:
        """Upload ``<program_name>.dat`` and ``<program_name>.src`` into ``KRC:/R1/Program`` over OPC-UA."""
        self._run(self._async_upload_krl_program())

    async def _async_upload_krl_program(self) -> None:
        prog_dir = self._client.get_node(_KRLFILE_PROGRAM_DIR)
        delete_method = self._client.get_node(f"{_KRLFILE_PROGRAM_DIR}#Delete")
        create_method = self._client.get_node(f"{_KRLFILE_PROGRAM_DIR}#CreateFile")
        files = {
            f"{self.program_name.lower()}.dat": KRL_DAT_TEMPLATE.format(program_name=self.program_name).replace(
                "\n", "\r\n"
            ),
            f"{self.program_name.lower()}.src": KRL_SRC_TEMPLATE.format(program_name=self.program_name).replace(
                "\n", "\r\n"
            ),
        }
        existing_children = {
            (await child.read_browse_name()).Name.lower(): child for child in await prog_dir.get_children()
        }
        for filename, content in files.items():
            if filename.lower() in existing_children:
                await prog_dir.call_method(delete_method, existing_children[filename.lower()].nodeid)
            res = await prog_dir.call_method(
                create_method,
                self._ua.Variant(filename, self._ua.VariantType.String),
                self._ua.Variant(True, self._ua.VariantType.Boolean),
            )
            file_node_out, handle = res[0], res[1]
            file_node = self._client.get_node(file_node_out)
            write_method = self._client.get_node(f"{file_node_out.to_string()}#Write")
            close_method = self._client.get_node(f"{file_node_out.to_string()}#Close")
            try:
                await file_node.call_method(
                    write_method,
                    self._ua.Variant(handle, self._ua.VariantType.UInt32),
                    self._ua.Variant(content.encode("ascii"), self._ua.VariantType.ByteString),
                )
            finally:
                await file_node.call_method(
                    close_method,
                    self._ua.Variant(handle, self._ua.VariantType.UInt32),
                )

    def send_joint_command(self, joints_degrees: np.ndarray, velocity_percent: int) -> int:
        """Send a joint setpoint to the running KRL server program and return its command ID."""
        return int(self._run(self._async_send_joint_command(joints_degrees, velocity_percent)))

    async def _async_send_joint_command(self, joints_degrees: np.ndarray, velocity_percent: int) -> int:
        try:
            ready, current_ack, prog_state = await self._client.read_values(
                [
                    self._py_ready_node,
                    self._py_ack_id_node,
                    self._client.get_node(_TASK_PROGRAM_STATE_NODE),
                ]
            )
        except Exception as exception:
            raise RuntimeError(
                f"KRL program {self.program_name!r} is not loaded or accessible on the KUKA controller. "
                f"Upload and start /R1/{self.program_name} on the smartPAD (see kuka_setup.md)."
            ) from exception

        if not bool(ready):
            try:
                task_name = await self._client.get_node(_TASK_PROGRAM_NAME_NODE).read_value()
                state_detail = f" (TaskProgramName={task_name!r}, ProgramState={prog_state!r})"
            except Exception:
                state_detail = ""
            raise RuntimeError(
                f"KRL program /R1/{self.program_name} is not initialized (PY_READY is False){state_detail}. "
                f"Select /R1/{self.program_name} on the smartPAD, confirm drives are green (I), and start the program."
            )

        if str(prog_state) != "P_Active":
            logger.warning(
                f"KRL program /R1/{self.program_name} is currently paused (ProgramState={prog_state!r}). "
                "In T1 mode, hold the rear enabling switch + green Start (+) button on the smartPAD; "
                "in AUT mode, turn drives on (I green) and press Start (+) once so R is green."
            )

        try:
            await self._write_node(self._ov_pro_node, 100, self._ua.VariantType.Int32)
        except Exception:
            pass

        for node, angle_deg in zip(self._py_axis_nodes, joints_degrees):
            await self._write_node(node, float(angle_deg), self._ua.VariantType.Float)
        await self._write_node(self._py_vel_node, int(velocity_percent), self._ua.VariantType.Int32)
        next_cmd_id = int(current_ack) + 1
        await self._write_node(self._py_cmd_id_node, next_cmd_id, self._ua.VariantType.Int32)
        return next_cmd_id

    def is_command_completed(self, cmd_id: int) -> bool:
        """Return ``True`` when the KRL program has acknowledged ``cmd_id`` and is no longer moving."""
        ack_id, moving = self._run(self._client.read_values([self._py_ack_id_node, self._py_moving_node]))
        return int(ack_id) >= int(cmd_id) and not bool(moving)

    def wait_until_krl_ready_and_idle(self, timeout: float = 120.0) -> None:
        """Block until the KRL server program is running (`P_Active`), ready (`PY_READY`), and idle."""
        deadline = time.time() + timeout
        logged_hint = False
        while time.time() < deadline:
            ready, moving, cmd_id, ack_id, prog_state = self._run(
                self._client.read_values(
                    [
                        self._py_ready_node,
                        self._py_moving_node,
                        self._py_cmd_id_node,
                        self._py_ack_id_node,
                        self._client.get_node(_TASK_PROGRAM_STATE_NODE),
                    ]
                )
            )
            if bool(ready) and not bool(moving) and int(cmd_id) == int(ack_id) and str(prog_state) == "P_Active":
                return
            if not logged_hint:
                logger.info(
                    f"Waiting for KRL program /R1/{self.program_name} to become active and idle "
                    f"(ProgramState={prog_state!r}, PY_READY={ready}, PY_MOVING={moving}, "
                    f"PY_CMD_ID={cmd_id}, PY_ACK_ID={ack_id}). "
                    "Hold the rear grey enabling switch (I green) + green Play (+) button (R green) on the smartPAD."
                )
                logged_hint = True
            time.sleep(0.2)
        raise TimeoutError(
            f"Timed out after {timeout:.1f}s waiting for /R1/{self.program_name} to become active on the smartPAD."
        )

    def close(self) -> None:
        """Disconnect the OPC-UA client and stop the background event loop thread."""
        if self._connected and self._client is not None:
            try:
                self._run(self._client.disconnect())
            except Exception as exception:
                logger.debug(f"Ignoring error while disconnecting KUKA OPC-UA client: {exception!r}")
            self._connected = False
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)


def _create_opcua_bridge(
    ip_address: str,
    port: int,
    username: str,
    password: str,
    program_name: str,
    timeout: float,
) -> Any:
    """Create the OPC-UA bridge instance (monkeypatch target for unit tests)."""
    return _KukaOpcUaBridge(
        ip_address=ip_address,
        port=port,
        username=username,
        password=password,
        program_name=program_name,
        timeout=timeout,
    )


class Kuka(PositionManipulator):
    """Control a KUKA KRC4 robot (such as the KR 3 R540) over ``KUKA.DeviceConnector`` OPC-UA.

    AIRO represents joint configurations in radians and Cartesian poses as 4x4 homogeneous matrices
    in metres. KUKA represents joint angles in degrees and Cartesian poses as
    ``[X, Y, Z, A, B, C]`` in millimetres and degrees, where ``A`` is rotation around Z, ``B``
    around Y, and ``C`` around X (``R = Rz(A) @ Ry(B) @ Rx(C)``). This class converts between
    the two conventions at the API boundary.

    State queries (:meth:`get_joint_configuration` and :meth:`get_tcp_pose`) and analytical
    kinematics (:meth:`forward_kinematics` and :meth:`inverse_kinematics`) work as soon as the
    controller is reachable over OPC-UA. Motion commands dispatch setpoints to the small KRL
    motion server program (``PY_SLOW`` by default) running in ``KRC:\\R1\\Program``; see
    ``kuka_setup.md`` next to this module for smartPAD setup instructions.

    Args:
        ip_address: IP address of the KUKA controller (e.g. ``"192.168.1.2"``).
        port: OPC-UA server port on the controller. Defaults to ``4840``.
        username: OPC-UA username with read/write variable access. Defaults to ``"OpcUaSecurityAdmin"``.
        password: OPC-UA password. Defaults to ``"kuka"``.
        program_name: Name of the KRL motion server module in ``KRC:/R1/Program``. Defaults to ``"PY_SLOW"``.
        upload_krl_program: If ``True``, automatically upload ``<program_name>.dat`` and ``.src`` into
            ``KRC:/R1/Program`` upon connecting. Defaults to ``False``.
        manipulator_specs: Optional specification override. Defaults to the KUKA KR 3 R540 datasheet
            velocity limits.
        gripper: Optional parallel position gripper attached to the robot.
        timeout: OPC-UA request timeout in seconds. Defaults to ``10.0``.
    """

    def __init__(
        self,
        ip_address: str,
        port: int = DEFAULT_OPCUA_PORT,
        username: str = DEFAULT_OPCUA_USERNAME,
        password: str = DEFAULT_OPCUA_PASSWORD,
        program_name: str = DEFAULT_KRL_PROGRAM,
        upload_krl_program: bool = False,
        manipulator_specs: ManipulatorSpecs | None = None,
        gripper: ParallelPositionGripper | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.ip_address = ip_address
        self.port = port
        self.program_name = program_name.upper()
        self._bridge = _create_opcua_bridge(
            ip_address=ip_address,
            port=port,
            username=username,
            password=password,
            program_name=self.program_name,
            timeout=timeout,
        )
        self._closed = False

        if upload_krl_program:
            self._bridge.upload_krl_program()

        metadata = self._bridge.read_controller_metadata()
        self.dof = KUKA_DOF
        self.model = str(metadata["model"])
        self._geometry: KukaKinematicGeometry = metadata["geometry"]
        self._base_transform = self._convert_kuka_pose_to_homogeneous_pose(
            np.asarray(metadata["base_pose"], dtype=float)
        )
        self._tool_transform = self._convert_kuka_pose_to_homogeneous_pose(
            np.asarray(metadata["tool_pose"], dtype=float)
        )
        self._base_transform_inv = np.linalg.inv(self._base_transform)
        self._tool_transform_inv = np.linalg.inv(self._tool_transform)

        if manipulator_specs is None:
            manipulator_specs = ManipulatorSpecs(
                max_joint_speeds=np.radians(KR3_R540_MAX_JOINT_SPEEDS_DEG_S).tolist(),
                max_linear_speed=KR3_R540_MAX_LINEAR_SPEED_M_S,
            )
        if manipulator_specs.dof != self.dof:
            self.close()
            raise ValueError(
                f"Manipulator specs have {manipulator_specs.dof} joints, but the connected robot has {self.dof}."
            )

        super().__init__(manipulator_specs, gripper)

        self._joint_lower_limits = np.radians(np.asarray(metadata["lower_limits_deg"], dtype=float))
        self._joint_upper_limits = np.radians(np.asarray(metadata["upper_limits_deg"], dtype=float))
        self._validate_controller_array(self._joint_lower_limits, "minimum joint limits")
        self._validate_controller_array(self._joint_upper_limits, "maximum joint limits")

        self._pose_reached_l2_threshold = 0.01
        self._joint_configuration_reached_l2_threshold = 0.01

        logger.info(
            f"Connected to KUKA robot model {self.model} at {self.ip_address}:{self.port} ({self.dof} DoF)."
        )

    def close(self) -> None:
        """Close the OPC-UA connection to the KUKA controller."""
        if self._closed:
            return
        self._closed = True
        self._bridge.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exception_type: Any, exception: Any, traceback: Any) -> None:
        self.close()

    def upload_krl_program(self) -> None:
        """Upload the KRL motion server module (``.dat`` and ``.src``) to ``KRC:/R1/Program``."""
        self._bridge.upload_krl_program()

    def wait_until_ready(self, timeout: float = 120.0) -> None:
        """Wait until the KRL motion server module on the controller is active (`P_Active`) and idle."""
        self._bridge.wait_until_krl_ready_and_idle(timeout=timeout)

    def get_joint_configuration(self) -> JointConfigurationType:
        """Get the current measured joint configuration in radians."""
        joint_degrees = np.asarray(self._bridge.get_joint_degrees(), dtype=float)
        self._validate_controller_array(joint_degrees, "joint configuration")
        return np.radians(joint_degrees)

    def get_tcp_pose(self) -> HomogeneousMatrixType:
        """Get the current TCP pose in the active ``$BASE`` frame as a 4x4 homogeneous matrix (metres)."""
        kuka_pose = np.asarray(self._bridge.get_kuka_tcp_pose(), dtype=float)
        return self._convert_kuka_pose_to_homogeneous_pose(kuka_pose)

    def move_to_joint_configuration(
        self,
        joint_configuration: JointConfigurationType,
        joint_speed: float | None = None,
    ) -> AwaitableAction:
        """Move to a target joint configuration using controller PTP planning."""
        self._assert_joint_configuration_is_valid(joint_configuration)
        speed = self.default_joint_speed if joint_speed is None else joint_speed
        self._assert_positive_speed(speed, "joint_speed")
        self._assert_joint_speed_is_valid(speed)
        velocity_percent = self._speed_to_percentage(speed, min(self.manipulator_specs.max_joint_speeds))

        target = np.asarray(joint_configuration, dtype=float)
        cmd_id = self._bridge.send_joint_command(np.degrees(target), velocity_percent)
        return AwaitableAction(
            lambda: self._bridge.is_command_completed(cmd_id)
            and bool(
                np.linalg.norm(self.get_joint_configuration() - target)
                < self._joint_configuration_reached_l2_threshold
            )
        )

    def move_to_tcp_pose(
        self,
        tcp_pose: HomogeneousMatrixType,
        joint_speed: float | None = None,
    ) -> AwaitableAction:
        """Move the TCP to a target pose in joint space."""
        self._assert_pose_is_valid(tcp_pose)
        speed = self.default_joint_speed if joint_speed is None else joint_speed
        self._assert_positive_speed(speed, "joint_speed")
        self._assert_joint_speed_is_valid(speed)
        velocity_percent = self._speed_to_percentage(speed, min(self.manipulator_specs.max_joint_speeds))

        joint_target = self.inverse_kinematics(tcp_pose)
        if joint_target is None or not self._is_joint_configuration_reachable(joint_target):
            raise RobotConfigurationException(
                f"pose {tcp_pose} is not reachable, could be because of kinematic constraints or safety constraints"
            )
        cmd_id = self._bridge.send_joint_command(np.degrees(joint_target), velocity_percent)
        return self._pose_motion_action(cmd_id, tcp_pose)

    def move_linear_to_tcp_pose(
        self,
        tcp_pose: HomogeneousMatrixType,
        linear_speed: float | None = None,
    ) -> AwaitableAction:
        """Move the TCP to a target pose, verifying straight-line Cartesian reachability."""
        self._assert_pose_is_valid(tcp_pose)
        speed = self.default_linear_speed if linear_speed is None else linear_speed
        self._assert_positive_speed(speed, "linear_speed")
        self._assert_linear_speed_is_valid(speed)

        current_joints = self.get_joint_configuration()
        start_pose = self.forward_kinematics(current_joints)
        linear_distance = float(np.linalg.norm(tcp_pose[:3, 3] - start_pose[:3, 3]))
        num_checks = max(2, int(np.ceil(linear_distance / 0.02)))
        seed = current_joints
        for step in range(1, num_checks + 1):
            alpha = step / num_checks
            waypoint_pose = self._interpolate_se3(start_pose, tcp_pose, alpha)
            waypoint_joints = self.inverse_kinematics(waypoint_pose, seed)
            if waypoint_joints is None or not self._is_joint_configuration_reachable(waypoint_joints):
                raise RobotConfigurationException(
                    f"Cartesian straight-line path to {tcp_pose} is not reachable at interpolation fraction {alpha:.2f}."
                )
            seed = waypoint_joints

        velocity_percent = self._speed_to_percentage(speed, self.manipulator_specs.max_linear_speed)
        cmd_id = self._bridge.send_joint_command(np.degrees(seed), velocity_percent)
        return self._pose_motion_action(cmd_id, tcp_pose)

    def servo_to_joint_configuration(
        self,
        joint_configuration: JointConfigurationType,
        duration: float,
    ) -> AwaitableAction:
        """Send a best-effort joint setpoint over OPC-UA for the specified duration."""
        self._assert_positive_duration(duration)
        joint_configuration_array = np.asarray(joint_configuration, dtype=float)
        if joint_configuration_array.shape != (self.dof,):
            raise ValueError(f"Expected a joint configuration with shape ({self.dof},).")
        self._assert_joint_configuration_is_valid(joint_configuration_array)

        current_joints = self.get_joint_configuration()
        max_delta = float(np.max(np.abs(joint_configuration_array - current_joints)))
        estimated_speed = max(max_delta / duration, 1e-3)
        max_allowed_speed = min(self.manipulator_specs.max_joint_speeds)
        velocity_percent = self._speed_to_percentage(min(estimated_speed, max_allowed_speed), max_allowed_speed)
        self._bridge.send_joint_command(np.degrees(joint_configuration_array), velocity_percent)
        return self._duration_action(duration)

    def servo_to_tcp_pose(
        self,
        tcp_pose: HomogeneousMatrixType,
        duration: float,
    ) -> AwaitableAction:
        """Send a best-effort TCP pose setpoint over OPC-UA for the specified duration."""
        self._assert_positive_duration(duration)
        joint_target = self.inverse_kinematics(tcp_pose)
        if joint_target is None or not self._is_joint_configuration_reachable(joint_target):
            raise RobotConfigurationException(
                f"pose {tcp_pose} is not reachable, could be because of kinematic constraints or safety constraints"
            )
        return self.servo_to_joint_configuration(joint_target, duration)

    def forward_kinematics(self, joint_configuration: JointConfigurationType) -> HomogeneousMatrixType:
        """Calculate the TCP pose in the active ``$BASE`` frame for a given joint configuration (radians)."""
        joint_configuration_array = np.asarray(joint_configuration, dtype=float)
        if joint_configuration_array.shape != (self.dof,):
            raise ValueError(f"Expected a joint configuration with shape ({self.dof},).")
        flange_in_root = kuka_flange_forward_kinematics(joint_configuration_array, self._geometry)
        return self._base_transform_inv @ flange_in_root @ self._tool_transform

    def inverse_kinematics(
        self,
        tcp_pose: HomogeneousMatrixType,
        joint_configuration_near: JointConfigurationType | None = None,
    ) -> JointConfigurationType | None:
        """Solve analytical 6-DOF spherical-wrist inverse kinematics for ``tcp_pose``.

        Args:
            tcp_pose: Desired TCP pose as a 4x4 homogeneous matrix in the active ``$BASE`` frame (metres).
            joint_configuration_near: Reference joint configuration in radians used to select the closest
                branch. Defaults to the current joint configuration.

        Returns:
            The closest reachable joint configuration in radians, or ``None`` if no reachable solution exists.
        """
        if joint_configuration_near is None:
            joint_configuration_near = self.get_joint_configuration()
        near = np.asarray(joint_configuration_near, dtype=float)
        if near.shape != (self.dof,):
            raise ValueError(f"Expected a joint configuration with shape ({self.dof},).")

        flange_in_root = self._base_transform @ np.asarray(tcp_pose, dtype=float) @ self._tool_transform_inv
        raw_solutions = kuka_flange_inverse_kinematics_all(
            flange_in_root,
            self._geometry,
            q4_reference=float(near[3]),
        )
        if not raw_solutions:
            logger.warning(
                f"Inverse kinematics failed for TCP position {tcp_pose[:3, 3].round(4).tolist()} m: "
                "no analytical solution exists (target is out of reach)."
            )
            return None

        valid_candidates: list[np.ndarray] = []
        blocking_joints: set[int] = set()
        for base_solution in raw_solutions:
            # Adjust 2*pi multiples (e.g. A6 allows [-350, 350] deg) to pick the turn closest to `near`.
            adjusted = base_solution.copy()
            for idx in range(self.dof):
                best_angle = adjusted[idx]
                best_dist = abs(best_angle - near[idx])
                for shift in (-2.0 * np.pi, 2.0 * np.pi):
                    candidate_angle = base_solution[idx] + shift
                    if self._joint_lower_limits[idx] <= candidate_angle <= self._joint_upper_limits[idx]:
                        dist = abs(candidate_angle - near[idx])
                        if dist < best_dist:
                            best_angle = candidate_angle
                            best_dist = dist
                adjusted[idx] = best_angle

            if self._is_joint_configuration_reachable(adjusted):
                valid_candidates.append(adjusted)
            else:
                over_limit = np.nonzero((adjusted < self._joint_lower_limits) | (adjusted > self._joint_upper_limits))[
                    0
                ]
                blocking_joints.update(int(j) + 1 for j in over_limit)

        if not valid_candidates:
            joints_str = ", ".join(f"A{j}" for j in sorted(blocking_joints))
            logger.warning(
                f"Inverse kinematics failed for TCP position {tcp_pose[:3, 3].round(4).tolist()} m: "
                f"{len(raw_solutions)} analytical solution(s) exist, but all exceed joint limits "
                f"(blocking joint(s): {joints_str})."
            )
            return None

        best_solution = min(valid_candidates, key=lambda q: float(np.linalg.norm(q - near)))
        return best_solution

    def _is_joint_configuration_reachable(self, joint_configuration: JointConfigurationType) -> bool:
        joint_configuration_array = np.asarray(joint_configuration, dtype=float)
        return bool(
            joint_configuration_array.shape == (self.dof,)
            and np.all(joint_configuration_array >= self._joint_lower_limits)
            and np.all(joint_configuration_array <= self._joint_upper_limits)
        )

    def _pose_motion_action(self, cmd_id: int, target_pose: HomogeneousMatrixType) -> AwaitableAction:
        return AwaitableAction(
            lambda: self._bridge.is_command_completed(cmd_id)
            and bool(np.linalg.norm(self.get_tcp_pose() - target_pose) < self._pose_reached_l2_threshold)
        )

    def _validate_controller_array(self, values: np.ndarray, name: str) -> None:
        if values.shape != (self.dof,):
            raise RuntimeError(
                f"KUKA controller returned {name} with shape {values.shape}; expected ({self.dof},)."
            )

    @staticmethod
    def _interpolate_se3(
        start_pose: HomogeneousMatrixType,
        end_pose: HomogeneousMatrixType,
        alpha: float,
    ) -> HomogeneousMatrixType:
        slerp = Slerp([0.0, 1.0], Rotation.from_matrix([start_pose[:3, :3], end_pose[:3, :3]]))
        pose = np.eye(4, dtype=float)
        pose[:3, :3] = slerp([alpha])[0].as_matrix()
        pose[:3, 3] = (1.0 - alpha) * start_pose[:3, 3] + alpha * end_pose[:3, 3]
        return pose

    @staticmethod
    def _speed_to_percentage(speed: float, maximum_speed: float) -> int:
        return int(np.clip(np.ceil(100.0 * speed / maximum_speed), 1, 100))

    @staticmethod
    def _assert_positive_speed(speed: float, parameter_name: str) -> None:
        if speed <= 0:
            raise ValueError(f"{parameter_name} must be greater than zero.")

    @staticmethod
    def _assert_positive_duration(duration: float) -> None:
        if duration <= 0:
            raise ValueError("duration must be greater than zero.")

    @staticmethod
    def _duration_action(duration: float) -> AwaitableAction:
        action_sent_time = time.perf_counter_ns()
        return AwaitableAction(
            lambda: time.perf_counter_ns() - action_sent_time > duration * 1e9,
            default_timeout=2 * duration,
            default_sleep_resolution=0.002,
        )

    @staticmethod
    def _convert_kuka_pose_to_homogeneous_pose(kuka_pose: KukaPoseType) -> HomogeneousMatrixType:
        """Convert KUKA ``[X_mm, Y_mm, Z_mm, A_deg, B_deg, C_deg]`` to a 4x4 homogeneous matrix (metres).

        KUKA defines ``R = Rz(A) @ Ry(B) @ Rx(C)``, which matches extrinsic XYZ Euler angles
        ``[roll=rad(C), pitch=rad(B), yaw=rad(A)]`` in ``SE3Container.from_euler_angles_and_translation``.
        """
        if kuka_pose.shape != (6,):
            raise ValueError(f"Expected a KUKA pose with shape (6,), received {kuka_pose.shape}.")
        translation_m = kuka_pose[:3] / 1000.0
        euler_xyz_rad = np.radians([kuka_pose[5], kuka_pose[4], kuka_pose[3]])
        return SE3Container.from_euler_angles_and_translation(euler_xyz_rad, translation_m).homogeneous_matrix

    @staticmethod
    def _convert_homogeneous_pose_to_kuka_pose(homogeneous_pose: HomogeneousMatrixType) -> KukaPoseType:
        """Convert a 4x4 homogeneous matrix (metres) to KUKA ``[X_mm, Y_mm, Z_mm, A_deg, B_deg, C_deg]``."""
        pose = SE3Container.from_homogeneous_matrix(homogeneous_pose)
        translation_mm = pose.translation * 1000.0
        roll_c, pitch_b, yaw_a = np.degrees(pose.orientation_as_euler_angles)
        return np.array([translation_mm[0], translation_mm[1], translation_mm[2], yaw_a, pitch_b, roll_c], dtype=float)


if __name__ == "__main__":
    import click
    from airo_robots.manipulators.hardware.manual_manipulator_testing import manual_test_robot

    @click.command()
    @click.option("--ip-address", required=True, help="IP address of the KUKA controller (e.g. 192.168.1.2).")
    @click.option("--port", default=DEFAULT_OPCUA_PORT, show_default=True, help="KUKA OPC-UA port.")
    @click.option(
        "--upload-krl",
        is_flag=True,
        default=False,
        help="Upload the PY_SLOW KRL motion server module to KRC:/R1/Program before testing.",
    )
    def test_kuka(ip_address: str, port: int, upload_krl: bool) -> None:
        """Run the manual manipulator tests against a KUKA robot."""
        with Kuka(ip_address, port=port, upload_krl_program=upload_krl) as robot:
            manual_test_robot(robot)

    test_kuka()
