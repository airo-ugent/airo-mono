import time
from collections import deque
from multiprocessing import Array, Lock, Process, Value
from multiprocessing.sharedctypes import Synchronized, SynchronizedArray
from multiprocessing.synchronize import Lock as LockType
from typing import List, Optional

import numpy as np
from airo_robots.grippers import ParallelPositionGripper

# from airo_robots.manipulators.position_manipulator import ManipulatorSpecs, PositionManipulator
from airo_robots.manipulators.position_manipulator import ManipulatorSpecs
from airo_spatial_algebra import SE3Container
from airo_typing import HomogeneousMatrixType, JointConfigurationType
from loguru import logger
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

RotVecPoseType = np.ndarray
""" a 6D pose [tx,ty,tz,rotvecx,rotvecy,rotvecz]"""


"""
How to use:

from airo_robots.manipulators.hardware.ur_rtde_torque import URrtdeTorque as URrtde
robot = URrtde("10.42.0.162", URrtde.UR3E_CONFIG,initial_joint_configuration=np.array([-1.58, - 1.74, - 0.71, - 1.51 , 1.47,  3.109]))
robot.enable_torque_control()

    while True:
        robot.target_pos = np.array(target_joint_configuration)
        #don't move too much in every step!
"""


# Max log entries: ~3 minutes at 500Hz
MAX_LOG_ENTRIES = 90000


def _torque_worker(
    ur_ip: str,
    torque_limit: List[float],
    target_pos_shared: SynchronizedArray,
    pos_cache_shared: SynchronizedArray,
    tcp_cache_shared: SynchronizedArray,
    force_cache_shared: SynchronizedArray,
    running_flag: Synchronized,
    shared_lock: LockType,
) -> None:

    """
    Executes a real-time PD torque control loop (500Hz) for the UR manipulator.

    This function acts as a worker process that calculates and sends torque commands
    to the robot controller via the RTDE interface. It uses a PD controller to
    drive the robot towards the target positions found in the shared memory.

    It also synchronizes the robot's current state (joint positions, TCP pose, TCP force)
    back to shared memory for the main process to read.

    Args:
        ur_ip: The IP address of the UR robot.
        target_pos_shared: Shared memory array (size 6) containing
            the desired joint positions. The worker reads from this.
        pos_cache_shared: Shared memory array (size 6) where the
            worker writes the current joint positions (radians).
        tcp_cache_shared: Shared memory array (size 6) where the
            worker writes the current TCP pose [x, y, z, rx, ry, rz].
        force_cache_shared: Shared memory array (size 6) where the
            worker writes the current TCP force [Fx, Fy, Fz, Tx, Ty, Tz].
        running_flag: Shared boolean flag. The loop runs as long
            as this is True. Set to False from the main process to stop the worker.

    Returns:
        None
    """
    freq = 500.0
    rtde_c = RTDEControlInterface(ur_ip, freq)
    rtde_r = RTDEReceiveInterface(ur_ip, freq)
    dt = 1 / freq
    # max_vel = 1
    # max_step = max_vel * dt
    # need tuning
    max_torque = np.asarray(torque_limit) * 0.8

    Kd = np.array([12.0, 12.0, 10.0, 2.4, 2.0, 1.0])
    Kp = np.array([120.0, 120.0, 100.0, 30.0, 30.0, 30.0])
    # Kd = np.array([10.0, 10.0, 8.0, 1.5, 1.5, 1.0])

    # Ki = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    print(f"New Kp: {Kp}")
    print(f"New Kd: {Kd}")

    log_buffer = deque(maxlen=MAX_LOG_ENTRIES)
    # alpha = 1.0 no filter
    # vel_filter_alpha = 0.8
    t0 = time.time()
    # internal_target = np.array(rtde_r.getActualQ())
    # vel_filter_alpha = 0.5
    # qd_target_prev = np.zeros(6)

    # Omega: Natural Frequency， Zeta: Damping Ratio
    traj_omega = 40
    traj_zeta = 1.05
    # inner state
    ref_pos = np.array(rtde_r.getActualQ())
    ref_vel = np.zeros(6)

    # Filter parameter (alpha=0.1 means 10% new value, 90% history to smooth noise)
    qd_act_filtered = np.zeros(6)
    act_filter_alpha = 0.1

    try:
        while running_flag.value:
            t_start = rtde_c.initPeriod()
            q_act = np.array(rtde_r.getActualQ())
            qd_act = np.array(rtde_r.getActualQd())
            tcp_pose = np.array(rtde_r.getActualTCPPose())
            tcp_force = np.array(rtde_r.getActualTCPForce())
            with shared_lock:
                for i in range(6):
                    pos_cache_shared[i] = q_act[i]
                    tcp_cache_shared[i] = tcp_pose[i]
                    force_cache_shared[i] = tcp_force[i]
                target = np.array([target_pos_shared[i] for i in range(6)])

            qd_act_filtered = act_filter_alpha * qd_act + (1 - act_filter_alpha) * qd_act_filtered

            # Second-Order Trajectory Generator
            # a_ref = w^2 * (target - p) - 2*z*w * v
            pos_err_traj = target - ref_pos
            ref_acc = (traj_omega * traj_omega * pos_err_traj) - (2.0 * traj_zeta * traj_omega * ref_vel)

            ref_vel = ref_vel + ref_acc * dt
            ref_pos = ref_pos + ref_vel * dt

            q_err = ref_pos - q_act
            qd_err = ref_vel - qd_act_filtered

            torque_p = Kp * q_err
            torque_d = Kd * qd_err

            torque_d = np.clip(torque_d, -0.8 * max_torque, 0.8 * max_torque)
            torque_target = torque_p + torque_d
            torque_target = np.clip(torque_target, -max_torque, max_torque)

            time.time() - t0

            # save log for tuning
            # log_buffer.append(np.concatenate([[t_now], target, q_act, torque_p, torque_d, torque_target]))

            rtde_c.directTorque(torque_target.tolist())
            rtde_c.waitPeriod(t_start)
    except KeyboardInterrupt:
        logger.info("Control loop interrupted by user.")

    except Exception as e:
        logger.error(f"Unexpected error in torque worker: {e}")

    finally:

        try:
            if len(log_buffer) > 0:
                logger.info(f"Saving {len(log_buffer)} log entries to 'torque_log.npz'...")
                data_array = np.array(log_buffer)
                np.savez("torque_log.npz", data=data_array)
                logger.info("Log saved successfully.")
            else:
                logger.warning("Log buffer is empty, nothing to save.")
            zero_torque = [0.0] * 6
            for _ in range(20):
                t_start = rtde_c.initPeriod()
                rtde_c.directTorque(zero_torque)
                rtde_c.waitPeriod(t_start)
            rtde_c.stopScript()
            rtde_c.disconnect()
            rtde_r.disconnect()
        except Exception as e:
            logger.warning(f"Failed to stop Torque Loop: {e}")


class URrtdeTorque:
    """Implementation of the Position-controlled manipulator class for UR robots.
    This Implementation uses the ur-rtde library to address the robot's RTDE API.

    Configuring the TCP should be done manually on the robot, so that IK can be done in the TCP frame instead of the flange frame.

    No (self-) collision checking nor obstacle avoidance is performed.

    As a quick reminder: the UR e-series can be controlled at up to 500hz, the non-e-series up to 125Hz.
     This is mostly relevant for the servo methods.

    Finally, the ur-rtde library has more functionality than is exposed in the interface, you can always address the
    control/receive interface attributes directly if you need them.
    """

    # ROBOT SPEC CONFIGURATIONS

    # https://www.universal-robots.com/media/1807464/ur3e-rgb-fact-sheet-landscape-a4.pdf
    # Torque values: https://www.universal-robots.com/articles/ur/robot-care-maintenance/max-joint-torques-cb3-and-e-series/
    UR3E_CONFIG = ManipulatorSpecs([1.0, 1.0, 1.0, 2.0, 2.0, 2.0], 1.0, [54.0, 54.0, 28.0, 9.0, 9.0, 9.0])
    # https://www.universal-robots.com/media/240787/ur3_us.pdf
    UR3_CONFIG = ManipulatorSpecs([1.0, 1.0, 1.0, 2.0, 2.0, 2.0], 1.0, [54.0, 54.0, 28.0, 9.0, 9.0, 9.0])
    UR5E_CONFIG = ManipulatorSpecs([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 1.0, [150.0, 150.0, 150.0, 28.0, 28.0, 28.0])

    def __init__(
        self,
        ip_address: str,
        manipulator_specs: ManipulatorSpecs,
        gripper: Optional[ParallelPositionGripper] = None,
        initial_joint_configuration: Optional[JointConfigurationType] = None,
    ) -> None:
        """
        Args:
            initial_joint (Optional[JointConfigurationType]): A specific joint configuration to move the robot
                BEFORE starting the torque control loop.
                If provided, the robot will perform a blocking move to this position during initialization.
                This is highly recommended in torque_mode to ensure the robot starts from a known, safe configuration.
        """
        self.default_torque = manipulator_specs.max_torque
        self.gripper = gripper
        self.ip_address = ip_address
        try:
            if manipulator_specs.max_torque is None:
                raise ValueError("Cannot enable torque mode without setting the maximum allowed torques.")

            self._torque_process: Optional[Process] = None
            self._shared_lock = Lock()
            self.target_pos_shared = Array("d", [0.0] * 6)
            self.pos_cache_shared = Array("d", [0.0] * 6)
            self.tcp_cache_shared = Array("d", [0.0] * 6)
            self.force_cache_shared = Array("d", [0.0] * 6)
            self.running_flag = Value("b", False)
            self.tmp_move(initial_joint_configuration)

        except RuntimeError:
            raise RuntimeError(
                "Could not connect to the robot. Is the robot in remote control? Is the IP correct? Is the network connection ok?"
            )

    def tmp_move(self, target_joint_configuration):
        self.disable_torque_control()
        tmp_recv = RTDEReceiveInterface(self.ip_address)
        tmp_ctrl = RTDEControlInterface(self.ip_address)
        try:
            if target_joint_configuration is not None:
                tmp_ctrl.moveJ(target_joint_configuration)
            q0 = np.array(tmp_recv.getActualQ(), dtype=float)
            tcp0 = np.array(tmp_recv.getActualTCPPose(), dtype=float)
            for i in range(6):
                self.target_pos_shared[i] = q0[i]
                self.pos_cache_shared[i] = q0[i]
            for i in range(6):
                self.tcp_cache_shared[i] = tcp0[i]
        finally:
            tmp_recv.disconnect()
            tmp_ctrl.disconnect()

        self.enable_torque_control()

    @staticmethod
    def _convert_rotvec_pose_to_homogeneous_pose(ur_pose: RotVecPoseType) -> HomogeneousMatrixType:
        return SE3Container.from_rotation_vector_and_translation(ur_pose[3:], ur_pose[:3]).homogeneous_matrix

    @staticmethod
    def _convert_homogeneous_pose_to_rotvec_pose(homogeneous_pose: HomogeneousMatrixType) -> RotVecPoseType:
        se3 = SE3Container.from_homogeneous_matrix(homogeneous_pose)
        rotation = se3.orientation_as_rotation_vector
        translation = se3.translation
        return np.concatenate([translation, rotation])

    @property
    def target_pos(self) -> JointConfigurationType:
        with self._shared_lock:
            return np.array(self.target_pos_shared[:], dtype=float)

    @target_pos.setter
    def target_pos(self, q_target: JointConfigurationType) -> None:
        q_target = np.asarray(q_target, dtype=float)
        assert q_target.shape == (6,)
        with self._shared_lock:
            for i in range(6):
                self.target_pos_shared[i] = float(q_target[i])

    def get_cached_joint_configuration(self) -> JointConfigurationType:
        with self._shared_lock:
            return np.array(self.pos_cache_shared[:], dtype=float)

    def get_cached_tcp_pose(self) -> HomogeneousMatrixType:
        with self._shared_lock:
            return self._convert_rotvec_pose_to_homogeneous_pose(np.array(self.tcp_cache_shared[:], dtype=float))

    def get_cached_tcp_force(self) -> np.ndarray:
        """Returns cached TCP force/torque [Fx, Fy, Fz, Tx, Ty, Tz] in N and Nm."""
        with self._shared_lock:
            return np.array(self.force_cache_shared[:], dtype=float)

    def enable_torque_control(self) -> None:
        if self._torque_process is not None and self._torque_process.is_alive():
            return
        self.running_flag.value = True
        self._torque_process = Process(
            target=_torque_worker,
            args=(
                self.ip_address,
                self.default_torque,
                self.target_pos_shared,
                self.pos_cache_shared,
                self.tcp_cache_shared,
                self.force_cache_shared,
                self.running_flag,
                self._shared_lock,
            ),
            daemon=True,
        )
        self._torque_process.start()
        logger.info("Torque process started")

    def disable_torque_control(self) -> None:
        if self._torque_process is None:
            return
        logger.info("Stopping torque process...")
        self.running_flag.value = False
        self._torque_process.join(timeout=5.0)
        if self._torque_process.is_alive():
            logger.warning("Torque process is still alive!")
        else:
            logger.info("Torque process stopped successfully")
        self._torque_process = None
