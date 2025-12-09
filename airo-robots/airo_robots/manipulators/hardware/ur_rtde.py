import time
from multiprocessing import Array, Process, Value
from multiprocessing.sharedctypes import Synchronized, SynchronizedArray
from typing import Optional

import numpy as np
from airo_robots.awaitable_action import AwaitableAction
from airo_robots.grippers import ParallelPositionGripper
from airo_robots.manipulators.position_manipulator import ManipulatorSpecs, PositionManipulator
from airo_spatial_algebra import SE3Container
from airo_typing import HomogeneousMatrixType, JointConfigurationType, WrenchType
from loguru import logger
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

RotVecPoseType = np.ndarray
""" a 6D pose [tx,ty,tz,rotvecx,rotvecy,rotvecz]"""


def _torque_worker(
    ur_ip: str,
    torque_limit: WrenchType,
    target_pos_shared: SynchronizedArray,
    pos_cache_shared: SynchronizedArray,
    tcp_cache_shared: SynchronizedArray,
    running_flag: Synchronized,
    log_path: Optional[str] = "torque_log.csv",
) -> None:

    """
    Executes a real-time PD torque control loop (500Hz) for the UR manipulator.

    This function acts as a worker process that calculates and sends torque commands
    to the robot controller via the RTDE interface. It uses a PD controller to
    drive the robot towards the target positions found in the shared memory.

    It also synchronizes the robot's current state (joint positions, TCP pose) back to shared memory for the main process to read.

    Args:
        ur_ip: The IP address of the UR robot.
        target_pos_shared: Shared memory array (size 6) containing
            the desired joint positions. The worker reads from this.
        pos_cache_shared: Shared memory array (size 6) where the
            worker writes the current joint positions (radians).
        tcp_cache_shared: Shared memory array (size 6) where the
            worker writes the current TCP pose [x, y, z, rx, ry, rz].
        running_flag: Shared boolean flag. The loop runs as long
            as this is True. Set to False from the main process to stop the worker.
        log_path: Path to save the control loop data
            (targets, actuals, torques) as a CSV file. Defaults to "torque_log.csv".

    Returns:
        None
    """
    rtde_c = RTDEControlInterface(ur_ip, 500.0)
    rtde_r = RTDEReceiveInterface(ur_ip, 500.0)

    # need tuning
    max_torque = torque_limit * 0.8
    Kp = max_torque / np.array([0.25, 0.25, 0.5, 0.35, 0.3, 0.3])

    Kd_ratios = np.array([0.04, 0.04, 0.04, 0.04, 0.04, 0.04])
    Kd = Kd_ratios * Kp

    log_buffer = []
    t0 = time.time()
    qd_act_filtered = np.zeros(6)
    vel_filter_alpha = 0.12

    try:
        while running_flag.value:
            t_start = rtde_c.initPeriod()

            q_act = np.array(rtde_r.getActualQ(), dtype=float)
            qd_act = np.array(rtde_r.getActualQd(), dtype=float)
            tcp_pose = np.array(rtde_r.getActualTCPPose(), dtype=float)

            for i in range(6):
                pos_cache_shared[i] = q_act[i]
                tcp_cache_shared[i] = tcp_pose[i]

            target = np.array([target_pos_shared[i] for i in range(6)], dtype=float)

            # new
            qd_act_filtered = vel_filter_alpha * qd_act + (1 - vel_filter_alpha) * qd_act_filtered

            q_err = target - q_act
            for k in range(6):
                threshold = 0.005 if k < 3 else 0.002  #
                if abs(q_err[k]) < threshold:
                    q_err[k] = 0.0
            # qd_err = -qd_act
            qd_err = -qd_act_filtered
            torque_p = Kp * q_err
            torque_d = Kd * qd_err
            torque_d = np.clip(torque_d, -0.2 * max_torque, 0.2 * max_torque)
            torque_target = torque_p + torque_d
            torque_target = np.clip(torque_target, -max_torque, max_torque)

            t_now = time.time() - t0
            log_buffer.append(np.concatenate([[t_now], target, q_act, torque_p, torque_d, torque_target]))

            rtde_c.directTorque(torque_target.tolist())
            rtde_c.waitPeriod(t_start)

    finally:
        try:
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

        if log_path is not None and len(log_buffer) > 0:
            log_buffer = np.array(log_buffer)
            header = (
                ["time"]
                + [f"target_{i}" for i in range(6)]
                + [f"q_act_{i}" for i in range(6)]
                + [f"torque_p_{i}" for i in range(6)]
                + [f"torque_d_{i}" for i in range(6)]
                + [f"torque_cmd_{i}" for i in range(6)]
            )
            np.savetxt(log_path, log_buffer, delimiter=",", header=",".join(header), comments="")
            logger.success(f"Saved {len(log_buffer)} samples to {log_path}")


class URrtde(PositionManipulator):
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
    UR3E_CONFIG = ManipulatorSpecs([1.0, 1.0, 1.0, 2.0, 2.0, 2.0], 1.0, np.array([54.0, 54.0, 28.0, 9.0, 9.0, 9.0]))
    # https://www.universal-robots.com/media/240787/ur3_us.pdf
    UR5E_CONFIG = ManipulatorSpecs(
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 1.0, np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])
    )

    def __init__(
        self,
        ip_address: str,
        manipulator_specs: ManipulatorSpecs,
        gripper: Optional[ParallelPositionGripper] = None,
        torque_mode: bool = False,
        initial_joint: Optional[JointConfigurationType] = None,
    ) -> None:
        """
        Args:
            initial_joint (Optional[JointConfigurationType]): A specific joint configuration to move the robot
                BEFORE starting the torque control loop.
                If provided, the robot will perform a blocking move to this position during initialization.
                This is highly recommended in torque_mode to ensure the robot starts from a known, safe configuration.
        """
        super().__init__(manipulator_specs, gripper)
        self.ip_address = ip_address
        try:
            if torque_mode:
                self._torque_process = None
                self.torque_mode = True
                self.target_pos_shared = Array("d", [0.0] * 6)
                self.pos_cache_shared = Array("d", [0.0] * 6)
                self.tcp_cache_shared = Array("d", [0.0] * 6)
                self.running_flag = Value("b", False)
                tmp_recv = RTDEReceiveInterface(self.ip_address)
                tmp_ctrl = RTDEControlInterface(self.ip_address)
                if initial_joint is not None:
                    tmp_ctrl.moveJ(initial_joint)
                q0 = np.array(tmp_recv.getActualQ(), dtype=float)
                tcp0 = np.array(tmp_recv.getActualTCPPose(), dtype=float)
                for i in range(6):
                    self.target_pos_shared[i] = q0[i]
                    self.pos_cache_shared[i] = q0[i]
                for i in range(6):
                    self.tcp_cache_shared[i] = tcp0[i]
                tmp_recv.disconnect()
                tmp_ctrl.disconnect()

                self.enable_torque_control()

            else:
                self.rtde_control = RTDEControlInterface(self.ip_address)
                self.rtde_receive = RTDEReceiveInterface(self.ip_address)

        except RuntimeError:
            raise RuntimeError(
                "Could not connect to the robot. Is the robot in remote control? Is the IP correct? Is the network connection ok?"
            )

        self.default_linear_acceleration = 1.2  # m/s^2
        self.default_leading_axis_joint_acceleration = 1.2  # rad/s^2
        """
        Leading axis = the axis that has to move the most to reach the target configuration. All other joint accelerations
        will be scaled accordingly.
        """

        # sensible values but you might need to tweak them for your purposes.
        self.servo_proportional_gain = 100
        self.servo_lookahead_time = 0.1

        # some thresholds for the awaitable actions, to check if a move command has been completed
        self._pose_reached_L2_threshold = 0.01
        self._joint_config_reached_L2_threshold = 0.01

    def get_joint_configuration(self) -> JointConfigurationType:
        return np.array(self.rtde_receive.getActualQ())

    def get_tcp_pose(self) -> HomogeneousMatrixType:
        tpc_rotvec_pose = self.rtde_receive.getActualTCPPose()
        return self._convert_rotvec_pose_to_homogeneous_pose(tpc_rotvec_pose)

    def move_linear_to_tcp_pose(
        self, tcp_pose: HomogeneousMatrixType, linear_speed: Optional[float] = None
    ) -> AwaitableAction:
        self.rtde_control.servoStop()  # stop any ongoing servo commands to avoid "another thread is controlling the robot" errors
        self._assert_pose_is_valid(tcp_pose)
        linear_speed = linear_speed or self.default_linear_speed
        self._assert_linear_speed_is_valid(linear_speed)

        tcp_rotvec_pose = self._convert_homegeneous_pose_to_rotvec_pose(tcp_pose)
        self.rtde_control.moveL(tcp_rotvec_pose, linear_speed, self.default_linear_acceleration, asynchronous=True)
        return AwaitableAction(
            lambda: bool(np.linalg.norm(self.get_tcp_pose() - tcp_pose) < self._pose_reached_L2_threshold)
            and self._is_move_command_finished()
        )

    def move_to_tcp_pose(
        self, tcp_pose: HomogeneousMatrixType, joint_speed: Optional[float] = None
    ) -> AwaitableAction:
        self.rtde_control.servoStop()  # stop any ongoing servo commands to avoid "another thread is controlling the robot" errors
        self._assert_pose_is_valid(tcp_pose)
        joint_speed = joint_speed or self.default_joint_speed
        # don't know what the leading axis will be, so check that joint speed < min(max_joint_speeds)
        self._assert_joint_speed_is_valid(joint_speed)

        tcp_rotvec_pose = self._convert_homegeneous_pose_to_rotvec_pose(tcp_pose)
        # this is a convenience function of the ur-rtde library that combines IK with moveJ
        self.rtde_control.moveJ_IK(
            tcp_rotvec_pose, joint_speed, self.default_leading_axis_joint_acceleration, asynchronous=True
        )
        return AwaitableAction(
            lambda: bool(np.linalg.norm(self.get_tcp_pose() - tcp_pose) < self._pose_reached_L2_threshold)
            and self._is_move_command_finished()
        )

    def move_to_joint_configuration(
        self, joint_configuration: JointConfigurationType, joint_speed: Optional[float] = None
    ) -> AwaitableAction:
        self.rtde_control.servoStop()  # stop any ongoing servo commands to avoid "another thread is controlling the robot" errors
        # check joint limits
        self._assert_joint_configuration_is_valid(joint_configuration)

        joint_speed = joint_speed or self.default_joint_speed
        self._assert_joint_speed_is_valid(joint_speed)

        self.rtde_control.moveJ(
            joint_configuration, joint_speed, self.default_leading_axis_joint_acceleration, asynchronous=True
        )
        return AwaitableAction(
            lambda: bool(
                np.linalg.norm(self.get_joint_configuration() - joint_configuration)
                < self._joint_config_reached_L2_threshold
            )
            and self._is_move_command_finished()
        )

    def servo_to_tcp_pose(self, tcp_pose: HomogeneousMatrixType, duration: float) -> AwaitableAction:
        # cannot check reachability here of the pose, since that takes ~ms, which is too expensive for high frequency.

        tcp_rotvec_pose = self._convert_homegeneous_pose_to_rotvec_pose(tcp_pose)
        # note that servoL does not really exist in the UR API, it is a convenient combination IK + servoJ of the ur-rtde library
        # also note that we do not check that the pose is reachable, since that takes ~ms, which is too expensive for high frequency.
        # TODO: if we use IKFast, this would be faster than sending the pose to the robot and waiting for the response

        self.rtde_control.servoL(
            tcp_rotvec_pose, 0.0, 0.0, duration, self.servo_lookahead_time, self.servo_proportional_gain
        )

        # be careful when waiting for the action to be completed in a high frequency control loop,
        # use small granularity in the wait method to minimize latency between the action completion and the return of the wait method
        action_sent_time = time.time_ns()
        return AwaitableAction(
            lambda: time.time_ns() - action_sent_time > duration * 1e9,
            default_timeout=2 * duration,
            default_sleep_resolution=0.002,
        )

    def servo_to_joint_configuration(
        self, joint_configuration: JointConfigurationType, duration: float
    ) -> AwaitableAction:
        # cannot check reachability here of the pose, since that takes ~ms, which is too expensive for high frequency.

        # the UR robot uses a proportional gain (kp) and lookahead time (kd/proportional gain kp) to create a PID like tracking behaviour
        # that somehow determines target joint positions for their low-level control.
        # as stated here https://www.universal-robots.com/articles/ur/programming/servoj-command/#:~:text=Download%20section%20HERE-,Servoj%20can%20be%20used%20for%20online%20realtime%20control%20of%20joint,reaction%20the%20robot%20will%20have.

        # however if you do a servoJ for a really long time, the motion seems to be at constant joint velocity
        # so they also seem to do a linear interpolation to determine servoJ setpoints
        # and to only then use the gain and lookeahead time parameters to determine how the robot
        # tries to track these setpoints.

        self.rtde_control.servoJ(
            joint_configuration, 0.0, 0.0, duration, self.servo_lookahead_time, self.servo_proportional_gain
        )
        # servoJ call will block the robot for `time` seconds, but the ur-rtde call is non-blocking
        # after completing the servoJ, the robot will look for the most recent command. This means that if you send multiple commands
        # in a row, the robot will only execute the last one.

        # be careful when waiting for the action to be completed in a high frequency control loop,
        # use small granularity in the wait method to minimize latency between the action completion and the return of the wait method
        action_sent_time = time.time_ns()
        return AwaitableAction(
            lambda: time.time_ns() - action_sent_time > duration * 1e9,
            default_timeout=2 * duration,
            default_sleep_resolution=0.002,
        )

    def inverse_kinematics(
        self, tcp_pose: HomogeneousMatrixType, joint_configuration_guess: Optional[JointConfigurationType] = None
    ) -> JointConfigurationType:
        tcp_rotvec_pose = self._convert_homegeneous_pose_to_rotvec_pose(tcp_pose)

        q_near = joint_configuration_guess or np.array([])
        return self.rtde_control.getInverseKinematics(tcp_rotvec_pose, q_near)

    def forward_kinematics(self, joint_configuration: JointConfigurationType) -> HomogeneousMatrixType:
        # tcp_rotvec_pose = self.rtde_control.getForwardKinematics(joint_configuration)
        # print(f"{tcp_rotvec_pose=}")
        # return self._convert_rotvec_pose_to_homogeneous_pose(tcp_rotvec_pose)

        # this function seems to be broken in the ur-rtde library. It returns completely unrealistic rotvec poses.
        # If you need forward kinematics, consider using IKFast.
        raise NotImplementedError

    def is_tcp_pose_reachable(self, tcp_pose: HomogeneousMatrixType) -> bool:
        """check if tcp pose is reachable by the robot.
        Overrides default implementation with a specific method for the UR robot."""
        tcp_rotvec_pose = self._convert_homegeneous_pose_to_rotvec_pose(tcp_pose)
        return self.rtde_control.isPoseWithinSafetyLimits(tcp_rotvec_pose)

    def _is_joint_configuration_reachable(self, joint_configuration: JointConfigurationType) -> bool:
        return self.rtde_control.isJointsWithinSafetyLimits(joint_configuration)

    @staticmethod
    def _convert_rotvec_pose_to_homogeneous_pose(ur_pose: RotVecPoseType) -> HomogeneousMatrixType:
        return SE3Container.from_rotation_vector_and_translation(ur_pose[3:], ur_pose[:3]).homogeneous_matrix

    @staticmethod
    def _convert_homegeneous_pose_to_rotvec_pose(homogeneous_pose: HomogeneousMatrixType) -> RotVecPoseType:
        se3 = SE3Container.from_homogeneous_matrix(homogeneous_pose)
        rotation = se3.orientation_as_rotation_vector
        translation = se3.translation
        return np.concatenate([translation, rotation])

    def _is_move_command_finished(self) -> bool:
        """check if the robot has finished executing the last move command."""
        progress = self.rtde_control.getAsyncOperationProgress()
        return progress < 0

    def get_tcp_force(self) -> WrenchType:
        return np.array(self.rtde_receive.getActualTCPForce())

    def stop_script(self):
        self.rtde_control.stopScript()

    @property
    def target_pos(self) -> JointConfigurationType:
        if not self.torque_mode:
            raise RuntimeError("target_pos only valid in torque_mode")
        return np.array(self.target_pos_shared[:], dtype=float)

    @target_pos.setter
    def target_pos(self, q_target: JointConfigurationType) -> None:
        if not self.torque_mode:
            raise RuntimeError("target_pos only valid in torque_mode")
        q_target = np.asarray(q_target, dtype=float)
        assert q_target.shape == (6,)
        for i in range(6):
            self.target_pos_shared[i] = float(q_target[i])

    def get_cached_joint_configuration(self) -> JointConfigurationType:
        if not self.torque_mode:
            return np.array(self.rtde_receive.getActualQ(), dtype=float)
        return np.array(self.pos_cache_shared[:], dtype=float)

    def get_cached_tcp_pose(self) -> HomogeneousMatrixType:
        if not self.torque_mode:
            tpc_rotvec_pose = self.rtde_receive.getActualTCPPose()
            return self._convert_rotvec_pose_to_homogeneous_pose(tpc_rotvec_pose)
        return self._convert_rotvec_pose_to_homogeneous_pose(np.array(self.tcp_cache_shared[:], dtype=float))

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
                self.running_flag,
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


if __name__ == "__main__":
    """test script for UR rtde.
    e.g. python airo-robots/airo_robots/manipulators/hardware/ur_rtde.py --ip_address 10.42.0.162
    """
    import click
    from airo_robots.manipulators.hardware.manual_manipulator_testing import manual_test_robot

    @click.command()
    @click.option("--ip_address", help="IP address of the UR robot")
    def test_ur_rtde(ip_address: str) -> None:
        print(f"{ip_address=}")
        ur3e = URrtde(ip_address, URrtde.UR3E_CONFIG)
        manual_test_robot(ur3e)

    test_ur_rtde()
