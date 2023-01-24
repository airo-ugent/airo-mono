import time
import warnings
from typing import Optional

import numpy as np
from airo_robots.grippers.parallel_position_gripper import ParallelPositionGripper
from airo_robots.manipulators.hardware.manual_manipulator_testing import manual_test_robot
from airo_robots.manipulators.position_manipulator import ManipulatorSpecs, PositionManipulator
from airo_spatial_algebra import SE3Container
from airo_typing import HomogeneousMatrixType, JointConfigurationType
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

RotVecPoseType = np.ndarray
""" a 6D pose [tx,ty,tz,rotvecx,rotvecy,rotvecz]"""


class UR_RTDE(PositionManipulator):
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
    UR3E_CONFIG = ManipulatorSpecs(6, [1.0, 1.0, 1.0, 2.0, 2.0, 2.0], 1.0)
    # https://www.universal-robots.com/media/240787/ur3_us.pdf
    UR3_CONFIG = ManipulatorSpecs(6, [1.0, 1.0, 1.0, 2.0, 2.0, 2.0], 1.0)

    def __init__(
        self, ip_address: str, manipulator_specs: ManipulatorSpecs, gripper: Optional[ParallelPositionGripper] = None
    ) -> None:
        super().__init__(manipulator_specs, gripper)
        self.ip_address = ip_address
        try:
            self.rtde_control = RTDEControlInterface(self.ip_address)
            self.rtde_receive = RTDEReceiveInterface(self.ip_address)

        except TimeoutError:
            raise TimeoutError(
                "Could not connect to the robot. Is the robot in remote control? Is the IP correct? Is the network connection ok?"
            )

        self.default_linear_speed = 0.1  # m/s
        self.default_linear_acceleration = 1.2  # m/s^2
        self.default_leading_axis_joint_speed = 1  # rad/s
        self.default_leading_axis_joint_acceleration = 1.2  # rad/s^2

        # sensible values but you might need to tweak them for your purposes.
        self.servo_proportional_gain = 200
        self.servo_lookahead_time = 0.1

    def get_joint_configuration(self) -> JointConfigurationType:
        return np.array(self.rtde_receive.getActualQ())

    def get_tcp_pose(self) -> HomogeneousMatrixType:
        tpc_rotvec_pose = self.rtde_receive.getActualTCPPose()
        return self._convert_rotvec_pose_to_homogeneous_pose(tpc_rotvec_pose)

    def move_linear_to_tcp_pose(self, tcp_pose: HomogeneousMatrixType, linear_speed: Optional[float] = None) -> None:
        if not self.is_tcp_pose_kinematically_reachable(tcp_pose):
            warnings.warn(f"Pose is not reachable :\n{tcp_pose}")
            return
        tcp_rotvec_pose = self._convert_homegeneous_pose_to_rotvec_pose(tcp_pose)
        linear_speed = linear_speed or self.default_linear_speed
        linear_speed = np.clip(linear_speed, 0.0, self.manipulator_specs.max_linear_speed)
        return self.rtde_control.moveL(tcp_rotvec_pose, linear_speed, self.default_linear_acceleration)

    def move_to_tcp_pose(self, tcp_pose: HomogeneousMatrixType, joint_speed: Optional[float] = None) -> None:
        if not self.is_tcp_pose_kinematically_reachable(tcp_pose):
            warnings.warn(f"Pose is not reachable :\n{tcp_pose}")
            return

        tcp_rotvec_pose = self._convert_homegeneous_pose_to_rotvec_pose(tcp_pose)
        joint_speed = joint_speed or self.default_leading_axis_joint_speed
        # don't know what leading axis is atm, so take min of all max joint speeds
        joint_speed = np.clip(joint_speed, 0.0, min(self.manipulator_specs.max_joint_speeds))

        # this is a convenience function of the ur-rtde library that combines IK with moveJ
        self.rtde_control.moveJ_IK(tcp_rotvec_pose, joint_speed, self.default_leading_axis_joint_acceleration)

    def move_to_joint_configuration(
        self, joint_configuration: JointConfigurationType, joint_speed: Optional[float] = None
    ) -> None:
        # check joint limits
        if not self.rtde_control.isJointsWithinSafetyLimits(joint_configuration):
            warnings.warn(f"joint configuration {joint_configuration} is not reachable.")
            return

        joint_speed = joint_speed or self.default_leading_axis_joint_speed
        # don't know what leading axis is atm, so take min of all max joint speeds
        joint_speed = np.clip(joint_speed, 0.0, min(self.manipulator_specs.max_joint_speeds))
        self.rtde_control.moveJ(joint_configuration, joint_speed, self.default_leading_axis_joint_acceleration)
        return

    def servo_to_tcp_pose(self, tcp_pose: HomogeneousMatrixType, duration: float) -> None:
        # cannot check reachability here of the pose, since that takes ~ms, which is too expensive for high frequency.

        tcp_rotvec_pose = self._convert_homegeneous_pose_to_rotvec_pose(tcp_pose)
        # note that servoL does not really exist in the UR API, it is a convenient combination IK + servoJ of the ur-rtde library
        self.rtde_control.servoL(
            tcp_rotvec_pose, 0.0, 0.0, duration, self.servo_lookahead_time, self.servo_proportional_gain
        )
        # servoL call will block the robot for `time` seconds, but the ur-rtde call is non-blocking
        # so add explicit sleep here to make the call synchronous
        time.sleep(duration)

    def servo_to_joint_configuration(self, joint_configuration: JointConfigurationType, duration: float) -> None:
        # the UR robot uses a proportional gain (kp) and lookahead time (kd/proportional gain kp) to create a PID like tracking behaviour
        # that somehow determines target joint positions for their low-level control.
        # as stated here https://www.universal-robots.com/articles/ur/programming/servoj-command/#:~:text=Download%20section%20HERE-,Servoj%20can%20be%20used%20for%20online%20realtime%20control%20of%20joint,reaction%20the%20robot%20will%20have.

        # however if you do a servoJ for a really long time, the motion seems to be at constant joint velocity
        # so they seem to do a linear interpolation to determine servoJ setpoints
        # and to only then use the gain and lookeahead time parameters to determine how the robot
        # tries to track these setpoints.

        self.rtde_control.servoJ(
            joint_configuration, 0.0, 0.0, duration, self.servo_lookahead_time, self.servo_proportional_gain
        )
        # servoJ call will block the robot for `time` seconds, but the ur-rtde call is non-blocking
        # so add explicit sleep here to make the call synchronous
        # after completing the servoJ will look for a next command so say that you have sent 5 servoJ calls in the meantime
        # only the last would be exectued, which can result in shaky behaviour
        # this is why the explict sleep is added so that the function only returns once the function has completed.
        # cf the example of the ur-rtde library where they use the 'wait' function for this
        # but on non-realtime linux kernels it seems that the sleep accuracy is on the <ms order, so seems fine to do it like this
        # https://stackoverflow.com/questions/1133857/how-accurate-is-pythons-time-sleep
        # if it needs to be more accurate, we need to switch to a real-time kernel
        time.sleep(duration)

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

    def is_tcp_pose_kinematically_reachable(self, tcp_pose: HomogeneousMatrixType) -> bool:
        tcp_rotvec_pose = self._convert_homegeneous_pose_to_rotvec_pose(tcp_pose)
        return self.rtde_control.isPoseWithinSafetyLimits(tcp_rotvec_pose)

    @staticmethod
    def _convert_rotvec_pose_to_homogeneous_pose(ur_pose: RotVecPoseType) -> HomogeneousMatrixType:
        return SE3Container.from_rotation_vector_and_translation(ur_pose[3:], ur_pose[:3]).homogeneous_matrix

    @staticmethod
    def _convert_homegeneous_pose_to_rotvec_pose(homogeneous_pose: HomogeneousMatrixType) -> RotVecPoseType:
        se3 = SE3Container.from_homogeneous_matrix(homogeneous_pose)
        rotation = se3.orientation_as_rotation_vector
        translation = se3.translation
        return np.concatenate([translation, rotation])


if __name__ == "__main__":
    """test script for UR rtde.
    ex. python airo-robots/airo_robots/manipulators/ur3e.py --ip_address 10.42.0.162 for Victor
    """
    import click

    @click.command()
    @click.option("--ip_address", help="IP address of the UR robot")
    def test_ur_rtde(ip_address: str) -> None:
        print(f"{ip_address=}")
        ur3e = UR_RTDE(ip_address, UR_RTDE.UR3E_CONFIG)
        manual_test_robot(ur3e)

    test_ur_rtde()
