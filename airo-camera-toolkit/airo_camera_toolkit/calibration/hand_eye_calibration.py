"""functions and script (see __main__) for hand-eye calibration. Both eye-in-hand and eye-to-hand are supported."""
import json
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
from airo_camera_toolkit.calibration.fiducial_markers import (
    detect_aruco_markers,
    detect_charuco_corners,
    draw_frame_on_image,
    get_pose_of_charuco_board,
    visualize_aruco_detections,
    visualize_charuco_detection,
)
from airo_camera_toolkit.interfaces import RGBCamera
from airo_camera_toolkit.utils import ImageConverter
from airo_dataset_tools.pose import EulerAngles, Pose, Position
from airo_spatial_algebra import SE3Container
from airo_typing import HomogeneousMatrixType


def compute_hand_eye_calibration_error(
    tcp_poses_in_base: List[HomogeneousMatrixType],
    marker_poses_in_camera: List[HomogeneousMatrixType],
    camera_pose: HomogeneousMatrixType,
) -> float:
    """compute the error between the left and right side of the AX =XB equation to have an estimate of the error of the calibration
    Decent calibrations should have an average error (way) below 0.01
    Args:
        tcp_poses_in_base (List[HomogeneousMatrixType]): list of tcp poses in base frame
        marker_poses_in_camera (List[HomogeneousMatrixType]): list of marker poses in camera frame
        camera_pose (HomogeneousMatrixType): camera pose in base frame (eye-to-hand) or camera pose in tcp frame(eye-in-hand))"""
    error = 0.0
    for i in range(len(tcp_poses_in_base) - 1):
        tcp_pose_in_base = tcp_poses_in_base[i]
        marker_pose_in_camera = marker_poses_in_camera[i]

        tcp_pose_in_base_2 = tcp_poses_in_base[i + 1]
        marker_pose_in_camera_2 = marker_poses_in_camera[i + 1]

        # cf https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gaebfc1c9f7434196a374c382abf43439b
        # for the AX=XB equation
        left_side = tcp_pose_in_base @ camera_pose @ marker_pose_in_camera
        right_side = tcp_pose_in_base_2 @ camera_pose @ marker_pose_in_camera_2
        error += float(np.linalg.norm(left_side - right_side))
    return error / (len(tcp_poses_in_base) - 1)


def eye_in_hand_pose_estimation(
    tcp_poses_in_base: List[HomogeneousMatrixType], marker_poses_in_camera: List[HomogeneousMatrixType]
) -> Tuple[Optional[HomogeneousMatrixType], Optional[float]]:
    """wrapper around the opencv eye-in-hand extrinsics calibration function."""
    tcp_orientations_as_rotvec_in_base = [
        SE3Container.from_homogeneous_matrix(tcp_pose).orientation_as_rotation_vector for tcp_pose in tcp_poses_in_base
    ]
    tcp_positions_in_base = [
        SE3Container.from_homogeneous_matrix(tcp_pose).translation for tcp_pose in tcp_poses_in_base
    ]

    marker_orientations_as_rotvec_in_camera = [
        SE3Container.from_homogeneous_matrix(marker_pose).orientation_as_rotation_vector
        for marker_pose in marker_poses_in_camera
    ]
    marker_positions_in_camera = [
        SE3Container.from_homogeneous_matrix(marker_pose).translation for marker_pose in marker_poses_in_camera
    ]

    camera_rotation_matrix, camera_translation = cv2.calibrateHandEye(
        tcp_orientations_as_rotvec_in_base,
        tcp_positions_in_base,
        marker_orientations_as_rotvec_in_camera,
        marker_positions_in_camera,
        None,
        None,
    )

    if camera_rotation_matrix is None or camera_translation is None:
        return None, None

    camera_pose_in_tcp_frame = SE3Container.from_rotation_matrix_and_translation(
        camera_rotation_matrix, camera_translation
    ).homogeneous_matrix

    calibration_error = compute_hand_eye_calibration_error(
        tcp_poses_in_base, marker_poses_in_camera, camera_pose_in_tcp_frame
    )
    return camera_pose_in_tcp_frame, calibration_error


def eye_to_hand_pose_estimation(
    tcp_poses_in_base: List[HomogeneousMatrixType], marker_poses_in_camera: List[HomogeneousMatrixType]
) -> Tuple[Optional[HomogeneousMatrixType], Optional[float]]:
    """wrapper around the opencv eye-to-hand extrinsics calibration function."""

    # the AX=XB problem for the eye-to-hand is equivalent to the AX=XB problem for the eye-in-hand if you invert the poses of the tcp
    # cf https://docs.opencv.org/4.5.4/d9/d0c/group__calib3d.html#gaebfc1c9f7434196a374c382abf43439b
    # cf https://forum.opencv.org/t/eye-to-hand-calibration/5690/2

    base_pose_in_tcp_frame = [np.linalg.inv(tcp_pose) for tcp_pose in tcp_poses_in_base]

    camera_pose_in_base, calibration_error = eye_in_hand_pose_estimation(
        base_pose_in_tcp_frame, marker_poses_in_camera
    )
    return camera_pose_in_base, calibration_error


if __name__ == "__main__":  # noqa C901 - ignore complexity warning
    """script for hand-eye calibration. Both eye-in-hand and eye-to-hand are supported."""
    import click
    from airo_camera_toolkit.calibration.fiducial_markers import (
        AIRO_DEFAULT_ARUCO_DICT,
        AIRO_DEFAULT_CHARUCO_BOARD,
        ArucoDictType,
        CharucoDictType,
    )
    from airo_camera_toolkit.cameras.zed2i import Zed2i
    from airo_robots.manipulators.hardware.ur_rtde import URrtde
    from loguru import logger

    def do_camera_robot_calibration(
        mode: str, aruco_dict: ArucoDictType, charuco_board: CharucoDictType, camera: RGBCamera, robot: URrtde
    ) -> Optional[HomogeneousMatrixType]:
        """script to do hand-eye calibration with an UR robot and a ZED2i camera.
        Will open camera stream and visualize the detected markers.
        Press S to capture pose, press F to finish. Make sure the detections look good (corners/contours are accurate) before capturing.
        Once you have at least 5 markers, you can press F to finish and get the extrinsics pose.
        But gathering more poses will improve the accuracy of the calibration.
        This function is added in the __main__ to avoid having a dependency on the airo-robots package in the module."""

        # for now, the robot is assumed to be a UR robot with RTDE interface, as we make use of the teach mode functions.
        # TODO: make this more generic? either assume the teachmode is available for all robots, OR use teleop instead of teachmode.

        min_poses = 3
        tcp_poses_in_base = []
        marker_poses_in_camera = []
        camera_pose = None
        calibration_error = None

        print(
            "Press S to capture pose, press F to finish. Make sure the detections look good (corners/contours are accurate) before capturing."
        )
        robot.rtde_control.teachMode()
        while True:
            image = camera.get_rgb_image()
            image = ImageConverter.from_numpy_format(image).image_in_opencv_format

            aruco_result = detect_aruco_markers(image, aruco_dict)
            if not aruco_result:
                continue
            charuco_result = detect_charuco_corners(image, aruco_result, charuco_board)
            if not charuco_result:
                continue

            charuco_pose = get_pose_of_charuco_board(charuco_result, charuco_board, camera.intrinsics_matrix(), None)

            # visualize
            image = visualize_aruco_detections(image, aruco_result)
            image = visualize_charuco_detection(image, charuco_result)
            if charuco_pose is not None:
                image = draw_frame_on_image(image, charuco_pose, camera.intrinsics_matrix())

            if camera_pose is not None and mode == "eye_to_hand":
                # visualize the pose of the robot in the camera on the image, to get visual feedback
                image = draw_frame_on_image(image, np.linalg.inv(camera_pose), camera.intrinsics_matrix())
            image = cv2.resize(image, (1920, 1080))
            cv2.imshow("image", image)

            key = cv2.waitKey(1)
            if key == ord("s"):
                robot.rtde_control.endTeachMode()
                time.sleep(0.5)

                # retake charuco pose, to make sure that the robot pose corresponds to the image
                calibration_image = camera.get_rgb_image()
                calibration_image = ImageConverter.from_numpy_format(calibration_image).image_in_opencv_format
                aruco_result = detect_aruco_markers(calibration_image, aruco_dict)
                if not aruco_result:
                    continue
                charuco_result = detect_charuco_corners(calibration_image, aruco_result, charuco_board)
                if not charuco_result:
                    continue

                charuco_pose = get_pose_of_charuco_board(
                    charuco_result, charuco_board, camera.intrinsics_matrix(), None
                )

                if charuco_pose is None:
                    logger.warning("No charuco pose detected, please try again.")
                    continue
                marker_poses_in_camera.append(charuco_pose)
                tcp_poses_in_base.append(robot.get_tcp_pose())
                logger.info(f"{len(tcp_poses_in_base)} poses captured")
                time.sleep(0.5)
                robot.rtde_control.teachMode()

                if len(tcp_poses_in_base) >= min_poses and len(marker_poses_in_camera) >= min_poses:
                    print(len(tcp_poses_in_base))
                    print(len(marker_poses_in_camera))
                    if mode == "eye_in_hand":
                        # pose of camera in tcp frame
                        camera_pose, calibration_error = eye_in_hand_pose_estimation(
                            tcp_poses_in_base, marker_poses_in_camera
                        )
                    elif mode == "eye_to_hand":
                        # pose of camera in base frame
                        camera_pose, calibration_error = eye_to_hand_pose_estimation(
                            tcp_poses_in_base, marker_poses_in_camera
                        )
                    else:
                        raise ValueError(f"Unknown mode {mode}")
                    logger.info(f"camera pose: {camera_pose}")
                    logger.info(f"calibration error: {calibration_error}, should be < 0.01 for good calibration")

            elif key == ord("f"):
                if len(tcp_poses_in_base) < min_poses:
                    logger.warning(f"Not enough poses captured, please capture at least {min_poses}poses.")
                    continue
                robot.rtde_control.endTeachMode()
                break

        return camera_pose

    robot = URrtde("10.42.0.162", URrtde.UR3_CONFIG)
    camera = Zed2i()
    aruco_dict = AIRO_DEFAULT_ARUCO_DICT
    charuco_board = AIRO_DEFAULT_CHARUCO_BOARD

    @click.command()
    @click.option("--mode", default="eye_in_hand", help="eye_in_hand or eye_to_hand")
    @click.option("--robot_ip", default="10.42.0.162", help="robot ip address")
    def calibrate(mode: str, robot_ip: str) -> None:
        pose = do_camera_robot_calibration(mode, aruco_dict, charuco_board, camera, robot)

        if pose is None:
            logger.warning("Calibration failed, exiting.")
            return

        pose_se3 = SE3Container.from_homogeneous_matrix(pose)
        x, y, z = pose_se3.translation
        roll, pitch, yaw = pose_se3.orientation_as_euler_angles

        pose_saveable = Pose(
            position_in_meters=Position(x=x, y=y, z=z),
            rotation_euler_xyz_in_radians=EulerAngles(roll=roll, pitch=pitch, yaw=yaw),
        )
        with open("camera_pose.json", "w") as f:
            json.dump(pose_saveable.dict(), f, indent=4)

    calibrate()
