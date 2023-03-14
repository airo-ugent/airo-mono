"""functions and script (see __main__) for hand-eye calibration. Both eye-in-hand and eye-to-hand are supported."""
import time
from typing import Any, List, Optional

import click
import cv2
import numpy as np
from airo_camera_toolkit.calibration.fiducial_markers import (
    AIRO_DEFAULT_ARUCO_DICT,
    AIRO_DEFAULT_CHARUCO_BOARD,
    ArucoDictType,
    CharucoDictType,
    detect_aruco_markers,
    detect_charuco_corners,
    draw_frame_on_image,
    get_pose_of_charuco_board,
    visualize_aruco_detections,
    visualize_charuco_detection,
)
from airo_camera_toolkit.interfaces import RGBCamera
from airo_camera_toolkit.utils import ImageConverter
from airo_spatial_algebra import SE3Container
from airo_typing import HomogeneousMatrixType
from loguru import logger


def eye_in_hand_pose_estimation(
    tcp_poses_in_base: List[HomogeneousMatrixType], marker_poses_in_camera: List[HomogeneousMatrixType]
) -> Optional[HomogeneousMatrixType]:
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
        return None

    camera_pose_in_tcp_frame = SE3Container.from_rotation_matrix_and_translation(
        camera_rotation_matrix, camera_translation
    ).homogeneous_matrix
    return camera_pose_in_tcp_frame


def eye_to_hand_pose_estimation(
    tcp_poses_in_base: List[HomogeneousMatrixType], marker_poses_in_camera: List[HomogeneousMatrixType]
) -> Optional[HomogeneousMatrixType]:
    """wrapper around the opencv eye-to-hand extrinsics calibration function."""

    # the AX=XB problem for the eye-to-hand is equivalent to the AX=XB problem for the eye-in-hand if you invert the poses of the tcp
    # cf https://docs.opencv.org/4.5.4/d9/d0c/group__calib3d.html#gaebfc1c9f7434196a374c382abf43439b
    # cf https://forum.opencv.org/t/eye-to-hand-calibration/5690/2

    base_pose_in_tcp_frame = [np.linalg.inv(tcp_pose) for tcp_pose in tcp_poses_in_base]

    camera_pose_in_base = eye_in_hand_pose_estimation(base_pose_in_tcp_frame, marker_poses_in_camera)
    if camera_pose_in_base is None:
        return None
    return camera_pose_in_base


def do_camera_robot_calibration(  # noqa: C901 - too complex
    mode: str, aruco_dict: ArucoDictType, charuco_board: CharucoDictType, camera: RGBCamera, robot: Any
) -> Optional[HomogeneousMatrixType]:
    """function to do hand-eye calibration with an UR robot and a ZED2i camera.
    Will open camera stream and visualize the detected markers.
    Press S to capture pose, press F to finish. Make sure the detections look good (corners/contours are accurate) before capturing.
    Once you have at least 5 markers, you can press F to finish and get the extrinsics pose.
    But gathering more poses will improve the accuracy of the calibration.
    """

    # for now, the robot is assumed to be a UR robot with RTDE interface, as we make use of the teach mode functions.
    # TODO: make this more generic? either assume the teachmode is available for all robots, OR use teleop instead of teachmode.

    # TODO: the type of the robot is now set to Any, because we don't want to have a dependency on the airo-robots package? But this is not ideal.
    min_poses = 5
    tcp_poses_in_base = []
    marker_poses_in_camera = []
    camera_pose = None

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

        image = cv2.resize(image, (1920, 1080))
        cv2.imshow("image", image)

        key = cv2.waitKey(1)
        if key == ord("s"):
            robot.rtde_control.endTeachMode()
            time.sleep(0.5)
            if charuco_pose is None:
                logger.warning("No charuco pose detected, please try again.")
                continue
            marker_poses_in_camera.append(charuco_pose)
            tcp_poses_in_base.append(robot.get_tcp_pose())
            logger.info(f"{len(tcp_poses_in_base)} poses captured")
            time.sleep(0.5)
            robot.rtde_control.teachMode()

        elif key == ord("f"):
            if len(tcp_poses_in_base) < min_poses:
                logger.warning(f"Not enough poses captured, please capture at least {min_poses}poses.")
                continue
            robot.rtde_control.endTeachMode()
            break

        if len(tcp_poses_in_base) >= min_poses and len(marker_poses_in_camera) >= min_poses:
            if mode == "eye_in_hand":
                # pose of camera in tcp frame
                camera_pose = eye_in_hand_pose_estimation(tcp_poses_in_base, marker_poses_in_camera)
            elif mode == "eye_to_hand":
                # pose of camera in base frame
                camera_pose = eye_to_hand_pose_estimation(tcp_poses_in_base, marker_poses_in_camera)

    return camera_pose


if __name__ == "__main__":
    """script that performs hand-eye calibration with a UR robot and a ZED2i camera."""

    from airo_camera_toolkit.cameras.zed2i import Zed2i
    from airo_robots.manipulators import URrtde

    robot = URrtde("10.42.0.162", URrtde.UR3_CONFIG)
    camera = Zed2i()
    aruco_dict = AIRO_DEFAULT_ARUCO_DICT
    charuco_board = AIRO_DEFAULT_CHARUCO_BOARD

    @click.command()
    @click.option("--mode", default="eye_in_hand", help="eye_in_hand or eye_to_hand")
    def calibrate(mode: str) -> None:
        pose = do_camera_robot_calibration(mode, aruco_dict, charuco_board, camera, robot)
        print(pose)
        # TODO: serialize and save the extrinsics to the to-be-determined airo-mono extrinsics format

    calibrate()
