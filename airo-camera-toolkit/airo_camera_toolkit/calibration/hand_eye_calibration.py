"""Function for the hand-eye calibration tool. Both eye-in-hand and eye-to-hand are supported."""
import json
import os
from typing import List, Optional

import cv2
from airo_camera_toolkit.calibration.collect_calibration_data import (
    create_calibration_data_dir,
    save_calibration_sample,
)
from airo_camera_toolkit.calibration.compute_calibration import (
    compute_calibration_all_methods,
    draw_base_pose_on_image,
)
from airo_camera_toolkit.calibration.fiducial_markers import (
    ArucoDictType,
    CharucoBoardType,
    detect_and_visualize_charuco_pose,
)
from airo_camera_toolkit.interfaces import RGBCamera
from airo_camera_toolkit.utils.image_converter import ImageConverter
from airo_dataset_tools.data_parsers.camera_intrinsics import CameraIntrinsics
from airo_robots.manipulators.hardware.ur_rtde import URrtde
from airo_robots.manipulators.position_manipulator import PositionManipulator
from airo_typing import HomogeneousMatrixType, OpenCVIntImageType
from loguru import logger


def do_camera_robot_calibration(
    mode: str,
    aruco_dict: ArucoDictType,
    charuco_board: CharucoBoardType,
    camera: RGBCamera,
    robot: PositionManipulator,
    calibration_dir: Optional[str],
) -> None:
    """Do hand-eye calibration, both eye-in-hand and eye-to-hand are supported.

    Args:
        mode: eye_in_hand or eye_to_hand
        aruco_dict: The aruco dictionary used for the charuco board.
        charuco_board: The charuco board used for calibration.
        camera: The camera being used to collect the data.
        robot: The robot being used to collect the data.
        calibration_dir: The directory to save the calibration samples and results to, will be created if None
    """

    data_dir = create_calibration_data_dir(calibration_dir)
    calibration_dir = os.path.dirname(data_dir)  # TODO clean this up

    logger.info(f"Saving calibration data to {data_dir}")
    logger.info("Press S to save a sample, Q to quit.")

    resolution = camera.resolution

    intrinsics = camera.intrinsics_matrix()

    # Saving the intrinsics
    camera_intrinsics = CameraIntrinsics.from_matrix_and_resolution(intrinsics, resolution)
    intrinsics_filepath = os.path.join(data_dir, "intrinsics.json")
    with open(intrinsics_filepath, "w") as f:
        json.dump(camera_intrinsics.model_dump(exclude_none=True), f, indent=4)

    window_name = "Hand-eye calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # For now, the robot is assumed to be a UR robot with RTDE interface, as we make use of the teach mode functions.
    # TODO: make this more generic by providing a teach mode function in the PositionManipulator interface?
    assert isinstance(robot, URrtde), "Only UR robots are supported for now."
    robot.rtde_control.teachMode()

    MIN_POSES = 3
    tcp_poses_in_base: List[HomogeneousMatrixType] = []
    images: List[OpenCVIntImageType] = []
    camera_pose_best = None

    while True:
        # Live visualization of board detection
        image_rgb = camera.get_rgb_image_as_int()
        image = ImageConverter.from_numpy_int_format(image_rgb).image_in_opencv_format
        detect_and_visualize_charuco_pose(image, intrinsics, aruco_dict, charuco_board)
        tcp_pose = robot.get_tcp_pose()
        draw_base_pose_on_image(image, intrinsics, camera_pose_best, mode, tcp_pose)
        cv2.imshow(window_name, image)

        key = cv2.waitKey(1)
        if key == ord("q"):
            robot.rtde_control.endTeachMode()
            break

        if key == ord("s"):
            # TODO reject samples where no board was detected?
            sample_index = len(tcp_poses_in_base)
            tcp_pose, image_bgr = save_calibration_sample(sample_index, robot, camera, data_dir)
            logger.info(f"Saved {sample_index + 1} sample(s).")

            tcp_poses_in_base.append(tcp_pose)
            images.append(image_bgr)

            n_samples = len(tcp_poses_in_base)
            if n_samples < MIN_POSES:
                continue

            # The the calibration with the new set of samples
            results_dir = os.path.join(calibration_dir, f"results_n={n_samples}")
            os.makedirs(results_dir)
            logger.info(f"Running calibration with {n_samples} (image, tcp_pose) pairs")
            logger.info(f"Saving calibration results to {results_dir}")
            poses_dict, errors_dict = compute_calibration_all_methods(
                results_dir, images, tcp_poses_in_base, intrinsics, mode, aruco_dict, charuco_board
            )

            min_error_key = min(errors_dict, key=lambda x: errors_dict.get(x) or float("inf"))
            camera_pose_best = poses_dict[min_error_key]
