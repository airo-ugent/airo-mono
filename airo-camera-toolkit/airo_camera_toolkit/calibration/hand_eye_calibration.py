"""functions and script (see __main__) for hand-eye calibration. Both eye-in-hand and eye-to-hand are supported."""
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
    AIRO_DEFAULT_ARUCO_DICT,
    AIRO_DEFAULT_CHARUCO_BOARD,
    ArucoDictType,
    CharucoBoardType,
    detect_and_visualize_charuco_pose,
)
from airo_camera_toolkit.cameras.camera_discovery import click_camera_options, discover_camera
from airo_camera_toolkit.interfaces import RGBCamera
from airo_camera_toolkit.utils import ImageConverter
from airo_dataset_tools.data_parsers.camera_intrinsics import CameraIntrinsics
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
    """Script to do hand-eye calibration, both eye-in-hand and eye-to-hand are supported.

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


if __name__ == "__main__":
    import click
    from airo_robots.manipulators.hardware.ur_rtde import URrtde

    aruco_dict = AIRO_DEFAULT_ARUCO_DICT
    charuco_board = AIRO_DEFAULT_CHARUCO_BOARD

    @click.command()
    @click.option("--mode", default="eye_in_hand", help="eye_in_hand or eye_to_hand")
    @click.option("--robot_ip", default="10.42.0.162", help="robot ip address")
    @click.option(
        "--calibration_dir", type=click.Path(exists=False), help="directory to save the calibration data to."
    )
    @click_camera_options
    def calibrate_with_ur(
        mode: str,
        robot_ip: str,
        calibration_dir: Optional[str] = None,
        camera_brand: Optional[str] = None,
        camera_serial_number: Optional[str] = None,
    ) -> None:
        """Script to do hand-eye calibration with an UR robot. Will open camera stream and visualize the detected board
        pose. Press S to capture pose, press Q to finish. Make sure the detections look good (corners/contours are
        accurate) before capturing. Once you have collected at least 3 samples, the solving the calibration will be
        attempted. To check the quality of the calibration, check the residual errors and the base pose visualizations
        in the results directory. When you are satisfied with the results, you can copy the found camera pose (saved as
        a json file) to use it in your application.

        Notes:

         * An increase in residual error when an additional sample is added does not necessarily mean that the
        calibration has become worse.

         * If residual error is low but the visualization looks wrong, you might have use the wrong mode (eye_in_hand
         or eye_to_hand). Try rerunning the calibration with the compute_calibration.py script.
        """
        robot = URrtde(robot_ip, URrtde.UR3_CONFIG)

        camera = discover_camera(camera_brand, camera_serial_number)
        do_camera_robot_calibration(mode, aruco_dict, charuco_board, camera, robot, calibration_dir)

    calibrate_with_ur()
