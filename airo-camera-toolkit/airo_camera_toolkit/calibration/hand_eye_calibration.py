"""functions and script (see __main__) for hand-eye calibration. Both eye-in-hand and eye-to-hand are supported."""
import json
import os
from typing import Union

import cv2
from airo_camera_toolkit.calibration.collect_calibration_data import (
    create_calibration_data_dir,
    detect_and_draw_charuco,
    save_calibration_sample,
)
from airo_camera_toolkit.calibration.compute_calibration import compute_calibration_all_methods
from airo_camera_toolkit.calibration.fiducial_markers import (
    AIRO_DEFAULT_ARUCO_DICT,
    AIRO_DEFAULT_CHARUCO_BOARD,
    ArucoDictType,
    CharucoDictType,
)
from airo_camera_toolkit.interfaces import RGBCamera, RGBDCamera
from airo_camera_toolkit.utils import ImageConverter
from airo_dataset_tools.data_parsers.camera_intrinsics import CameraIntrinsics
from airo_robots.manipulators.position_manipulator import PositionManipulator
from loguru import logger


def do_camera_robot_calibration(
    mode: str,
    aruco_dict: ArucoDictType,
    charuco_board: CharucoDictType,
    camera: Union[RGBCamera, RGBDCamera],
    robot: PositionManipulator,
    calibration_dir: str,
    save_pointclouds: bool,
):
    """script to do hand-eye calibration with an UR robot and a ZED2i camera.
    Will open camera stream and visualize the detected markers.
    Press S to capture pose, press Q to finish. Make sure the detections look good (corners/contours are accurate) before capturing.
    Once you have at least 5 markers, you can press F to finish and get the extrinsics pose.
    But gathering more poses will improve the accuracy of the calibration.
    """

    # for now, the robot is assumed to be a UR robot with RTDE interface, as we make use of the teach mode functions.
    # TODO: make this more generic? either assume the teachmode is available for all robots, OR use teleop instead of teachmode.
    if save_pointclouds and not isinstance(camera, RGBDCamera):
        raise ValueError("save_pointclouds is True but camera is not an RGBDCamera")

    data_dir = create_calibration_data_dir(calibration_dir)
    calibration_dir = os.path.dirname(data_dir)  # TODO clean this up

    logger.info(f"Saving calibration data to {data_dir}")
    logger.info("Press S to save a sample, Q to quit.")

    resolution = camera.resolution_sizes[camera.resolution]
    intrinsics = camera.intrinsics_matrix()

    # Saving the intrinsics
    camera_intrinsics = CameraIntrinsics.from_matrix_and_resolution(intrinsics, resolution)
    intrinsics_filepath = os.path.join(data_dir, "intrinsics.json")
    with open(intrinsics_filepath, "w") as f:
        json.dump(camera_intrinsics.dict(exclude_none=True), f, indent=4)

    window_name = "Hand-eye calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    robot.rtde_control.teachMode()  # This does generalize to all robots

    MIN_POSES = 3
    tcp_poses_in_base = []
    images = []

    while True:
        # Live visualization of board detection
        image_rgb = camera.get_rgb_image_as_int()
        image = ImageConverter.from_numpy_int_format(image_rgb).image_in_opencv_format
        detect_and_draw_charuco(image, intrinsics, aruco_dict, charuco_board)
        cv2.imshow(window_name, image)

        key = cv2.waitKey(1)
        if key == ord("q"):
            robot.rtde_control.endTeachMode()
            break

        if key == ord("s"):
            # TODO reject samples where no board was detected?
            sample_index = len(tcp_poses_in_base)
            tcp_pose, image_bgr = save_calibration_sample(sample_index, robot, camera, data_dir, save_pointclouds)
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
            compute_calibration_all_methods(
                results_dir, images, tcp_poses_in_base, intrinsics, mode, aruco_dict, charuco_board
            )


if __name__ == "__main__":  # noqa C901 - ignore complexity warning
    """script for hand-eye calibration. Both eye-in-hand and eye-to-hand are supported."""
    import click
    from airo_camera_toolkit.cameras.zed2i import Zed2i
    from airo_robots.manipulators.hardware.ur_rtde import URrtde

    aruco_dict = AIRO_DEFAULT_ARUCO_DICT
    charuco_board = AIRO_DEFAULT_CHARUCO_BOARD

    @click.command()
    @click.option("--mode", default="eye_in_hand", help="eye_in_hand or eye_to_hand")
    @click.option("--robot_ip", default="10.42.0.162", help="robot ip address")
    @click.option(
        "--camera_serial_number",
        default=None,
        type=int,
        help="serial number of the camera to use if you have multiple cameras connected.",
    )
    @click.option(
        "--calibration_dir", type=click.Path(exists=False), help="directory to save the calibration data to."
    )
    @click.option(
        "--save_pointclouds",
        is_flag=True,
        default=False,
        help="save pointclouds in addition to images and tcp poses.",
    )
    def calibrate(
        mode: str, robot_ip: str, camera_serial_number: int, calibration_dir: str, save_pointclouds: bool
    ) -> None:
        robot = URrtde(robot_ip, URrtde.UR3_CONFIG)
        print(f"zed serial numbers: {Zed2i.list_camera_serial_numbers()}")

        depth_mode = Zed2i.NONE_DEPTH_MODE
        if save_pointclouds:
            depth_mode = Zed2i.NEURAL_DEPTH_MODE

        camera = Zed2i(serial_number=camera_serial_number, depth_mode=depth_mode)
        do_camera_robot_calibration(mode, aruco_dict, charuco_board, camera, robot, calibration_dir, save_pointclouds)

    calibrate()
