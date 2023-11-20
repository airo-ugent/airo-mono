import datetime
import json
import os
import time
from typing import Union

import click
import cv2
from airo_camera_toolkit.calibration.fiducial_markers import (
    AIRO_DEFAULT_ARUCO_DICT,
    AIRO_DEFAULT_CHARUCO_BOARD,
    detect_aruco_markers,
    detect_charuco_corners,
    draw_frame_on_image,
    get_pose_of_charuco_board,
    visualize_aruco_detections,
    visualize_charuco_detection,
)
from airo_camera_toolkit.interfaces import RGBCamera, RGBDCamera
from airo_camera_toolkit.utils import ImageConverter
from airo_dataset_tools.data_parsers.camera_intrinsics import CameraIntrinsics
from airo_dataset_tools.data_parsers.pose import Pose
from airo_robots.manipulators.position_manipulator import PositionManipulator


def create_calibration_data_dir(calibration_dir: str) -> None:
    if calibration_dir is None:
        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        calibration_dir = os.path.join(os.getcwd(), f"calibration_{datetime_str}")

    # Check whether calibration_dir is empty
    if os.path.exists(calibration_dir) and len(os.listdir(calibration_dir)) != 0:
        raise ValueError(f"calibration_dir {calibration_dir} exists and is not empty")

    data_dir = os.path.join(calibration_dir, "data")
    os.makedirs(os.path.join(calibration_dir, "data"), exist_ok=True)
    return data_dir


def save_calibration_sample(sample_index, robot, camera, data_dir):
    # Stop freedrive so robot is completely still at moment of the image capture
    robot.rtde_control.endTeachMode()
    time.sleep(0.5)

    image_rgb = camera.get_rgb_image_as_int()
    image_bgr = ImageConverter.from_numpy_int_format(image_rgb).image_in_opencv_format

    tcp_pose = robot.get_tcp_pose()

    suffix = f"{sample_index:04d}"
    image_filename = f"image_{suffix}.png"
    tcp_pose_filename = f"tcp_pose_{suffix}.json"
    image_filepath = os.path.join(data_dir, image_filename)
    tcp_pose_filepath = os.path.join(data_dir, tcp_pose_filename)

    cv2.imwrite(image_filepath, image_bgr)

    pose = Pose.from_homogeneous_matrix(tcp_pose)
    with open(tcp_pose_filepath, "w") as f:
        json.dump(pose.model_dump(), f, indent=4)

    robot.rtde_control.teachMode()

    return tcp_pose, image_bgr


def detect_and_draw_charuco(
    image, intrinsics, aruco_dict=AIRO_DEFAULT_ARUCO_DICT, charuco_board=AIRO_DEFAULT_CHARUCO_BOARD
):
    aruco_result = detect_aruco_markers(image, aruco_dict)
    if not aruco_result:
        return
    image = visualize_aruco_detections(image, aruco_result)

    charuco_result = detect_charuco_corners(image, aruco_result, charuco_board)
    if not charuco_result:
        return

    image = visualize_charuco_detection(image, charuco_result)

    charuco_pose = get_pose_of_charuco_board(charuco_result, charuco_board, intrinsics, None)
    if charuco_pose is None:
        return

    image = draw_frame_on_image(image, charuco_pose, intrinsics)


def collect_calibration_data(
    robot: PositionManipulator, camera: Union[RGBCamera, RGBDCamera], calibration_dir: str
) -> None:
    """collect calibration data for hand-eye calibration.

    Args:
        robot: the robot to use for collecting the data.
        camera: the camera to use for collecting the data.
        calibration_dir: directory to save the calibration data to.
    """
    from loguru import logger

    data_dir = create_calibration_data_dir(calibration_dir)

    logger.info(f"Saving calibration data to {data_dir}")
    logger.info("Press S to save a sample, Q to quit.")

    resolution = camera.resolution_sizes[camera.resolution]
    intrinsics = camera.intrinsics_matrix()

    # Saving the intrinsics
    camera_intrinsics = CameraIntrinsics.from_matrix_and_resolution(intrinsics, resolution)
    intrinsics_filepath = os.path.join(data_dir, "intrinsics.json")
    with open(intrinsics_filepath, "w") as f:
        json.dump(camera_intrinsics.model_dump(exclude_none=True), f, indent=4)

    window_name = "Calibration data collection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    robot.rtde_control.teachMode()  # This does generalize to all robots
    sample_index = 0

    while True:
        # Live visualization of board detection
        image_rgb = camera.get_rgb_image_as_int()
        image = ImageConverter.from_numpy_int_format(image_rgb).image_in_opencv_format
        detect_and_draw_charuco(image, intrinsics)
        cv2.imshow(window_name, image)

        key = cv2.waitKey(1)
        if key == ord("q"):
            robot.rtde_control.endTeachMode()
            break

        if key == ord("s"):
            save_calibration_sample(sample_index, robot, camera, data_dir)
            sample_index += 1
            logger.info(f"Saved {sample_index} sample(s).")


@click.command()
@click.option("--robot_ip", default="10.42.0.162", help="robot ip address")
@click.option(
    "--camera_serial_number",
    default=None,
    type=int,
    help="serial number of the camera to use if you have multiple cameras connected.",
)
@click.option("--calibration_dir", type=click.Path(exists=False), help="directory to save the calibration data to.")
def collect_calibration_data_with_ur_and_zed(robot_ip: str, camera_serial_number: int, calibration_dir: str) -> None:
    from airo_camera_toolkit.cameras.zed2i import Zed2i
    from airo_robots.manipulators.hardware.ur_rtde import URrtde

    robot = URrtde(robot_ip, URrtde.UR3_CONFIG)

    print(f"ZED serial numbers: {Zed2i.list_camera_serial_numbers()}")
    camera = Zed2i(depth_mode=Zed2i.NEURAL_DEPTH_MODE, serial_number=camera_serial_number)

    collect_calibration_data(robot, camera, calibration_dir)


if __name__ == "__main__":
    collect_calibration_data_with_ur_and_zed()
