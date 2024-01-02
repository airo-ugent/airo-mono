import datetime
import json
import os
import time
from typing import Optional, Tuple

import click
import cv2
from airo_camera_toolkit.calibration.fiducial_markers import detect_and_visualize_charuco_pose
from airo_camera_toolkit.cameras.camera_discovery import click_camera_options, discover_camera
from airo_camera_toolkit.interfaces import RGBCamera
from airo_camera_toolkit.utils.image_converter import ImageConverter
from airo_dataset_tools.data_parsers.camera_intrinsics import CameraIntrinsics
from airo_dataset_tools.data_parsers.pose import Pose
from airo_robots.manipulators.hardware.ur_rtde import URrtde
from airo_robots.manipulators.position_manipulator import PositionManipulator
from airo_typing import HomogeneousMatrixType, OpenCVIntImageType


def create_calibration_data_dir(calibration_dir: Optional[str] = None) -> str:
    """Ensures that a calibration_dir exists and has an empty "data" subfolder where new calibration samples can be
    stored.

    Args:
        calibration_dir: Directory to save the calibration data to. If None, a directory with the current timestamp
            will be created in the current working directory.

    Returns:
        Path to the "data" subfolder of the calibration_dir.
    """
    if calibration_dir is None:
        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        calibration_dir = os.path.join(os.getcwd(), f"calibration_{datetime_str}")

    os.makedirs(calibration_dir, exist_ok=True)

    data_dir = os.path.join(calibration_dir, "data")

    # If data_dir already exists, check whether it is empty
    if os.path.exists(data_dir) and len(os.listdir(data_dir)) != 0:
        raise ValueError(f"The data subfolder of {calibration_dir} already exists and is not empty.")

    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def save_calibration_sample(
    sample_index: int, robot: URrtde, camera: RGBCamera, data_dir: str
) -> Tuple[HomogeneousMatrixType, OpenCVIntImageType]:
    """Collect a single calibration sample and save to the data_dir.
    A calibration data sample consists of an image and a TCP pose.

    Args:
        sample_index: index of the sample used for the names of the saved files.
        robot: The robot being used to collect the data.
        camera: The camera being used to collect the data.
        data_dir: The directory to save the sample to.

    Returns:
        If charuco board detection is succesful, the TCP pose and the image, else None.
    """
    # Stop freedrive so robot is completely still at moment of the image capture
    robot.rtde_control.endTeachMode()

    ROBOT_STOP_WAIT_TIME = 0.5
    time.sleep(ROBOT_STOP_WAIT_TIME)

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


def collect_calibration_data(robot: PositionManipulator, camera: RGBCamera, calibration_dir: Optional[str]) -> None:
    """Collect calibration data samples for hand-eye calibration.

    Args:
        robot: the robot to use for collecting the data.
        camera: the camera to use for collecting the data.
        calibration_dir: directory to save the calibration data to, if None a directory will be created
    """
    from loguru import logger

    data_dir = create_calibration_data_dir(calibration_dir)

    logger.info(f"Saving calibration data to {data_dir}")
    logger.info("Press S to save a sample, Q to quit.")

    resolution = camera.resolution
    intrinsics = camera.intrinsics_matrix()

    # Saving the intrinsics
    camera_intrinsics = CameraIntrinsics.from_matrix_and_resolution(intrinsics, resolution)
    intrinsics_filepath = os.path.join(data_dir, "intrinsics.json")
    with open(intrinsics_filepath, "w") as f:
        json.dump(camera_intrinsics.model_dump(exclude_none=True), f, indent=4)

    window_name = "Calibration data collection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # For now, the robot is assumed to be a UR robot with RTDE interface, as we make use of the teach mode functions.
    robot.rtde_control.teachMode()  # type: ignore
    sample_index = 0

    while True:
        # Live visualization of board detection
        image_rgb = camera.get_rgb_image_as_int()
        image = ImageConverter.from_numpy_int_format(image_rgb).image_in_opencv_format
        detect_and_visualize_charuco_pose(image, intrinsics)
        cv2.imshow(window_name, image)

        key = cv2.waitKey(1)
        if key == ord("q"):
            robot.rtde_control.endTeachMode()  # type: ignore
            break

        if key == ord("s"):
            save_calibration_sample(sample_index, robot, camera, data_dir)  # type: ignore
            sample_index += 1
            logger.info(f"Saved {sample_index} sample(s).")


@click.command()
@click.option("--robot_ip", default="10.42.0.162", help="robot ip address")
@click.option("--calibration_dir", type=click.Path(exists=False), help="directory to save the calibration data to.")
@click_camera_options
def collect_calibration_data_with_ur(
    robot_ip: str,
    calibration_dir: Optional[str] = None,
    camera_brand: Optional[str] = None,
    camera_serial_number: Optional[str] = None,
) -> None:
    """Script to collect calibration data for hand-eye calibration with a UR robot."""
    from airo_robots.manipulators.hardware.ur_rtde import URrtde

    robot = URrtde(robot_ip, URrtde.UR3_CONFIG)
    camera = discover_camera(camera_brand, camera_serial_number)
    collect_calibration_data(robot, camera, calibration_dir)


if __name__ == "__main__":
    collect_calibration_data_with_ur()
