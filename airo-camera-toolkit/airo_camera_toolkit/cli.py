"""CLI interface for this package"""

from typing import Optional

import click
from airo_camera_toolkit.calibration.fiducial_markers import AIRO_DEFAULT_ARUCO_DICT, AIRO_DEFAULT_CHARUCO_BOARD
from airo_camera_toolkit.calibration.hand_eye_calibration import do_camera_robot_calibration
from airo_camera_toolkit.cameras.camera_discovery import click_camera_options, discover_camera


@click.group()
def cli() -> None:
    """CLI entrypoint for airo-camera-toolkit"""


@cli.command(name="hand-eye-calibration")
@click.option("--mode", default="eye_in_hand", help="eye_in_hand or eye_to_hand")
@click.option("--robot_ip", default="10.42.0.162", help="robot ip address")
@click.option("--calibration_dir", type=click.Path(exists=False), help="directory to save the calibration data to.")
@click_camera_options
def calibrate_with_ur(
    mode: str,
    robot_ip: str,
    calibration_dir: Optional[str] = None,
    camera_brand: Optional[str] = None,
    camera_serial_number: Optional[str] = None,
) -> None:
    """Do hand-eye calibration with a UR robot. Will open camera stream and visualize the detected board
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
    from airo_robots.manipulators.hardware.ur_rtde import URrtde

    aruco_dict = AIRO_DEFAULT_ARUCO_DICT
    charuco_board = AIRO_DEFAULT_CHARUCO_BOARD

    robot = URrtde(robot_ip, URrtde.UR3_CONFIG)

    camera = discover_camera(camera_brand, camera_serial_number)
    do_camera_robot_calibration(mode, aruco_dict, charuco_board, camera, robot, calibration_dir)


if __name__ == "__main__":
    cli()
