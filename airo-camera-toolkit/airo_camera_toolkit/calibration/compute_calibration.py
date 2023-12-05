import datetime
import glob
import json
import os
from typing import List, Optional, Tuple

import click
import cv2
import numpy as np
from airo_camera_toolkit.calibration.fiducial_markers import (
    AIRO_DEFAULT_ARUCO_DICT,
    AIRO_DEFAULT_CHARUCO_BOARD,
    ArucoDictType,
    CharucoBoardType,
    detect_charuco_board,
    draw_frame_on_image,
)
from airo_dataset_tools.data_parsers.camera_intrinsics import CameraIntrinsics
from airo_dataset_tools.data_parsers.pose import Pose
from airo_spatial_algebra import SE3Container
from airo_typing import CameraIntrinsicsMatrixType, CameraResolutionType, HomogeneousMatrixType, OpenCVIntImageType
from loguru import logger

cv2_CALIBRATION_METHODS = {
    "Tsai": cv2.CALIB_HAND_EYE_TSAI,
    "Park": cv2.CALIB_HAND_EYE_PARK,
    "Haraud": cv2.CALIB_HAND_EYE_HORAUD,
    "Andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "Daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def compute_hand_eye_calibration_error(
    tcp_poses_in_base: List[HomogeneousMatrixType],
    board_poses_in_camera: List[HomogeneousMatrixType],
    camera_pose: HomogeneousMatrixType,
) -> float:
    """Compute the error between the left and right side of the AX=XB equation to have an estimate of the error of the
    calibration. In our experience, average error below 0.01 are pretty good.

    Args:
        tcp_poses_in_base: list of tcp poses in base frame
        board_poses_in_camera: list of marker poses in camera frame
        camera_pose: camera pose in base frame (eye-to-hand) or camera pose in tcp frame(eye-in-hand))
    """
    error = 0.0
    for i in range(len(tcp_poses_in_base) - 1):
        tcp_pose_in_base = tcp_poses_in_base[i]
        board_pose_in_camera = board_poses_in_camera[i]

        tcp_pose_in_base_2 = tcp_poses_in_base[i + 1]
        board_pose_in_camera_2 = board_poses_in_camera[i + 1]

        # cf https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gaebfc1c9f7434196a374c382abf43439b
        # for the AX=XB equation
        left_side = tcp_pose_in_base @ camera_pose @ board_pose_in_camera
        right_side = tcp_pose_in_base_2 @ camera_pose @ board_pose_in_camera_2
        error += float(np.linalg.norm(left_side - right_side))
    return error / (len(tcp_poses_in_base) - 1)


def eye_in_hand_pose_estimation(
    tcp_poses_in_base: List[HomogeneousMatrixType],
    board_poses_in_camera: List[HomogeneousMatrixType],
    method: int = cv2.CALIB_HAND_EYE_ANDREFF,
) -> Tuple[Optional[HomogeneousMatrixType], Optional[float]]:
    """Wrapper around the opencv eye-in-hand extrinsics calibration function.

    Args:
        tcp_poses_in_base: list of tcp poses in base frame
        board_poses_in_camera: list of marker poses in camera frame
        method: one of the cv2.CALIB_HAND_EYE_* methods
    """
    tcp_orientations_as_rotvec_in_base = [
        SE3Container.from_homogeneous_matrix(tcp_pose).orientation_as_rotation_vector for tcp_pose in tcp_poses_in_base
    ]
    tcp_positions_in_base = [
        SE3Container.from_homogeneous_matrix(tcp_pose).translation for tcp_pose in tcp_poses_in_base
    ]

    marker_orientations_as_rotvec_in_camera = [
        SE3Container.from_homogeneous_matrix(board_pose).orientation_as_rotation_vector
        for board_pose in board_poses_in_camera
    ]
    marker_positions_in_camera = [
        SE3Container.from_homogeneous_matrix(board_pose).translation for board_pose in board_poses_in_camera
    ]

    # When running with duplicated tcp poses, I've had this error:
    # error: (-7:Iterations do not converge) Rotation normalization issue: determinant(R) is null in function 'normalizeRotation'
    try:
        camera_rotation_matrix, camera_translation = cv2.calibrateHandEye(
            tcp_orientations_as_rotvec_in_base,
            tcp_positions_in_base,
            marker_orientations_as_rotvec_in_camera,
            marker_positions_in_camera,
            None,
            None,
            method,
        )
    except cv2.error:
        return None, None

    if camera_rotation_matrix is None or camera_translation is None:
        return None, None

    # We've noticed that the OpenCV output can contains NaNs, which crashes here.
    try:
        camera_pose_in_tcp_frame = SE3Container.from_rotation_matrix_and_translation(
            camera_rotation_matrix, camera_translation
        ).homogeneous_matrix
    except ValueError:
        return None, None

    camera_pose_in_tcp_frame = SE3Container.from_rotation_matrix_and_translation(
        camera_rotation_matrix, camera_translation
    ).homogeneous_matrix

    calibration_error = compute_hand_eye_calibration_error(
        tcp_poses_in_base, board_poses_in_camera, camera_pose_in_tcp_frame
    )
    return camera_pose_in_tcp_frame, calibration_error


def eye_to_hand_pose_estimation(
    tcp_poses_in_base: List[HomogeneousMatrixType],
    board_poses_in_camera: List[HomogeneousMatrixType],
    method: int = cv2.CALIB_HAND_EYE_ANDREFF,
) -> Tuple[Optional[HomogeneousMatrixType], Optional[float]]:
    """Wrapper around the opencv eye-to-hand extrinsics calibration function.

    Args:
        tcp_poses_in_base: list of tcp poses in base frame
        board_poses_in_camera: list of marker poses in camera frame
        method: one of the cv2.CALIB_HAND_EYE_* methods
    """
    #  Invert the tcp_poses to make the AX=XB problem for eye_to_hand mode equivalent to the eye_in_hand mode.
    # cf https://docs.opencv.org/4.5.4/d9/d0c/group__calib3d.html#gaebfc1c9f7434196a374c382abf43439b
    # cf https://forum.opencv.org/t/eye-to-hand-calibration/5690/2
    base_pose_in_tcp_frame = [np.linalg.inv(tcp_pose) for tcp_pose in tcp_poses_in_base]

    camera_pose_in_base, calibration_error = eye_in_hand_pose_estimation(
        base_pose_in_tcp_frame, board_poses_in_camera, method
    )
    return camera_pose_in_base, calibration_error


def compute_calibration(
    board_poses_in_camera: List[HomogeneousMatrixType],
    tcp_poses_in_base: List[HomogeneousMatrixType],
    mode: str = "eye_in_hand",
    method: int = cv2.CALIB_HAND_EYE_ANDREFF,
) -> Tuple[Optional[HomogeneousMatrixType], Optional[float]]:
    """Compute the calibration for a given mode and method.

    Args:
        board_poses_in_camera: list of marker poses in camera frame
        tcp_poses_in_base: list of tcp poses in base frame
        mode: one of "eye_in_hand" or "eye_to_hand"
        method: one of the cv2.CALIB_HAND_EYE_* methods

    Returns:
        camera_pose: if successful, camera pose in base frame (eye-to-hand) or camera pose in tcp frame(eye-in-hand))
        calibration_error: error of the calibration
    """
    if mode == "eye_in_hand":
        # pose of camera in tcp frame
        camera_pose, calibration_error = eye_in_hand_pose_estimation(tcp_poses_in_base, board_poses_in_camera, method)
    elif mode == "eye_to_hand":
        # pose of camera in base frame
        camera_pose, calibration_error = eye_to_hand_pose_estimation(tcp_poses_in_base, board_poses_in_camera, method)
    else:
        raise ValueError(f"Unknown mode {mode}")

    return camera_pose, calibration_error


def save_board_detections(
    results_dir: str,
    board_poses_in_camera: List[Optional[HomogeneousMatrixType]],
    images: List[OpenCVIntImageType],
    intrinsics: CameraIntrinsicsMatrixType,
) -> None:
    """Convenience function to that saves jpg images of with the board pose drawn on it.

    Args:
        results_dir: directory to save the results to, must exist
        board_poses_in_camera: list of marker poses in camera frame, may contain None
        images: list of images of the calibration board
        intrinsics: camera intrinsics
    """

    board_detections_dir = os.path.join(results_dir, "board_detections")
    os.makedirs(board_detections_dir)
    for i, (board_pose, image) in enumerate(zip(board_poses_in_camera, images)):
        image_annotated = image.copy()
        if board_pose is None:
            continue
        draw_frame_on_image(image_annotated, board_pose, intrinsics)
        detection_filepath = os.path.join(board_detections_dir, f"board_detection_{i:04d}.jpg")
        cv2.imwrite(detection_filepath, image_annotated)


def draw_base_pose_on_image(
    image: OpenCVIntImageType,
    intrinsics: CameraIntrinsicsMatrixType,
    camera_pose: Optional[HomogeneousMatrixType],
    mode: str = "eye_in_hand",
    tcp_pose: Optional[HomogeneousMatrixType] = None,
) -> None:
    """Draws the robot's base pose on an image, using the camera_pose resulting from the calibration.

    Args:
        image: image to draw on
        camera_pose: camera pose in base frame (eye-to-hand) or camera pose in tcp frame(eye-in-hand))
        intrinsics: camera intrinsics
        mode: one of "eye_in_hand" or "eye_to_hand"
        tcp_pose: tcp pose in base frame that corresponds to the image
    """
    if camera_pose is None:
        return

    if mode == "eye_to_hand":
        X_B_C = camera_pose  # Camera in base frame
        X_C_B = np.linalg.inv(X_B_C)
    if mode == "eye_in_hand":
        if tcp_pose is None:
            return  # tcp pose is required to visualize base in eye_in_hand mode

        X_TCP_C = camera_pose  # Camera in TCP frame
        X_B_TCP = tcp_pose
        X_C_TCP = np.linalg.inv(X_TCP_C)
        X_TCP_B = np.linalg.inv(X_B_TCP)
        X_C_B = X_C_TCP @ X_TCP_B

    base_pose_in_camera = X_C_B
    draw_frame_on_image(image, base_pose_in_camera, intrinsics)


def compute_calibration_all_methods(
    results_dir: str,
    images: List[OpenCVIntImageType],
    tcp_poses_in_base: List[HomogeneousMatrixType],
    intrinsics: CameraIntrinsicsMatrixType,
    mode: str = "eye_in_hand",
    aruco_dict: ArucoDictType = AIRO_DEFAULT_ARUCO_DICT,
    charuco_board: CharucoBoardType = AIRO_DEFAULT_CHARUCO_BOARD,
) -> Tuple[dict, dict]:
    """Computes the calibration solution for all methods available in OpenCV and saves the results to a directory.

    Args:
        results_dir: directory to save the results to, must exist
        images: list of images of the calibration board
        tcp_poses_in_base: list of tcp poses in base frame
        intrinsics: camera intrinsics
        mode: one of "eye_in_hand" or "eye_to_hand"
        aruco_dict: aruco dictionary
        charuco_board: charuco board

    Returns:
        calibration_result_poses: dictionary of the camera pose for each method
        calibration_errors: dictionary of the calibration error for each method
    """
    calibration_errors_filepath = os.path.join(results_dir, "residual_errors.json")
    calibration_errors = {}
    calibration_result_poses = {}

    board_poses_in_camera = [
        detect_charuco_board(image, intrinsics, aruco_dict=aruco_dict, charuco_board=charuco_board) for image in images
    ]

    save_board_detections(results_dir, board_poses_in_camera, images, intrinsics)

    # Removes poses where no board was detected
    tcp_poses_in_base = [
        tcp_poses_in_base[i] for i, board_pose in enumerate(board_poses_in_camera) if board_pose is not None
    ]
    board_poses_in_camera: List[HomogeneousMatrixType] = [  # type: ignore
        board_pose for board_pose in board_poses_in_camera if board_pose is not None
    ]
    logger.info(f"Board poses were detected in {len(board_poses_in_camera)} of the calibration samples.")

    for name, method in cv2_CALIBRATION_METHODS.items():
        camera_pose, calibration_error = compute_calibration(board_poses_in_camera, tcp_poses_in_base, mode, method)  # type: ignore
        if calibration_error is None:
            calibration_error = np.inf

        logger.info(f"Residual error {name}: {calibration_error:.4f}")

        calibration_errors[name] = calibration_error
        calibration_result_poses[name] = camera_pose

        with open(calibration_errors_filepath, "w") as f:
            json.dump(calibration_errors, f, indent=4)

        if camera_pose is None:
            continue

        # Save the camera pose
        pose_path = os.path.join(results_dir, f"camera_pose_{name}.json")
        pose_saveable = Pose.from_homogeneous_matrix(camera_pose)
        with open(pose_path, "w") as f:
            json.dump(pose_saveable.model_dump(), f, indent=4)

        # Save an image with the pose drawn on it (use last image taken)
        image = images[-1].copy()
        draw_base_pose_on_image(image, intrinsics, camera_pose, mode, tcp_poses_in_base[-1])

        # Write residual error on image
        error_str = f"{name}: {calibration_error:.4f}"
        cv2.putText(image, error_str, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(results_dir, f"base_pose_in_camera_{name}.jpg"), image)

    return calibration_result_poses, calibration_errors


def load_calibration_data(
    calibration_dir: str,
) -> Tuple[List[OpenCVIntImageType], List[HomogeneousMatrixType], CameraIntrinsicsMatrixType, CameraResolutionType]:
    """Function to load calibration samples and camera parameters from a "data" directory in a calibration_dir

    Args:
        calibration_dir: directory containing the "data" directory

    Returns:
        Calibration data and camera parameters
    """
    data_dir = os.path.join(calibration_dir, "data")

    # Loading the intrinsics and resolution
    intrinsics_path = os.path.join(data_dir, "intrinsics.json")
    with open(intrinsics_path, "r") as f:
        camera_intrinsics = CameraIntrinsics.model_validate_json(f.read())

    resolution = camera_intrinsics.image_resolution.as_tuple()
    intrinsics = camera_intrinsics.as_matrix()

    image_paths = sorted(glob.glob(os.path.join(data_dir, "image_*.png")))
    pose_paths = sorted(glob.glob(os.path.join(data_dir, "tcp_pose_*.json")))

    images = [cv2.imread(image_path) for image_path in image_paths]
    tcp_poses = []
    for filepath in pose_paths:
        with open(filepath, "r") as f:
            pose = Pose.model_validate_json(f.read())
        tcp_poses.append(pose.as_homogeneous_matrix())

    return images, tcp_poses, intrinsics, resolution


@click.command()
@click.argument(
    "calibration_dir",
    type=click.Path(exists=True),
)
@click.option("--mode", default="eye_in_hand", help="eye_in_hand or eye_to_hand")
def compute_calibration_from_saved_data(calibration_dir: str, mode: str = "eye_in_hand") -> None:
    """Runs all OpenCV calibration methods on the data saved in a calibration directory."""
    images, tcp_poses, intrinsics, _ = load_calibration_data(calibration_dir)

    results_dir = os.path.join(calibration_dir, "results")
    if os.path.exists(results_dir):
        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        results_dir = os.path.join(calibration_dir, f"results_{datetime_str}")
    os.makedirs(results_dir)

    logger.info(f"Saving calibration results to {results_dir}")

    compute_calibration_all_methods(results_dir, images, tcp_poses, intrinsics, mode)


if __name__ == "__main__":
    compute_calibration_from_saved_data()
