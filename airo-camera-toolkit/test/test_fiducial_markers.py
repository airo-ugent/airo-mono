from test.test_config import _CalibrationTest

import cv2
import numpy as np
from airo_camera_toolkit.calibration.fiducial_markers import (
    AIRO_DEFAULT_ARUCO_DICT,
    AIRO_DEFAULT_CHARUCO_BOARD,
    detect_aruco_markers,
    detect_charuco_corners,
    get_pose_of_charuco_board,
    get_poses_of_aruco_markers,
)


def test_empty_aruco_marker_detection():
    empty_marker_image = cv2.imread(str(_CalibrationTest._empty_image_path))
    detections = detect_aruco_markers(empty_marker_image, AIRO_DEFAULT_ARUCO_DICT)
    assert detections is None


def test_aruco_marker_detection():
    charuco_board_image = cv2.imread(str(_CalibrationTest._default_charuco_board_path))
    detections = detect_aruco_markers(charuco_board_image, AIRO_DEFAULT_ARUCO_DICT)

    n_arucos = 17
    assert detections is not None
    assert len(detections.ids) == n_arucos
    assert detections.corners.shape == (n_arucos, 1, 4, 2)


def test_aruco_pose_estimation_on_marker_image():
    intrinsics = np.eye(3)
    charuco_board_image = cv2.imread(str(_CalibrationTest._default_charuco_board_path))
    detections = detect_aruco_markers(charuco_board_image, AIRO_DEFAULT_ARUCO_DICT)
    assert detections is not None
    poses = get_poses_of_aruco_markers(detections, 1.0, intrinsics, None)

    # charuco_board_image = visualize_aruco_detections(charuco_board_image, detections)
    # charuco_board_image = draw_frame_on_image(charuco_board_image, poses[-1], intrinsics)
    # cv2.imshow("aruco", charuco_board_image)
    # cv2.waitKey(0)

    assert isinstance(poses, list)
    assert len(poses) == len(detections.ids)
    assert poses[0].shape == (4, 4)

    top_left_marker_pose = poses[np.where(detections.ids == 0)[0][0]]

    # the top left marker should be rotated 180 degrees around the x axis
    # since the z-axis should point towards the camera and the y-axis should point up
    # (this is how aruco frames are defined)
    x_rotated_matrix = np.eye(3)
    x_rotated_matrix[1, 1] = x_rotated_matrix[2, 2] = -1
    assert np.isclose(top_left_marker_pose[:3, :3], x_rotated_matrix, atol=1e-3).all()

    # this was manually measured for the identity intrinsic matrix
    assert np.isclose(top_left_marker_pose[:3, 3], np.array([2.26, 9.66e-1, 4.10e-3]), atol=1e-2).all()


def test_charuco_pose_estimation_on_marker_image():
    intrinsics = np.eye(3)

    charuco_board_image = cv2.imread(str(_CalibrationTest._default_charuco_board_path))
    detections = detect_aruco_markers(charuco_board_image, AIRO_DEFAULT_ARUCO_DICT)
    assert detections is not None
    charuco_corners = detect_charuco_corners(charuco_board_image, detections, AIRO_DEFAULT_CHARUCO_BOARD)
    assert charuco_corners is not None
    pose = get_pose_of_charuco_board(charuco_corners, AIRO_DEFAULT_CHARUCO_BOARD, intrinsics, None)

    # can uncomment this to visualize the pose estimation

    # charuco_board_image = visualize_charuco_detection(charuco_board_image, charuco_corners)
    # charuco_board_image = draw_frame_on_image(charuco_board_image, pose, intrinsics)
    # cv2.imshow("charuco", charuco_board_image)
    # cv2.waitKey(0)
    assert pose.shape == (4, 4)
    # these are the expected values for the default charuco board
    # which where measured by hand
    assert np.isclose(pose[:3, :3], np.eye(3), atol=1e-4).all()
    assert np.isclose(pose[:3, 3], np.array([0.01, 0.01, 0]), atol=1e-3).all()
