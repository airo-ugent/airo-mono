from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
from airo_camera_toolkit.reprojection import project_frame_to_image_plane
from airo_spatial_algebra import SE3Container
from airo_typing import CameraIntrinsicsMatrixType, HomogeneousMatrixType, OpenCVIntImageType
from cv2 import aruco


def to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


@dataclass
class ArucoMarkerDetectionResult:
    corners: np.ndarray  # (N,4,2)
    ids: np.ndarray  # (N,1)
    image: OpenCVIntImageType


@dataclass
class CharucoCornerDetectionResult(ArucoMarkerDetectionResult):
    # corners: np.ndarray(M,1,2)
    # ids: np.ndarray(M,1)
    # image: OpenCVIntImageType
    pass


###############
# Calibration #
###############


def detect_aruco_markers(image: OpenCVIntImageType, dictionary: aruco.Dictionary) -> ArucoMarkerDetectionResult:
    """Detect `aruco_dict` markers in `image`."""
    marker_corners, marker_ids, _ = aruco.detectMarkers(image, dictionary)
    print(type(marker_corners))
    print(type(marker_corners[0]))
    marker_corners = np.stack(marker_corners, dtype=np.float32)
    marker_corners = refine_corner_detection(image, marker_corners)
    result = ArucoMarkerDetectionResult(marker_corners, marker_ids, image)
    return result


def detect_charuco_corners(
    image: OpenCVIntImageType, markers_detection_result: ArucoMarkerDetectionResult, charuco_board
) -> CharucoCornerDetectionResult:
    """Detect CharuCo corners in `image` using the detected markers in `markers_detection_result`."""
    nb_corners, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
        markerCorners=markers_detection_result.corners,
        markerIds=markers_detection_result.ids,
        image=image,
        board=charuco_board,
    )
    charuco_corners = refine_corner_detection(image, charuco_corners)
    result = CharucoCornerDetectionResult(charuco_corners, charuco_ids, image)
    return result


def refine_corner_detection(image: OpenCVIntImageType, corners: np.ndarray) -> np.ndarray:
    """Refine the corners with sub-pixel accuracy."""

    # https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga354e0d7c86d0d9da75de9b9701a9a87e
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 50, 0.1)
    corners_shape = corners.shape
    corners = np.reshape(corners, (-1, 2))
    corners = cv2.cornerSubPix(to_gray(image), corners, (5, 5), (-1, -1), term)
    corners = np.reshape(corners, corners_shape)
    return corners


# pose estimation


def get_poses_of_aruco_markers(
    markers_detection_result: ArucoMarkerDetectionResult,
    marker_size: float,
    camera_matrix: CameraIntrinsicsMatrixType,
    dist_coeffs=None,
) -> List[HomogeneousMatrixType]:
    """Get the pose of the detected markers in `markers_detection_result`."""
    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
        corners=markers_detection_result.corners,
        markerLength=marker_size,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
    )
    if rvecs is None and tvecs is None:
        return None
    elif rvecs.shape != tvecs.shape:
        raise ValueError("rvecs and tvecs should have the same shape. Do you have multiple markers with the same ID?")

    # combine the rvecs and tvecs into a single pose matrix
    poses = [
        SE3Container.from_rotation_vector_and_translation(rvec[0], tvec).homogeneous_matrix
        for rvec, tvec in zip(rvecs, tvecs)
    ]
    return poses


def draw_frame_on_image(image, world_in_camera, camera_matrix):
    project_points = (
        np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        * 0.001
    )

    origin, x_pos, x_neg, y_pos, y_neg, z_pos = project_frame_to_image_plane(
        project_points, camera_matrix, world_in_camera
    ).astype(int)
    image = cv2.circle(image, origin, 10, (0, 255, 255), thickness=2)
    image = cv2.line(image, x_pos, origin, color=(0, 0, 255), thickness=2)
    # image = cv2.line(image, x_neg, origin, color=(100, 100, 255), thickness=2)
    image = cv2.line(image, y_pos, origin, color=(0, 255, 0), thickness=2)
    # image = cv2.line(image, y_neg, origin, color=(150, 255, 150), thickness=2)
    image = cv2.line(image, z_pos, origin, color=(255, 0, 0), thickness=2)
    return image


def get_pose_of_charuco_board(
    charuco_corners_detection_result: CharucoCornerDetectionResult,
    charuco_board,
    camera_matrix,
    dist_coeffs=None,
) -> np.ndarray:
    """Get the pose of the detected markers in `markers_detection_result`."""
    valid, rvec, tvec = aruco.estimatePoseCharucoBoard(
        charucoCorners=charuco_corners_detection_result.corners,
        charucoIds=charuco_corners_detection_result.ids,
        board=charuco_board,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
        rvec=None,
        tvec=None,
    )
    if (rvec is None and tvec is None) or not valid:
        return None
    # combine the rvec and tvec into a single pose matrix
    pose = SE3Container.from_rotation_vector_and_translation(rvec.flatten(), tvec.flatten()).homogeneous_matrix
    return pose


if __name__ == "__main__":
    import pathlib

    path = pathlib.Path(__file__).parent

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    image = cv2.imread(str(path / "charuco_board.png"))
    print(image.shape)
    aruco_result = detect_aruco_markers(image, aruco_dict)

    aruco.drawDetectedMarkers(image, [x for x in aruco_result.corners], aruco_result.ids)

    charuco_board = aruco.CharucoBoard((4, 3), 0.04, 0.031, aruco_dict)
    # img = charuco_board.generateImage((600, 600))
    # cv2.imwrite("charuco_board.png", img)

    # result.ids = result.ids[:-1]
    # result.corners = result.corners[:-1]
    result = detect_charuco_corners(image, aruco_result, charuco_board)
    print(np.array(result.corners))
    print(np.array(result.ids))
    aruco.drawDetectedCornersCharuco(image, np.array(result.corners), np.array(result.ids), (255, 255, 0))

    poses = get_poses_of_aruco_markers(aruco_result, 0.04, np.eye(3), np.zeros(5))
    pose = get_pose_of_charuco_board(result, charuco_board, np.eye(3), np.zeros(5))
    print(poses[0])
    print(pose)
    image = draw_frame_on_image(image, poses[0], np.eye(3))
    image = draw_frame_on_image(image, pose, np.eye(3))

    cv2.imshow("image", image)
    cv2.waitKey(0)
