"""Aruco marker & Charuco Board detection and pose estimation.
Use charuco whenever you can, as their checkerboard corners can be detected more accurately and there are more points for the PnP solver to work with.

This code is partially based on a codebase by Peter De Roovere (https://github.com/pderoovere),
so part of the credit for this code goes to him.
"""
from dataclasses import dataclass
from typing import Any, List, Optional

import cv2
import numpy as np
from airo_camera_toolkit.reprojection import project_frame_to_image_plane
from airo_spatial_algebra import SE3Container
from airo_typing import CameraIntrinsicsMatrixType, HomogeneousMatrixType, OpenCVIntImageType
from cv2 import aruco

# explicitly type these to Any, as the cv2.aruco module is not typed
# and would be cast by mypy to Any implictly, which results in a warning
ArucoDictType = Any
CharucoDictType = Any

# see the pdf file in the airo-camera-toolkit/docs folder
AIRO_DEFAULT_ARUCO_DICT: ArucoDictType = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
AIRO_DEFAULT_CHARUCO_BOARD: CharucoDictType = aruco.CharucoBoard((7, 5), 0.04, 0.031, AIRO_DEFAULT_ARUCO_DICT)


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


#############
# detection #
#############


def detect_aruco_markers(image: OpenCVIntImageType, dictionary: ArucoDictType) -> Optional[ArucoMarkerDetectionResult]:
    """Detect markers from `aruco_dict` dictionary in the `image`."""
    marker_corners, marker_ids, _ = aruco.detectMarkers(image, dictionary)
    if marker_corners is None or marker_ids is None:
        return None
    marker_corners = np.stack(marker_corners)
    marker_corners = refine_corner_detection(image, marker_corners)
    result = ArucoMarkerDetectionResult(marker_corners, marker_ids, image)
    return result


def detect_charuco_corners(
    image: OpenCVIntImageType, markers_detection_result: ArucoMarkerDetectionResult, charuco_board: CharucoDictType
) -> Optional[CharucoCornerDetectionResult]:
    """Detect CharuCo corners in the image using the detected markers in `markers_detection_result`."""
    nb_corners, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
        markerCorners=markers_detection_result.corners,
        markerIds=markers_detection_result.ids,
        image=image,
        board=charuco_board,
    )
    if charuco_corners is None or charuco_ids is None:
        return None
    charuco_corners = refine_corner_detection(image, charuco_corners)
    result = CharucoCornerDetectionResult(charuco_corners, charuco_ids, image)
    return result


def refine_corner_detection(image: OpenCVIntImageType, corners: np.ndarray) -> np.ndarray:
    """Refine detected corners with sub-pixel accuracy."""

    # https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga354e0d7c86d0d9da75de9b9701a9a87e
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 0.1)
    corners_shape = corners.shape
    corners = np.reshape(corners, (-1, 2))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # use a small window size, to avoid influence of a neighboring marker/ checkerboard tile
    # even then this sometimes gave worse results than without the refinement, so keep an eye on this
    corners = cv2.cornerSubPix(gray_image, corners, (3, 3), (-1, -1), term)
    corners = np.reshape(corners, corners_shape)
    return corners


###################
# pose estimation #
###################


def get_poses_of_aruco_markers(
    markers_detection_result: ArucoMarkerDetectionResult,
    marker_size: float,
    camera_matrix: CameraIntrinsicsMatrixType,
    dist_coeffs: Optional[np.ndarray] = None,
) -> Optional[List[HomogeneousMatrixType]]:
    """Get the poses of the detected markers in `markers_detection_result`."""
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
    marker_poses_in_camera_frame = [
        SE3Container.from_rotation_vector_and_translation(rvec[0], tvec).homogeneous_matrix
        for rvec, tvec in zip(rvecs, tvecs)
    ]
    return marker_poses_in_camera_frame


def get_pose_of_charuco_board(
    charuco_corners_detection_result: CharucoCornerDetectionResult,
    charuco_board: CharucoDictType,
    camera_matrix: CameraIntrinsicsMatrixType,
    dist_coeffs: Optional[np.ndarray] = None,
) -> Optional[HomogeneousMatrixType]:
    """Get the pose of the detected CharuCo board in `charuco_corners_detection_result`.
    The origin of the charuco frame is defined in the topleft corner of the board."""
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
    charuco_pose_in_camera_frame = SE3Container.from_rotation_vector_and_translation(
        rvec.flatten(), tvec.flatten()
    ).homogeneous_matrix
    return charuco_pose_in_camera_frame


#################
# visualization #
#################


def draw_frame_on_image(
    image: OpenCVIntImageType, frame_pose_in_camera: HomogeneousMatrixType, camera_matrix: CameraIntrinsicsMatrixType
) -> OpenCVIntImageType:
    """draws a 2D projection of a frame on the iamge. Be careful when interpreting this visually, it is often hard to estimate the true 3D direction of an axis' 2D projection."""
    project_points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    origin, x_pos, y_pos, z_pos = project_frame_to_image_plane(
        project_points, camera_matrix, frame_pose_in_camera
    ).astype(int)
    image = cv2.line(image, x_pos, origin, color=(0, 0, 255), thickness=2)
    image = cv2.line(image, y_pos, origin, color=(0, 255, 0), thickness=2)
    image = cv2.line(image, z_pos, origin, color=(255, 0, 0), thickness=2)
    return image


def visualize_aruco_detections(
    image: OpenCVIntImageType, aruco_result: ArucoMarkerDetectionResult
) -> OpenCVIntImageType:
    """draws the aruco marker countours/corners and their IDs on the image"""
    image = aruco.drawDetectedMarkers(image, [x for x in aruco_result.corners], aruco_result.ids)
    return image


def visualize_charuco_detection(image: OpenCVIntImageType, result: CharucoCornerDetectionResult) -> OpenCVIntImageType:
    """draws the charuco checkerboard corners and their IDs on the image"""
    image = aruco.drawDetectedCornersCharuco(image, np.array(result.corners), np.array(result.ids), (255, 255, 0))
    return image


if __name__ == "__main__":  # noqa: C901 - ignore complexity
    """CLI script for live visualisation of marker detection and pose estimation
    run python -m <file-name> --help to see the available options in the terminal.
    Defaults to the AIRO_DEFAULT_CHARUCO_BOARD.
    """
    import click
    from airo_camera_toolkit.cameras.zed2i import Zed2i
    from airo_camera_toolkit.utils import ImageConverter

    @click.command()
    @click.option("--aruco_marker_size", default=0.031, help="Size of the aruco marker in meters")
    @click.option("--charuco_x_count", default=7, help="Number of checkerboard tiles in the x direction")
    @click.option("--charuco_y_count", default=5, help="Number of checkerboard tiles in the y direction")
    @click.option("--charuco_tile_size", default=0.04, help="Size of the charuco checkerboard tiles in meters")
    def visualize_marker_detections(
        aruco_marker_size: float,
        charuco_x_count: Optional[int] = None,
        charuco_y_count: Optional[int] = None,
        charuco_tile_size: Optional[int] = None,
    ) -> None:

        aruco_dict = AIRO_DEFAULT_ARUCO_DICT
        detect_charuco = charuco_x_count is not None and charuco_y_count is not None and charuco_tile_size is not None
        if detect_charuco:
            charuco_board = aruco.CharucoBoard(
                (charuco_x_count, charuco_y_count), charuco_tile_size, aruco_marker_size, aruco_dict
            )

        camera = Zed2i()

        print("press Q to quit")
        while True:
            aruco_result = None
            charuco_result = None
            aruco_poses = None
            charuco_pose = None
            image = camera.get_rgb_image()
            image = ImageConverter.from_numpy_format(image).image_in_opencv_format

            intrinsics = camera.intrinsics_matrix()

            aruco_result = detect_aruco_markers(image, aruco_dict)

            if aruco_result:
                aruco_poses = get_poses_of_aruco_markers(aruco_result, 0.04, intrinsics)

            if aruco_result:
                charuco_result = detect_charuco_corners(image, aruco_result, charuco_board)

                if charuco_result:
                    charuco_pose = get_pose_of_charuco_board(charuco_result, charuco_board, intrinsics)
            if aruco_result:
                image = visualize_aruco_detections(image, aruco_result)
                if aruco_poses is not None:
                    image = draw_frame_on_image(image, aruco_poses[0], intrinsics)

            if charuco_result:
                image = visualize_charuco_detection(image, charuco_result)
                if charuco_pose is not None:
                    image = draw_frame_on_image(image, charuco_pose, intrinsics)

            image = cv2.resize(image, (1024, 768))
            cv2.imshow("image", image)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break

    visualize_marker_detections()
