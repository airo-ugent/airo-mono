from dataclasses import dataclass

import cv2
import numpy as np
from airo_typing import OpenCVIntImageType
from cv2 import aruco

Image = OpenCVIntImageType


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
    # marker_corners = np.concatenate(marker_corners, axis=0,dtype=np.float32)
    # marker_corners = refine_corner_detection(image,marker_corners)
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
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 50, 0.01)
    corner_group = corners.shape[1]
    corners = np.reshape(corners, (-1, 2))
    corners = cv2.cornerSubPix(to_gray(image), corners, (5, 5), (-1, -1), term)
    corners = np.reshape(corners, (-1, corner_group, 2))
    return corners


if __name__ == "__main__":
    import pathlib

    path = pathlib.Path(__file__).parent

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    image = cv2.imread(str(path / "charuco_3x4.png"))
    print(image.shape)
    result = detect_aruco_markers(image, aruco_dict)
    print(result.corners[0].dtype)

    aruco.drawDetectedMarkers(image, np.array(result.corners), result.ids)

    charuco_board = aruco.CharucoBoard((4, 3), 0.04, 0.031, aruco_dict)
    # img = charuco_board.generateImage((600, 600))
    # cv2.imwrite("charuco_board.png", img)
    result = detect_charuco_corners(image, result, charuco_board)
    print(np.array(result.corners))
    print(np.array(result.ids))
    aruco.drawDetectedCornersCharuco(image, np.array(result.corners), np.array(result.ids), (255, 255, 0))

    cv2.imshow("image", image)
    cv2.waitKey(0)
