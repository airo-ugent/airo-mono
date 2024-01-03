from typing import Union

from airo_spatial_algebra.operations import _HomogeneousPoints
from airo_typing import CameraIntrinsicsMatrixType, Vector2DArrayType, Vector3DArrayType, Vector3DType


def project_points_to_image_plane(
    positions_in_camera_frame: Union[Vector3DArrayType, Vector3DType],
    camera_intrinsics: CameraIntrinsicsMatrixType,
) -> Vector2DArrayType:
    """Projects an array of points from the 3D camera frame to their 2D pixel coordinates on the image plane.

    Make sure to transform them to the camera frame first if they are in the world frame.

    Args:
        positions_in_camera_frame: numpy array of shape (N, 3) containing the 3D positions of the points in the camera frame
        camera_intrinsics: camera intrinsics matrix as a numpy array of shape (3, 3)

    Returns:
        numpy array of shape (N, 2) containing the 2D pixel coordinates of the points on the image plane
    """

    homogeneous_positions_in_camera_frame = _HomogeneousPoints(positions_in_camera_frame).homogeneous_points.T
    homogeneous_positions_on_image_plane = camera_intrinsics @ homogeneous_positions_in_camera_frame[:3, ...]
    positions_on_image_plane = (
        homogeneous_positions_on_image_plane[:2, ...] / homogeneous_positions_on_image_plane[2, ...]
    )
    return positions_on_image_plane.T
