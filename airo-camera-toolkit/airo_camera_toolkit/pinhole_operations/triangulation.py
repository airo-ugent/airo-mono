from typing import List

import numpy as np
from airo_typing import CameraExtrinsicMatrixType, CameraIntrinsicsMatrixType, Vector2DArrayType, Vector3DType


def multiview_triangulation_midpoint(
    extrinsics_matrices: List[CameraExtrinsicMatrixType],
    intrinsics_matrices: List[CameraIntrinsicsMatrixType],
    image_coordinates: Vector2DArrayType,
) -> Vector3DType:
    """triangulates a point from multiple views using the midpoint method.
    This method minimizes the L2 distance in the camera space between the estimated point and all rays
    cf. https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8967077

    Args:
        extrinsics_matrices: list of extrinsics matrices for each viewpoint
        intrinsics_matrices: list of intrinsics matrices for each viewpoint
        image_points: list of image coordinates of the 3D point for each viewpoint

    Returns:
        estimated 3D position in the world frame as a numpy array of shape (3,).
    """

    # determine the rays for each camera in the world frame
    rays = []
    for extrinsics_matrix, intrinsics_matrix, image_point in zip(
        extrinsics_matrices, intrinsics_matrices, image_coordinates
    ):
        ray = (
            extrinsics_matrix[:3, :3]
            @ np.linalg.inv(intrinsics_matrix)
            @ np.array([image_point[0], image_point[1], 1])
        )
        ray = ray / np.linalg.norm(ray)
        rays.append(ray)

    lhs = 0
    rhs = 0
    for i, ray in enumerate(rays):
        rhs += (np.eye(3) - ray[:, np.newaxis] @ ray[np.newaxis, :]) @ extrinsics_matrices[i][:3, 3]
        lhs += np.eye(3) - ray[:, np.newaxis] @ ray[np.newaxis, :]

    lhs_inv = np.linalg.inv(lhs)
    midpoint = lhs_inv @ rhs
    return midpoint


def calculate_triangulation_errors(
    extrinsics_matrices: List[CameraExtrinsicMatrixType],
    intrinsics_matrices: List[CameraIntrinsicsMatrixType],
    image_coordinates: Vector2DArrayType,
    point: Vector3DType,
) -> List[float]:
    """calculates the Euclidean distances in the camera space between the estimated point and all rays, as a measure of triangulation error

    Args:
        extrinsics_matrices: list of extrinsics matrices for each viewpoint
        intrinsics_matrices: list of intrinsics matrices for each viewpoint
        image_points: list of image coordinates of the 3D point for each viewpoint
        point: 3D point in the world frame

    Returns:
        list of Euclidean distances between the point and the rays for each viewpoint
    """
    errors = []
    for extrinsics_matrix, intrinsics_matrix, image_point in zip(
        extrinsics_matrices, intrinsics_matrices, image_coordinates
    ):
        ray = (
            extrinsics_matrix[:3, :3]
            @ np.linalg.inv(intrinsics_matrix)
            @ np.array([image_point[0], image_point[1], 1])
        )
        ray = ray / np.linalg.norm(ray)
        error = np.linalg.norm(
            (np.eye(3) - ray[:, np.newaxis] @ ray[np.newaxis, :]) @ ((extrinsics_matrix[:3, 3]) - point)
        )
        errors.append(error.item())
    return errors
