"""
This file contains some operations on points and poses.

It also defines a helper class for a collection of points, but this is only used internally
to allow the user to directly interact with np.arrays to reduce friction.

i.e. the user can  call <transform>(points) where points is just a numpy array,
    instead of having to first convert the points to a specific format.
"""

import numpy as np
from airo_typing import HomogeneousMatrixType, Vector3DArrayType, Vectors3DType


class _HomogeneousPoints:
    """Helper class to facilitate multiplicating 4x4 matrices with one or more 3D points.
    This class internally handles the addition / removal of a dimension to the points.
    """

    # TODO: extend to generic dimensions (1D,2D,3D).
    def __init__(self, points: Vectors3DType):
        if not self.is_valid_points_type(points):
            raise ValueError(f"Invalid argument for {_HomogeneousPoints.__name__}.__init__ ")

        points = _HomogeneousPoints.ensure_array_2d(points)
        self._homogeneous_points = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)

    @staticmethod
    def is_valid_points_type(points: Vectors3DType) -> bool:
        if len(points.shape) == 1:
            if len(points) == 3:
                return True
        elif len(points.shape) == 2:
            if points.shape[1] == 3:
                return True
        return False

    @staticmethod
    def ensure_array_2d(points: Vectors3DType) -> Vector3DArrayType:
        """If points is a single shape (3,) point, then it is reshaped to (1,3)."""
        if len(points.shape) == 1:
            if len(points) != 3:
                raise ValueError("points has only one dimension, but it's length is not 3")
            points = points.reshape((1, 3))
        return points

    @property
    def homogeneous_points(self) -> np.ndarray:
        """Nx4 matrix representing the homogeneous points"""
        return self._homogeneous_points

    @property
    def points(self) -> Vectors3DType:
        """Nx3 matrix representing the points"""
        # normalize points (for safety, should never be necessary with affine transforms)
        # but we've had bugs of this type with projection operations, so better safe than sorry?
        scalars = self._homogeneous_points[:, 3][:, np.newaxis]
        points = self.homogeneous_points[:, :3] / scalars
        # TODO: if the original poitns was (1,3) matrix, then the resulting points would be a (3,) vector.
        #  Is this desirable? and if not, how to avoid it?
        if points.shape[0] == 1:
            # single point -> create vector from 1x3 matrix
            return points[0]
        else:
            return points

    def apply_transform(self, homogeneous_transform_matrix: HomogeneousMatrixType) -> None:
        self._homogeneous_points = (homogeneous_transform_matrix @ self.homogeneous_points.transpose()).transpose()


def transform_points(homogeneous_transform_matrix: HomogeneousMatrixType, points: Vectors3DType) -> Vectors3DType:
    """Applies a transform to a (set of) point(s).

    Args:
        homogeneous_transform_matrix (HomogeneousMatrixType): _description_
        points (PointsType): _description_
    Returns:
        PointsType: (3,) vector or (N,3) matrix.
    """
    homogeneous_points = _HomogeneousPoints(points)
    homogeneous_points.apply_transform(homogeneous_transform_matrix)
    return homogeneous_points.points
