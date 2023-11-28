from __future__ import annotations

from typing import Optional  # use class as type for class methods

import numpy as np
from airo_typing import (
    AxisAngleType,
    EulerAnglesType,
    HomogeneousMatrixType,
    QuaternionType,
    RotationMatrixType,
    RotationVectorType,
    Vector3DType,
)
from scipy.spatial.transform import Rotation
from spatialmath import SE3, UnitQuaternion
from spatialmath.base import trnorm


class SE3Container:
    """A container class for SE3 elements. These elements are used to represent the 3D Pose of an element A in a frame B,
    or stated differently the transform from frame B to frame A.

    Conventions:
    translations are in meters,rotations in radians.
    quaternions are scalar-last and normalized.
    euler angles are the angles of consecutive rotations around the original X-Y-Z axis (in that order).

    Note that ther exist many different types of euler angels that differ in the order of axes,
    and in whether they rotate around the original and the new axes. We chose this convention as it is the most common in robotics
    and also easy to reason about. use the Scipy.transform.Rotation class if you need to convert from/to other formats.

    This is a wrapper around the SE3 class of Peter Corke's Spatial Math Library: https://petercorke.github.io/spatialmath-python/
    The scope if this class is not to perform arbitrary calculations on SE3 elements,
    it is merely a 'simplified and more readable' wrapper
    that facilitates creating/retrieving position and/or orientations in various formats.

    If you need support for calculations and/or more ways to create SE3 elements, use Peter Corke's Spatial Math Library directly.
    You can decide this on the fly as you can always access the SE3 attribute of this class or instantiate this class from an SE3 object
    """

    def __init__(self, se3: SE3) -> None:  # type: ignore
        self.se3 = se3

    @classmethod
    def random(cls) -> SE3Container:
        """A random SE3 element with translations in the [-1,1]^3 cube."""
        return cls(SE3.Rand())

    @classmethod
    def from_translation(cls, translation: Vector3DType) -> SE3Container:
        """creates a translation-only SE3 element"""
        return cls(SE3.Trans(translation.tolist()))

    @classmethod
    def from_homogeneous_matrix(cls, matrix: HomogeneousMatrixType) -> SE3Container:
        _assert_is_se3_matrix(matrix)
        return cls(SE3(matrix))

    @classmethod
    def from_rotation_matrix_and_translation(
        cls, rotation_matrix: RotationMatrixType, translation: Optional[Vector3DType] = None
    ) -> SE3Container:
        _assert_is_so3_matrix(rotation_matrix)
        return cls(SE3.Rt(rotation_matrix, translation))

    @classmethod
    def from_rotation_vector_and_translation(
        cls, rotation_vector: RotationVectorType, translation: Optional[Vector3DType] = None
    ) -> SE3Container:
        return cls(SE3.Rt(Rotation.from_rotvec(rotation_vector).as_matrix(), translation))

    @classmethod
    def from_quaternion_and_translation(
        cls, quaternion: QuaternionType, translation: Optional[Vector3DType] = None
    ) -> SE3Container:
        q = UnitQuaternion(quaternion[3], quaternion[:3])  # scalar-first in math lib
        return cls(SE3.Rt(q.R, translation))

    @classmethod
    def from_euler_angles_and_translation(
        cls, euler_angels: EulerAnglesType, translation: Optional[Vector3DType] = None
    ) -> SE3Container:
        # convert from extrinsic XYZ to rotmatrix
        # bc SE3.Eul does not accept translation
        rot_matrix = Rotation.from_euler("xyz", euler_angels, degrees=False).as_matrix()
        return cls.from_rotation_matrix_and_translation(rot_matrix, translation)

    @classmethod
    def from_orthogonal_base_vectors_and_translation(
        cls,
        x_axis: Vector3DType,
        y_axis: Vector3DType,
        z_axis: Vector3DType,
        translation: Optional[Vector3DType] = None,
    ) -> SE3Container:
        # create orientation matrix with base vectors as columns
        orientation_matrix = np.zeros((3, 3))
        for i, axis in enumerate([x_axis, y_axis, z_axis]):
            orientation_matrix[:, i] = axis / np.linalg.norm(axis)

        _assert_is_so3_matrix(orientation_matrix)

        return cls(SE3.Rt(orientation_matrix, translation))

    @property
    def orientation_as_quaternion(self) -> QuaternionType:
        angle, vec = self.se3.angvec()
        scalar_first_quaternion = UnitQuaternion.AngVec(angle, vec).A
        return self.scalar_first_quaternion_to_scalar_last(scalar_first_quaternion)

    @property
    def orientation_as_euler_angles(self) -> EulerAnglesType:
        zyx_ordered_angles = self.se3.eul()
        # convert from intrinsic ZYZ  to extrinsic xyz
        return Rotation.from_euler("ZYZ", zyx_ordered_angles, degrees=False).as_euler("xyz", degrees=False)

    @property
    def orientation_as_axis_angle(self) -> AxisAngleType:
        angle, axis = self.se3.angvec()
        return axis.astype(np.float64), float(angle)

    @property
    def orientation_as_rotation_vector(self) -> Vector3DType:
        axis, angle = self.orientation_as_axis_angle
        if axis is None:
            return np.zeros(3)
        return angle * axis

    @property
    def rotation_matrix(self) -> RotationMatrixType:
        return self.se3.R

    @property
    def homogeneous_matrix(self) -> HomogeneousMatrixType:
        return self.se3.A

    @property
    def translation(self) -> Vector3DType:
        # TODO: should this be named position or translation?
        return self.se3.t

    @property
    def x_axis(self) -> Vector3DType:
        """also called normal vector. This is the first column of the rotation matrix"""
        return self.se3.n

    @property
    def y_axis(self) -> Vector3DType:
        """also colled orientation vector. This is the second column of the rotation matrix"""
        return self.se3.o

    @property
    def z_axis(self) -> Vector3DType:
        """also called approach vector. This is the third column of the rotation matrix"""
        return self.se3.a

    def __str__(self) -> str:
        return str(f"SE3 -> \n {self.homogeneous_matrix}")

    @staticmethod
    def scalar_first_quaternion_to_scalar_last(scalar_first_quaternion: np.ndarray) -> QuaternionType:
        scalar_last_quaternion = np.roll(scalar_first_quaternion, -1)
        return scalar_last_quaternion

    @staticmethod
    def scalar_last_quaternion_to_scalar_first(scalar_last_quaternion: QuaternionType) -> np.ndarray:
        scalar_first_quaternion = np.roll(scalar_last_quaternion, 1)
        return scalar_first_quaternion


def normalize_so3_matrix(matrix: np.ndarray) -> np.ndarray:
    """normalize an SO3 matrix (i.e. a rotation matrix) to be orthogonal and have determinant 1 (right-handed coordinate system)
    see https://en.wikipedia.org/wiki/3D_rotation_group

    Can be used to fix numerical issues with rotation matrices

    will make sure x,y,z are unit vectors, then
    will construct new x vector as y cross z, then construct new y vector as z cross x, so that x,y,z are orthogonal

    """
    assert matrix.shape == (3, 3), "matrix is not a 3x3 matrix"
    return trnorm(matrix)


def _assert_is_so3_matrix(matrix: np.ndarray) -> None:
    """check if matrix is a valid SO3 matrix
    this requires the matrix to be orthogonal (base vectors are perpendicular) and have determinant 1 (right-handed coordinate system)
    see https://en.wikipedia.org/wiki/3D_rotation_group

    This function will raise a ValueError if the matrix is not valid

    """
    if matrix.shape != (3, 3):
        raise ValueError("matrix is not a 3x3 matrix")
    if not np.allclose(matrix @ matrix.T, np.eye(3)):
        raise ValueError(
            "matrix is not orthnormal, i.e. its base vectors are not perpendicular. If you are sure this is a numerical issue, use normalize_so3_matrix()"
        )
    if not np.allclose(np.linalg.det(matrix), 1):
        raise ValueError("matrix does not have determinant 1 (not right-handed)")


def _assert_is_se3_matrix(matrix: np.ndarray) -> None:
    """check if matrix is a valid SE3 matrix (i.e. a valid pose)
    this requires the rotation part to be a valid SO3 matrix and the translation part to be a 3D vector

    This function will raise a ValueError if the matrix is not valid
    """
    if matrix.shape != (4, 4):
        raise ValueError("matrix is not a 4x4 matrix")
    if not np.allclose(matrix[3, :], np.array([0, 0, 0, 1])):
        raise ValueError("last row of matrix is not [0,0,0,1]")
    _assert_is_so3_matrix(matrix[:3, :3])
