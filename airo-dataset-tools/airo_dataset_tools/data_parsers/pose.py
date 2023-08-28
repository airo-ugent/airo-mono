from __future__ import annotations

import numpy as np
from airo_spatial_algebra.se3 import SE3Container
from airo_typing import HomogeneousMatrixType
from pydantic import BaseModel


class Position(BaseModel):
    """Position in 3D space, all units are in meters."""

    x: float
    y: float
    z: float


class EulerAngles(BaseModel):
    roll: float
    pitch: float
    yaw: float


class Pose(BaseModel):
    """Pose of an object in 3D space, all units are in meters and radians.

    The euler angles are  extrinsic (rotations about the axes xyz of the original coordinate system, which is assumed
    to remain motionless).
    """

    position_in_meters: Position
    rotation_euler_xyz_in_radians: EulerAngles

    @classmethod
    def from_homogeneous_matrix(cls, matrix: HomogeneousMatrixType) -> Pose:
        """Creates a Pose object from a 4x4 homogeneous transformation matrix."""
        se3_pose = SE3Container.from_homogeneous_matrix(matrix)
        position = se3_pose.translation
        euler_angles = se3_pose.orientation_as_euler_angles

        position_model = Position(x=position[0], y=position[1], z=position[2])
        euler_angles_model = EulerAngles(roll=euler_angles[0], pitch=euler_angles[1], yaw=euler_angles[2])

        pose = cls(position_in_meters=position_model, rotation_euler_xyz_in_radians=euler_angles_model)
        return pose

    def as_homogeneous_matrix(self) -> HomogeneousMatrixType:
        """Returns the pose as a 4x4 homogeneous transformation matrix."""
        position = self.position_in_meters
        euler_angles = self.rotation_euler_xyz_in_radians

        position_array = np.array([position.x, position.y, position.z])
        euler_angles_array = np.array([euler_angles.roll, euler_angles.pitch, euler_angles.yaw])

        pose_matrix = SE3Container.from_euler_angles_and_translation(
            euler_angles_array, position_array
        ).homogeneous_matrix

        return pose_matrix
