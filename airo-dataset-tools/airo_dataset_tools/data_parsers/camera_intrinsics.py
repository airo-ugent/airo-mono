from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from airo_typing import CameraIntrinsicsMatrixType, CameraResolutionType
from pydantic import BaseModel


class Resolution(BaseModel):
    width: int
    height: int

    def as_tuple(self) -> CameraResolutionType:
        return self.width, self.height


class FocalLengths(BaseModel):
    fx: float
    fy: float


class PrincipalPoint(BaseModel):
    cx: float
    cy: float


RadialDistortionCoefficients = List[float]
TangentialDistortionCoefficients = List[float]


class CameraIntrinsics(BaseModel):
    """A format for storing the camera intrinsics at a specific image resolution."""

    image_resolution: Resolution
    focal_lengths_in_pixels: FocalLengths
    principal_point_in_pixels: PrincipalPoint

    # Distortion coefficients are stored so you can add as many as you want.
    radial_distortion_coefficients: Optional[RadialDistortionCoefficients] = None
    tangential_distortion_coefficients: Optional[TangentialDistortionCoefficients] = None

    @classmethod
    def from_matrix_and_resolution(
        cls, intrinsics_matrix: CameraIntrinsicsMatrixType, resolution: Tuple[int, int]
    ) -> CameraIntrinsics:
        """Creates a CameraIntrinsics object from a 3x3 matrix and an image resolution (width, height)."""
        fx = intrinsics_matrix[0, 0]
        fy = intrinsics_matrix[1, 1]
        cx = intrinsics_matrix[0, 2]
        cy = intrinsics_matrix[1, 2]

        width, height = resolution

        camera_intrinsics = cls(
            image_resolution=Resolution(width=width, height=height),
            focal_lengths_in_pixels=FocalLengths(fx=fx, fy=fy),
            principal_point_in_pixels=PrincipalPoint(cx=cx, cy=cy),
        )
        return camera_intrinsics

    def as_matrix(self) -> CameraIntrinsicsMatrixType:
        """Returns the camera intrinsics as a 3x3 matrix, often called K."""
        fx = self.focal_lengths_in_pixels.fx
        fy = self.focal_lengths_in_pixels.fy
        cx = self.principal_point_in_pixels.cx
        cy = self.principal_point_in_pixels.cy

        intrinsics_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        return intrinsics_matrix
