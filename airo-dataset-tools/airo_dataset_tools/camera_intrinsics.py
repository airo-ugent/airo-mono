from typing import List, Optional

from pydantic import BaseModel


class Resolution(BaseModel):
    width: int
    height: int


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
    radial_distortion_coefficients: Optional[RadialDistortionCoefficients]
    tangential_distortion_coefficients: Optional[TangentialDistortionCoefficients]
