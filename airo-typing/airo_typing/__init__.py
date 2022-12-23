from typing import Tuple, Union

import numpy as np

# TODO: see if we can specify the shape of these types for mypy

# spatial algebra types
Vector2DType = np.ndarray
"""a (2,) np array that represents a 2D position/translation/direction"""

Vector2DArrayType = np.ndarray
"""a (N,2) np array that represents N 2D positions/translations/directions"""

Vectors2DType = Union[Vector2DType, Vector2DArrayType]
"""a convenience type that represents a (2,) 2D vector or (N,2) array of 3D vectors."""


Vector3DType = np.ndarray
""" a (3,) np array that represents a 3D position/translation/direction
"""
Vector3DArrayType = np.ndarray
""" a (N,3) np array that represents N 3D positions/translations/directions
"""
Vectors3DType = Union[Vector3DType, Vector3DArrayType]
""" a convenience type that represents a (3,) 3D vector or (N,3) array of 3D vectors.
"""
QuaternionType = np.ndarray
"""scalar-last quaternion that represents a rotation around the <x,y,z> axis with angle <theta>
as <x cos(theta), y cos(theta), z cos(theta), sin(theta)>
"""
RotationMatrixType = np.ndarray
"""3x3 rotation matrix
"""
EulerAnglesType = np.ndarray
"""XYZ angles of rotation around the axes of the original frame (extrinsic).
First rotate around X, then around Y, finally around Z.
"""
AxisAngleType = Tuple[Vector3DType, float]

RotationVectorType = np.ndarray
""" Rotation vector <x*theta,y*theta,z*theta> that represents a rotation around the <x,y,z> axis with angle <theta>
"""

HomogeneousMatrixType = np.ndarray
"""4x4 homogeneous transform matrix
<<R,T>|<0,0,0,1>>
"""

# camera related types

OpenCVImageType = np.ndarray
"""an image in the OpenCV format: BGR, uint8, (H,W,C)"""

NumpyFloatImageType = np.ndarray
""" an image in the numpy format: RGB, float (0-1), (H,W,C)"""

NumpyIntImageType = np.ndarray
""" an image in the numpy format: RGB, uint8 (0-255), (H,W,C)"""
TorchImageType = np.ndarray
""" an image in the torch format: RGB, float(0-1), (C,H,W)"""

NumpyDepthMapType = np.ndarray
""" a depth map (z-buffer),float, (H,W)"""

CameraIntrinsicsMatrixType = np.ndarray
"""3x3 camera intrinsics matrix

K = [[fx,s,cx],[0,fy,cy],[0,0,1]]
see e.g. https://ksimek.github.io/2013/08/13/intrinsic/ for more details """


CameraExtrinsicMatrixType = HomogeneousMatrixType
"""4x4 camera extrinsic matrix,
this is the homogeneous matrix that describes the camera pose in the world frame"""
