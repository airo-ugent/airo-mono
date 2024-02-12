from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

# TODO: see if we can specify the shape of these types for mypy

#######################
# spatial algebra types
#######################
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
"""scalar-last quaternion <x,y,z,w> that represents a rotation around the <x,y,z> axis with angle <theta>
as <x sin(theta), y sin(theta), z sin(theta), cos(theta)>
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
<<R,T>|<0,0,0,1>> that represents the pose of a frame A in another frame B

Shorthand notation is T^A_B.
"""

# Changing the applied-on frame requires the Adjoint of the transform between the two frames.
# Changing the expressed-in frame requires multiplication by the rotation matrix of the transform between the two frames.
# notation is taken from https://manipulation.csail.mit.edu/clutter.html#section3
WrenchType = np.ndarray
""" a (6,) numpy array that represents a wrench applied on a frame and expressed in a (possibly different) frame as [Fx,Fy,Fz,Tx,Ty,Tz]

shorthand notation is W^F_E, where F is the frame the wrench is applied on, and E is the frame the wrench is expressed in.
"""

# Changing the measured-in frame requires the Adjoint of the transform between the two frames.
# Changing the expressed-in frame requires multiplication by the rotation matrix of the transform between the two frames.
# Notation is taken from https://manipulation.csail.mit.edu/pick.html#jacobian
TwistType = np.ndarray
""" a (6,) numpy array that represents the spatial velocity or an incremental motion of one frame as measured in another frame (and possibly expressed in a third frame)

shorthand notation is ^C T^B_A, where C is the frame the velocity is measured in, B is the frame the velocity is expressed in.
"""

#####################
# Manipulator types #
#####################

JointConfigurationType = np.ndarray
"""an (N,) numpy array that represents the joint angles for a robot"""

JointPathType = np.ndarray
""" a (T, N) array of joint states (can be position/velocity/acceleration) that describe a path in joint space"""

TimesType = np.ndarray
""" a (T,) array of monotonically increasing times (float), corresponding to a path"""


@dataclass
class JointPathContainer:
    positions: Optional[JointPathType] = None
    velocities: Optional[JointPathType] = None
    accelerations: Optional[JointPathType] = None
    efforts: Optional[JointPathType] = None


@dataclass
class SingleArmTrajectory:
    times: TimesType  # time (seconds) from start of trajectory
    path: JointPathContainer
    gripper_path: Optional[JointPathContainer] = None


@dataclass
class DualArmTrajectory:
    times: TimesType  # time (seconds) from start of trajectory
    path_left: JointPathContainer
    path_right: JointPathContainer
    gripper_path_left: Optional[JointPathContainer] = None
    gripper_path_right: Optional[JointPathContainer] = None


PosePathType = np.ndarray
""" a (T, 4, 4) list of homogeneous matrices that describe a path in cartesian space"""


@dataclass
class PoseTrajectory:
    times: TimesType
    poses: PosePathType


ForwardKinematicsFunctionType = Callable[[JointConfigurationType], HomogeneousMatrixType]
""" a function that computes the forward kinematics of a given joint configuration"""

InverseKinematicsFunctionType = Callable[[HomogeneousMatrixType], List[JointConfigurationType]]
""" a function that computes one or more inverse kinematics solutions of a given TCP pose"""

JointConfigurationCheckerType = Callable[[JointConfigurationType], bool]
""" a function that checks a certain condition on a joint configuration, e.g. collision checking"""

######################
# camera related types
######################

OpenCVIntImageType = np.ndarray
"""an image in the OpenCV format: BGR, uint8, (H,W,C)"""

NumpyFloatImageType = np.ndarray
""" a float image in the numpy format: RGB, float (0-1), (H,W,C)"""
NumpyIntImageType = np.ndarray
""" an int image in the numpy format: RGB, uint8 (0-255), (H,W,C)"""
TorchFloatImageType = np.ndarray
""" an image in the torch format: RGB, float(0-1), (C,H,W)"""

NumpyDepthMapType = np.ndarray
""" a depth map (z-buffer),float, (H,W)"""

CameraResolutionType = Tuple[int, int]
""" a tuple of image (width, height) in pixels"""

CameraIntrinsicsMatrixType = np.ndarray
"""3x3 camera intrinsics matrix

K = [[fx,s,cx],[0,fy,cy],[0,0,1]]
see e.g. https://ksimek.github.io/2013/08/13/intrinsic/ for more details """


CameraExtrinsicMatrixType = HomogeneousMatrixType
"""4x4 camera extrinsic matrix,
this is the homogeneous matrix that describes the camera pose in the world frame"""

##########################
# 3D and point cloud types
##########################

BoundingBox3DType = Tuple[Tuple[float, float, float], Tuple[float, float, float]]
""" a tuple of two tuples that represent the min and max corners of a 3D bounding box"""

PointCloudPositionsType = Vector3DArrayType
""" a (N,3) float32 numpy array that represents a point cloud"""

PointCloudColorsType = np.ndarray
""" a (N,3) uint8 numpy array that represents the RGB colors of a point cloud"""

PointCloudAttributesType = Dict[str, np.ndarray]
""" a dictionary of numpy arrays that represent additional attributes of a point cloud, e.g. normals, confidence, etc. """


@dataclass
class PointCloud:
    points: PointCloudPositionsType
    colors: Optional[PointCloudColorsType] = None
    attributes: Optional[PointCloudAttributesType] = None
