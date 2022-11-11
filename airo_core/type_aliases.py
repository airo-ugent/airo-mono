import numpy as np
from typing import Tuple
#TODO: see if we can specify the shape of these types for mypy  

VectorType = np.ndarray
""" 3D vector that represents a position/translation/direction in cartesian space
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
AxisAngleType = Tuple[VectorType, float]

HomogeneousMatrixType = np.ndarray 
"""4x4 homogeneous transform matrix 
<<R,T>|<0,0,0,1>>
"""