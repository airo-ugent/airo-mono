from abc import ABC
from dataclasses import dataclass

import numpy as np
from airo_ipc.cyclone_shm.idl_shared_memory.base_idl import BaseIdl


@dataclass
class Buffer(ABC, BaseIdl):
    pass


@dataclass
class RGBFrameBuffer(Buffer):
    # Color image data (height x width x channels)
    rgb: np.ndarray


@dataclass
class StereoRGBFrameBuffer(Buffer):
    # Color image data (height x width x channels)
    rgb_left: np.ndarray
    rgb_right: np.ndarray
    # Intrinsic camera parameters (camera matrix)
    intrinsics_left: np.ndarray
    intrinsics_right: np.ndarray
    # Extrinsic camera parameters (camera matrix)
    pose_right_in_left: np.ndarray


@dataclass
class DepthFrameBuffer(Buffer):
    # Depth image data (height x width)
    depth_image: np.ndarray
    # Depth map (height x width)
    depth_map: np.ndarray
    # Confidence map (height x width)
    confidence_map: np.ndarray


@dataclass
class PointCloudBuffer(Buffer):
    # Point cloud positions (height * width x 3)
    point_cloud_positions: np.ndarray
    # Point cloud colors (height * width x 3)
    point_cloud_colors: np.ndarray
    # Valid point cloud points (scalar), for sparse point clouds
    point_cloud_valid: np.ndarray


@dataclass
class CameraMetadataBuffer(Buffer):
    # Timestamp of the frame (seconds)
    timestamp: np.ndarray
    # (2,) uint32 array: width, height.
    resolution: np.ndarray
    # (1,) float32 scalar array: fps.
    fps: np.ndarray
    # (3, 3) float32 array containing intrinsics.
    intrinsics_matrix: np.ndarray
