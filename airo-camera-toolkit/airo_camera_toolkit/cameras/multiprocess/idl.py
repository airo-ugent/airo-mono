import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from airo_camera_toolkit.interfaces import DepthCamera, RGBCamera, RGBDCamera, StereoRGBDCamera
from airo_ipc.cyclone_shm.idl_shared_memory.base_idl import BaseIdl
from airo_typing import CameraResolutionType, PointCloud


@dataclass
class Buffer(ABC, BaseIdl):
    @abstractmethod
    def allocate_empty(resolution: CameraResolutionType) -> BaseIdl:
        pass

    @abstractmethod
    def allocate_from_camera(camera: RGBCamera) -> BaseIdl:
        pass


@dataclass
class RGBFrameBuffer(BaseIdl):
    # Timestamp of the frame (seconds)
    timestamp: np.ndarray
    # Color image data (height x width x channels)
    rgb: np.ndarray

    @staticmethod
    def allocate_empty(resolution: CameraResolutionType) -> BaseIdl:
        width, height = resolution

        return RGBFrameBuffer(
            timestamp=np.empty((1,), dtype=np.float64),
            rgb=np.empty((height, width, 3), dtype=np.uint8),
        )

    @staticmethod
    def allocate_from_camera(camera: RGBCamera) -> BaseIdl:
        image = camera._retrieve_rgb_image_as_int()

        timestamp = time.time()
        return RGBFrameBuffer(
            timestamp=np.array([timestamp], dtype=np.float64),
            rgb=image,
        )


@dataclass
class StereoRGBFrameBuffer(BaseIdl):
    # Timestamp of the frame (seconds)
    timestamp: np.ndarray
    # Color image data (height x width x channels)
    rgb_left: np.ndarray
    rgb_right: np.ndarray
    # Intrinsic camera parameters (camera matrix)
    intrinsics_left: np.ndarray
    intrinsics_right: np.ndarray
    # Extrinsic camera parameters (camera matrix)
    pose_right_in_left: np.ndarray

    @staticmethod
    def allocate_empty(resolution: CameraResolutionType) -> BaseIdl:
        width, height = resolution

        return StereoRGBFrameBuffer(
            timestamp=np.empty((1,), dtype=np.float64),
            rgb_left=np.empty((height, width, 3), dtype=np.uint8),
            rgb_right=np.empty((height, width, 3), dtype=np.uint8),
            intrinsics_left=np.empty((3, 3), dtype=np.float64),
            intrinsics_right=np.empty((3, 3), dtype=np.float64),
            pose_right_in_left=np.empty((4, 4), dtype=np.float64),
        )

    @staticmethod
    def allocate_from_camera(camera: StereoRGBDCamera) -> BaseIdl:
        image_left = camera._retrieve_rgb_image_as_int(StereoRGBDCamera.LEFT_RGB)
        image_right = camera._retrieve_rgb_image_as_int(StereoRGBDCamera.RIGHT_RGB)
        intrinsics_left = camera.intrinsics_matrix(StereoRGBDCamera.LEFT_RGB)
        intrinsics_right = camera.intrinsics_matrix(StereoRGBDCamera.RIGHT_RGB)
        pose_right_in_left = camera.pose_of_right_view_in_left_view

        timestamp = time.time()
        return StereoRGBFrameBuffer(
            timestamp=np.array([timestamp], dtype=np.float64),
            rgb_left=image_left,
            rgb_right=image_right,
            intrinsics_left=intrinsics_left,
            intrinsics_right=intrinsics_right,
            pose_right_in_left=pose_right_in_left,
        )


@dataclass
class DepthFrameBuffer(BaseIdl):
    # Timestamp of the frame (seconds)
    timestamp: np.ndarray
    # Depth image data (height x width)
    depth_image: np.ndarray
    # Depth map (height x width)
    depth_map: np.ndarray
    # Confidence map (height x width)
    confidence_map: np.ndarray

    @staticmethod
    def allocate_empty(resolution: CameraResolutionType) -> BaseIdl:
        width, height = resolution

        return DepthFrameBuffer(
            timestamp=np.empty((1,), dtype=np.float64),
            depth_image=np.empty((height, width, 3), dtype=np.uint8),
            depth_map=np.empty((height, width), dtype=np.float32),
            confidence_map=np.empty((height, width), dtype=np.float32),
        )

    @staticmethod
    def allocate_from_camera(camera: DepthCamera) -> BaseIdl:
        depth_image = camera._retrieve_depth_image()
        depth_map = camera._retrieve_depth_map()
        confidence_map = camera._retrieve_confidence_map()

        timestamp = time.time()
        return DepthFrameBuffer(
            timestamp=np.array([timestamp], dtype=np.float64),
            depth_image=depth_image,
            depth_map=depth_map,
            confidence_map=confidence_map,
        )


@dataclass
class PointCloudBuffer(BaseIdl):
    # Timestamp of the frame (seconds)
    timestamp: np.ndarray
    # Point cloud positions (height * width x 3)
    point_cloud_positions: np.ndarray
    # Point cloud colors (height * width x 3)
    point_cloud_colors: np.ndarray
    # Valid point cloud points (scalar), for sparse point clouds
    point_cloud_valid: np.ndarray

    @property
    def point_cloud(self) -> PointCloud:
        num_points = self.point_cloud_valid.item()
        positions = self.point_cloud_positions[:num_points]
        colors = self.point_cloud_colors[:num_points]
        return PointCloud(positions, colors)

    @staticmethod
    def allocate_empty(resolution: CameraResolutionType) -> BaseIdl:
        width, height = resolution

        return PointCloudBuffer(
            timestamp=np.empty((1,), dtype=np.float64),
            point_cloud_positions=np.empty((height * width, 3), dtype=np.float32),
            point_cloud_colors=np.empty((height * width, 3), dtype=np.uint8),
            point_cloud_valid=np.empty((1,), dtype=np.uint32),
        )

    @staticmethod
    def allocate_from_camera(camera: RGBDCamera) -> BaseIdl:
        point_cloud = camera._retrieve_colored_point_cloud()

        pcd_pos_buf = np.full(
            (camera.resolution[0] * camera.resolution[1], 3),
            fill_value=np.nan,
            dtype=np.float32,
        )
        pcd_col_buf = np.zeros(
            (camera.resolution[0] * camera.resolution[1], 3),
            dtype=np.uint8,
        )

        pcd_pos_buf[: point_cloud.points.shape[0]] = point_cloud.points
        if point_cloud.colors is not None:
            pcd_col_buf[: point_cloud.colors.shape[0]] = point_cloud.colors
        else:
            pcd_col_buf[: point_cloud.points.shape[0]] = 0  # If no colors, use black.
        point_cloud_valid = np.array([point_cloud.points.shape[0]], dtype=np.uint32)

        timestamp = time.time()
        return PointCloudBuffer(
            timestamp=np.array([timestamp], dtype=np.float64),
            point_cloud_positions=pcd_pos_buf,
            point_cloud_colors=pcd_col_buf,
            point_cloud_valid=point_cloud_valid,
        )


@dataclass
class CameraMetadataBuffer(BaseIdl):
    # (2,) uint32 array: width, height.
    resolution: np.ndarray
    # (1,) float32 scalar array: fps.
    fps: np.ndarray
    # (3, 3) float32 array containing intrinsics.
    intrinsics_matrix: np.ndarray

    @staticmethod
    def allocate_empty(resolution: CameraResolutionType) -> BaseIdl:
        return CameraMetadataBuffer(
            resolution=np.empty_like(resolution, dtype=np.uint32),
            intrinsics_matrix=np.empty((3, 3), dtype=np.float32),
            fps=np.empty((1,), dtype=np.float32),
        )

    @staticmethod
    def allocate_from_camera(camera: RGBCamera) -> BaseIdl:
        return CameraMetadataBuffer(
            resolution=np.array(camera.resolution).astype(np.uint32),
            intrinsics_matrix=camera.intrinsics_matrix().astype(np.float32),
            fps=np.array([camera.fps], dtype=np.float32),
        )
