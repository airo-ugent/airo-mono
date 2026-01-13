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
    def fill_from_camera(self, camera: RGBCamera) -> BaseIdl:
        pass


@dataclass
class RGBFrameBuffer(Buffer):
    # Color image data (height x width x channels)
    rgb: np.ndarray

    @staticmethod
    def allocate_empty(resolution: CameraResolutionType) -> BaseIdl:
        width, height = resolution

        return RGBFrameBuffer(
            rgb=np.empty((height, width, 3), dtype=np.uint8),
        )

    def fill_from_camera(self, camera: RGBCamera) -> BaseIdl:
        image = camera._retrieve_rgb_image_as_int()

        self.rgb = image

        return self


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

    @staticmethod
    def allocate_empty(resolution: CameraResolutionType) -> BaseIdl:
        width, height = resolution

        return StereoRGBFrameBuffer(
            rgb_left=np.empty((height, width, 3), dtype=np.uint8),
            rgb_right=np.empty((height, width, 3), dtype=np.uint8),
            intrinsics_left=np.empty((3, 3), dtype=np.float64),
            intrinsics_right=np.empty((3, 3), dtype=np.float64),
            pose_right_in_left=np.empty((4, 4), dtype=np.float64),
        )

    def fill_from_camera(self, camera: StereoRGBDCamera) -> BaseIdl:
        image_left = camera._retrieve_rgb_image_as_int(StereoRGBDCamera.LEFT_RGB)
        image_right = camera._retrieve_rgb_image_as_int(StereoRGBDCamera.RIGHT_RGB)
        intrinsics_left = camera.intrinsics_matrix(StereoRGBDCamera.LEFT_RGB)
        intrinsics_right = camera.intrinsics_matrix(StereoRGBDCamera.RIGHT_RGB)
        pose_right_in_left = camera.pose_of_right_view_in_left_view

        self.rgb_left = image_left
        self.rgb_right = image_right
        self.intrinsics_left = intrinsics_left
        self.intrinsics_right = intrinsics_right
        self.pose_right_in_left = pose_right_in_left

        return self


@dataclass
class DepthFrameBuffer(Buffer):
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
            depth_image=np.empty((height, width, 3), dtype=np.uint8),
            depth_map=np.empty((height, width), dtype=np.float32),
            confidence_map=np.empty((height, width), dtype=np.float32),
        )

    def fill_from_camera(self, camera: DepthCamera) -> BaseIdl:
        depth_image = camera._retrieve_depth_image()
        depth_map = camera._retrieve_depth_map()
        confidence_map = camera._retrieve_confidence_map()

        self.depth_image = depth_image
        self.depth_map = depth_map
        self.confidence_map = confidence_map

        return self


@dataclass
class PointCloudBuffer(Buffer):
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
            point_cloud_positions=np.empty((height * width, 3), dtype=np.float32),
            point_cloud_colors=np.empty((height * width, 3), dtype=np.uint8),
            point_cloud_valid=np.empty((1,), dtype=np.uint32),
        )

    def fill_from_camera(self, camera: RGBDCamera) -> BaseIdl:
        point_cloud = camera._retrieve_colored_point_cloud()

        self.point_cloud_positions[: point_cloud.points.shape[0]] = point_cloud.points
        if point_cloud.colors is not None:
            self.point_cloud_colors[: point_cloud.colors.shape[0]] = point_cloud.colors
        else:
            self.point_cloud_colors[: point_cloud.colors.shape[0]] = 0  # If no colors, use black.
        self.point_cloud_valid[0] = point_cloud.points.shape[0]

        return self


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

    @staticmethod
    def allocate_empty(resolution: CameraResolutionType) -> BaseIdl:
        return CameraMetadataBuffer(
            timestamp=np.empty((1,), dtype=np.float64),
            resolution=np.empty_like(resolution, dtype=np.uint32),
            intrinsics_matrix=np.empty((3, 3), dtype=np.float32),
            fps=np.empty((1,), dtype=np.float32),
        )

    def fill_from_camera(self, camera: RGBCamera) -> BaseIdl:
        self.timestamp[0] = time.time()
        self.resolution = np.array(camera.resolution).astype(np.uint32)
        self.intrinsics_matrix = camera.intrinsics_matrix().astype(np.float32)
        self.fps[0] = camera.fps

        return self
