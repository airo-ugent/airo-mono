"""Helpers for staging a (possibly sparse) point cloud into fixed-size shared-memory buffers.

Several publishers (RGBD, stereo RGBD, Zed) embed a point cloud in their frame buffer. Since the
buffer has a fixed size (one slot per pixel) but cameras can return fewer points than pixels, the
same "allocate once, fill every frame" logic is shared here instead of being duplicated per publisher.
"""

from typing import Tuple

import numpy as np
from airo_typing import PointCloud


def allocate_pointcloud_buffers(width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    """Allocate position/color buffers sized for one point per pixel.

    Args:
        width: Image width.
        height: Image height.

    Returns:
        A ``(positions, colors)`` tuple of zero-initialized ``(width * height, 3)`` arrays,
        with dtypes ``float32`` and ``uint8`` respectively.
    """
    num_points = width * height
    positions = np.zeros((num_points, 3), dtype=np.float32)
    colors = np.zeros((num_points, 3), dtype=np.uint8)
    return positions, colors


def fill_pointcloud_buffers(positions_buf: np.ndarray, colors_buf: np.ndarray, point_cloud: PointCloud) -> int:
    """Stage a (possibly sparse) point cloud into fixed-size buffers, in place.

    Positions beyond the valid point count are filled with NaN, so a receiver that doesn't check
    the returned count still sees invalid data rather than a previous, larger point cloud's
    leftovers. Missing colors are filled with black.

    Args:
        positions_buf: Pre-allocated ``(N, 3)`` float32 buffer to fill in place.
        colors_buf: Pre-allocated ``(N, 3)`` uint8 buffer to fill in place.
        point_cloud: The point cloud to stage.

    Returns:
        The number of valid points written (``point_cloud.points.shape[0]``).
    """
    num_points = point_cloud.points.shape[0]
    positions_buf.fill(np.nan)
    positions_buf[:num_points] = point_cloud.points
    if point_cloud.colors is not None:
        colors_buf[:num_points] = point_cloud.colors
    else:
        colors_buf[:num_points] = 0  # Use black if no colors
    return num_points
