from typing import Any

import numpy as np
from airo_spatial_algebra.operations import transform_points
from airo_typing import BoundingBox3DType, HomogeneousMatrixType, PointCloud


def filter_point_cloud(point_cloud: PointCloud, mask: Any) -> PointCloud:
    """Creates a new point cloud that is filtered by the given mask.
    Will also filter the colors and attributes if they are present.

    Args:
        point_cloud: the point cloud to filter
        mask: the mask to filter the point cloud by, used to index the attribute arrays, can be boolean or indices

    Returns:
        the new filtered point cloud
    """
    points = point_cloud.points[mask]
    colors = None if point_cloud.colors is None else point_cloud.colors[mask]

    attributes = None
    if point_cloud.attributes is not None:
        attributes = {}
        for key, value in point_cloud.attributes.items():
            attributes[key] = value[mask]

    point_cloud_filtered = PointCloud(points, colors, attributes)
    return point_cloud_filtered


def generate_point_cloud_crop_mask(point_cloud: PointCloud, bounding_box: BoundingBox3DType) -> np.ndarray:
    """Creates a mask that can be used to filter a point cloud to the given bounding box.

    Args:
        bounding_box: the bounding box that surrounds the points to keep
        point_cloud: the point cloud to crop

    Returns:
        the mask that can be used to filter the point cloud
    """
    points = point_cloud.points
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    (x_min, y_min, z_min), (x_max, y_max, z_max) = bounding_box
    crop_mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max) & (z >= z_min) & (z <= z_max)
    return crop_mask


def crop_point_cloud(
    point_cloud: PointCloud,
    bounding_box: BoundingBox3DType,
) -> PointCloud:
    """Creates a new point cloud that is cropped to the given bounding box.
    Will also crop the colors and attributes if they are present.

    Args:
        bounding_box: the bounding box that surrounds the points to keep
        point_cloud: the point cloud to crop

    Returns:
        the new cropped point cloud
    """
    crop_mask = generate_point_cloud_crop_mask(point_cloud, bounding_box)
    return filter_point_cloud(point_cloud, crop_mask.nonzero())


def transform_point_cloud(point_cloud: PointCloud, frame_transformation: HomogeneousMatrixType) -> PointCloud:
    """Creates a new point cloud for which the points are transformed to the desired frame.
    Will keep colors and attributes if they are present.

    The `frame_transformation` is a homogeneous matrix expressing the current point cloud frame in the target point cloud frame.
    For example, if you capture a point cloud from a camera with the extrinsics matrix `X_W_C`, expressing the camera's pose in
    the world frame, then you can express the point cloud in the world frame with:

    `point_cloud_in_world = transform_point_cloud(point_cloud, X_W_C)`

    Args:
        point_cloud: The point cloud to transform.
        frame_transformation: The transformation matrix from the current point cloud frame to the new desired frame.

    Returns:
        The new transformed point cloud."""
    new_points = transform_points(frame_transformation, point_cloud.points)
    return PointCloud(new_points, point_cloud.colors, point_cloud.attributes)
