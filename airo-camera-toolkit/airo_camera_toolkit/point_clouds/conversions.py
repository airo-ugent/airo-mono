from typing import Any

import open3d as o3d
import open3d.core as o3c
from airo_typing import PointCloud


def point_cloud_to_open3d(point_cloud: PointCloud) -> Any:  # TODO: change Any back to o3d.t.geometry.PointCloud
    """Converts a PointCloud dataclass object to an open3d tensor point cloud.
    Note that the memory buffers of the underlying numpy arrays are shared between the two.

    Args:
        point_cloud: the point cloud to convert

    Returns:
        pcd: the open3d tensor point cloud
    """
    positions = o3c.Tensor.from_numpy(point_cloud.points)

    map_to_tensors = {
        "positions": positions,
    }

    if point_cloud.colors is not None:
        colors = o3c.Tensor.from_numpy(point_cloud.colors)
        map_to_tensors["colors"] = colors

    if point_cloud.attributes is not None:
        for attribute_name, array in point_cloud.attributes.items():
            map_to_tensors[attribute_name] = o3c.Tensor.from_numpy(array)

    pcd = o3d.t.geometry.PointCloud(map_to_tensors)
    return pcd


def open3d_to_point_cloud(pcd: Any) -> PointCloud:  # TODO: change Any back to o3d.t.geometry.PointCloud
    """Converts an open3d point cloud to a PointCloud dataclass object.
    Note that the memory buffers of the underlying numpy arrays are shared between the two.

    Args:
        pcd: the open3d tensor point cloud
    """
    points = pcd.point.positions.numpy()
    colors = pcd.point.colors.numpy() if "colors" in pcd.point else None

    attributes = {}
    for attribute_name, array in pcd.point.items():
        if attribute_name in ["positions", "colors"]:
            continue
        attributes[attribute_name] = array.numpy()

    return PointCloud(points, colors)
