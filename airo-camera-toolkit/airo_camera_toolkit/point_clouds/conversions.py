import numpy as np
import open3d as o3d
from airo_typing import PointCloud


def open3d_to_point_cloud(pcd: o3d.geometry.PointCloud) -> PointCloud:
    points = np.asarray(pcd.points).astype(np.float32)
    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8) if pcd.has_colors() else None
    return PointCloud(points, colors)


def point_cloud_to_open3d(pointcloud: PointCloud) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud.points.astype(np.float64))
    if pointcloud.colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(pointcloud.colors.astype(np.float64) / 255)
    return pcd
