import numpy as np
import open3d as o3d
from airo_typing import ColoredPointCloudType


def pointcloud_open3d_to_numpy(pcd: o3d.geometry.PointCloud) -> ColoredPointCloudType:
    points = np.asarray(pcd.points).astype(np.float32)
    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    return points, colors


def pointcloud_numpy_to_open3d(pointcloud: ColoredPointCloudType) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud[0].astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(pointcloud[1].astype(np.float64) / 255)
    return pcd
