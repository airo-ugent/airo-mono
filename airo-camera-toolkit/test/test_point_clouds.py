import numpy as np
from airo_camera_toolkit.point_clouds.conversions import point_cloud_to_open3d
from airo_camera_toolkit.point_clouds.operations import crop_point_cloud
from airo_typing import PointCloud


def test_crop_known():
    n = 11
    points_x = np.arange(0, n)  # 11 points, 0 to 10
    points = np.zeros((n, 3))
    points[:, 0] = points_x

    point_cloud = PointCloud(points)

    bbox = (2.5, -1, -1), (7.5, 1, 1)  # should include points (3, 0, 0) up to (7, 0, 0)

    point_cloud_cropped = crop_point_cloud(point_cloud, bbox)

    assert np.all(point_cloud_cropped.points == np.array([[3, 0, 0], [4, 0, 0], [5, 0, 0], [6, 0, 0], [7, 0, 0]]))


def test_crop_is_inside_bbox():
    n = 100
    points = np.random.rand(n, 3)  # 100 3D points in the unit cube
    bbox = (0.25, 0.25, 0.25), (0.75, 0.75, 0.75)

    point_cloud = PointCloud(points)
    point_cloud_cropped = crop_point_cloud(point_cloud, bbox)

    assert np.all(point_cloud_cropped.points >= np.array(bbox[0]))
    assert np.all(point_cloud_cropped.points <= np.array(bbox[1]))


def test_crop_equal_to_open3d():
    import open3d as o3d

    n = 100
    points = np.random.rand(n, 3)  # 100 3D points in the unit cube

    bbox = (0.25, 0.25, 0.25), (0.75, 0.75, 0.75)
    bbox_o3d = o3d.t.geometry.AxisAlignedBoundingBox(*bbox)

    point_cloud = PointCloud(points)
    pcd = point_cloud_to_open3d(point_cloud)

    point_cloud_cropped = crop_point_cloud(point_cloud, bbox)
    pcd_cropped = pcd.crop(bbox_o3d)

    assert np.all(np.asarray(pcd_cropped.point.positions.numpy()) == point_cloud_cropped.points)
