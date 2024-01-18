from test.test_config import _ImageTestValues

import numpy as np
from airo_camera_toolkit.pinhole_operations import (
    extract_depth_from_depthmap_heuristic,
    multiview_triangulation_midpoint,
    project_points_to_image_plane,
    unproject_using_depthmap,
)
from PIL import Image


def test_world_to_image_plane_projection():
    position_sets = [_ImageTestValues._positions_in_camera_frame, _ImageTestValues._positions_in_camera_frame[0]]
    coords_sets = [_ImageTestValues._positions_on_image_plane, _ImageTestValues._positions_on_image_plane[0]]

    for positions, coords in zip(position_sets, coords_sets):
        projected_points = project_points_to_image_plane(positions, _ImageTestValues._intrinsics_matrix)
        assert np.isclose(projected_points, coords, atol=1e-2).all()


def _load_depth_image():
    # load image, convert to grayscale and then convert to numpy array
    depth_map = Image.open(_ImageTestValues._depth_image_path).convert("L")
    depth_map = np.array(depth_map)
    return depth_map


def _load_depthmap():
    return np.load(_ImageTestValues._depth_map_path)


def test_depth_heuristic():
    # load image, convert to grayscale and then convert to numpy array
    depth_map = _load_depthmap()
    depths = extract_depth_from_depthmap_heuristic(_ImageTestValues._positions_on_image_plane, depth_map, mask_size=51)
    assert np.isclose(depths, _ImageTestValues._depth_z_values, atol=1e-2).all()


def test_reproject_camera_frame():
    depth_map = _load_depthmap()
    reprojected_points = unproject_using_depthmap(
        _ImageTestValues._positions_on_image_plane, depth_map, _ImageTestValues._intrinsics_matrix
    )
    assert np.isclose(reprojected_points, _ImageTestValues._positions_in_camera_frame, atol=1e-2).all()


def test_triangulation():
    # construct a simple example
    position = np.array([0.02, 0.01, 0.01])
    # extrinsics of camera 1 meter to the left looking at the origin
    extrinsics1 = np.array([[0, 0, -1, 1], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    # extrinsics of camera 1 meter to the front looking at the origin
    extrinsics2 = np.array([[1, 0, 0, 0], [0, 0, -1, 1], [0, 1, 0, 0], [0, 0, 0, 1]])

    # intrinsics of both cameras: 500x500 pixels, 1000px focal length
    intrinsics = np.array([[1000, 0, 2000], [0, 1000, 2000], [0, 0, 1]])
    # position in the camera frames
    position1 = (np.linalg.inv(extrinsics1) @ np.append(position, 1))[:3]
    position2 = (np.linalg.inv(extrinsics2) @ np.append(position, 1))[:3]

    # image coordinates of the point in both cameras
    image_coordinates_1 = project_points_to_image_plane(position1, intrinsics)[0]
    image_coordinates_2 = project_points_to_image_plane(position2, intrinsics)[0]

    ## actual test

    # triangulate the point
    triangulated_point = multiview_triangulation_midpoint(
        [extrinsics1, extrinsics2], [intrinsics, intrinsics], [image_coordinates_1, image_coordinates_2]
    )

    assert np.isclose(triangulated_point, position, atol=1e-2).all()
