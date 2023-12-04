from test.test_config import _ImageTestValues

import numpy as np
from airo_camera_toolkit.reprojection import (
    extract_depth_from_depthmap_heuristic,
    project_frame_to_image_plane,
    reproject_to_frame,
)
from PIL import Image


def test_world_to_image_plane_projection():
    world_pose_in_camera_frame = np.linalg.inv(_ImageTestValues._extrinsics_matrix)
    projected_points = project_frame_to_image_plane(
        _ImageTestValues._positions_in_world_frame, _ImageTestValues._intrinsics_matrix, world_pose_in_camera_frame
    )
    assert np.isclose(projected_points, _ImageTestValues._positions_on_image_plane, atol=1e-2).all()


def test_world_to_image_plane_projection_single_vector():
    # is this test necessary?
    # if the Homogeneous points class works as expected, this should work too..
    world_pose_in_camera_frame = np.linalg.inv(_ImageTestValues._extrinsics_matrix)
    projected_points = project_frame_to_image_plane(
        _ImageTestValues._positions_in_world_frame[0], _ImageTestValues._intrinsics_matrix, world_pose_in_camera_frame
    )
    assert np.isclose(projected_points, _ImageTestValues._positions_on_image_plane[0], atol=1e-2).all()


def test_camera_to_image_plane_projection():
    projected_points = project_frame_to_image_plane(
        _ImageTestValues._positions_in_camera_frame, _ImageTestValues._intrinsics_matrix
    )
    assert np.isclose(projected_points, _ImageTestValues._positions_on_image_plane, atol=1e-2).all()


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


def test_reproject_world_frame():
    depth_map = _load_depthmap()
    reprojected_points = reproject_to_frame(
        _ImageTestValues._positions_on_image_plane,
        _ImageTestValues._intrinsics_matrix,
        _ImageTestValues._extrinsics_matrix,
        depth_map,
    )
    assert np.isclose(reprojected_points, _ImageTestValues._positions_in_world_frame, atol=1e-2).all()


def test_reproject_camera_frame():
    depth_map = _load_depthmap()
    reprojected_points = reproject_to_frame(
        _ImageTestValues._positions_on_image_plane, _ImageTestValues._intrinsics_matrix, np.eye(4), depth_map
    )
    assert np.isclose(reprojected_points, _ImageTestValues._positions_in_camera_frame, atol=1e-2).all()
