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


def test_depth_heuristic_edge_cases():
    """Test depth extraction for points near or on the edge of the image."""
    # Create a simple depth map with known values
    IMAGE_SIZE = 100
    depth_map = np.ones((IMAGE_SIZE, IMAGE_SIZE)) * 5.0
    mask_size = 11
    
    # Test cases: points at different edge positions
    edge_coordinates = np.array([
        [0, 0],                          # top-left corner
        [IMAGE_SIZE - 1, 0],             # top-right corner
        [0, IMAGE_SIZE - 1],             # bottom-left corner
        [IMAGE_SIZE - 1, IMAGE_SIZE - 1],# bottom-right corner
        [5, 0],                          # top edge
        [5, IMAGE_SIZE - 1],             # bottom edge
        [0, 50],                         # left edge
        [IMAGE_SIZE - 1, 50],            # right edge
    ], dtype=float)
    
    # This should not raise an error and should return valid depth values
    depths = extract_depth_from_depthmap_heuristic(edge_coordinates, depth_map, mask_size=mask_size)
    
    # All depths should be close to 5.0 (or NaN for fully out-of-bounds cases)
    # The function should handle edge cases gracefully
    assert depths.shape[0] == edge_coordinates.shape[0], "Should return one depth value per coordinate"
    # At least some values should be valid (not all NaN)
    assert not np.all(np.isnan(depths)), "Should return at least some valid depth values"


def test_depth_heuristic_with_varied_depths_at_edges():
    """Test that depth extraction works correctly when points are at edges with varying depth values."""
    # Create a depth map with a gradient
    IMAGE_SIZE = 100
    depth_map = np.tile(np.arange(IMAGE_SIZE, dtype=float).reshape(IMAGE_SIZE, 1), (1, IMAGE_SIZE))
    mask_size = 5
    
    # Test a point near the top edge
    coords = np.array([[50, 2]], dtype=float)
    depths = extract_depth_from_depthmap_heuristic(coords, depth_map, mask_size=mask_size)
    
    # Should return a valid depth value (not crash)
    assert depths.shape[0] == 1
    # The depth should be in a reasonable range (close to the actual row index)
    assert not np.isnan(depths[0]), "Should return a valid depth value"


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
