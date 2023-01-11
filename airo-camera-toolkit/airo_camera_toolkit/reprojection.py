from typing import Optional, Union

import numpy as np
from airo_spatial_algebra.operations import _HomogeneousPoints
from airo_typing import (
    CameraIntrinsicsMatrixType,
    HomogeneousMatrixType,
    NumpyDepthMapType,
    Vector2DArrayType,
    Vector3DArrayType,
    Vector3DType,
)


def reproject_to_frame_z_plane(
    image_coords: Vector2DArrayType,
    camera_intrinsics: CameraIntrinsicsMatrixType,
    camera_in_frame_pose: HomogeneousMatrixType,
    height: float = 0.0,
) -> Vector3DArrayType:
    """Reprojects points from the image plane to a Z-plane of the specified frame

    This is useful if you known the height of the object in the world frame,
    which is the case for 2D items (cloth!) or for rigid, known 3D objects with a fixed orientation.

    If the target frame is the world frame, the camera_in_frame_pose is the extrinsics matrix.

    Returns:
        positions in the world frame on the Z=height plane wrt to the frame.
    """
    # convert to homogeneous coordinates and transpose to column vectors
    homogeneous_coords = np.ones((image_coords.shape[0], 3))
    homogeneous_coords[:, :2] = image_coords
    homogeneous_coords = np.transpose(homogeneous_coords)

    camera_frame_ray_vector = np.linalg.inv(camera_intrinsics) @ homogeneous_coords

    translation = camera_in_frame_pose[0:3, 3]
    rotation_matrix = camera_in_frame_pose[0:3, 0:3]

    world_frame_ray_vectors = rotation_matrix @ camera_frame_ray_vector
    world_frame_ray_vectors = np.transpose(world_frame_ray_vectors)
    t = (height - translation[2]) / world_frame_ray_vectors[:, 2]
    points = t[:, np.newaxis] * world_frame_ray_vectors + translation
    return points


def reproject_to_frame(
    coordinates: Vector2DArrayType,
    camera_intrinsics: CameraIntrinsicsMatrixType,
    camera_in_frame_pose: HomogeneousMatrixType,
    depth_map: NumpyDepthMapType,
    mask_size=11,
    depth_percentile=0.05,
) -> np.ndarray:
    """
    Reprojects coordinates on the image plane to a base frame, as defined by the camera in frame pose.
    Args:
    Returns: (3, N) np.array containing the coordinates of the point in the camera frame. Each column is a set of
                coordinates.
    """
    # TODO: make this more generic for 2D as well  (HomogeneousPoints class)
    homogeneous_coords = np.ones((coordinates.shape[0], 3))
    homogeneous_coords[:, :2] = coordinates
    homogeneous_coords = np.transpose(homogeneous_coords)
    rays_in_camera_frame = (
        np.linalg.inv(camera_intrinsics) @ homogeneous_coords
    )  # shape is cast by numpy to column vector!

    z_values_in_camera_frame = extract_depth_from_depthmap_heuristic(
        coordinates, depth_map, mask_size, depth_percentile
    )

    t = z_values_in_camera_frame / rays_in_camera_frame[2, :]

    positions_in_camera_frame = t * rays_in_camera_frame

    homogeneous_positions_in_camera_frame = _HomogeneousPoints(positions_in_camera_frame.T).homogeneous_points.T
    homogeneous_positions_in_frame = camera_in_frame_pose @ homogeneous_positions_in_camera_frame
    return homogeneous_positions_in_frame[:3, ...].T


def extract_depth_from_depthmap_heuristic(
    coordinates: Vector2DArrayType,
    depth_map: NumpyDepthMapType,
    mask_size: int = 11,
    depth_percentile: float = 0.05,
) -> np.ndarray:
    """
    A simple heuristic to get more robust depth values of the depth map. Especially with keypoints we are often interested in points
    on the edge of an object, or even worse on a corner. Not only are these regions noisy by themselves but the keypoints could also be
    be a little off.

    This function takes the percentile of a region around the specified point and assumes we are interested in the nearest object present.
    This is not always true (think about the backside of a box looking under a 45 degree angle) but it serves as a good proxy. The more confident
    you are of your keypoints and the better the heatmaps are, the lower you could set the mask size and percentile. If you are very, very confident
    you could directly take the pointcloud as well instead of manually querying the heatmap, but I find that they are more noisy.

    Also note that this function assumes there are no negative infinity values (no objects closer than 30cm!)

    Returns:
        (np.ndarray) a 1D array of the depth values for the specified coordinates
    """

    assert mask_size % 2, "only odd sized markers allowed"
    assert (
        depth_percentile < 0.25
    ), "For straight corners, about 75 percent of the region will be background.. Are your sure you want the percentile to be lower?"
    # check all coordinates are within the size of the depth map to avoid unwanted wrapping of the array indices
    assert np.max(coordinates[:, 1]) < depth_map.shape[0], "V coordinates out of bounds"
    assert np.max(coordinates[:, 0]) < depth_map.shape[1], "U coordinates out of bounds"
    assert np.min(coordinates) >= 0, "coordinates out of bounds"

    # convert coordinates to integers
    coordinates = coordinates.astype(np.uint32)

    # extract depth values by taking the percentile of the depth values in a region around the point
    depth_regions = np.empty((coordinates.shape[0], (mask_size) ** 2))
    for i in range(coordinates.shape[0]):
        depth_region = depth_map[
            coordinates[i, 1] - mask_size // 2 : coordinates[i, 1] + mask_size // 2 + 1,
            coordinates[i, 0] - mask_size // 2 : coordinates[i, 0] + mask_size // 2 + 1,
        ]
        depth_regions[i, :] = depth_region.flatten()
    depth_values = np.nanquantile(depth_regions, depth_percentile, axis=1)

    return depth_values


def project_frame_to_image_plane(
    positions_in_frame: Union[Vector3DArrayType, Vector3DType],
    camera_matrix: CameraIntrinsicsMatrixType,
    frame_to_camera_transform: Optional[HomogeneousMatrixType] = None,
) -> Vector2DArrayType:
    """Projects an array of points from a 3D world frame to the 2D image plane.

    Projecting to the world frame is a special case of this projection operation.
    In this case, the frame_to_camera_transform is the inverse of the camera extrinsics matrix.

    Projecting from the camera frame is also a special case, in this case the frame_to_camera_transform is the identity matrix.
    If no is given, this function assumes that the points are already in the camera frame.
    """
    # TODO: should we add assert statements to validate the input?

    if frame_to_camera_transform is None:
        # if no transform is given, the positions are assumed to be in the camera frame
        # in this case, the transform is the identity matrix
        frame_to_camera_transform = np.eye(4)

    homogeneous_positions_in_world_frame = _HomogeneousPoints(positions_in_frame).homogeneous_points
    homogeneous_positions_in_world_frame = homogeneous_positions_in_world_frame.T
    homogeneous_positions_in_camera_frame = frame_to_camera_transform @ homogeneous_positions_in_world_frame
    homogeneous_positions_on_image_plane = camera_matrix @ homogeneous_positions_in_camera_frame[:3, ...]
    positions_on_image_plane = (
        homogeneous_positions_on_image_plane[:2, ...] / homogeneous_positions_on_image_plane[2, ...]
    )
    return positions_on_image_plane.T
