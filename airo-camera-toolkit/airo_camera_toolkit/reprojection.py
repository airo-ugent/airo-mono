from typing import Union

import numpy as np


def homogeneous_vector(points: np.ndarray):
    """
    Args:
        points: (3, N) array of points, each column is a set of coordinates
    """
    points_homog = np.vstack((points, np.ones(points.shape[1])))
    return points_homog


def homogeneous_matrix(translation_vector: np.ndarray, rotation_matrix: np.ndarray):
    translation_vector = translation_vector.reshape((3,))
    if rotation_matrix.shape != (3, 3):
        raise ValueError(f"Rotation matrix should be shape (3,3) and not {rotation_matrix.shape}")

    hommat = np.zeros((4, 4))
    hommat[0:3, 0:3] = rotation_matrix
    hommat[0:3, 3] = translation_vector
    hommat[3, :] = [0, 0, 0, 1]

    return hommat


def reproject_to_world_z_plane(
    image_coords: np.ndarray, camera_matrix: np.ndarray, world_in_camera_frame_pose: np.ndarray, height: float = 0.0
):
    """Reprojects points from the camera plane to the specified Z-plane of the world frame
    (as this is often the frame in which you have the z-information)
    This is useful if you known the height of the object in the world frame,
     which is the case for 2D items (cloth!) or for rigid, known 3D objects (that do not tumble)
    Args:
        image_coords (_type_): 2D (Nx2) numpy vector with (u,v) coordinates of a point w.r.t. the camera matrix
        camera_matrix (_type_): 2D 3x3 numpy array with camera matrix
        world_in_camera_frame_pose (_type_): 2D 4x4 numpy array with the transformation in homogeneous coordinates from the camera to the world origin.
    Returns:
        _type_ Nx3 numpy axis with world coordinates on the Z=height plane wrt to the world frame.
    """
    if image_coords.shape[0] == 0:
        return []

    coords = np.ones((image_coords.shape[0], 3))
    coords[:, :2] = image_coords
    image_coords = np.transpose(coords)

    camera_in_world_frame_pose = np.linalg.inv(world_in_camera_frame_pose)

    camera_frame_ray_vector = np.linalg.inv(camera_matrix) @ image_coords

    translation = camera_in_world_frame_pose[0:3, 3]
    rotation_matrix = camera_in_world_frame_pose[0:3, 0:3]

    world_frame_ray_vectors = rotation_matrix @ camera_frame_ray_vector
    world_frame_ray_vectors = np.transpose(world_frame_ray_vectors)
    t = (height - translation[2]) / world_frame_ray_vectors[:, 2]
    points = t[:, np.newaxis] * world_frame_ray_vectors + translation
    return points


def reproject_to_world_frame(
    _u: Union[int, np.ndarray, list],
    _v: Union[int, np.ndarray, list],
    camera_intrinsics_matrix: np.ndarray,
    camera_extrinsics_hommat: np.ndarray,
    depth_map: np.ndarray,
    mask_size=11,
    depth_percentile=0.05,
) -> np.ndarray:
    """
    Reprojects points on the image plane to a base frame, as defined by an extrinsics matrix.
    point = (_u[i], _v[i], 0) with origin in the top left corner of the img and y-axis pointing down
    Args:
        _u: (N,) array of u-coordinates
        _v: (N,) array of v-coordinates
        camera_intrinsics_matrix: 3x3 camera matrix
        camera_extrinsics_hommat: 4x4 homogeneous extrinsics matrix
        depth_map: LxM depth map, depth_map at coord (u,v) gives the z-value of the position of that pixel in the
                    camera frame (!not the distance to the camera!)
        depthmap_mask_size: see use in extract_depth_from_depthmap_heuristic
        depth_percentile: see use in extract_depth_from_depthmap_heuristic
    Returns: (3, N) np.array containing the coordinates of the point in the camera frame. Each column is a set of
                coordinates.
    """
    points_in_camera_frame = reproject_to_camera_frame(
        _u, _v, camera_intrinsics_matrix, depth_map, mask_size, depth_percentile
    )
    homogeneous_points = homogeneous_vector(points_in_camera_frame)
    point_in_base_frame = camera_extrinsics_hommat @ homogeneous_points
    return point_in_base_frame[0:3, :]


def reproject_to_camera_frame(
    _u: Union[int, np.ndarray, list],
    _v: Union[int, np.ndarray, list],
    camera_matrix: np.ndarray,
    depth_map: np.ndarray,
    depthmap_mask_size: int = 11,
    depth_percentile: float = 0.05,
) -> np.ndarray:
    """
    Reprojects points on the image plane to the 3D frame of the camera.
    point = (_u[i], _v[i], 0) with origin in the top left corner of the img and y-axis pointing down
    Args:
        _u: (N,) array of u-coordinates
        _v: (N,) array of v-coordinates
        camera_matrix: 3x3 camera matrix
        depth_map: LxM depth map, depth_map at coord (u,v) gives the z-value of the position of that pixel in the
                    camera frame (!not the distance to the camera!)
        depthmap_mask_size: see use in extract_depth_from_depthmap_heuristic
        depth_percentile: see use in extract_depth_from_depthmap_heuristic
    Returns: (3, N) np.array containing the coordinates of the point in the camera frame. Each column is a set of
                coordinates.
    """
    # ensure proper functionality when integers are passed for u and v
    u = np.array(_u).flatten()
    v = np.array(_v).flatten()

    img_coords = np.array([u, v, [1.0 for _ in range(u.size)]])
    rays_in_camera_frame = np.linalg.inv(camera_matrix) @ img_coords  # shape is cast by numpy to column vector!

    z_values_in_camera_frame = extract_depth_from_depthmap_heuristic(
        u, v, depth_map, depthmap_mask_size, depth_percentile
    )
    t = z_values_in_camera_frame / rays_in_camera_frame[2, :]

    positions_in_camera_frame = t * rays_in_camera_frame
    return positions_in_camera_frame


def extract_depth_from_depthmap_heuristic(
    _u: Union[int, np.ndarray, list],
    _v: Union[int, np.ndarray, list],
    depth_map: np.ndarray,
    mask_size: int = 11,
    depth_percentile: float = 0.05,
) -> Union[float, np.ndarray]:
    """
    A simple heuristic to get more robust depth values of the depth map. Especially with keypoints we are often interested in points
    on the edge of an object, or even worse on a corner. Not only are these regions noisy by themselves but the keypoints could also be
    be a little off.
    This function takes the percentile of a region around the specified point and assumes we are interested in the nearest object present.
    This is not always true (think about the backside of a box looking under a 45 degree angle) but it serves as a good proxy. The more confident
    you are of your keypoints and the better the heatmaps are, the lower you could set the mask size and percentile. If you are very, very confident
    you could directly take the pointcloud as well instead of manually querying the heatmap, but I find that they are more noisy.
    Also note that this function assumes there are no negative infinity values (no objects closer than 30cm!)
    Args:
        _u: 1D array of u-coordinates
        _v: 1D array of v-coordinates
    """
    # ensure proper functionality when integers are passed for u and v
    u = np.array(_u).flatten()
    v = np.array(_v).flatten()

    assert u.shape == v.shape, "u and v arrays have dissimilar dimensions"
    assert mask_size % 2, "only odd sized markers allowed"
    assert (
        depth_percentile < 0.25
    ), "For straight corners, about 75 percent of the region will be background.. Are your sure you want the percentile to be lower?"

    depth_regions = np.zeros((u.size, (mask_size - 1) ** 2))
    for i in range(u.size):
        depth_region = depth_map[
            v[i] - mask_size // 2 : v[i] + mask_size // 2, u[i] - mask_size // 2 : u[i] + mask_size // 2
        ]
        depth_regions[i, :] = depth_region.flatten()
    depth_values = np.nanquantile(depth_regions, depth_percentile, axis=1)
    if depth_values.size == 1:
        return float(depth_values)
    else:
        return depth_values


def project_world_to_image_plane(
    point: np.ndarray, world_to_camera_transform: np.ndarray, camera_matrix: np.ndarray
) -> np.ndarray:
    """Projects a point from the 3D world frame to the 2D image plane.

    Works in two steps. First transforms the 3D point to a 3D point in camera frame.
    Then projects the point onto the image plane. Note the normalization by the third coordinate."""
    point = np.array(point).reshape((3, 1))
    point_homogeneous = np.append(point, [[1.0]])
    point_camera_homogeneous = world_to_camera_transform @ point_homogeneous
    point_camera = point_camera_homogeneous[:3]
    point_image_homogeneous = camera_matrix @ point_camera
    point_image = point_image_homogeneous[:2] / point_image_homogeneous[2]
    return point_image
