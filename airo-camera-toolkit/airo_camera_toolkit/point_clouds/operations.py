from airo_typing import BoundingBox3DType, PointCloud


def crop_pointcloud(
    bounding_box: BoundingBox3DType,
    point_cloud: PointCloud,
) -> PointCloud:
    """Creates a new point cloud that is cropped to the given bounding box.
    Will also crop the colors and attributes if they are present.

    Args:
        bounding_box: the bounding box that surrounds the points to keep
        point_cloud: the point cloud to crop

    Returns:
        the new cropped point cloud
    """
    (x_min, y_min, z_min), (x_max, y_max, z_max) = bounding_box

    points = point_cloud.points

    mask = (
        (points[:, 0] >= x_min)
        & (points[:, 0] <= x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] <= y_max)
        & (points[:, 2] >= z_min)
        & (points[:, 2] <= z_max)
    )
    points_cropped = points[mask]
    colors_cropped = None if point_cloud.colors is None else point_cloud.colors[mask]

    attributes_cropped = None
    if point_cloud.attributes is not None:
        attributes_cropped = {}
        for key, value in point_cloud.attributes.items():
            attributes_cropped[key] = value[mask]

    point_cloud_cropped = PointCloud(points_cropped, colors_cropped, attributes_cropped)
    return point_cloud_cropped
