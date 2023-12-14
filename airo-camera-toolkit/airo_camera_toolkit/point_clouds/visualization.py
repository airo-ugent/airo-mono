from typing import Any, Tuple

import open3d as o3d
from airo_typing import Vector3DType


def open3d_point(
    position: Vector3DType, color: Tuple[float, float, float], radius: float = 0.01
) -> Any:  # Change Any back to o3d.geometry.TriangleMesh
    """Creates a small sphere mesh for visualization in open3d.

    Args:
        position: 3D position of the point
        color: RGB color of the point as 0-1 floats
        radius: radius of the sphere

    Returns:
        sphere: an open3d mesh
    """
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(position)
    sphere.paint_uniform_color(color)
    sphere.compute_vertex_normals()
    return sphere
