"""
This file contains functions to calculate bounding boxes of one or more Blender objects.
We also have functions to visualize some of these bouding boxes in a Blender scene (name start wirh add_).

This could be:
- Global axis-aligned bounding box ()
- Object-oriented bounding box, i.e. the bounding box of an object in its local coordinate system
- Arbirary oriented bounding box, e.g. the minimum volume bounding box.

Additionally, we might want to support 2D screen space bounding boxes.

Current only the first type is implemented, but contributions are welcome for the others.

References:
    https://en.wikipedia.org/wiki/Minimum_bounding_box
"""

from typing import Union

import bpy
import numpy as np
from bpy.types import Object


def add_axis_aligned_bounding_box(min_corner: np.ndarray, max_corner: np.ndarray) -> bpy.types.Object:
    """Add an axis aligned bounding box to the scene.

    Args:
        min_corner: Minimum corner of the bounding box.
        max_corner: Maximum corner of the bounding box.
    """
    bpy.ops.mesh.primitive_cube_add()
    cube = bpy.context.object
    cube.location = (min_corner + max_corner) / 2
    cube.scale = (max_corner - min_corner) / 2  # Divide by 2 because the default size is 2
    cube.name = "Axis aligned bounding box"
    return cube


def axis_aligned_bounding_box(objects: Union[Object, list[Object]]) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the axis aligned bounding box of one or more objects.
    Collection instance objects are allowed.

    # TODO Test more extensively e.g. with rotated and scaled objects.

    Args:
        objects: List of objects to calculate the joint bounding box of.

    Returns:
        Two diagonal corners of the bounding box, the minimum and the maximum corner.
    """
    # Allow passing a single object
    if isinstance(objects, Object):
        objects = [objects]

    min_corner = np.array([np.inf, np.inf, np.inf])
    max_corner = np.array([-np.inf, -np.inf, -np.inf])

    for object in objects:
        if object.type == "EMPTY" and object.instance_type == "COLLECTION" and object.instance_collection:
            # Collection instances are a special case, we want the bounding box of the collection it references
            local_bounding_box = axis_aligned_bounding_box(object.instance_collection.objects)
            _min_local_corner, _max_local_corner = local_bounding_box
        else:
            all_local_corners = np.array(object.bound_box)  # 8 corners of the bounding box in local coordinates
            _min_local_corner = all_local_corners.min(axis=0)
            _max_local_corner = all_local_corners.max(axis=0)

        # Transform the bounding box into world coordinates
        _min_corner = (np.array(object.matrix_world) @ np.append(_min_local_corner, 1))[:3]
        _max_corner = (np.array(object.matrix_world) @ np.append(_max_local_corner, 1))[:3]

        # Update the joint global bounding box
        min_corner = np.minimum(min_corner, _min_corner)
        max_corner = np.maximum(max_corner, _max_corner)

    return min_corner, max_corner
