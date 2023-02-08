import bpy


def add_material(blender_object: bpy.types.Object, color: tuple[float, float, float], roughness: float = 0.5):
    """Adds a new material to a blender object.

    Args:
        blender_object: The blender object to add the material to.
        color: The color of the material. RGB or RGBA values between 0 and 1.
        roughness: The roughness of the material. A value between 0 and 1.
    """
    if len(color) == 3:
        color = (*color, 1.0)

    material = bpy.data.materials.new(name="Material")
    material.diffuse_color = color  # Viewport color, doesn't affect render
    material.use_nodes = True

    bdsf = material.node_tree.nodes["Principled BSDF"]
    bdsf.inputs["Base Color"].default_value = color
    bdsf.inputs["Roughness"].default_value = roughness

    blender_object.data.materials.append(material)
    return material
