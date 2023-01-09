import bpy


# TODO: docstring and type hints
# TODO: handle length 3 tuples for colors
# TODO: add roughness parameter
def add_material(blender_object, color):
    material = bpy.data.materials.new(name="Material")
    material.diffuse_color = color
    material.use_nodes = True
    bdsf = material.node_tree.nodes["Principled BSDF"]
    bdsf.inputs["Base Color"].default_value = color
    blender_object.data.materials.append(material)
    return material
