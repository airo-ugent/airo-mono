import airo_blender as ab
import bpy

# TODO Refactor this into a real test.

# Delete the default cube
bpy.ops.object.delete()

# Untransformed sphere
bpy.ops.mesh.primitive_uv_sphere_add()
sphere0 = bpy.context.object
bounding_box = ab.axis_aligned_bounding_box(sphere0)
ab.add_axis_aligned_bounding_box(*bounding_box)

# Transformed sphere
bpy.ops.mesh.primitive_uv_sphere_add()
sphere1 = bpy.context.object
sphere1.location.z += 3.0
bpy.context.view_layer.update()  # Propagate changed location to matrix_world (calling operators also does this)
bounding_box = ab.axis_aligned_bounding_box(sphere1)
ab.add_axis_aligned_bounding_box(*bounding_box)

# Collection instance of previous sphere put in a collection
# Select only sphere1
bpy.ops.object.select_all(action="DESELECT")
sphere1.select_set(True)
bpy.ops.object.link_to_collection(collection_index=0, is_new=True, new_collection_name="Transformed sphere")
bpy.ops.object.collection_instance_add(collection="Transformed sphere")
sphere2 = bpy.context.object
sphere2.location.x += 4.0
bpy.context.view_layer.update()

bounding_box = ab.axis_aligned_bounding_box(sphere2)
ab.add_axis_aligned_bounding_box(*bounding_box)
