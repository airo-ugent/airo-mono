import json

import airo_blender as ab
import bpy
import numpy as np

# Create the cylinder
bpy.ops.mesh.primitive_cylinder_add()

# We can assign the blender Objects to variables for easy access
cube = bpy.data.objects["Cube"]
cylinder = bpy.data.objects["Cylinder"]

# Playing the objects' properties
cube.scale = (2.0, 2.0, 0.1)
cube.location.z -= 0.1
cylinder.scale = (0.5, 0.5, 1.0)
cylinder.location.z = 0.5
cylinder.rotation_euler.x = 3.14 / 2.0

# Adding a nice material
red = (1.0, 0.0, 0.0, 1.0)
ab.add_material(cylinder, red)

# Making the background brighter
bpy.data.worlds["World"].node_tree.nodes["Background"].inputs["Color"].default_value = (1.0, 0.9, 0.7, 1.0)

# Telling Blender to render with Cycles, and how many rays we want to cast per pixel
bpy.context.scene.render.engine = "CYCLES"
bpy.context.scene.cycles.samples = 64

bpy.context.scene.render.resolution_x = 1024
bpy.context.scene.render.resolution_y = 512

bpy.context.scene.render.filepath = "red_cylinder.jpg"

# Rendering the scene into an image
bpy.ops.render.render(write_still=True)

# Writing the cylinder pose to a file
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

camera = bpy.context.scene.camera

# Find the model to camera transform
# Use use the Drake notation here, X_ab means the transform from frame b to frame a
X_wm = cylinder.matrix_world  # world to model
X_wc = camera.matrix_world  # world to camera
X_mc = X_wm.inverted() @ X_wc  # model to camera

translation, rotation, scale = X_mc.decompose()

cam_t_m2c = list(1000.0 * translation)
cam_R_m2c = list(1000.0 * np.array(rotation.to_matrix()).flatten())

data = {
    "cam_R_m2c": cam_R_m2c,
    "cam_t_m2c": cam_t_m2c,
}

with open("cylinder_pose.json", "w") as file:
    json.dump(data, file, indent=4)
