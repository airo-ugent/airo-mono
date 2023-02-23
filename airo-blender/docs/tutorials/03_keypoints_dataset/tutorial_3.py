import json

import airo_blender as ab
import bpy
import numpy as np

# Set numpy random seed for reproducibility
random_seed = 0
np.random.seed(random_seed)

# Delete the default cube
bpy.ops.object.delete()

# Part 1: Create the towel geometry and material
width, length = 0.2, 0.3

vertices = [
    np.array([-width / 2, -length / 2, 0.0]),
    np.array([-width / 2, length / 2, 0.0]),
    np.array([width / 2, length / 2, 0.0]),
    np.array([width / 2, -length / 2, 0.0]),
]
edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
faces = [(0, 1, 2, 3)]

name = "Towel"
mesh = bpy.data.meshes.new(name)
mesh.from_pydata(vertices, edges, faces)
mesh.update()
towel = bpy.data.objects.new(name, mesh)
bpy.context.collection.objects.link(towel)

# Add a random color to the towel
random_rgb_color = np.random.uniform(0.0, 1.0, size=3)
ab.add_material(towel, random_rgb_color)

# Part 2: Load a background and a table
with open("asset_snapshot.json", "r") as file:
    assets = json.load(file)["assets"]

# Set an HDRI world background
worlds = [asset for asset in assets if asset["type"] == "worlds"]
woods_info = [asset for asset in worlds if asset["name"] == "woods"][0]
woods = ab.load_asset(**woods_info)
bpy.context.scene.world = woods


# Load a random table
def table_filter(asset_info: dict) -> bool:
    if asset_info["type"] != "collections":
        return False

    if "table" not in asset_info["tags"]:
        return False

    not_tables = ["desk_lamp_arm_01", "CoffeeCart_01", "wicker_basket_01"]
    if asset_info["name"] in not_tables:
        return False

    return True


tables = [asset for asset in assets if table_filter(asset)]
random_table = np.random.choice(tables)
table_collection = ab.load_asset(**random_table)
bpy.ops.object.collection_instance_add(collection=table_collection.name)
table = bpy.context.object

# Part 3: Place the towel on the table
# Bounding box of the table
_, max_corner = ab.axis_aligned_bounding_box(table_collection.objects)
_, _, z_max = max_corner

# Place the towel on the table with a random rotation
towel.location = (0.0, 0.0, z_max + 0.003)
towel.rotation_euler = (0.0, 0.0, np.random.uniform(0.0, 2 * np.pi))

# Part 4: Setting up the camera
camera = bpy.data.objects["Camera"]


def random_point_on_unit_sphere() -> np.ndarray:
    point_gaussian_3D = np.random.randn(3)
    point_on_unit_sphere = point_gaussian_3D / np.linalg.norm(point_gaussian_3D)
    return point_on_unit_sphere


# Sample a point on the top part of the unit sphere
high_point = random_point_on_unit_sphere()
while high_point[2] < 0.75:
    high_point = random_point_on_unit_sphere()

# Place the camera above the table
high_point[2] += z_max
camera.location = high_point

# Make the camera look at the towel center
camera_direction = towel.location - camera.location  # Note: these are mathutils Vectors
camera.rotation_euler = camera_direction.to_track_quat("-Z", "Y").to_euler()

# Set the camera focal length to 32 mm
camera.data.lens = 32

# Part 5: Render the image

# Telling Blender to render with Cycles, and how many rays we want to cast per pixel
bpy.context.scene.render.engine = "CYCLES"
bpy.context.scene.cycles.samples = 64

bpy.context.scene.render.resolution_x = 512
bpy.context.scene.render.resolution_y = 512

bpy.context.scene.render.filepath = f"{random_seed:08d}.jpg"

# Rendering the scene into an image
bpy.ops.render.render(write_still=True)
