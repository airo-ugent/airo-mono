import json

import airo_blender as ab
import bpy
import numpy as np

# Delete the default cube
bpy.ops.object.delete()

# Get the list of available assets
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

# Bounding box of the table
min_corner, max_corner = ab.axis_aligned_bounding_box(table_collection.objects)
x_min, y_min, _ = min_corner
x_max, y_max, z_max = max_corner

# Load the croissant collection
croissant_info = [asset for asset in assets if asset["name"] == "croissant"][0]
croissant_collection = ab.load_asset(**croissant_info)

# Create 12 croissant instances
for _ in range(12):
    bpy.ops.object.collection_instance_add(collection=croissant_collection.name)
    croissant = bpy.context.object

    # Place the croissant randomly on the table
    margin = 0.05
    x = np.random.uniform(x_min + margin, x_max - margin)
    y = np.random.uniform(y_min + margin, y_max - margin)
    z = z_max
    croissant.location = x, y, z

    # Randomize rotation around the z-axis
    rz = np.random.uniform(0, 2 * np.pi)
    croissant.rotation_euler = 0, 0, rz

    # Randomize the 3 scale components
    scale = np.random.uniform(0.6, 1.4, size=3)
    croissant.scale = scale
