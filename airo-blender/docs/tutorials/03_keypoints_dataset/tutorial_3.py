import json
import os

import airo_blender as ab
import bpy
import numpy as np
from airo_blender.coco_parser import CocoImage

# Set numpy random seed for reproducibility
random_seed = 0
np.random.seed(random_seed)

# Delete the default cube
bpy.ops.object.delete()

scene = bpy.context.scene

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
scene.world = woods


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
scene.render.engine = "CYCLES"
scene.cycles.samples = 64

image_width, image_height = 512, 512
scene.render.resolution_x = image_width
scene.render.resolution_y = image_height


# Make a directory to organize all the outputs
output_dir = f"{random_seed:08d}"
os.makedirs(output_dir, exist_ok=True)

# Set image format to PNG
image_name = f"{random_seed:08d}"

# Semantic segmentation of the towel
towel.pass_index = 1

scene.view_layers["ViewLayer"].use_pass_object_index = True
scene.use_nodes = True

# Add a file output node to the scene
tree = scene.node_tree
links = tree.links
nodes = tree.nodes
node = nodes.new("CompositorNodeOutputFile")
node.location = (500, 200)
node.base_path = output_dir
slot_image = node.file_slots["Image"]
slot_image.path = image_name
slot_image.format.color_mode = "RGB"

# Prevent the 0001 suffix from being added to the file name


segmentation_name = f"{image_name}_segmentation"
node.file_slots.new(segmentation_name)
slot_segmentation = node.file_slots[segmentation_name]

# slot_segmentation.path = f"{random_seed:08d}_segmentation"
slot_segmentation.format.color_mode = "BW"
slot_segmentation.use_node_format = False
slot_segmentation.save_as_render = False

render_layers_node = nodes["Render Layers"]
links.new(render_layers_node.outputs["Image"], node.inputs[0])

# Divide the IndexOB by 255 to get a 0-1 range
# math_node = nodes.new("CompositorNodeMath")
# math_node.operation = "DIVIDE"
# math_node.inputs[1].default_value = 255
# math_node.location = (300, 200)
# links.new(render_layers_node.outputs["IndexOB"], math_node.inputs[0])
# links.new(math_node.outputs[0], node.inputs[slot_segmentation.path])

# Other method, use the mask ID node
mask_id_node = nodes.new("CompositorNodeIDMask")
mask_id_node.index = 1
mask_id_node.location = (300, 200)
links.new(render_layers_node.outputs["IndexOB"], mask_id_node.inputs[0])
links.new(mask_id_node.outputs[0], node.inputs[slot_segmentation.path])

# Rendering the scene into an image
bpy.ops.render.render(animation=False)

# Annoying fix, because Blender adds a 0001 suffix to the file name which can't be disabled
image_path = os.path.join(output_dir, f"{image_name}0001.png")
image_path_new = os.path.join(output_dir, f"{image_name}.png")
os.rename(image_path, image_path_new)

segmentation_path = os.path.join(output_dir, f"{segmentation_name}0001.png")
segmentation_path_new = os.path.join(output_dir, f"{segmentation_name}.png")
os.rename(segmentation_path, segmentation_path_new)

# TODO get bounding box of the segmentation mask
# TODO load segmentation mask
# np.where(segmentation_mask == True) # get the coordinates
import cv2

segmentation_mask = cv2.imread(segmentation_path_new, cv2.IMREAD_GRAYSCALE)
mask_coords = np.where(segmentation_mask == 255)
x_min = np.min(mask_coords[1])
x_max = np.max(mask_coords[1])
y_min = np.min(mask_coords[0])
y_max = np.max(mask_coords[0])
print(x_min, x_max, y_min, y_max)

# drawing the rectangle
image_bgr = cv2.imread(image_path_new)
cv2.rectangle(image_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
image_annotated_path = os.path.join(output_dir, f"{image_name}_annotated.png")
# Draw a circle in the top left corner of the bounding box
cv2.circle(image_bgr, (x_min, y_min), 5, (0, 0, 255), -1)
cv2.imwrite(image_annotated_path, image_bgr)

coco_image = CocoImage(file_name=image_path_new, height=image_height, width=image_width, id=random_seed)
print(coco_image)
