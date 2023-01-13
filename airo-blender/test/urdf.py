import os

import bpy
import xmltodict  # ./blender/blender-3.4.1-linux-x64/3.4/python/bin/pip3 install xmltodict
from mathutils import Vector

bpy.ops.object.delete()

urdf_path = "/home/idlab185/urdfpy/tests/data/ur5/ur5.urdf"
# urdf_path = "/home/idlab185/robotiq_arg85_description/robots/robotiq_arg85_description.URDF"
# urdf_path = "/home/idlab185/rbo_dataset/rbo_dataset/objects/ikea/configuration_2017-07-22/ikea.urdf"
urdf_dir = os.path.dirname(urdf_path)

file = open(urdf_path, "r")
xml_content = file.read()

urdf_dict = xmltodict.parse(xml_content)

# import json
# print(json.dumps(urdf_dict, indent=4))

print("name of first link: ", urdf_dict["robot"]["link"][0]["@name"])

links = urdf_dict["robot"]["link"]
blender_links = {}

for i, link in enumerate(links):
    bpy.ops.object.empty_add(type="ARROWS", radius=0.05)
    empty = bpy.context.object
    empty.name = link["@name"]
    blender_links[link["@name"]] = empty
    empty.lock_rotation = (True, True, True)
    empty.lock_location = (True, True, True)

    if "visual" not in link:
        continue

    relative_mesh_path = link["visual"]["geometry"]["mesh"]["@filename"]
    mesh_path = os.path.join(urdf_dir, relative_mesh_path)

    old_objs = set(bpy.context.scene.objects)

    # if file ends with .dae, use collada importer
    if mesh_path.endswith(".dae"):
        bpy.ops.wm.collada_import(filepath=mesh_path)
    else:
        bpy.ops.import_mesh.stl(filepath=mesh_path)

    imported_objs = set(bpy.context.scene.objects) - old_objs
    print(link["@name"], "imported", len(imported_objs), "objects")
    for obj in imported_objs:
        if obj.type in ("CAMERA", "LIGHT"):
            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            bpy.ops.object.delete()
        else:
            obj.lock_rotation = (True, True, True)
            obj.lock_location = (True, True, True)
            obj.parent = empty

joints = urdf_dict["robot"]["joint"]
for j, joint in enumerate(joints):
    parent = blender_links[joint["parent"]["@link"]]
    child = blender_links[joint["child"]["@link"]]
    child.parent = parent
    origin = joint["origin"]
    translation = Vector(
        [float(f) for f in origin["@xyz"].split(" ") if f]
    )  # if f because rbo dataset has trailing space
    rotation = Vector([float(f) for f in origin["@rpy"].split(" ") if f])
    child.location = translation
    child.rotation_euler = rotation

    if joint["@type"] == "revolute":
        # Unlock the rotation axis of the revolute joint
        axis_split = joint["axis"]["@xyz"].split(" ")
        search = "1" in axis_split
        if "1" in axis_split:
            search = "1"
        elif "-1" in axis_split:
            search = "-1"
        else:
            continue
        axis_index = axis_split.index(search)
        child.lock_rotation[axis_index] = False
        child.empty_display_size = 0.1

# TODO free location and rotation of the base link (named world for ur5)
