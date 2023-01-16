import os

import airo_blender as ab
import bpy
import numpy as np
import xmltodict  # ./blender/blender-3.4.1-linux-x64/3.4/python/bin/pip3 install xmltodict
from mathutils import Vector

# reference

bpy.ops.object.delete()

urdf_path = "/home/idlab185/urdfpy/tests/data/ur5/ur5.urdf"
# urdf_path = "/home/idlab185/robotiq_arg85_description/robots/robotiq_arg85_description.URDF"
# urdf_path = "/home/idlab185/robotiq_2finger_grippers/robotiq_2f_85_gripper_visualization/urdf/robotiq2f85.urdf"
urdf_path = "/home/idlab185/partnet-mobility-sample/44853/mobility.urdf"  # cabinet
# urdf_path = "/home/idlab185/partnet-mobility-sample/7128/mobility.urdf"
# urdf_path = "/home/idlab185/partnet-mobility-sample/7130/mobility.urdf"
# urdf_path = "/home/idlab185/partnet-mobility-sample/103452/mobility.urdf" # washing machine

urdf_dir = os.path.dirname(urdf_path)

file = open(urdf_path, "r")
xml_content = file.read()

urdf_dict = xmltodict.parse(xml_content)


# print(json.dumps(urdf_dict, indent=4))

print("name of first link: ", urdf_dict["robot"]["link"][0]["@name"])

links = urdf_dict["robot"]["link"]
blender_links = {}


def parse_vector3(vector_string: str):
    vector = np.array([float(f) for f in vector_string.split(" ") if f])

    if len(vector) != 3:
        raise ValueError("Vector should have 3 elements")

    return vector


for i, link in enumerate(links):  # noqa: C901
    bpy.ops.object.empty_add(type="ARROWS", radius=0.05)
    empty = bpy.context.object
    empty.name = link["@name"]
    blender_links[link["@name"]] = empty
    empty.lock_rotation = (True, True, True)
    empty.lock_location = (True, True, True)

    if "visual" not in link:
        continue

    visuals = link["visual"]

    # Check if visual is a list of visuals
    if not isinstance(link["visual"], list):
        visuals = [visuals]

    for visual in visuals:
        print("visual", visual)
        if "geometry" not in visual:
            continue

        if "mesh" not in visual["geometry"]:
            continue

        relative_mesh_path = visual["geometry"]["mesh"]["@filename"]
        # mesh_scales = float(link["visual"]["geometry"]["mesh"]["@scale"])
        mesh_path = os.path.join(urdf_dir, relative_mesh_path)

        old_objs = set(bpy.context.scene.objects)

        # if file ends with .dae, use collada importer
        if mesh_path.endswith(".dae"):
            bpy.ops.wm.collada_import(filepath=mesh_path)
        elif mesh_path.endswith(".stl"):
            bpy.ops.import_mesh.stl(filepath=mesh_path)
        elif mesh_path.endswith(".obj"):
            # There axes where chosen for partnet mobility, I hope it works for all URDFs with objs
            bpy.ops.wm.obj_import(filepath=mesh_path, validate_meshes=True, forward_axis="Y", up_axis="Z")
            # bpy.ops.import_scene.obj(filepath=mesh_path, axis_forward="Y", axis_up="Z")
        else:
            print("Warning, currently only import .dae, .stl and .obj mesh files.")

        imported_objs = set(bpy.context.scene.objects) - old_objs
        print(link["@name"], "imported", len(imported_objs), "objects")
        for obj in imported_objs:
            if obj.type in ("CAMERA", "LIGHT"):
                bpy.ops.object.select_all(action="DESELECT")
                obj.select_set(True)
                bpy.ops.object.delete()

            else:
                # I don't like these lines below, but without it, one of the meshes for the robotiq2finger gripper was not
                # oriented correctly. Could also be due to a bad collada file.
                # This basically ignores the transform stored in the mesh file.
                obj.location = parse_vector3(visual["origin"]["@xyz"])
                if "@rpy" in visual["origin"]:
                    obj.rotation_euler = parse_vector3(visual["origin"]["@rpy"])

                # obj.location = 0.0, 0.0, 0.0
                # obj.rotation_euler = 0.0, 0.0, 0.0

                obj.lock_rotation = (True, True, True)
                obj.lock_location = (True, True, True)
                obj.parent = empty

                if "@scale" in visual["geometry"]["mesh"]:
                    mesh_scales = [float(f) for f in visual["geometry"]["mesh"]["@scale"].split(" ") if f]
                    obj.scale = Vector(mesh_scales)
                    bpy.ops.object.select_all(action="DESELECT")
                    obj.select_set(True)
                    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

                # TODO: simplify this
                if "material" in visual:
                    if "color" in visual["material"]:
                        if "@rgba" in visual["material"]["color"]:
                            color = visual["material"]["color"]["@rgba"]
                            color = [float(c) for c in color.split(" ") if c]
                            ab.add_material(obj, color)


joints = urdf_dict["robot"]["joint"]

# links_by_name = {link["@name"]: link for link in links}
joints_by_name = {joint["@name"]: joint for joint in joints}


def parse_axis(axis_string: str):
    axis_vector = np.array([float(f) for f in axis_string.split(" ") if f])

    if len(axis_vector) != 3:
        raise ValueError("Axis vector should have 3 elements")

    # if np.count_nonzero(axis_vector) != 1:
    #     raise NotImplementedError("Currently only axes with one non-zero element are supported.")

    axis_index = np.where(axis_vector != 0)[0][0]
    return axis_vector, axis_index


for j, joint in enumerate(joints):  # noqa: C901

    # if joint["@name"] == "left_inner_finger_pad_joint":
    #     break

    parent = blender_links[joint["parent"]["@link"]]
    child = blender_links[joint["child"]["@link"]]
    child.parent = parent
    origin = joint["origin"]

    if "@xyz" in origin:
        translation = Vector(
            [float(f) for f in origin["@xyz"].split(" ") if f]
        )  # if f because rbo dataset has trailing space
        child.location = translation

    if "@rpy" in origin:
        rotation = Vector([float(f) for f in origin["@rpy"].split(" ") if f])
        child.rotation_euler = rotation

    if joint["@type"] == "revolute":
        axis_vector, axis_index = parse_axis(joint["axis"]["@xyz"])

        if "mimic" not in joint:
            # Unlock the rotation axis of the revolute joint
            child.lock_rotation[axis_index] = False
            child.empty_display_size = 0.1
            print(
                "Free joint: ",
                joint["@name"],
                " movable child link: ",
                joint["child"]["@link"],
                " free axis: ",
                axis_index,
            )
        else:
            mimic_joint = joints_by_name[joint["mimic"]["@joint"]]
            mimic_child = blender_links[mimic_joint["child"]["@link"]]
            multiplier = float(joint["mimic"]["@multiplier"])

            bpy.ops.object.select_all(action="DESELECT")
            child.select_set(True)
            bpy.context.view_layer.objects.active = child
            print("Selecting: ", child.name)
            bpy.ops.object.constraint_add(type="COPY_ROTATION")
            print(child.constraints)
            constraint = child.constraints["Copy Rotation"]
            constraint.target = mimic_child
            constraint.mix_mode = "ADD"

            for i in range(3):
                if i != axis_index:
                    if i == 0:
                        constraint.use_x = False
                    elif i == 1:
                        constraint.use_y = False
                    elif i == 2:
                        constraint.use_z = False

            mimic_axis_vector, mimic_axis_index = parse_axis(mimic_joint["axis"]["@xyz"])

            # Invert the axis of the copy rotation constraint
            invert_axis = np.isclose(-1, multiplier * np.dot(mimic_axis_vector, axis_vector))
            if invert_axis:
                if axis_index == 0:
                    constraint.invert_x = True
                elif axis_index == 1:
                    constraint.invert_y = True
                elif axis_index == 2:
                    constraint.invert_z = True
    if joint["@type"] == "prismatic":
        axis_vector, axis_index = parse_axis(joint["axis"]["@xyz"])

        if np.count_nonzero(axis_vector) > 1:
            print("Non-axis aligned prismatic joint not supported: ", joint["@name"])
            continue

        child.lock_location[axis_index] = False
        child.empty_display_size = 0.1
        print(
            "Free joint: ",
            joint["@name"],
            " movable child link: ",
            joint["child"]["@link"],
            " free axis: ",
            axis_index,
        )


# TODO free location and rotation of the base link (named world for ur5)
