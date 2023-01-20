"""Very basic Blender URDF importer.

Features:
- Only relative or absolute paths are supported in the URDF files, so not paths with //:package.
- Only imports .dae, .stl and .obj mesh files.
- The imported URDFs are not rigged, so they must be posed in a "forwards kinematics" manner.
- Joint limits are ignored.
- We (will) return a dictionary with info over the free DoFs of the URDF, so that we can use it to generate random poses.

Implementation:
- We use xmltodict to parse the URDF file to a python dictionary.
  Child xml elements can be accessed from the dictionary by using the tag as key e.g. urdf_dict["robot"].
  Attributes can be accessed by using the "@" prefix e.g. link["@name"].
- Then we import all links as empty objects. The visual meshes are imported and set as children of the link_empties_by_name.
- By default we lock all degrees of freedom of the links.
- Joints are then parsed. TODO explain.
"""
import os

import airo_blender as ab
import bpy
import numpy as np
import xmltodict


def parse_vector_string(vector_string: str) -> list[float]:
    return [float(f) for f in vector_string.split(" ") if f]


def read_urdf_as_dictionary(urdf_path: str) -> dict:
    file = open(urdf_path, "r")
    xml_content = file.read()
    urdf_dict = xmltodict.parse(xml_content)
    return urdf_dict


def set_delta_transform_from_origin(object: bpy.types.Object, origin: dict) -> None:
    if "@xyz" in origin:
        object.delta_location = parse_vector_string(origin["@xyz"])
    if "@rpy" in origin:
        object.delta_rotation_euler = parse_vector_string(origin["@rpy"])


def import_mesh_from_urdf(mesh_path: str) -> list[bpy.types.Object]:
    objects_before_import = set(bpy.context.scene.objects)

    mesh_file_extenstion = os.path.splitext(mesh_path)[1].lower()
    if mesh_file_extenstion == ".dae":
        bpy.ops.wm.collada_import(filepath=mesh_path)
    elif mesh_file_extenstion == ".stl":
        bpy.ops.import_mesh.stl(filepath=mesh_path)
    elif mesh_file_extenstion == ".obj":
        # These axes where chosen for PartNet-Mobility, I hope it works for all URDFs with objs.
        bpy.ops.wm.obj_import(filepath=mesh_path, validate_meshes=True, forward_axis="Y", up_axis="Z")
    else:
        print(f"Ignoring mesh with extension {mesh_file_extenstion}. Not supported yet. Mesh path = {mesh_path}")

    objects_after_import = set(bpy.context.scene.objects)
    imported_objects = objects_after_import - objects_before_import

    geometry_objects = set([object for object in imported_objects if object.type == "MESH"])
    non_geometry_objects = list(imported_objects - geometry_objects)

    for object in non_geometry_objects:
        bpy.data.objects.remove(object, do_unlink=True)

    return list(geometry_objects)


def add_urdf_material_to_geometries(material: dict, geometry_objects: list[bpy.types.Object]):
    if "color" not in material:
        return
    if "@rgba" not in material["color"]:
        return

    rgba_color = material["color"]["@rgba"]
    rgba_color = parse_vector_string(rgba_color)

    for object in geometry_objects:
        ab.add_material(object, rgba_color)


def import_visual(visual: dict, parent: bpy.types.Object, urdf_dir: str) -> None:
    geometry = visual["geometry"]  # geometry is a required element of visual
    geometry_objects = []

    # Create the objects specified by the geometry element in this visual
    if "mesh" in geometry:
        relative_mesh_path = geometry["mesh"]["@filename"]
        mesh_path = os.path.join(urdf_dir, relative_mesh_path)
        geometry_objects = import_mesh_from_urdf(mesh_path)
        if "@scale" in geometry["mesh"]:
            scales = parse_vector_string(geometry["mesh"]["@scale"])
            bpy.ops.object.select_all(action="DESELECT")
            for object in geometry_objects:
                object.scale = scales
                object.location = (0, 0, 0)
                object.rotation_euler = (0, 0, 0)
                object.select_set(True)
            bpy.ops.object.transform_apply()

    if "box" in geometry:
        bpy.ops.mesh.primitive_cube_add()
        box = bpy.context.object
        scales = parse_vector_string(geometry["box"]["@size"])
        box.scale = 0.5 * np.array(scales)  # Blender cube have size of 2 by default.

        bpy.ops.object.select_all(action="DESELECT")
        box.select_set(True)
        bpy.ops.object.transform_apply()
        geometry_objects.append(box)

    # Additional processing for all geometry objects in this visual
    for object in geometry_objects:
        if "origin" in visual:
            set_delta_transform_from_origin(object, visual["origin"])

        object.lock_location = (True, True, True)
        object.lock_rotation = (True, True, True)
        object.parent = parent

    if "material" in visual:
        material = visual["material"]
        add_urdf_material_to_geometries(material, geometry_objects)


def import_link(link: dict, urdf_dir: str) -> bpy.types.Object:
    empty = create_locked_empty(link["@name"])
    if "visual" not in link:
        return empty

    # One link can contain multiple visuals.
    visuals = link["visual"]
    visuals = visuals if isinstance(visuals, list) else [visuals]
    for visual in visuals:
        import_visual(visual, empty, urdf_dir)
    return empty


def create_locked_empty(name: str) -> bpy.types.Object:
    bpy.ops.object.empty_add(type="ARROWS", radius=0.05)
    empty = bpy.context.object
    empty.name = name
    empty.lock_rotation = (True, True, True)
    empty.lock_location = (True, True, True)
    return empty


def make_transform_with_z_aligned_to_axis(axis: list) -> np.ndarray:
    """Creates a transform matrix that has its Z-axis aligned to the given axis.
    The X and Y axes are arbitrary, but orthogonal to the Z-axis and normalized.

    Args:
        axis (list): The axis to align the Z-axis to. Must be a normalized 3D vector.

    Returns:
        np.ndarray: a 4x4 homogeneous transform matrix.
    """
    Z = np.array(axis)

    # Create abitrary X and Y axes that are orthogonal to Z.
    temp = np.array([1, 0, 0])
    if np.abs(np.dot(temp, Z)) > 0.999:
        temp = np.array([0, 1, 0])
    X = np.cross(Z, temp)
    X = X / np.linalg.norm(X)
    Y = np.cross(Z, X)

    orientation = np.column_stack([X, Y, Z])
    transform = np.identity(4)
    transform[:3, :3] = orientation
    return transform


def insert_joint_axis_empty(joint: bpy.types.Object, child: bpy.types.Object, axis: list) -> bpy.types.Object:
    joint_axis_empty = create_locked_empty(f"{joint.name}_axis")
    joint_axis_transform = make_transform_with_z_aligned_to_axis(axis)

    joint_axis_empty.parent = joint
    joint_axis_empty.matrix_parent_inverse = np.linalg.inv(joint_axis_transform)

    # Parent the child to the joint axis empty, but keep the transform from the previous parenting
    bpy.ops.object.select_all(action="DESELECT")
    child.select_set(True)
    joint_axis_empty.select_set(True)
    bpy.context.view_layer.objects.active = joint_axis_empty
    bpy.ops.object.parent_no_inverse_set(keep_transform=True)
    return joint_axis_empty


def insert_joint(joint: dict, link_empties_by_name: dict) -> bpy.types.Object:
    joint_empty = create_locked_empty(joint["@name"])
    parent = link_empties_by_name[joint["parent"]["@link"]]
    child = link_empties_by_name[joint["child"]["@link"]]

    if "origin" in joint:
        set_delta_transform_from_origin(joint_empty, joint["origin"])

    joint_empty.parent = parent
    child.parent = joint_empty
    return joint_empty


def add_mimic_driver(
    driver_joint_empty: bpy.types.Object,
    mimic_joint_empty: bpy.types.Object,
    multiplier: float,
    offset: float,
    mimic_revolute: bool = True,
    driver_revolute: bool = True,
) -> None:
    mimic_property = "rotation_euler" if mimic_revolute else "location"
    driver_property = "rotation_euler" if driver_revolute else "location"
    driver = mimic_joint_empty.driver_add(mimic_property, 2).driver
    variable = driver.variables.new()
    variable.name = "var"
    variable.targets[0].id = driver_joint_empty
    variable.targets[0].data_path = f"{driver_property}.z"
    driver.expression = f"{multiplier} * {variable.name} + {offset}"


def configure_axis_joint(joint: dict, joint_empty: bpy.types.Object, child: bpy.types.object) -> bpy.types.Object:
    axis = [1.0, 0.0, 0.0]  # Default axis from spec
    if "axis" in joint:
        axis = parse_vector_string(joint["axis"]["@xyz"])
    joint_axis_empty = insert_joint_axis_empty(joint_empty, child, axis)
    # Make the joint axis more visible
    joint_axis_empty.empty_display_type = "SINGLE_ARROW"
    joint_axis_empty.empty_display_size = 0.30

    # Free up DoFs of the joint axis.
    if joint["@type"] in ["revolute", "continuous"]:
        joint_axis_empty.lock_rotation[2] = False
    elif joint["@type"] == "prismatic":
        joint_axis_empty.lock_location[2] = False
    elif joint["@type"] == "planar":  # UNTESTED
        joint_axis_empty.lock_location[0] = False
        joint_axis_empty.lock_location[1] = False
    return joint_axis_empty


def configure_mimic_joint(joint: dict, joint_empties_by_name: dict) -> None:
    joint_name = joint["@name"]
    mimic_joint_empty = joint_empties_by_name[joint_name]
    if joint_name + "_axis" in joint_empties_by_name:
        mimic_joint_empty = joint_empties_by_name[joint_name + "_axis"]

    mimic_joint_empty.empty_display_size = 0.1

    mimic = joint["mimic"]
    driver_joint_name = mimic["@joint"]
    driver_joint_empty = joint_empties_by_name[driver_joint_name]
    if driver_joint_name + "_axis" in joint_empties_by_name:
        driver_joint_empty = joint_empties_by_name[driver_joint_name + "_axis"]

    multiplier = 1.0 if "@multiplier" not in mimic else float(mimic["@multiplier"])
    offset = 0.0 if "@offset" not in mimic else float(mimic["@offset"])
    add_mimic_driver(driver_joint_empty, mimic_joint_empty, multiplier, offset)


def import_urdf(urdf_path: str) -> list[bpy.types.Object]:
    urdf_dict = read_urdf_as_dictionary(urdf_path)
    urdf_dir = os.path.dirname(urdf_path)

    # Importing the loose links and their visuals.
    links = urdf_dict["robot"]["link"]
    link_empties_by_name = {}
    for link in links:
        link_empty = import_link(link, urdf_dir)
        link_empties_by_name[link["@name"]] = link_empty

    # Inserting joint empties between links through parenting.
    joints = urdf_dict["robot"]["joint"]
    joint_empties_by_name = {}
    for joint in joints:
        joint_empty = insert_joint(joint, link_empties_by_name)
        joint_empties_by_name[joint["@name"]] = joint_empty

    # Setting up the joint types
    for joint in joints:
        joint_empty = joint_empties_by_name[joint["@name"]]
        joint_types_that_use_axis = ["revolute", "prismatic", "planar", "continuous"]
        if joint["@type"] in joint_types_that_use_axis:
            child = link_empties_by_name[joint["child"]["@link"]]
            joint_axis_empty = configure_axis_joint(joint, joint_empty, child)
            joint_empties_by_name[joint["@name"] + "_axis"] = joint_axis_empty
        elif joint["@type"] == "floating":
            joint_empty.lock_rotation = False, False, False
            joint_empty.lock_location = False, False, False

    # Configuring the mimic joints
    for joint in joints:
        if "mimic" in joint:
            configure_mimic_joint(joint, joint_empties_by_name)

    # Unlocking the root links
    child_links = [joint["child"]["@link"] for joint in joints]
    root_links = [link["@name"] for link in links if link["@name"] not in child_links]
    blender_root_links = [link_empties_by_name[root_link] for root_link in root_links]
    for root_link in blender_root_links:
        root_link.lock_location = False, False, False
        root_link.lock_rotation = False, False, False

    return blender_root_links


if __name__ == "__main__":
    bpy.ops.object.delete()  # Delete the default cube

    # Universal robots
    # urdf_path = "/home/idlab185/urdfpy/tests/data/ur5/ur5.urdf"
    # urdf_path = "/home/idlab185/urdf-workshop/universal_robots/ros/ur10e/ur10e.urdf"

    # Robotiq 2F-85 gripper
    # urdf_path = "/home/idlab185/robotiq_arg85_description/robots/robotiq_arg85_description.URDF"
    # urdf_path = "/home/idlab185/robotiq_2finger_grippers/robotiq_2f_85_gripper_visualization/urdf/robotiq2f85.urdf"

    # PartNet mobility samples
    # urdf_path = "/home/idlab185/partnet-mobility-sample/44853/mobility.urdf"  # cabinet
    # urdf_path = "/home/idlab185/partnet-mobility-sample/7128/mobility.urdf"
    # urdf_path = "/home/idlab185/partnet-mobility-sample/7130/mobility.urdf"
    urdf_path = "/home/idlab185/partnet-mobility-sample/103452/mobility.urdf"  # washing machine

    import_urdf(urdf_path)

    # Make the scene a bit prettier
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs["Color"].default_value = (1.0, 0.9, 0.7, 1.0)
    bpy.context.scene.render.engine = "CYCLES"
