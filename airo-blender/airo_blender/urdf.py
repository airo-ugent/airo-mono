"""Very basic Blender URDF importer.

Features:
- Only relative or absolute paths are supported in the URDF files, so not paths with //:package.
- Only imports .dae, .stl and .obj mesh files.
- The imported URDFs are not rigged, so they must be posed in a "forwards kinematics" manner.
- Joint limits are ignored.
- We (will) return a dictionary with info over the free DoFs of the URDF, so that we can use it to generate random poses.

Implementation:
- We use xmltodict to parse the URDF file to a python dictionary.
- Then we import all links as empty objects. The visual meshes are imported and set as children of the link_empties.
- By default we lock all degrees of freedom of the links.
- Joints are then parsed. TODO explain.
"""
import os

import airo_blender as ab
import bpy
import numpy as np
import xmltodict


def parse_vector_string(vector_string: str) -> list[float]:
    """Helper function to parse vector that are given as strings e.g. '1 2 3' -> [1, 2, 3].
    The reason we check for empty strings is that we've encountered URDF files with redundant spaces.

    Args:
        vector_string: The string from the URDF e.g. from attributes like @xyz, @rpy or @rgba.

    Returns:
        The parsed vector.
    """
    return [float(f) for f in vector_string.split(" ") if f]


def read_urdf(urdf_path: str) -> dict:
    """Reads a URDF (xml) file into a Python dictionary.
    Childern elements can be accessed by using the tag name as key e.g. urdf["robot"].
    Attributes can be accessed by using the "@" prefix e.g. link["@name"].

    Args:
        urdf_path: path to the URDF file.

    Returns:
        The URDF as a dictionary.
    """
    with open(urdf_path, "r") as file:
        xml_content = file.read()
    urdf = xmltodict.parse(xml_content)
    return urdf


def set_delta_transform_from_origin(object: bpy.types.Object, origin: dict) -> None:
    """Many URDF elements (e.g. joint, visual) have an origin element that specifies its transform.
    We found that to correctly pose their Blender objects, it was simplest to set their delta transforms.
    However it might be desirable to set the "regular" transforms instead, so we might change this in the future.

    Args:
        object: The Blender object (often an Empty) corresponding to the URDF element.
        origin: The origin element of the URDF element.
    """
    if "@xyz" in origin:
        object.delta_location = parse_vector_string(origin["@xyz"])
    if "@rpy" in origin:
        object.delta_rotation_euler = parse_vector_string(origin["@rpy"])


def import_mesh_from_urdf(mesh_path: str) -> list[bpy.types.Object]:
    """Imports a mesh from a mesh element of a URDF file.
    This comes down to calling the correct Blender import operator based on the file extension.
    We also remove all non-mesh objects (e.g. lights, cameras) that are imported, as they can clutter the scene.

    Args:
        mesh_path: The path to the mesh file.
    """
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


def setup_mesh_geometry(mesh: dict, urdf_dir: str) -> list[bpy.types.Object]:
    """Set up a mesh geometry element of a URDF file.
    This starts with importing the mesh file, and then applying the scale if it is specified.
    Also removes the transform of the imported meshes, as I've found they can cause issues.
    I'm not sure whether this is spec or standard practice, but it seems to work with the URDF files I've tested.

    Args:
        mesh: The mesh element of the URDF.
        urdf_dir: The directory of the URDF file, used to resolve relative paths.

    Returns:
        The imported meshes as Blender objects.
    """
    relative_mesh_path = mesh["@filename"]
    mesh_path = os.path.join(urdf_dir, relative_mesh_path)
    mesh_objects = import_mesh_from_urdf(mesh_path)
    if "@scale" in mesh:
        scales = parse_vector_string(mesh["@scale"])
        bpy.ops.object.select_all(action="DESELECT")
        for object in mesh_objects:
            object.scale = scales
            object.location = (0, 0, 0)
            object.rotation_euler = (0, 0, 0)
            object.select_set(True)
        bpy.ops.object.transform_apply()
    return mesh_objects


def setup_box_geometry(box: dict) -> bpy.types.Object:
    """Set up a box geometry element of link by creating and scaling a Blender cube.

    Args:
        box: The box element of the URDF.

    Returns:
        The Blender object for the box.
    """
    bpy.ops.mesh.primitive_cube_add()
    box_object = bpy.context.object
    scales = parse_vector_string(box["@size"])
    box_object.scale = 0.5 * np.array(scales)  # Blender cubes have size of 2 by default.

    bpy.ops.object.select_all(action="DESELECT")
    box_object.select_set(True)
    bpy.ops.object.transform_apply()
    return box_object


def add_urdf_material_to_geometries(material: dict, geometry_objects: list[bpy.types.Object]):
    """Adds a material to the geometry objects of a URDF link.

    Args:
        material: The material element of the link.
        geometry_objects: The Blender objects of the geometry of the link.
    """
    if "color" not in material:
        return
    if "@rgba" not in material["color"]:
        return

    rgba_color = material["color"]["@rgba"]
    rgba_color = parse_vector_string(rgba_color)

    for object in geometry_objects:
        ab.add_material(object, rgba_color)


def import_visual(visual: dict, parent: bpy.types.Object, urdf_dir: str) -> None:
    """Imports a visual element of a URDF link.

    Args:
        visual: The visual element of the link.
        parent: The Empty object of the link, all objects created for the visual will be parented to it.
        urdf_dir: The directory of the URDF file, used to resolve relative paths.
    """
    geometry = visual["geometry"]  # geometry is a required element of visual
    geometry_objects = []

    # Create the objects specified by the geometry element in this visual
    if "mesh" in geometry:
        mesh_geometry_objects = setup_mesh_geometry(geometry["mesh"], urdf_dir)
        geometry_objects.extend(mesh_geometry_objects)
    if "box" in geometry:
        box_object = setup_box_geometry(geometry["box"])
        geometry_objects.append(box_object)
    # TODO add support for sphere and cylinder geometries.

    # Additional processing for all geometry objects in this visual
    for object in geometry_objects:
        if "origin" in visual:
            set_delta_transform_from_origin(object, visual["origin"])
        object.parent = parent
        # We lock the transform of all geometry objects as the should be used through the Empty of their link.
        # This makes is less likely for users to accidentally mess up their robot models in Blender.
        object.lock_location = (True, True, True)
        object.lock_rotation = (True, True, True)

    # Add the material to all geometry objects in this visual
    if "material" in visual:
        material = visual["material"]
        add_urdf_material_to_geometries(material, geometry_objects)


def import_link(link: dict, urdf_dir: str) -> bpy.types.Object:
    """Imports a link element of a URDF file.
    This creates an Empty object for the link, and then imports the link's visuals as the Empty's childern.

    Args:
        link: The link element of the URDF.
        urdf_dir: The directory of the URDF file, used to resolve relative paths.
    """
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
    """Creates an Empty object with the given name and locks its transform.
     Note that Blender might add a number suffix if the name is already taken.

    Args:
        name: The base name of the Empty object.
    """
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
        axis: The axis to align the Z-axis to. Must be a normalized 3D vector.

    Returns:
        The transform as a 4x4 homogeneous matrix.
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
    """URDF joints can have an axis that is not aligned with the principal axes of the joint. However to easily lock
    the joint in Blender this has to be the case. Our solution is to insert a "joint axis" Empty object between the
    joint and its child link, and align the Empty's Z-axis to the joint's axis. To unlock DoFs, e.g. for a revolute
    joints, we can simply unlock the empty's rotation_euler.z

    Args:
        joint: The Blender object (an Empty) of the joint.
        child: The Blender object (an Empty) of the child link.
        axis: The axis of the joint

    Returns:
        The created intermediate Empty object for the joint axis.
    """
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


def insert_joint(joint: dict, link_empties: dict[str, bpy.types.Object]) -> bpy.types.Object:
    """Creates an Empty object for a joint and set its parent and child link through Blender parenting.

    Args:
        joint: The joint element of the URDF.
        link_empties: A dictionary mapping link names to their corresponding Empty objects.
    """
    joint_empty = create_locked_empty(joint["@name"])
    parent = link_empties[joint["parent"]["@link"]]
    child = link_empties[joint["child"]["@link"]]

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
    """A Blender driver is a mechanism to automatically update a property of an object based on the value of another.
    This function sets a driver up between the z of the location or rotation_euler of the driver and the mimicking
    object.
    """
    mimic_property = "rotation_euler" if mimic_revolute else "location"
    driver_property = "rotation_euler" if driver_revolute else "location"
    driver = mimic_joint_empty.driver_add(mimic_property, 2).driver
    variable = driver.variables.new()
    variable.name = "var"
    variable.targets[0].id = driver_joint_empty
    variable.targets[0].data_path = f"{driver_property}.z"
    driver.expression = f"{multiplier} * {variable.name} + {offset}"


def configure_axis_joint(joint: dict, joint_empty: bpy.types.Object, child: bpy.types.Object) -> bpy.types.Object:
    """First inserts an intermediate Empty for the joint axis. Then unlocks the correct degrees of freedom of the joint
    axis empty e.g. the z rotation for revolute joints.

    Args:
        joint: The joint element of the URDF.
        joint_empty: The Blender object (an Empty) of the joint.
        child: The Blender object (an Empty) of the child link of the joint.

    Returns:
        The created intermediate Empty object for the joint axis.
    """
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


def configure_mimic_joint(joint: dict, joint_empties: dict[str, bpy.types.Object]) -> None:
    """Sets up a mimic URDF joint. This is a joint that copies the value of another joint.

    Args:
        joint: The joint element of the URDF.
        joint_empties: A dictionary mapping joint names to their corresponding Empty objects.
    """
    joint_name = joint["@name"]
    mimic_joint_empty = joint_empties[joint_name]
    if joint_name + "_axis" in joint_empties:
        mimic_joint_empty = joint_empties[joint_name + "_axis"]

    mimic_joint_empty.empty_display_size = 0.1

    mimic = joint["mimic"]
    driver_joint_name = mimic["@joint"]
    driver_joint_empty = joint_empties[driver_joint_name]
    if driver_joint_name + "_axis" in joint_empties:
        driver_joint_empty = joint_empties[driver_joint_name + "_axis"]

    multiplier = 1.0 if "@multiplier" not in mimic else float(mimic["@multiplier"])
    offset = 0.0 if "@offset" not in mimic else float(mimic["@offset"])
    add_mimic_driver(driver_joint_empty, mimic_joint_empty, multiplier, offset)


def setup_joint_types(
    joints: dict, joint_empties: dict[str, bpy.types.Object], link_empties: dict[str, bpy.types.Object]
):
    """Sets up the different joint types of the URDF.

    Args:
        joints: The joint elements of the URDF.
        joint_empties: A dictionary mapping joint names to their corresponding Empty objects.
        link_empties: A dictionary mapping link names to their corresponding Empty objects.

    Returns:
        A dictionary mapping names of the free joints to their corresponding Empty objects.
    """
    joint_types_that_use_axis = ("revolute", "prismatic", "planar", "continuous")

    free_joint_empties = {}
    for joint in joints:
        joint_empty = joint_empties[joint["@name"]]
        if joint["@type"] in joint_types_that_use_axis:
            child = link_empties[joint["child"]["@link"]]
            joint_axis_empty = configure_axis_joint(joint, joint_empty, child)
            joint_axis_name = joint["@name"] + "_axis"
            joint_empties[joint_axis_name] = joint_axis_empty
            if "mimic" not in joint:
                free_joint_empties[joint["@name"]] = joint_axis_empty
        elif joint["@type"] == "floating":
            joint_empty.lock_rotation = False, False, False
            joint_empty.lock_location = False, False, False
            if "mimic" not in joint:  # I'm not even 100% sure floating mimic joints are allowed/used.
                free_joint_empties[joint["@name"]] = joint_empty
    return free_joint_empties


def unlock_root_links(link_empties: dict[str, bpy.types.Object]) -> None:
    """Unlocks the location and rotation of the root links.

    Args:
        link_empties: A dictionary mapping link names to their corresponding Empty objects.
    """
    for link in link_empties.values():
        if link.parent is None:
            link.lock_location = [False, False, False]
            link.lock_rotation = [False, False, False]


def import_urdf(urdf_path: str) -> tuple[dict[str, bpy.types.Object]]:
    """Imports a URDF file into Blender.

    Args:
        urdf_path: The path to the URDF file.

    Returns:
        Three dictionaries that map joint/link names to their corresponding Blender Empty objects:
            free_joint_empties: The joints with free DoFs. The values of these joints can be set to pose the robot.
            joint_empties: All joints. Some of these joints can be fixed.
            link_empties: All links.
    """
    urdf = read_urdf(urdf_path)
    urdf_dir = os.path.dirname(urdf_path)

    # Importing the loose links and their visuals.
    links = urdf["robot"]["link"]
    link_empties = {}
    for link in links:
        link_empty = import_link(link, urdf_dir)
        link_empties[link["@name"]] = link_empty

    # Inserting joint empties between links through parenting.
    joints = urdf["robot"]["joint"]
    joint_empties = {}
    for joint in joints:
        joint_empty = insert_joint(joint, link_empties)
        joint_empties[joint["@name"]] = joint_empty

    # Setting up the joint types, and meanwhile we can also find the free joints.
    free_joint_empties = setup_joint_types(joints, joint_empties, link_empties)

    # Configuring the mimic joints
    for joint in joints:
        if "mimic" in joint:
            configure_mimic_joint(joint, joint_empties)

    # Unlocking the root links
    unlock_root_links(link_empties)

    return free_joint_empties, joint_empties, link_empties


if __name__ == "__main__":
    bpy.ops.object.delete()  # Delete the default cube

    # Universal robots
    urdf_path = "/home/idlab185/urdfpy/tests/data/ur5/ur5.urdf"
    # urdf_path = "/home/idlab185/urdf-workshop/universal_robots/ros/ur10e/ur10e.urdf"

    # Robotiq 2F-85 gripper
    # urdf_path = "/home/idlab185/robotiq_arg85_description/robots/robotiq_arg85_description.URDF"
    # urdf_path = "/home/idlab185/robotiq_2finger_grippers/robotiq_2f_85_gripper_visualization/urdf/robotiq2f85.urdf"

    # PartNet mobility samples
    # urdf_path = "/home/idlab185/partnet-mobility-sample/44853/mobility.urdf"  # cabinet
    # urdf_path = "/home/idlab185/partnet-mobility-sample/7128/mobility.urdf"
    # urdf_path = "/home/idlab185/partnet-mobility-sample/7130/mobility.urdf"
    # urdf_path = "/home/idlab185/partnet-mobility-sample/103452/mobility.urdf"  # washing machine

    free_joint_empties, joint_empties, link_empties = import_urdf(urdf_path)
    urdf = read_urdf(urdf_path)
    joints_by_name = {joint["@name"]: joint for joint in urdf["robot"]["joint"]}

    print(f"Imported: {urdf_path} with {len(free_joint_empties)} free joints:")
    for joint_name, joint_empty in free_joint_empties.items():
        # You use the URDF dictionary to look up additional info about the joint with its name.
        joint = joints_by_name[joint_name]
        joint_type = joint["@type"]
        if joint_type != "continuous":
            # TODO: Handle case where there is no limit (it is only required for prismatic and revolute).
            # TODO: Handle case where lower or upper are not specified, default to 0 then.
            joint_lower = joint["limit"]["@lower"]
            joint_upper = joint["limit"]["@upper"]
        else:
            joint_lower = "-inf"
            joint_upper = "inf"
        print(f"- {joint_name} is of type {joint_type} with range ({joint_lower}, {joint_upper})")
    root_links = [link for link in link_empties.values() if link.parent is None]
    print(f"The root links of the model are: {root_links}")

    # Make the scene a bit prettier
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs["Color"].default_value = (1.0, 0.9, 0.7, 1.0)
    bpy.context.scene.render.engine = "CYCLES"
