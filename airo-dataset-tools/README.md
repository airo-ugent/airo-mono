# airo-dataset-tools

Package for creating, saving and loading datasets

This functionality is mainly provided in the form of [Pydantic](https://docs.pydantic.dev/) parsers.

[COCO](https://cocodataset.org/#format-data) is the preferred format for computer vision datasets. Other formats will be added if they are needed for dataset creation (think the format of a labeling tool) or for consumption of the dataset (think the YOLO format for training an object detector).

Besides datasets, we also plan to provide parsers for other persistent data such as camera intrinsics and extrinsics.

## Pose format

Our main use case for storing poses, is for camera extrinsics or other 6D object poses.
We value the following properties in a format:

- **human friendly**: readable, easily interpretable and editable
- can be understood as **little context** as possible
- **hard to makes mistakes** when filling in the values
- use of **SI units** (meters, radians, etc.)

For this reason we decided to explicitly name all scalar fields (this way you always specify the correct amount).
We also chose Euler angles for the rotation, because it is minimal (3 parameters), and gimbal lock is not an issue for storing poses.
The json output of our Pydantic parser looks like this:

```json
{
  "position_in_meters": {
    "x": 0.0,
    "y": 0.0,
    "z": 0.15
  },
  "rotation_euler_XYZ_in_radians": {
    "roll": 0.7853981633974483,
    "pitch": -1.5707963267948966,
    "yaw": 3.141592653589793
  }
}
```

Other formats that were considered, but we did not adopt:

- ROS [geometry_msgs/Pose](https://docs.ros2.org/latest/api/geometry_msgs/index-msg.html):
  uses quaternions which are hard to interpret
- ROS [URDF <element xyz= rpy= />](http://wiki.ros.org/urdf/XML/link):
  close to ours but XML and too terse
- BOP [cam_R_m2c and cam_t_m2c](https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md#ground-truth-annotations):
  uses millimeter as units and stores entire rotation matrices row-wise

### Coordinate system conventions :world_map:

The format above could be used with any coordinate system.
At AIRO we use right-handed coordinate systems exclusively.
The **Z-axis** generally means the **"up"** direction, _except for cameras._
The X-axis convention is more object-specific, and will be explained below.
The Y-axis should be used to complete the right-handed coordinate system (e.g. through `np.cross(Z, X)`).

> :information_source: See the docstring of the `Camera` class in the `airo-camera-toolkit` package for the convention used for cameras.
> In brief: Z points towards the scene, X points to the right, Y points down.

#### World :earth_africa:

In our "world" coordinates systems, increasing Z means going towards the sky.

#### Objects, e.g. chairs, mugs, etc. :coffee:

For objects there's a lot of room for interpretation.
In doubt, imagine placing an object on a table "in the most natural way".
Choose the Z-axis to point up, and if possible the X-axis to along another "back to front" or forward direction.
Choose object origins to lie at the center of the base of the object (this is used in 3D tools to allow drag-and-drop objects into scenes).
Note that origins may lie outside the model, e.g. for a chair it could lie between the 4 legs.

#### Robot grippers :fist:

For **robot grippers**, we define the Z direction to point in the main direction the gripper is pointing, i.e. the direction along which you would approach an object to be grasped.
For parallel grippers, the X-axis is defined as the direction along which the gripper opens and closes.

#### UR robot frames :robot:

Frames as defined by de UR controller (these are not the same as [the conventions in ROS](https://gavanderhoorn.github.io/rep/rep-0199.html)):

- **base**: Z up, Y is towards the cable connected to the base. Origin at the center of the base.
- **TCP** set to `(0, 0, 0)`: Z forward and -Y points towards the connector on the flange. Origin at the center of the flange ring (TODO check: not in the indent).
