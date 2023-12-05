# Data parsers
We call our Pydantic Model classes *data parsers* because we use them to define, load and store data in a specific format.
For documentation of the COCO data parsers, see [here](../coco_tools/README.md).
The other data parsers we provide are:
* [Pose](../../docs/pose.md)
* [CameraIntrinsics](../../docs/camera_intrinsics.md)


## Example usage

:floppy_disk: **Creating a Pydantic Model instance and saving it to json:**
```python
from airo_dataset_tools.data_parsers.pose import EulerAngles, Pose, Position

pose = Pose(
    position_in_meters=Position(x=1.0, y=2.0, z=3.0),
    rotation_euler_xyz_in_radians=EulerAngles(roll=np.pi / 4, pitch=-np.pi / 2, yaw=np.pi),
)

with open("pose.json", "w") as file:
    json.dump(pose.model_dump(exclude_none=True), file, indent=4)
```

:mag_right: **Loading a Pydantic Model instance from json:**
```python
with open("pose.json", "r") as file:
    pose2 = Pose.model_validate_json(file.read())

x = pose2.position_in_meters.x
print(x)
```