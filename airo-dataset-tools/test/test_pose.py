import json
import pathlib

import numpy as np
from airo_dataset_tools.data_parsers.pose import EulerAngles, Pose, Position


def test_pose_save_and_load(tmp_path: pathlib.Path):
    pose = Pose(
        position_in_meters=Position(x=1.0, y=2.0, z=3.0),
        rotation_euler_xyz_in_radians=EulerAngles(roll=np.pi / 4, pitch=-np.pi / 2, yaw=np.pi),
    )

    with open(tmp_path / "pose.json", "w") as file:
        json.dump(pose.model_dump(exclude_none=True), file, indent=4)

    with open(tmp_path / "pose.json", "r") as file:
        pose2 = Pose.model_validate_json(file.read())

    assert pose2.position_in_meters.x == 1.0
    assert pose2.position_in_meters.y == 2.0
    assert pose2.position_in_meters.z == 3.0
    assert pose2.rotation_euler_xyz_in_radians.roll == np.pi / 4
    assert pose2.rotation_euler_xyz_in_radians.pitch == -np.pi / 2
    assert pose2.rotation_euler_xyz_in_radians.yaw == np.pi


def test_homogeneous_coversion():
    pose = Pose(
        position_in_meters=Position(x=1.0, y=2.0, z=3.0),
        rotation_euler_xyz_in_radians=EulerAngles(roll=np.pi / 4, pitch=-np.pi / 3, yaw=np.pi / 6),  # no gimbal lock
    )

    pose_matrix = pose.as_homogeneous_matrix()
    pose2 = Pose.from_homogeneous_matrix(pose_matrix)

    # isclose is used because exact comparison failed due to tiny floating point rounding errors
    assert np.isclose(pose2.position_in_meters.x, pose.position_in_meters.x)
    assert np.isclose(pose2.position_in_meters.y, pose.position_in_meters.y)
    assert np.isclose(pose2.position_in_meters.z, pose.position_in_meters.z)
    assert np.isclose(pose2.rotation_euler_xyz_in_radians.roll, pose.rotation_euler_xyz_in_radians.roll)
    assert np.isclose(pose2.rotation_euler_xyz_in_radians.pitch, pose.rotation_euler_xyz_in_radians.pitch)
    assert np.isclose(pose2.rotation_euler_xyz_in_radians.yaw, pose.rotation_euler_xyz_in_radians.yaw)


if __name__ == "__main__":
    test_pose_save_and_load(pathlib.Path("."))
    test_homogeneous_coversion()
