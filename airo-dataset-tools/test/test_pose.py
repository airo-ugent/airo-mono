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
        json.dump(pose.dict(exclude_none=True), file, indent=4)

    pose2 = Pose.parse_file(tmp_path / "pose.json")
    assert pose2.position_in_meters.x == 1.0
    assert pose2.position_in_meters.y == 2.0
    assert pose2.position_in_meters.z == 3.0
    assert pose2.rotation_euler_xyz_in_radians.roll == np.pi / 4
    assert pose2.rotation_euler_xyz_in_radians.pitch == -np.pi / 2
    assert pose2.rotation_euler_xyz_in_radians.yaw == np.pi


if __name__ == "__main__":
    test_pose_save_and_load(pathlib.Path("."))
