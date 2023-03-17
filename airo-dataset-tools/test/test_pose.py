import json

import numpy as np
from airo_dataset_tools.pose import EulerAngles, Pose, Position


def test_pose_simple():
    pose = Pose(
        position_in_meters=Position(x=1.0, y=2.0, z=3.0),
        rotation_euler_XYZ_in_radians=EulerAngles(roll=np.pi / 4, pitch=-np.pi / 2, yaw=np.pi),
    )

    with open("pose.json", "w") as file:
        json.dump(pose.dict(exclude_none=True), file, indent=4)


if __name__ == "__main__":
    test_pose_simple()
