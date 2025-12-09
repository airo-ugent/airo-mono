import time

import numpy as np
import ur_rtde
from airo_robots.grippers import Robotiq2F85

robot = ur_rtde.URrtde(
    "10.42.0.162", ur_rtde.URrtde.UR3E_CONFIG, None, True, np.array([-1.57, -1.57, -1.57, -3.14, -2.07, 3.157])
)
gripper = Robotiq2F85("10.42.0.162")
robot.gripper = gripper
try:
    robot.enable_torque_control()

    while True:

        robot.target_pos = np.array([-1.57, -1.57, -1.57, -3.14, -2.07, 3.157])
        time.sleep(0.5)
        # print(robot.gripper.get_current_width())
        robot.target_pos = np.array([-1.57, -1.57, -1.57, -3.14, -1.87, 3.157])
        time.sleep(0.5)
        print(robot.get_cached_tcp_pose())
except KeyboardInterrupt:
    pass
finally:

    try:
        robot.disable_torque_control()
    except AttributeError:
        pass
