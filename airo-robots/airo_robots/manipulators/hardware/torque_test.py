import time

import ur_rtde
from airo_robots.manipulators import URrtde

robot = ur_rtde.URrtde("10.42.0.162", URrtde.UR3E_CONFIG, True, None, [-1.57, -1.57, -1.57, -3.14, -2.07, 3.157])
try:
    robot.enable_torque_control()

    while True:

        robot.target_pos = [-1.57, -1.57, -1.57, -3.14, -2.07, 3.157]
        time.sleep(0.5)
        robot.target_pos = [-1.57, -1.57, -1.57, -3.14, -1.87, 3.157]
        time.sleep(0.5)
        print(robot.get_cached_tcp_pose())
except KeyboardInterrupt:
    pass
finally:

    try:
        robot.disable_torque_control()
    except AttributeError:
        pass
