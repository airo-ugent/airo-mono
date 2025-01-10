# from schunk_gripper import SchunkGripper
import time

import rtde_control
import rtde_receive
from schunk_gripper_v2 import SchunkGripper  # integrate more functions

robot_ip = "10.42.0.162"
rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
rtde_c = rtde_control.RTDEControlInterface(robot_ip)
gripper = SchunkGripper(local_port=44875)
gripper.connect()
# rtde_c.servoJ(actual_q, 0.1, 0.1, 1, 0.2, 100) # for test rtdf with interpreter mode
gripper_index = 0
position1 = 10
position = 50
speed = 50
# gripper._getPosition()
gripper.acknowledge(gripper_index)
gripper.connect_server_socket()
# rtde control reader
# actual_q = rtde_r.getActualQ()
# print(actual_q)
# schunk contrl reader
# gripper.moveAbsolute(gripper_index, position1, speed)
# time.sleep(2)
# gripper.moveAbsolute(gripper_index, position, speed)
# time.sleep(1)
while True:
    response = gripper.getPosition()
    time.sleep(1)
    actual_q = rtde_r.getActualQ()
    print(actual_q)
# print(response)
gripper.disconnect()

"""
b'interpreter_mode()\n'
b'socket_open("10.42.0.162", 55050, "rpc_socket")\n'
b'socket_send_line("acknowledge(0)", "rpc_socket")\nsync()\n'
b'socket_close("rpc_socket")\n'
10.42.0.1 47013
b'socket_open("10.42.0.1", 47013, "socket_grasp_sensor")\n'
<socket.socket fd=6, family=AddressFamily.AF_INET, type=SocketKind.SOCK_STREAM, proto=0, laddr=('10.42.0.1', 47013), raddr=('10.42.0.162', 47996)>
b'socket_open("127.0.0.1", 55050, "rpc_socket")\n'
b'socket_send_line("getPosition(0)", "rpc_socket")\n'
b'response=socket_read_line("rpc_socket", 2)\n'
b'popup(response)\n'
b'socket_send_line(response, "socket_grasp_sensor")\n
"""
