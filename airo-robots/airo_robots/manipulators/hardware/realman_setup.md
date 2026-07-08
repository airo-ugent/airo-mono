# Realman Robot arm setup

## Hardware setup
connect both the power supply and safety button to the robot.  See https://develop.realman-robotics.com/en/robot4th/quickUseManual/
You can use the green button on the robot to freedrive it.

## Network
Connect the robot to the workstation using an ethernet cable. Make sure to configure the network connection appropriately.

The default IP address of the robot arm is `192.168.1.18`.
## robot configuration

Connect to the robot controller, which exposes a UI on port . The default username and password are `user` and `123` by default.

### TCP
You can set the TCP manually, including the weight and CoM of the end-effector.

### protective stop / collision safety

You can set the force level of the safety system, see https://develop.realman-robotics.com/en/robot4th/teachingPendantfour/setting/#collision-protection-level

Note that this is different from the certified system in a UR robot.

## Testing
To test that everything works, try running the [realman.py](realman.py) script.
If the robot has a gripper, also set the TCP to something reasonable to avoid collisions.
```bash
python realman.py --ip_address 192.168.1.18
```
