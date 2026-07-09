# Realman Robot arm setup
This Readme contains some basic information about the realman arm, for more information, see https://develop.realman-robotics.com/en/robot4th/summarize/
## Hardware setup
connect both the power supply and safety button to the robot.  See https://develop.realman-robotics.com/en/robot4th/quickUseManual/
You can use the green button on the robot to freedrive it.

## Network
Connect the robot to the workstation using an ethernet cable. Make sure to configure the network connection appropriately:

- On ubuntu, the easy option is to create a new network profile, give it any name e.g. `realman` and in the IPv4 tab select `Shared to other computers`.

- On MacOS (or ubuntu):  create a network profile with IPv4 configured manually at 192.168.1.200 (subnet mask 255.255.255.0) and configure IPv4 automatically.

The default IP address of the robot arm is `192.168.1.18`. Make sure you can ping it before continuing.

## robot configuration

Connect to the robot controller in the browser: [http://192.168.1.18/](http://192.168.1.18/) . The default username and password are `user` and `123`.

### Robot Type

to see the exact type of the robot arm, go to `> Configuration > Robotic Arm Configuration > Version Information`. Note that there are **2** versions of the RM-75-6F (F/T variant), when looking up D-H params or using a URDF, make sure to double check this.

### End-effector Configuration
You should set the end-effector properties, including the TCP frame, weight and CoM of the end-effector. This is configured in `> Configuration > Robotic Arm Configuration > Tool Calibration`

If you do not know the TCP frame wrt to the robot `tool0` / `flange` frame, an easy option is to move to the robot vertically down till it hits a surface with known z-distance from the robot base frame. If that is not possible, a more elabore six-point touch calibration procedure is available in the UI.

The same goes for the CoM and mass. Often this is specified in the manual of the gripper/.., but a calibration tool exists in the UI. Keep in mind that this tool is not super accurate.

Inaccurate end-effector information will 1) mess up the IK, since the IK is calculated with the TCP frame, and mess up freedrive since gravity compensation will not work.

### F/T sensor

If your realman arm has a F/T sensor (see above to check version), you can configure it at `> Configuration > Robotic Arm Configuration > Force Sensor Configuration`. The realman seems to not use the CoM params from the End-effector for gravity compensation of the F/T wrench by default. Make sure to perform a calibration of the sensor for you end-effector, see [here](https://develop.realman-robotics.com/en/robot4th/teachingPendantfour/setting/#force-config) for details.

To see the F/T readings, go to  `> Configuration > Robotic Arm Configuration > Force Sensor Data`

### protective stop / collision safety

You can set the force threshold of the safety system, see https://develop.realman-robotics.com/en/robot4th/teachingPendantfour/setting/#collision-protection-level

Note that this is different from the certified system in a UR robot, more freedom, less safety guarantees.

### freedrive
To freedrive the robot, keep pressing the green button on the wrist of the robot, and then move it around. 


## Testing
To test that everything works, try running the [realman.py](realman.py) script.
If the robot has a gripper, also set the TCP to something reasonable to avoid collisions.
```bash
python realman.py --ip_address 192.168.1.18
```
