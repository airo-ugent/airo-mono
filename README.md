# airo_core

## goals
- create base classes / interfaces for abstracting (simulated) hardware to allow for interchanging different simulators/ hardware on the one hand and different 'decision making frameworks' on the other hand.
- provide functionality that is required to quickly create robot behavior à la Folding competition @ IROS22
- facilitate research by providing, wrapping or linking to common operations for Robotic Perception/control


The idea is that by having the hardware interface, that we do not have to choose permanently between Moveit or Drake as planning framework and ROS vs. ur_rtde for communication with the robot, or whether to communicate sensor data over ROS or not.

The camera class for example, could easily be wrapped in a ROS node later after all functional code has already been written in a class that inherits from the base RGBD base class. In a simulator you can also easily use the same base class (with partial implementation of the interface), which allows for using the same API and possible for easily swapping out real and simulated hardware.

If we ever were to use Moveit, the IK of the real robot could for example send a request to moveit using the ros_bridge. Or you could directly interact with Moveit w/o using the AIRO core interface, so you are also not limited by it?

In short, it should provide a clear interface that can be implement in any HW/simulation and used to execute plans devised in any motion planning framework. If desired, you could bypass the Interface.

An initial layout might look like:
```
.
├── airo_core/
│   ├── robot/
│   │   └── ?
│   ├── gripper/
│   │   └── parallel_gripper.py -> baseclass for parallel grippers
│   ├── camera/
│   │   ├── rgbd_camera.py -> baseclass for RGB-D camera
│   │   ├── transforms.py -> geometric transforms for images and back-transforming coordinates
│   │   └── projection.py -> code for 2D-3D conversion
│   ├── spatial_algebra /
│   │   ├── se3.py --> wrapper for se3
│   │   └── operations.py --> some util code for interacting with poses, transforms and positions
│   └── visualisation/
│       └── live_visualisation.py --> code for live visualising images, metrics,... (rosboard + rosbridge)
├── airo_robotiq2F85/
│   └── robotiq2F85.py -> implements parallel_gripper from airo_core for Robotiq2F85 over TCP URcap
├── airo_ZED2i/
│   └── airo_zed2i.py -> implements rgb-d camera interface for ZED2i cameras
└── airo_drake/
    └── -dual_arm -> creates configuration for dual-arm UR3e w/ 2F-85 grippers?
```
## Developer guide
### Coding style and testing
- formatting with black
- linting with flake8
- above is enforced w/ pre-commit.
- typing with mypy?
- docstrings in reST (Sphinx) format ([most used](https://stackoverflow.com/questions/3898572/what-are-the-most-common-python-docstring-formats))
- testing with pytest. All tests are grouped in `/test`


### Design
- attributes that require complex getter/setter behaviour should use python properties
- the easiest code to maintain is no code -> does it already exists somewhere?