The goals of this repo are
- first and foremost to
    - facilitate our research and development in robotic perception and control
    - facilitate the creation of demos/applications à la Folding competition @ IROS22
by  provide either wrappers to existing libraries or implementions for common functionalities and operations.
- to do this in an opt-in fashion. Users should be able to choose which components to use and which not on the one hand and which frameworks/tools/... to use this repo with, e.g. to allow for interchanging different simulators/ hardware on the one hand and different 'decision making frameworks' on the other hand. This is a.o. achieved by creating base classes / interfaces for abstracting (simulated) hardware.
- to make it easy to use this repo. We want to avoid a steep learning curve as this repo is also meant te be used by people without too much experience in robotic manipulation.
- to avoid reinventing the wheel while doing the above by levering existing libraries if possible. Possible means that existing alternatives have the desired features (obviously) and have an acceptable ease-of-use, level of documentation, robustness and code quality.

Particularly important to note is that , in the spirit of this last item, this repo does not offer advanced robotics features such as optimization-based motion planning, collision checking... If you need such things, you have to use an existing framework such as Moveit, OMPL, Drake,... However these frameworks often come with a steeper learning curve due to their genericity, which is why we offer some basic functionalities in 'barebone python'. Simple things should be simple. You could however still use certain functionalities (such as converting pixels to SE3 poses), then use a framework like Drake to generate a collision-free trajectory to the robot and then interface the robot to drake using our drivers.

## summary of 'why make this repo?'

The idea is that by having the hardware interface, that we do not have to choose permanently between Moveit or Drake as planning framework and ROS vs. ur_rtde for communication with the robot, or whether to communicate sensor data over ROS or not.

The camera class for example, could easily be wrapped in a ROS node later after all functional code has already been written in a class that inherits from the base RGBD base class. In a simulator you can also easily use the same base class (with partial implementation of the interface), which allows for using the same API and possible for easily swapping out real and simulated hardware.

If we ever were to use Moveit, the IK of the real robot could for example send a request to moveit using the ros_bridge. Or you could directly interact with Moveit w/o using the AIRO core interface, so you are also not limited by it?

In short, it should provide a clear interface that can be implement in any HW/simulation and used to execute plans devised in any motion planning framework. If desired, you could bypass the interface.


inspiration sources:
- Berkeley Autolab: [core](https://github.com/BerkeleyAutomation/autolab_core) [ur python](https://github.com/BerkeleyAutomation/ur5py)
- CMU [frankapy](https://github.com/iamlab-cmu/frankapy) [paper](https://arxiv.org/abs/2011.02398?s=09)


Why not use ROS for everyting?
ROS is a great tool, but hard to create clean code and so generic that it makes easy things hard... Furthermore ROS2 still But this is not a provably right choice, it is driven by (limited) personal experience, the desire for a flat learning curve to serve short-lived projects such as master thesis etc.


## SCOPE
## HISTORY
## Alternatives, and why this repo exists next to them.