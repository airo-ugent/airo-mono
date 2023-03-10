The goals of this repo are to
    - facilitate our research and development in robotic perception and control
    - facilitate the creation of demos/applications à la Folding competition @ IROS22
by  provide either wrappers to existing libraries or implementions for common functionalities and operations.

Furtermore we wish to
- do this in an opt-in fashion. Users should be able to choose which components to use and which frameworks/tools/... to use them with, e.g. to allow for interchanging different simulators/ hardware on the one hand and different 'decision making frameworks' on the other hand. To this end we try to be framework-agnostic.

- to make it easy to use this repo. We want to avoid a steep learning curve as this repo is also meant te be used by people without too much experience in robotic manipulation.

- to avoid reinventing the wheel while doing the above by levering existing libraries if possible. Possible means that existing alternatives have the desired features (obviously) and have an acceptable ease-of-use, level of documentation, robustness and code quality.

Particularly important to note is that , in the spirit of this last item, this repo does not offer advanced robotics features such as optimization-based motion planning, collision checking... If you need such things, you have to use an existing framework such as Moveit, OMPL, Drake,... However these frameworks often come with a steeper learning curve due to their genericity, which is why we offer some basic functionalities in 'barebone python'. Simple things should be simple.

You could however still use this repo's functionalities (such as reprojecting pixels to SE3 poses), then use a framework like Drake to generate a collision-free trajectory to the robot and then interface with the robot in drake using our drivers. Or you could wrap a package in a ROS node and quickly embed it in the ROS ecosystem.


### Why not use ROS for everyting?

Some of the functionality in this repo is indeed availabe in ROS, most notably the entire stack to  control hardware and all the controllers that are available on top of it. ROS also has the tf2 package for working with transforms for example. This begs the question, why not create ros packages in this repo instead of pure python packages?

ROS 2 is a great tool but makes it hard to create clean code (duplication of launch/config/urdf files), comes with some unavoidable overhead  (e.g. building the packages vs simply running python code) and is so generic that it makes easy things hard... Furthermore ROS2 is still a work in progress and key components such as the python api for Moveit and the camera calibration package are still not ported.
As it is such a generic framework, ROS also comes with a steep learning curve, and we have quite some short-term (student) projects where we don't want the student spending 50% of the project on learning ROS in order to simply move a robot from A to B.
Furthermore, we also want to be able to use our code in static/dynamic simulations, and setting up an entire ros stack for the simulated hardware is sometimes too much overhead.

Therefore we opted to provide 'framework-agnostic code' in the form of python packages. This is not a provably right choice, it is driven by (limited) personal experience.

It is important to keep an eye on the scope of this repo, to avoid reinventing the weel too much. Once you have complex use cases, you should definitely consider using ROS 2 and the frameworks built on top of it, or similar frameworks such as Drake. As we provide our code in barebone python, you can easily wrap our code in a ROS node and be on your way.




