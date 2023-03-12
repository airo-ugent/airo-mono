# Airo-robots
This package contains code for controlling robot manipulators and grippers, specifically:
- base classes (interfaces) for the different hardware types (grippers, manipulators, F/T sensors), i.e. defines the 'driver' interfaces.
- implementations of those interfaces for some of the hardware used at AIRO.
- code to manually test these implementations, when the hardware is attached to the computer.

The following combinations of hardware and communication options are currently implemented:
| Hardware | Communication | Implementation |
|----------|:----------|----------------|
| UR robots | RTDE | [ur_rtde.py](airo_robots/manipulators/hardware/ur_rtde.py) |
| Robotiq 2F85 gripper | URCap web API | [robotiq_2f85_urcap.py](airo_robots/grippers/hardware/robotiq_2f85_urcap.py) |

Each hardware implementation module will have a `__main__` codeblock that runs the tests for that hardware implementation. This is useful to check if the hardware is connected correctly and the implementation is working as expected. But it is also the place to be to get an idea of how to use the implementation.

## Async interactions
Certain hardware commands can require some time before completion. Think about sending a move command to a robot, the exact time will depend on the velocity (profile) and the distance between the current pose and target pose, but it can easily be multiple seconds. You might nog want to be busy waiting on that command to complete before the method returns. For this reason, certain commands will send the command to the hardware, formulate a condition that signals when the command has finished and then return an [`AwaitableAction`](airo_robots/awaitable_action.py). You can then decide when (and if) you want to start busy waiting on the command to finish (according to the specified condition). This allows you to do some other useful things in the meantime.

When you decide to wait you have to specify the max waiting time (the timeout value) and the granularity of the waiting. Make sure to take a look in the code so that you understand these values and can use appropriate values, especially have a look at the part about the `time.sleep` accuracy. An example interaction might look like this:

```python
action = robot.move(target_pose)
# do other stuff, such as computing the next action
new_action = compute_new_action()
action.wait(timeout=2,resolution=0.2)
# send new action
robot.do_new_action(new_action)
```

**This is only guaranteed to work when you explicitly wait on a previous command before sending a new command**. If you send multiple commands, it is up to the robot controller to decide in what order the commands are executed (if they are not preempted). At that point, the termination condition in the `AwaitableAction` will probably make no sense anymore and result in a timeout. So make sure you know what you are doing if you send multiple actions and don't use the `wait()` method on them unless you know very well what you are doing.

See [below](#notes-on-the-different-types-of-interfaces-for-hardware-interaction) for a lengthier discussion of (async) hardware interactions and why the interface is implemented like this.
## Installation
You can simply pip install this package. Note that some hardware implementations have additional dependencies, which are for now included in the setup.py but might be separated later on.
## Structure
a more detailled overview of the structure and content of this package:

```
airo_robots/
    manipulators/
        position_manipulator.py             # base classes for position-controlled manipulators
        force_torque_sensor.py              # base class for FT sensors
        bimanual_position_manipulator.py    # base class for bimanual manipulators
        hardware/                           # contains the implementations of the inferfaces
            manual_gripper_testing.py       # code for manually testing hw implementations
            ur_rtde.py                      # implementation of the interfaces for UR robots over the RTDE interface

    grippers/
        parallel_position_gripper.py        # base classes for parallel-finger, position-controlled grippers
        hardware/
            manual_gripper_testing.py           # code for manually testing hw implementations
            robotiq_2f85_urcap.py                 # implementations for robotiq_2F85 gripper over the URscript TCP API
```

## Adding new hardware

### Adding new hardware implementations
If an interface already exists for the hardware you want to use, you can simply use that interface and implement the hardware-specific code in a new module under the appriate `hardware/` folder. For the async methods, use an appropriate condition in the Awaitable Action object to signal when the command has finished.

Don't forget to add a `__main__` codeblock to the new module that runs the tests for that hardware implementation.
Also don't forget to add the implementation in the table above and import it in the `__init__.py` of the appropriate submodule.


### Adding new interfaces
Best to look at the existing interfaces for inspiration. The main thing to keep in mind is that the interface should be as general as possible, while still being useful. So don't add too many methods that are specific to a single hardware implementation. Methods for which it makes sense should return an Awaitable Action object.

Dont forget to add the interface in the `__init__.py` of the appropriate submodule.
## Notes on the different types of interfaces for Hardware interaction

There are basically three ways to send commands to hardware:
- send a command and never look back (asynchronous)
- send a command, continue and later on check if it was succesful (asynchronous + awaitable)
- send a command and wait for it to finish (synchronous)

In ROS these are respectively sending to a topic, sending to an action server and sending + waiting on the action server.

A first thing to note is that not all commands have a clear 'termination condition'. E.g. if you send a new waypoint to an admittance controller, when has has the controller succesfully handled your request? There is no clear 'end' as with a MoveL signal that has a target pose and really has to get there.
So for these commands only the first option is available.

Another complication with async commands is about what should happen when you send multiple commands in a row. Should the robot execute them in that order, or should the robot simply take the most recent command and ignore the previous ones? This is a design choice that should be made by the user of the interface. ROS (and most hardware controllers) take the last option, i.e. the robot will execute the most recent command and ignore the previous ones. These would be marked as *preempted* in ROS.

To implement the awaitable behavior in python, there are a number of options:

1) use AsyncIO. This might actually be the most pythonic way, but it requires all downstream code to also be async. This is not always possible, e.g. when using this code in ROS. Even when it is possible, it is not desirable imo.

2) Have a custom return object on which you can wait actively when desired. This is the approach we have taken in this package.

To signal preemption of commands to the user, you have to manually queue all commands and mark all other commands as preempted when popping the most recent command from the queue. This in turn requires a separate controll process (otherwise your single thread (mind the GIL), would spend lots of times checking if the current command is finished, even if you don't care about that at all). We considered this complexity unjustified and hence do not detect/enforce/control preemptions. It is up to the user to either avoid sending multiple commands or to check how the hardware controllers deals with them. Note that waiting on preempted or stale commands will result in a timeout.
