# Airo-robots

This package contains code for controlling robot manipulators and grippers:
- it contains base classes (interfaces) for the different hardware types (grippers, manipulators, F/T sensors), i.e. defines the 'driver' interfaces.
- it contains implementations of those interfaces for some of the hardware used at AIRO.
- it contains code to manually test these implementations, when the hardware is attached to the computer. Each hw implementation module while have a `__main__` codeblock that runs the tests for that hardware implementation.


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
        manual_gripper_testing.py           # code for manually testing hw implementations
        robotiq_2f85_tcp.py                 # implementations for robotiq_2F85 gripper over the URscript TCP API
```

## Some notes on the different types of interfaces for Hardware interaction

For Hardware there are usually three ways to interact:
- send a command and never look back (asynchronous)
- send a command, continue and later on check if it was succesful (asynchronous + awaitable)
- send a command and wait for it to finish (synchronous)

In ROS these are respectively sending to a topic, sending to an action server and sending + waiting on the action server.

Also important to note that some commands don't have clear 'termination conditions'. E.g. if you send a new waypoint to an admittance controller, when has has the controller succesfully handled your request? There is no clear 'end' as with a MoveL signal that has a target pose and really has to get there.


In python I see two options:
- using asyncio and it's awaitables.
- using the concurrent.futures module

Asyncio forces downstream code to use the 'async' definition everywhere but comes with useful functionality so there is a trade-off between 'opt-in'-ness (don't force downstream users to use asyncio if they don't need it) and having to reinvent the wheel partially. Atm I perceived the formers as a bigger downside than the latter so chose to work with futures and not with asyncio.

There is an additional consideration, where you could argue that you want to hide the async details for a synchronous user. I.e. the user does not have to 'wait' manually, the function call does this behind the scenes. To accomodate this, there are multiple interfaces:

A synchronous interface for which functions simply return when their command is executed.
An async, awaitable interface for which functions return a Future object when the command is queued (guaranteed to execute). You can explicitly wait for completion with the Future.
An async, non-awaitable interface that simply registers the command (guaranteed to execute) but provides no feedback whatsoever.

Ofc. you don't want to defined these manually for each hardware piece so there are
wrappers that take a sync or async interface and convert it into the other interfaces.
An async, non-awaitable interface can ofc not be converted to an other interface.


This is very much a WIP, so changes will most likely be made in the future.


## Usage

To use the hardware implementations:
- go check out the interface to see what methods etc. are available.
- (optional) run the implementation module for the desired hardware to get a feeling with the interface implementation and check if everything works as expected.

- import the implementation in your script and start controlling the hardware.
