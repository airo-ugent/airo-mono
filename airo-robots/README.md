# Airo-robots

This package contains code for controlling robot manipulators and grippers:
- it contains base classes (interfaces) for the different hardware types (grippers, manipulators, F/T sensors), i.e. defines the 'driver' interfaces.
- it contains implementations of those interfaces for some of the hardware used at AIRO.
- it contains code to manually test these implementations, when the hardware is attached to the computer. Each hw implementation module while have a `__main__` codeblock that runs the tests for that hardware implementation.

a more detailled overview of the structure and content of this package:
```
airo_robots/
    manipulators/
        position_manipulator.py             # base class for position-controlled manipulators
        force_torque_sensor.py              # base class for FT sensors
        bimanual_position_manipulator.py    # base class for bimanual manipulators
        hardware/                           # contains the implementations of the inferfaces
            manual_gripper_testing.py       # code for manually testing hw implementations
            ur_rtde.py                      # implementation of the interfaces for UR robots over the RTDE interface

    grippers/
        parallel_position_gripper.py        # base class for parallel-finger, position-controlled grippers
        manual_gripper_testing.py           # code for manually testing hw implementations
        robotiq_2f85_tcp.py                 # implementations for robotiq_2F85 gripper over the URscript TCP API
```


## Usage

To use the hardware implementations:
- go check out the interface to see what methods etc. are available.
- (optional) run the implementation module for the desired hardware to get a feeling with the interface implementation and check if everything works as expected.

- import the implementation in your script and start controlling the hardware.
