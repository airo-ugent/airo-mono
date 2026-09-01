# Halberd BLE grippers

Drivers for grippers built on the [Dwengo](https://www.dwengo.org/) **Halberd** board (nRF52840), running the
`HalberdGripper` Arduino firmware library. Communication happens over Bluetooth Low Energy using the
**Airo Gripper Protocol (AGP)**; the protocol specification lives with the firmware library
(`PROTOCOL.md` in the `HalberdGripper` library of the Dwengo board support package).

Two drivers are provided:

| Class | Use case |
|-------|----------|
| `HalberdBLEGripper` | Standard parallel grippers; implements the `ParallelPositionGripper` interface (widths in meters). |
| `GenericHalberdGripper` | Exotic designs with multiple axes and named poses (e.g. soft grippers, multi-finger hands). |

## Installation

The BLE backend uses [bleak](https://github.com/hbldh/bleak), which is an optional dependency:

```bash
pip install "airo-robots[halberd]"
```

On Linux, `bleak` talks to BlueZ over D-Bus; make sure the `bluetooth` service is running and your user may
use it (usually being in the `bluetooth` group or using the default policy is enough).

## Identifying grippers

Each gripper firmware is given a **user-assigned name** (e.g. `gripper-left`) in its sketch via
`gripper.begin("gripper-left")`. This name is the primary identity used when connecting:

```python
from airo_robots.grippers import HalberdBLEGripper

gripper = HalberdBLEGripper.connect(name="gripper-left")
```

If several advertising grippers share the same name, connecting **fails loudly** with an
`AmbiguousGripperError` listing the candidates; you can then disambiguate with the BLE address:

```python
gripper = HalberdBLEGripper.connect(name="gripper-left", address="AA:BB:CC:DD:EE:FF")
```

Use `HalberdBLEGripper.discover()` to list the grippers that are currently advertising. Note that a
connected gripper stops advertising, so it will not show up in scans of other clients.

To physically identify a gripper, call `gripper.identify()`: the board blinks its LED for ~2 seconds.

## Parallel gripper usage

`HalberdBLEGripper` follows the standard `ParallelPositionGripper` interface:

```python
gripper = HalberdBLEGripper.connect(name="gripper-left")
gripper.move(0.02, speed=0.05).wait()   # move to 2 cm opening
gripper.open().wait()
gripper.close().wait()
print(gripper.get_current_width())
print(gripper.is_an_object_grasped())
gripper.disconnect()
```

Notes:

- Axis 0 of the firmware descriptor is interpreted as the gripper width; its range and max speed populate
  `gripper_specs`.
- The firmware descriptor currently carries no force limits, so `gripper_specs.max_force`/`min_force` fall
  back to defaults. `speed` and `max_grasp_force` are locally cached last-set values (the protocol has no
  read-back for them).
- `open()`/`close()` prefer the firmware-defined `open`/`closed` poses and fall back to a width move.

## Exotic gripper usage

`GenericHalberdGripper` exposes whatever axes and poses the firmware declares:

```python
from airo_robots.grippers import GenericHalberdGripper

gripper = GenericHalberdGripper.connect(name="tentacle")
print(gripper.axes)      # axis descriptors (id, min, max, maxSpeed)
print(gripper.poses)     # named poses defined in the firmware
gripper.move_axes({0: 0.5, 1: 1.57}).wait()
gripper.move_pose("curl").wait()
```

## Sensors

Firmware can declare scalar sensor channels (force, pressure, proximity, ...) via
`gripper.configureSensor(id, name, unit, min, max)` and stream values with
`gripper.reportSensor(id, value)`. The values arrive over a dedicated notify characteristic
(rate-limited, full snapshots) and are exposed on both driver classes:

```python
print(gripper.sensors)              # sensor descriptors (id, name, unit, min, max)
print(gripper.sensor_values)        # latest snapshot, ascending sensor-id order
print(gripper.get_sensor("force"))  # look up by descriptor name
```

The firmware only notifies when a value changes, so reading sensors before the first
snapshot arrives raises; grippers that declare no sensors return an empty tuple.

## Manual hardware test

With a gripper advertising nearby:

```bash
python -m airo_robots.grippers.hardware.halberd_ble.gripper --name gripper-left
```

This runs the standard `manually_test_gripper_implementation` routine.
