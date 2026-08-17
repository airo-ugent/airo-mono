# Robotiq 2F-85 on a FANUC controller

[`robotiq_2f85_fanuc.py`](robotiq_2f85_fanuc.py) drives a Robotiq 2F-85 that is wired to a FANUC
controller, through the [airo-fanuc](https://github.com/airo-ugent/airo-fanuc) driver. This is a
different implementation from [`robotiq_2f85_urcap.py`](robotiq_2f85_urcap.py), which talks to the
gripper's own Modbus registers over a UR control box and supports the full
`ParallelPositionGripper` interface. Over a FANUC, most of that interface is not reachable.

A FANUC controller has no gripper API, so the driver actuates the gripper the only way RMI allows: it
writes three numeric registers, and a `GRIPDISP` teach-pendant program you install watches them and
does the work. The driver's
[gripper](https://github.com/airo-ugent/airo-fanuc/blob/main/docs/gripper.md) page documents that
mechanism — the registers, the write order, the polling and the dispatcher contract. This page covers
what it costs on the airo-mono side. See [fanuc_setup.md](../../manipulators/hardware/fanuc_setup.md)
for setting up the arm itself.

## Prerequisites

- **`GRIPDISP` and `GRPRUN` installed on controller flash.** A site-installation prerequisite rather
  than a Python dependency — `pip install` cannot supply them, and the driver's preflight refuses a
  gripper-enabled session without them. `GRPRUN.LS` ships with the driver; `GRIPDISP` is specific to
  how your tool is wired and is written against the specification in the driver's gripper docs.
- **The gripper wired to the controller's tool I/O**, and working from the pendant before you try it
  from Python.
- **`DriverPolicy(enable_gripper=True)`** (the default), so the driver builds the gripper worker and
  launches the dispatcher during bring-up.

**Connecting opens the gripper.** Bring-up probes whether a dispatcher is already running by writing
a benign open, so the fingers move before your code issues any command. Keep hands clear while
connecting.

## What is and is not reachable

A register argument is a bucket index that `GRIPDISP` interprets — not a width in millimeters and not
a force in newtons — so only a handful of openings and force classes exist. The figures below are the
nominal values the module's constants carry: they select a bucket, and nothing reads them back.

| Open bucket | Opening | | Close force class | Force | For |
|---|---|---|---|---|---|
| `OPEN_FULL` | ~85 mm, fully open | | `FORCE_LIGHT` | ~100 N | rigid or easily-crushed parts |
| `OPEN_MID` | ~60 mm | | `FORCE_MEDIUM` | ~140 N | most things (the default) |
| `OPEN_NARROW` | ~35 mm | | `FORCE_HARD` | ~220 N | compressible parts that only hold once squeezed |

With nothing between the fingers, `FORCE_LIGHT` and `FORCE_MEDIUM` end at ~4 mm and `FORCE_HARD` at
0 mm. With an object between them the gripper stops on the object, which is the point of closing.

**Supported, with snapping:**

| Method | Behaviour |
|---|---|
| `move(width)` | snaps to the nearest reachable opening, and logs the snap when it lands more than a millimeter from what was asked. A width that snaps to the closed position is sent as a *close* at the current force class, so the gripper stops on an object rather than at a position |
| `open()` / `close()` | `OPEN_FULL` / a close at the current force class |
| `max_grasp_force` | snaps to the nearest force class, and applies to this and every later close (as the interface specifies). `close_force_class` sets it by class instead of by newtons |
| `last_commanded_width` | the nominal opening of the last command — a command, not a measurement |
| `last_result` | the dispatcher's verdict, `{"success": bool, "message": str}` |

**Not supported** (raises `NotImplementedError`), because nothing on this path can carry it:

| Method | Why |
|---|---|
| `get_current_width` | the gripper reports no position. Use `last_commanded_width` if the last commanded opening is good enough |
| `speed` (get and set) | `GRIPDISP` decides the finger speed; it is neither readable nor settable from here |
| `is_an_object_grasped` | the gripper's object-detection status never reaches the controller |
| `move(..., speed=...)` | same as `speed` |

`move()` returns an `AwaitableAction` that completes when the dispatcher has cleared the trigger.
**`success: True` is the dispatcher's word:** it means the teach-pendant program said it finished, not
that a width was reached or an object is held — nothing in this mechanism can tell you that. The
awaitable cannot hang, because the driver's worker always reaches a verdict within its own dispatch
timeout; a failure is logged and `last_result` carries it.

## Usage

The `Fanuc` class wraps the driver's gripper worker automatically when the driver brought one up:

```python
from airo_fanuc import DriverConfig, DriverPolicy
from airo_robots.manipulators.hardware.fanuc import Fanuc, create_crx10ial_profile

policy = DriverPolicy(config=DriverConfig(profile=create_crx10ial_profile()), enable_gripper=True)

with Fanuc("192.168.1.100", policy) as robot:
    gripper = robot.gripper                  # a Robotiq2F85Fanuc

    gripper.open().wait()
    gripper.move(0.035).wait()               # OPEN_NARROW

    gripper.max_grasp_force = 100            # snaps to FORCE_LIGHT
    gripper.close().wait()
    print(gripper.last_result)
```

Constructing it directly from a driver you brought up yourself is equivalent:
`Robotiq2F85Fanuc(driver)`. Bucket-level control, if you would rather name the bucket than a width, is
on the driver's own worker — `robot.driver.gripper.open_gripper(OPEN_MID)` and
`close_gripper(FORCE_HARD)`.

## A different gripper on a FANUC

The register mechanism is generic, so this class implements one specific protocol — the driver's
shipped `ROBOTIQ_2F85` preset — and its constructor refuses a driver configured with any other,
because the bucket values would mean something else. For another gripper, write its dispatcher and a
`RegisterGripperProtocol` for it (see the driver's gripper docs) and model a `ParallelPositionGripper`
on this one. `Fanuc` will leave `robot.gripper` as `None` with a warning rather than guess, and
`robot.driver.gripper` stays available for direct use.

## Testing

```bash
python robotiq_2f85_fanuc.py --ip_address 192.168.1.100
```

Runs `manually_test_fanuc_gripper`: every opening bucket, an in-between width to show the snapping,
a light close and a full-force close (put an object between the fingers for those), and then a check
that the unsupported methods refuse. The arm does not move.
`python ../../manipulators/hardware/fanuc.py --ip_address <ip> --gripper` runs the arm tests first
and these afterwards.

Both can be dry-run against the driver's fake controller without hardware — it implements the
register handshake too. See the *Without hardware* section of
[fanuc_setup.md](../../manipulators/hardware/fanuc_setup.md).

Note that **the gripper path is not covered by the driver's own validation ladder**: every script in
its `examples/` directory runs with the gripper disabled, so passing those says nothing about this.
The driver's gripper page lists its other known limits, including why bring-up with the gripper
enabled can need more than one attempt on a cold controller.
