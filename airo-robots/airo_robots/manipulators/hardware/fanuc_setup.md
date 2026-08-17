# FANUC robot arm setup

This README covers what you need to control a FANUC arm from airo-mono. The
[airo-fanuc](https://github.com/airo-ugent/airo-fanuc) driver's own documentation is the authoritative
source for the driver itself, and this page links to it rather than restating it. Read its
[safety](https://github.com/airo-ugent/airo-fanuc/blob/main/docs/safety.md) and
[portability](https://github.com/airo-ugent/airo-fanuc/blob/main/docs/portability.md) pages before a
first bring-up.

Developed and measured against a **FANUC CRX-10iA/L on an R-30iB-class controller**.

## Installation

```bash
pip install "airo-robots[fanuc]"
```

The driver is **Linux only** and ships prebuilt wheels; see its
[README](https://github.com/airo-ugent/airo-fanuc#installation) for the platforms it covers.

## Controller requirements

- **Controller option S636** (J519 Stream Motion and R912 RMI) — an ordered option, so check that
  your controller has it: `python -m airo_fanuc.controller_probe --ip <controller ip>` reads the
  ordered options off the controller, read-only, and is safe to run against a live cell.
- **Controller in AUTO**, drives powered, E-stop released, no active alarm, general override at
  100%. T1/T2 will connect but nothing will move.
- **Nothing else talking to the controller**: one Stream Motion peer per controller is a hardware
  constraint, and a second peer receives no status at all. The driver's advisory `flock` guards only
  part of that — it enforces *one driver per host*, so a two-arm cell needs a distinct `lock_path`
  per arm.

## Network

Connect the controller to the workstation over ethernet. The driver uses UDP port `60015` (Stream
Motion) and TCP port `16001` (RMI); make sure neither is firewalled. Ping the controller before
going further.

The IP address is whatever is configured on the pendant (`MENU > SETUP > Host Comm`); `192.168.1.100`
is the address of the cell at AIRO, not a FANUC-wide default.

## Robot profile

The driver ships **no arm profile** and requires one: it holds the limits its real-time core clamps
every command against. Read your own arm's off its controller rather than typing it by hand, and see
the driver's
[configuration](https://github.com/airo-ugent/airo-fanuc/blob/main/docs/configuration.md) page for
which fields the probe cannot supply:

```bash
python -m airo_fanuc.controller_probe --ip 192.168.1.100 --emit-profile
```

[`fanuc.py`](fanuc.py)'s `create_crx10ial_profile()` is the profile of the CRX-10iA/L at AIRO. Copy
it as a starting point rather than importing it into your own code: those limits are the *active
configuration of one controller*, not a property of the model.

## Usage

```python
from airo_fanuc import DriverConfig, DriverPolicy
from airo_robots.manipulators.hardware.fanuc import Fanuc, create_crx10ial_profile

policy = DriverPolicy(config=DriverConfig(profile=create_crx10ial_profile()), enable_gripper=False)

# Construct-and-go: this blocks until the arm is streaming and commandable, or raises with a reason.
with Fanuc("192.168.1.100", policy) as robot:
    print(robot.get_joint_configuration())      # radians
    print(robot.get_tcp_pose())                 # the controller's own TCP pose, with its active UTOOL

    q = robot.get_joint_configuration()
    q[5] += 0.1
    robot.move_to_joint_configuration(q).wait()
```

`DriverPolicy` and `DriverConfig` carry everything the driver's behaviour depends on; every field has
a default except the profile. Anything this interface does not expose stays reachable on
`robot.driver` (recovery, the ARM gate, timing statistics, the RMI session, register access).

## No kinematics

**The driver does no kinematics: no URDF, no forward kinematics, no inverse kinematics.** So on this
class every Cartesian command raises `NotImplementedError`:

| Method | Available |
|---|---|
| `get_joint_configuration`, `move_to_joint_configuration`, `servo_to_joint_configuration`, `execute_trajectory` | yes |
| `get_tcp_pose`, `get_flange_pose` | yes — the *controller* computes these, they are not derived here |
| `move_to_tcp_pose`, `move_linear_to_tcp_pose`, `servo_to_tcp_pose` | no |
| `inverse_kinematics`, `forward_kinematics`, `is_tcp_pose_reachable` | no |
| `start_freedrive` / `stop_freedrive` | no, see below |

Plan in joint space with a numerical solver on a FANUC URDF — the CRX-10iA/L URDF is in
[airo-models](https://github.com/airo-ugent/airo-models) — and command the resulting joint
configurations. [curobo](https://curobo.org/) and [drake](https://drake.mit.edu/) both work; the
`airo-drake` helpers used elsewhere in airo-mono (`time_parametrize_toppra`,
`discretize_drake_joint_trajectory`) produce trajectories `execute_trajectory` accepts directly.

Note that the controller's TCP pose follows the **UTOOL that is active on the pendant**. Set it
there for the tool you have mounted, or `get_tcp_pose()` reports a point that is not your tool tip
(on our cell the Robotiq gripper is a 175 mm offset along tool +Z).

## Trajectory execution

`execute_trajectory` is overridden for this robot: the base-class implementation streams a trajectory
servo command by servo command from Python, while the FANUC driver takes the whole timeline in one
submission and its C++ core owns playback, so a late Python thread costs nothing.

What the driver requires of a trajectory (its
[api](https://github.com/airo-ugent/airo-fanuc/blob/main/docs/api.md) page has the details):

- It **starts where the arm is**, within the driver's capture window around the commanded pose. A
  plan the arm has already partly executed cannot be joined part-way — replan it.
- The **first knot's velocity** must be inside the capture envelope. Starting from rest always is; a
  path that demands its peak velocity at `t=0` (a plain sine, say) is not. A raised cosine is.
- Joint velocities stay within the profile's velocity limits (a violation is refused).
- Joint **positions are not validated**: the core silently clamps them into the profile's position
  limits each tick. If your planner can leave the envelope, validate the path yourself, or the
  executed path is not the one you submitted.

Velocities are optional. A path without them is handed to the driver as-is, which derives the
playback tangents itself — one derivation, and the one that is guaranteed to satisfy the first-knot
envelope.

## Freedrive

A FANUC CRX is hand-guided with the buttons on the arm, but only while nothing holds the motion
group — and this driver holds it for as long as it is connected, so `start_freedrive()` raises.

To hand-guide the arm while reading its joint angles (for hand-eye calibration, say), close the
driver and use `airo_fanuc.FanucReceiveInterface` instead: it polls joint angles and status over
RMI and never takes the motion group.

## Gripper

Our Robotiq 2F-85 is wired to the controller's tool I/O and driven through
[`robotiq_2f85_fanuc.py`](../../grippers/hardware/robotiq_2f85_fanuc.py). With
`DriverPolicy(enable_gripper=True)` (the default) the `Fanuc` class wraps the driver's gripper worker
automatically, so `robot.gripper` is a ready `Robotiq2F85Fanuc`.

Two things to know before you enable it: the `GRIPDISP` and `GRPRUN` teach-pendant programs must be
installed on controller flash, and only discrete openings and force classes are reachable while
nothing is readable. [fanuc_robotiq.md](../../grippers/hardware/fanuc_robotiq.md) covers both.

## Testing

To test that everything works, run the [fanuc.py](fanuc.py) script. It moves the arm around the
configuration it is already in, so park it somewhere clear first.

```bash
python fanuc.py --ip_address 192.168.1.100
python fanuc.py --ip_address 192.168.1.100 --gripper   # also runs the gripper tests
```

The gripper alone can be tested with
[robotiq_2f85_fanuc.py](../../grippers/hardware/robotiq_2f85_fanuc.py):

```bash
python robotiq_2f85_fanuc.py --ip_address 192.168.1.100
```

Before either, walk the driver's own `examples/` directory once on a new cell: it is an ordered
bring-up ladder from a no-motion connect to streamed servoing, each step ending in a PASS/FAIL
verdict with the real-time loop's measured timing. It is a better first hardware run than anything
here, and `--fake` runs it offline.

### Without hardware

The driver ships an in-process fake controller, which
[`test/test_fanuc_fake_controller.py`](../../../test/test_fanuc_fake_controller.py) runs the same
paths against (skipped unless the driver is installed):

```bash
pytest airo-robots/test/test_fanuc_fake_controller.py
```

To dry-run the two scripts above, serve the fake on the driver's default ports in one terminal and
point the scripts at `127.0.0.1` in another. It implements the gripper register handshake too, so
`--gripper` works.

```python
from airo_fanuc.testing import FakeCRXConfig, FakeCRXController

controller = FakeCRXController(FakeCRXConfig(available_version=3, itp_s=0.008), sm_port=60015, rmi_bootstrap_port=16001)
controller.start()
controller.start_realtime(speed=1.0)
input("fake controller up, press enter to stop")
```

## Troubleshooting

The driver's
[troubleshooting](https://github.com/airo-ugent/airo-fanuc/blob/main/docs/troubleshooting.md) page is
symptom-first and covers what you will actually hit. What this class adds on top of it:

- **`OwnershipError`** — another process holds the controller.
- **A `RuntimeError` saying the robot is "not commandable"** — the driver's `RobotFaultedError`,
  translated. A fault is latched, or the ARM gate is set; `robot.driver.get_state()["fault_reason"]`
  and `["operator_hint"]` say which and what to do. After an e-stop, recovery deliberately leaves
  motion inhibited until `robot.driver.arm()` is called: that is a human decision, never something to
  put in a retry loop.
- **`InvalidTrajectoryException`** — the driver's own trajectory refusal, translated. If it mentions
  the capture window, anchor the first knot on `robot.get_commanded_joint_configuration()`.
- **`is_steady()` never becomes `True`** — a servo stream is still holding its last target. Call
  `robot.hold()`.
