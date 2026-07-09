# Universal Robots Torque Control

The [`URrtdeTorque`](ur_rtde_torque.py) class extends [`URrtde`](ur_rtde.py) with a **joint-space torque control mode**.
When enabled, a dedicated process runs a PD control loop at 500Hz that computes joint torques to track a target joint
configuration and streams them to the robot with `directTorque`. You update the target from your own code at whatever
rate you like; the controller smooths and tracks it.

Use this when you need compliant, high-frequency control (e.g. contact-rich tasks, learned policies, teleoperation).
For ordinary point-to-point motions and trajectories, stick with the position-control methods of `URrtde`.

## Requirements

* A UR **e-series** robot (the 500Hz control rate and torque interface are not available on CB-series robots).
  `URrtdeTorque` detects the robot model on connection and refuses to construct for a CB-series robot.
  Note that the `ManipulatorSpecs` of CB-series robots still list `max_joint_torques`: those are physical
  specs of the robot, not an indication that torque *control* is supported.
* `ur-rtde >= 1.6.0` (earlier versions do not expose `directTorque`).
* The robot must be in **Remote Control** mode, see [universal_robots_setup.md](universal_robots_setup.md).
* The robot's `ManipulatorSpecs` must have `max_joint_torques` set. The built-in specs of `URrtde` (UR3, UR3e, UR5e)
  already include these values.

## Safety

**Read this before running anything on real hardware.**

* In torque mode, *your* controller is responsible for stabilizing the robot. Badly tuned gains can cause oscillation
  or runaway motion. Keep the emergency stop within reach.
* The default PD gains were tuned on a **UR3e**. On any other model or with a different payload, validate the gains
  first: start near the robot's home configuration, command small target steps, and increase amplitude gradually.
* Commanded torques are always clipped to 80% of the robot's rated joint torques
  (`TORQUE_LIMIT_SAFETY_FACTOR`), but clipping is a last resort, not a substitute for sane gains.
* Set targets **close to the current configuration**. Large jumps are smoothed by the controller's reference
  trajectory generator, but small increments remain the safest way to command motion.

## Quickstart

```python
import numpy as np
from airo_robots.manipulators.hardware.ur_rtde_torque import URrtdeTorque

robot = URrtdeTorque("10.42.0.162")  # behaves exactly like URrtde at this point

# Move to a start configuration using regular position control.
start = np.array([-1.58, -1.74, -0.71, -1.51, 1.47, 3.11])
robot.move_to_joint_configuration(start).wait()

# Hand the robot over to the torque control loop.
# It will actively hold the current configuration until you set a target.
robot.enable_torque_control()

# Command motion by updating the target, e.g. at 100Hz from a policy or teleop device.
import time
t0 = time.time()
while time.time() - t0 < 10.0:
    target = start.copy()
    target[5] += 0.3 * np.sin(2 * np.pi * 0.2 * (time.time() - t0))
    robot.target_joint_configuration = target
    time.sleep(0.01)

# Return to position control.
robot.disable_torque_control()
robot.move_to_joint_configuration(start).wait()
```

You can try this end-to-end with the built-in manual test script (the robot will wiggle its wrist joint):

```bash
python ur_rtde_torque.py --ip_address 10.42.0.162
```

## Switching between modes

`enable_torque_control()` and `disable_torque_control()` can be alternated freely. Only one mode can be active at a
time, because the robot accepts commands from a single control script:

| | Position mode (default) | Torque mode |
|---|---|---|
| `move_*`, `servo_*`, `execute_trajectory` | ✅ | ❌ raises `RuntimeError` |
| `inverse_kinematics`, `is_tcp_pose_reachable` | ✅ | ❌ raises `RuntimeError` |
| `target_joint_configuration` (get/set) | ❌ raises `RuntimeError` | ✅ |
| `get_joint_configuration`, `get_tcp_pose`, `get_tcp_force` | ✅ | ✅ |
| Gripper | ✅ | ✅ |

`disable_torque_control()` ramps the commanded torque to zero before stopping, and is also registered as an `atexit`
handler so the robot is released cleanly if your script exits while torque control is active.

Use `robot.is_torque_control_active` to query the current mode. If the control process ever dies unexpectedly
(e.g. a protective stop), reading or setting the target raises a `RuntimeError`; call `disable_torque_control()` to
restore position control.

## Tuning

The gains are per-joint arrays passed to the constructor:

```python
robot = URrtdeTorque(
    "10.42.0.162",
    kp=np.array([120.0, 120.0, 100.0, 30.0, 30.0, 30.0]),  # [Nm/rad]
    kd=np.array([12.0, 12.0, 10.0, 2.4, 2.0, 1.0]),        # [Nm/(rad/s)]
)
```

Guidelines:

* Increase `kp` for stiffer, more accurate tracking; decrease it for more compliance.
* `kd` damps the motion. Too low → oscillation; too high → amplified velocity-measurement noise.
* Tune one joint at a time, starting from the wrist (lowest inertia) and working towards the base.

The controller itself is the `JointSpacePDController` dataclass in [ur_rtde_torque.py](ur_rtde_torque.py). It exposes
further parameters (reference trajectory natural frequency and damping, velocity filter constant) and is pure Python
with no hardware dependency, so you can simulate and unit-test gain changes before touching the robot — see
[test_joint_space_pd_controller.py](../../../test/test_joint_space_pd_controller.py) for examples.

## How it works

```
main process                          worker process (500Hz)
────────────                          ──────────────────────
URrtdeTorque                          _torque_control_worker
 ├─ rtde_receive (state queries)       ├─ rtde_control (owns the robot)
 ├─ target_joint_configuration ──────▶ ├─ reads target from shared memory
 │        (shared memory + lock)       ├─ JointSpacePDController:
 └─ enable/disable_torque_control      │    reference trajectory → PD → clip
                                       └─ directTorque(...)
```

* Only one `RTDEControlInterface` can be connected to a robot, so `enable_torque_control()` disconnects the main
  process's control interface and the worker creates its own; `disable_torque_control()` reverses this.
* The control loop runs in a separate *process* (not thread) so Python's GIL and your application code cannot
  introduce jitter into the 2ms control cycle.
* If you still observe missed cycles or jerky behavior on a busy machine, consider a `PREEMPT_RT` kernel and/or
  pinning the worker process to an isolated CPU core.
