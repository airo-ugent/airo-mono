# KUKA KRC4 Robot Setup (`KUKA.DeviceConnector` OPC-UA)

This guide explains how to prepare a KUKA KRC4 controller (such as a **KR C4 compact** driving a **KR 3 R540**) for use with [`Kuka`](kuka.py).

## 1. Installation

Install `airo-robots` with the `kuka` extra, which pulls in `asyncua`:

```bash
pip install "airo-robots[kuka]"
```

## 2. Network Connection

1. Connect your workstation to the KUKA controller's **KLI** Ethernet port (`192.168.1.2` on our setup).
2. Configure a static IPv4 address on the same `/24` subnet on your workstation (e.g. `192.168.1.100/24`):
   ```bash
   sudo ip addr add 192.168.1.100/24 dev <ethernet-interface>
   sudo ip link set <ethernet-interface> up
   ```
3. Verify connectivity to the controller and its `KUKA.DeviceConnector` OPC-UA endpoint on port `4840`:
   ```bash
   ping -c 2 192.168.1.2
   ```

## 3. Controller & smartPAD Preparation

`Kuka` reads robot state (`$AXIS_ACT`, `$POS_ACT`, `$BASE`, `$TOOL`, `$SOFTN_END`, `$SOFTP_END`, and machine kinematic geometry) directly over OPC-UA using the controller's `OpcUaSecurityAdmin` account.

To execute motion commands from Python, a small KRL server program (`PY_SLOW` in `KRC:\R1\Program\py_slow.src` and `py_slow.dat`) must be loaded and running on the controller:

1. **Upload the KRL program** (only needed once per controller):
   ```python
   from airo_robots.manipulators.hardware.kuka import Kuka

   with Kuka("192.168.1.2", upload_krl_program=True) as robot:
       pass
   ```
2. **Automatic External I/O check** (if drives **`I`** stay gray because no external PLC is attached):
   - Log in on the smartPAD as **Expert** or **Administrator** (`Main Menu -> Configuration -> User group`, default password `kuka`).
   - Open `Main Menu -> Configuration -> Inputs/outputs -> Automatic External`.
   - Ensure **`$MOVE_ENABLE`** and **`$DRIVES_OFF`** are set to **`1025`** (always `TRUE`) rather than an unpowered PLC input.
3. **Select and start `PY_SLOW` on the smartPAD**:
   - Navigate to `KRC:\R1\Program` and press **Select** on `py_slow`.
   - In **T1** mode, hold the rear enabling switch at the middle position so the **`I`** (drives) indicator turns green, then hold the green **Start (`+`)** key until the program reaches the `PTP $AXIS_ACT` line (`BCO` / `SAK` reached) and enters the `LOOP` (`PY_READY = TRUE`).
   - For hands-free operation in **AUT** mode, turn the key switch to set operating mode to **AUT**, tap the **`I`** indicator on the top bar -> press **`I`** to turn drives green, and press the green **Start (`+`)** button once so **`R`** turns green.

## 4. Usage & Manual Testing

```python
import numpy as np
from airo_robots.manipulators.hardware.kuka import Kuka

with Kuka("192.168.1.2") as robot:
    q = robot.get_joint_configuration()
    tcp = robot.get_tcp_pose()

    target = q.copy()
    target[0] += np.radians(5.0)
    robot.move_to_joint_configuration(target, joint_speed=np.radians(10.0)).wait(timeout=15.0)
```

To run the interactive manual test suite:

```bash
python -m airo_robots.manipulators.hardware.kuka --ip-address 192.168.1.2
```
