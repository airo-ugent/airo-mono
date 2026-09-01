"""Halberd BLE gripper support: AGP protocol codecs, BLE client and gripper drivers.

Keep this package importable without ``bleak`` installed: only actually connecting or
scanning requires it (``pip install "airo-robots[halberd]"``).
"""

from airo_robots.grippers.hardware.halberd_ble.gripper import (  # noqa: F401 - imported but unused
    GenericHalberdGripper,
    HalberdBLEGripper,
)
