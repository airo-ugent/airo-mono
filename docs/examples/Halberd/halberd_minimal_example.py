"""Minimal working example for a Halberd BLE gripper with a time-of-flight sensor.

Prerequisites:
- A Halberd board running a sketch based on the HalberdGripper Arduino library that:
    - starts as ``gripper.begin("gripper-left")`` (adjust GRIPPER_NAME below to match),
    - declares the ToF sensor: ``gripper.configureSensor(0, "tof", "m", 0.0, 2.0);``
    - streams readings from loop(): ``gripper.reportSensor(0, distanceMeters);``
- BLE support installed on this machine: ``pip install "airo-robots[halberd]"``

Run with:
    python halberd_minimal_example.py
"""

import time

from airo_robots.grippers import HalberdBLEGripper

GRIPPER_NAME = "gripper-left"  # the user-assigned name set in the sketch via gripper.begin()
TOF_SENSOR_NAME = "tof"  # the sensor name declared in the sketch via gripper.configureSensor()

if __name__ == "__main__":
    # Discovery is optional (connect() scans internally), but shows what is in range.
    print("Scanning for Halberd grippers...")
    for advertisement in HalberdBLEGripper.discover():
        print(f"  found {advertisement.name!r} at {advertisement.address} ({advertisement.rssi} dBm)")

    # Connect by user-assigned name. This fails loudly if several grippers share the name.
    gripper = HalberdBLEGripper.connect(name=GRIPPER_NAME)
    print(f"Connected to {gripper.gripper_specs.max_width * 1000:.0f} mm gripper {GRIPPER_NAME!r}.")

    # Blink the on-board LED so you can verify you are talking to the right gripper.
    gripper.identify()

    # Basic motion commands. Every motion returns an AwaitableAction; .wait() blocks
    # until the firmware reports DONE.
    gripper.open().wait()
    gripper.close().wait()

    # Move to a specific opening (meters), with an explicit speed (m/s).
    gripper.move(0.04, speed=0.05).wait()
    print(f"Current width: {gripper.get_current_width() * 1000:.1f} mm")
    print(f"Object grasped: {gripper.is_an_object_grasped()}")

    # Read the time-of-flight sensor declared by the firmware. Sensor snapshots stream
    # in automatically; get_sensor() simply returns the latest value.
    print(f"Sensors declared by the gripper: {gripper.sensors}")
    for _ in range(10):
        distance = gripper.get_sensor(TOF_SENSOR_NAME)
        print(f"ToF distance: {distance * 1000:.0f} mm")
        time.sleep(0.5)

    # Example: close only when something is within reach of the fingers.
    if gripper.get_sensor(TOF_SENSOR_NAME) < 0.10:
        print("Object detected within 10 cm, grasping.")
        gripper.close().wait()
    else:
        print("Nothing within 10 cm, opening.")
        gripper.open().wait()

    gripper.disconnect()
