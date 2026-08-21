"""Interactive keyboard control for a Halberd BLE gripper.

Prerequisites:
- A Halberd board advertising as ``GRIPPER_NAME`` (adjust below).
- BLE support installed: ``pip install "airo-robots[halberd]"``

Keys:
    o          Open fully.
    c          Close fully.
    Up arrow   Open by INCREMENT_M metres.
    Down arrow Close by INCREMENT_M metres.
    q / Ctrl-C Quit.

Run with:
    python halberd_keyboard_control.py
"""

import sys
import termios
import threading
import time
import tty

import numpy as np
from airo_robots.grippers import HalberdBLEGripper
from airo_robots.grippers.hardware.halberd_ble.client import HalberdConnectionError

GRIPPER_NAME = "gripper-left-sensors"  # must match gripper.begin() in the sketch
INCREMENT_M = 0.005  # 5 mm per arrow-key press
SENSOR_POLL_S = 0.1
TERMINAL_LOCK = threading.Lock()


def _format_sensor_snapshot(halberd_gripper: HalberdBLEGripper) -> str:
    """Format the latest sensor snapshot for terminal output."""
    sensors = halberd_gripper.sensors
    if not sensors:
        return "(no sensors declared)"

    try:
        values = halberd_gripper.sensor_values
    except HalberdConnectionError as error:
        return f"(sensor values unavailable: {error})"

    parts = []
    for sensor, value in zip(sensors, values):
        unit = f" {sensor.unit}" if sensor.unit else ""
        parts.append(f"{sensor.name}={value:.3f}{unit}")
    return " | ".join(parts)


def _terminal_print(message: str, *, end: str = "\n") -> None:
    """Serialize terminal writes so sensor updates do not interleave with commands."""
    with TERMINAL_LOCK:
        print(message, end=end, flush=True)


def _terminal_overwrite(message: str) -> None:
    """Overwrite the current terminal line with a short status message."""
    with TERMINAL_LOCK:
        sys.stdout.write(f"\r\033[2K{message}")
        sys.stdout.flush()


def _start_sensor_monitor(halberd_gripper: HalberdBLEGripper) -> tuple[threading.Event, threading.Thread | None]:
    """Print sensor snapshots when they change."""
    if not halberd_gripper.sensors:
        return threading.Event(), None

    stop_event = threading.Event()

    def _monitor() -> None:
        last_snapshot: tuple[float, ...] | None = None
        while not stop_event.is_set():
            try:
                snapshot = halberd_gripper.sensor_values
            except HalberdConnectionError:
                time.sleep(SENSOR_POLL_S)
                continue

            if snapshot != last_snapshot:
                _terminal_overwrite(f"Sensors: {_format_sensor_snapshot(halberd_gripper)}")
                last_snapshot = snapshot

            time.sleep(SENSOR_POLL_S)

    thread = threading.Thread(target=_monitor, daemon=True)
    thread.start()
    return stop_event, thread


def _read_key() -> str:
    """Read one keypress (blocking) from stdin in raw mode.

    Arrow keys are returned as the strings ``"UP"`` and ``"DOWN"``.
    All other keys are returned as their single character.
    """
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":  # start of an escape sequence
            seq = sys.stdin.read(2)
            if seq == "[A":
                return "UP"
            if seq == "[B":
                return "DOWN"
            return ch  # unknown escape — ignore
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


if __name__ == "__main__":
    _terminal_print("Connecting to gripper...")
    gripper = HalberdBLEGripper.connect(name=GRIPPER_NAME)
    max_width = gripper.gripper_specs.max_width
    _terminal_print(f"Connected to {max_width * 1000:.0f} mm gripper {GRIPPER_NAME!r}.")
    _terminal_print("Controls: o = open | c = close | ↑↓ = step | q = quit")
    if gripper.sensors:
        _terminal_print(f"Sensors: {', '.join(sensor.name for sensor in gripper.sensors)}")
    else:
        _terminal_print("Sensors: none declared")

    # Track width locally to avoid depending on state notifications from the firmware.
    current_width = max_width
    sensor_monitor_stop, sensor_monitor_thread = _start_sensor_monitor(gripper)

    try:
        while True:
            key = _read_key()

            _terminal_print("")

            if key in ("q", "\x03"):  # q or Ctrl-C
                _terminal_print("Quitting.")
                break

            elif key == "o":
                _terminal_print("Opening...")
                gripper.open().wait()
                current_width = max_width
                _terminal_print(f"  width: {current_width * 1000:.1f} mm")

            elif key == "c":
                _terminal_print("Closing...")
                gripper.close().wait()
                current_width = 0.0
                _terminal_print(f"  width: {current_width * 1000:.1f} mm")

            elif key == "UP":
                current_width = float(np.clip(current_width + INCREMENT_M, 0.0, max_width))
                _terminal_print(f"Opening to {current_width * 1000:.1f} mm...")
                gripper.move(current_width).wait()
                _terminal_print(f"  width: {current_width * 1000:.1f} mm")

            elif key == "DOWN":
                current_width = float(np.clip(current_width - INCREMENT_M, 0.0, max_width))
                _terminal_print(f"Closing to {current_width * 1000:.1f} mm...")
                gripper.move(current_width).wait()
                _terminal_print(f"  width: {current_width * 1000:.1f} mm")

    finally:
        sensor_monitor_stop.set()
        if sensor_monitor_thread is not None:
            sensor_monitor_thread.join(timeout=1.0)
        gripper.disconnect()
        _terminal_print("Disconnected.")
