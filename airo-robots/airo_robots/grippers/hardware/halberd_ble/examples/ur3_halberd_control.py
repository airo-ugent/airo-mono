"""Run a UR + Halberd pick/place sequence using saved TCP poses.

Edit the configuration constants below to match your setup.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from airo_robots.grippers import HalberdBLEGripper
from airo_robots.grippers.hardware.halberd_ble.client import HalberdConnectionError
from airo_robots.manipulators.hardware.ur_rtde import URrtde

np.set_printoptions(precision=3, suppress=True)

UR_IP_ADDRESS = "10.42.0.162"
HALBERD_GRIPPER_NAME = (
    "gripper-left-sensors"  # Make sure this matches the name set in the code running on Halberd via gripper.begin()
)
POSE_DIRECTORY = Path(__file__).resolve().parent / "poses"
POSE_FILENAMES = [
    "pose1_uptop_2.npy",
    "pose2_looking_down.npy",
    "pose3_near_servo_2.npy",
    "pose4_servo_pickup.npy",
    "pose5_servo_above_bin.npy",
    "pose6_servo_inside_bin.npy",
]
PLOT_SENSOR_NAMES = ("tof", "pressure")
SMOOTH_SENSOR_NAMES = {"pressure", "tof"}  # sensors to smooth in the live plot
SMOOTHING_WINDOW = 5  # number of samples for the moving average
SENSOR_Y_LIMITS: dict[str, tuple[float, float]] = {
    "tof": (0.0, 100.0),
    "pressure": (0.0, 3.5),
}


@dataclass
class SensorSeries:
    times: deque[float]
    values: deque[float]
    unit: str


class LiveSensorPlotter:
    """Background sensor polling + live plotting for selected sensor channels."""

    def __init__(
        self,
        gripper: HalberdBLEGripper,
        sensor_names: tuple[str, ...],
        poll_period_s: float = 0.05,
        window_s: float = 30.0,
    ) -> None:
        self._gripper = gripper
        self._sensor_names = sensor_names
        self._poll_period_s = poll_period_s
        self._window_s = window_s
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._start_t = time.monotonic()

        descriptor_by_name = {sensor.name.lower(): sensor for sensor in gripper.sensors}
        self._series: dict[str, SensorSeries] = {}
        self._lines = {}

        plt.ion()
        self._figure, axes = plt.subplots(len(sensor_names), 1, figsize=(8, 5), sharex=True)
        self._figure.suptitle("Sensor measurements", fontsize=40, fontweight="bold")
        try:
            manager = self._figure.canvas.manager
            if manager is not None:
                manager.window.wm_attributes("-zoomed", True)  # type: ignore[attr-defined]  # Tk on Linux: maximized
        except Exception:
            pass  # non-Tk backends: silently skip
        if len(sensor_names) == 1:
            axes = [axes]
        self._axes = axes

        for ax, sensor_name in zip(self._axes, sensor_names):
            descriptor = descriptor_by_name.get(sensor_name.lower())
            unit = descriptor.unit if descriptor and descriptor.unit else ""
            (line,) = ax.plot([], [], lw=2)
            ylabel = f"{sensor_name} [{unit}]" if unit else sensor_name
            ax.set_ylabel(ylabel, fontsize=30)
            ax.tick_params(axis="both", labelsize=14)
            ax.grid(True, alpha=0.3)
            self._series[sensor_name] = SensorSeries(deque(maxlen=5000), deque(maxlen=5000), unit)
            self._lines[sensor_name] = line

        self._axes[-1].set_xlabel("time [s]", fontsize=30)
        self._figure.tight_layout()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            t = time.monotonic() - self._start_t
            try:
                snapshot = self._gripper.sensor_values
            except HalberdConnectionError:
                time.sleep(self._poll_period_s)
                continue

            sensor_values_by_name = {
                sensor.name.lower(): snapshot[i] for i, sensor in enumerate(self._gripper.sensors) if i < len(snapshot)
            }

            with self._lock:
                for sensor_name in self._sensor_names:
                    value = sensor_values_by_name.get(sensor_name.lower())
                    if value is None:
                        continue
                    series = self._series[sensor_name]
                    series.times.append(t)
                    series.values.append(float(value))

            time.sleep(self._poll_period_s)

    def update_plot(self) -> None:
        """Redraw plots from the main thread.

        Tk-based matplotlib backends require all canvas operations on the main thread.
        """
        with self._lock:
            self._redraw_locked()

    def _redraw_locked(self) -> None:
        for ax, sensor_name in zip(self._axes, self._sensor_names):
            series = self._series[sensor_name]
            if not series.times:
                continue

            times = np.fromiter(series.times, dtype=float)
            values = np.fromiter(series.values, dtype=float)

            if sensor_name.lower() in SMOOTH_SENSOR_NAMES and len(values) >= SMOOTHING_WINDOW:
                kernel = np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW
                smoothed = np.convolve(values, kernel, mode="valid")
                times = times[SMOOTHING_WINDOW - 1 :]
                values = smoothed

            self._lines[sensor_name].set_data(times, values)

            x_max = float(times[-1])
            x_min = max(0.0, x_max - self._window_s)
            ax.set_xlim(x_min, max(self._window_s, x_max + 0.2))

            if sensor_name.lower() in SENSOR_Y_LIMITS:
                ax.set_ylim(SENSOR_Y_LIMITS[sensor_name.lower()])
            else:
                y_min = float(np.min(values))
                y_max = float(np.max(values))
                if abs(y_max - y_min) < 1e-6:
                    y_max += 1e-3
                    y_min -= 1e-3
                margin = 0.1 * (y_max - y_min)
                ax.set_ylim(y_min - margin, y_max + margin)

        self._figure.canvas.draw_idle()
        self._figure.canvas.flush_events()


def convert_angles_to_degrees(angles: np.ndarray) -> list[float]:
    return [round(float(np.rad2deg(angle)), 3) for angle in angles]


def print_robot_state(robot: URrtde) -> None:
    q = robot.get_joint_configuration()
    print("Current joint configuration [deg]:", convert_angles_to_degrees(q))
    print("Current TCP pose:\n", robot.get_tcp_pose())


def save_tcp_pose_as_numpy_array(tcp_pose: np.ndarray, filename: str) -> None:
    np.save(filename, tcp_pose)
    print(f"TCP pose saved to {filename}")


def load_tcp_pose_from_numpy_array(filename: Path) -> np.ndarray:
    tcp_pose = np.load(filename)
    if tcp_pose.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 pose matrix in {filename}, got shape {tcp_pose.shape}")
    print(f"TCP pose loaded from {filename}")
    return tcp_pose


def load_poses(pose_dir: Path) -> dict[int, np.ndarray]:
    poses = {}
    for i, filename in enumerate(POSE_FILENAMES, start=1):
        filepath = pose_dir / filename
        poses[i] = load_tcp_pose_from_numpy_array(filepath)
    return poses


def wait_for_sensor_stream(gripper: HalberdBLEGripper, timeout_s: float = 10.0) -> None:
    if not gripper.sensors:
        print("No sensors declared by the gripper firmware.")
        return

    deadline = time.monotonic() + timeout_s
    while True:
        try:
            _ = gripper.sensor_values
            return
        except HalberdConnectionError:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "No initial sensor snapshot received from gripper. "
                    "Ensure the firmware calls gripper.reportSensor(...) in loop()."
                )
            time.sleep(0.1)


def move_to_pose(robot: URrtde, pose: np.ndarray, label: str) -> None:
    print(f"Moving to {label}...")
    robot.move_to_tcp_pose(pose).wait()


def run_sequence(robot: URrtde, gripper: HalberdBLEGripper, poses: dict[int, np.ndarray]) -> None:
    # Make sure the gripper is open before starting the sequence
    print("Opening Halberd gripper...")
    gripper.open().wait()

    # Move to pose 1
    move_to_pose(robot, poses[1], "pose 1")

    # Move to pose 2
    move_to_pose(robot, poses[2], "pose 2")

    # Move to pose 3
    move_to_pose(robot, poses[3], "pose 3")

    # Close Halberd gripper
    print("Closing Halberd gripper...")
    gripper.close().wait()

    # Move to pose 4
    move_to_pose(robot, poses[4], "pose 4")

    # Move to pose 5
    move_to_pose(robot, poses[5], "pose 5")

    # Move to pose 6
    move_to_pose(robot, poses[6], "pose 6")

    # Open Halberd gripper
    print("Opening Halberd gripper...")
    gripper.open().wait()

    # Move to pose 5
    move_to_pose(robot, poses[5], "pose 5")

    # Move to pose 1
    move_to_pose(robot, poses[1], "pose 1")


def _run_sequence_in_thread(robot: URrtde, gripper: HalberdBLEGripper, pose_dir: Path) -> threading.Thread:
    """Start the pick/place sequence in a background thread and return it."""

    def _worker() -> None:
        poses = load_poses(pose_dir)
        run_sequence(robot, gripper, poses)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return thread


def _run_plot_loop(plotter: LiveSensorPlotter, sequence_thread: threading.Thread) -> None:
    """Drive the matplotlib event loop on the main thread until the sequence finishes."""
    while sequence_thread.is_alive():
        if plt.fignum_exists(plotter._figure.number):
            plotter.update_plot()
            plt.pause(0.05)
        else:
            time.sleep(0.05)

    sequence_thread.join()
    print("Sequence finished. Close the plot window or press Ctrl-C to exit.")
    while plt.fignum_exists(plotter._figure.number):
        plotter.update_plot()
        plt.pause(0.1)


def main() -> None:
    pose_dir = POSE_DIRECTORY

    if not pose_dir.exists():
        raise FileNotFoundError(f"Pose directory not found: {pose_dir}")

    robot = URrtde(UR_IP_ADDRESS, URrtde.UR3E_CONFIG)
    print_robot_state(robot)

    print(f"Connecting to Halberd gripper {HALBERD_GRIPPER_NAME!r}...")
    gripper = HalberdBLEGripper.connect(name=HALBERD_GRIPPER_NAME)
    print("Connected to Halberd gripper.")
    print("Declared sensors:", [sensor.name for sensor in gripper.sensors])

    wait_for_sensor_stream(gripper)

    plotter = LiveSensorPlotter(gripper, sensor_names=PLOT_SENSOR_NAMES)
    plotter.start()
    time.sleep(10.0)  # give the plotter a moment to start and collect initial data

    sequence_thread = _run_sequence_in_thread(robot, gripper, pose_dir)

    try:
        _run_plot_loop(plotter, sequence_thread)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        plotter.stop()
        gripper.disconnect()
        plt.close("all")


if __name__ == "__main__":
    main()
