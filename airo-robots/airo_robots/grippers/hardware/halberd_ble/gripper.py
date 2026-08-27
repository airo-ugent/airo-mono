"""Gripper drivers for Halberd-based BLE grippers.

- :class:`HalberdBLEGripper`: :class:`ParallelPositionGripper` implementation for grippers
  declaring the ``parallel`` profile (axis 0 = finger opening in meters).
- :class:`GenericHalberdGripper`: thin driver for exotic grippers (any axes/poses).

Grippers are selected by their user-assigned name, set in the Halberd sketch via
``gripper.begin("gripper-left")``. See halberd_ble.md for setup and usage.
"""

from typing import Optional

import numpy as np
from airo_robots.awaitable_action import AwaitableAction
from airo_robots.grippers.hardware.halberd_ble import protocol
from airo_robots.grippers.hardware.halberd_ble.client import HalberdAdvertisement, HalberdClient
from airo_robots.grippers.hardware.halberd_ble.client import connect as _connect
from airo_robots.grippers.hardware.halberd_ble.client import discover as _discover
from airo_robots.grippers.parallel_position_gripper import ParallelPositionGripper, ParallelPositionGripperSpecs
from loguru import logger

# The AGP descriptor does not (yet) carry force limits, so the specs use conservative
# defaults; pass explicit specs to the constructor when your gripper differs.
DEFAULT_MAX_FORCE = 100.0
DEFAULT_MIN_FORCE = 0.0
DEFAULT_MOVE_TIMEOUT = 30.0

OPEN_POSE = "open"
CLOSED_POSE = "closed"


class HalberdBLEGripper(ParallelPositionGripper):
    """A parallel gripper running on a Halberd board, controlled over BLE.
    For more exotic grippers, use :class:`GenericHalberdGripper`
    instead (see GenericHalberdGripper class below).

    The gripper firmware (HalberdGripper Arduino library) declares one axis: the finger
    opening in meters. ``open()``/``close()`` use the firmware's named poses, so custom
    finger geometries keep working as long as the sketch defines them.

    Notes on the interface contract:
    - ``speed``/``max_grasp_force`` setters are forwarded synchronously; the getters return
      the last value set from this client (AGP has no read-back for them).
    - specs force limits fall back to defaults because the descriptor does not carry them;
      pass ``gripper_specs`` to override.
    """

    def __init__(self, client: HalberdClient, gripper_specs: Optional[ParallelPositionGripperSpecs] = None) -> None:
        descriptor = client.descriptor
        if descriptor.profile != protocol.PARALLEL_PROFILE:
            raise ValueError(
                f"Gripper {descriptor.name!r} declares profile {descriptor.profile!r}, not "
                f"{protocol.PARALLEL_PROFILE!r}. Use GenericHalberdGripper for exotic grippers."
            )
        if not descriptor.axes:
            raise ValueError(f"Gripper {descriptor.name!r} declares no axes.")
        self._client = client
        self._generic = GenericHalberdGripper(client)
        self._width_axis = descriptor.axes[0]
        if gripper_specs is None:
            gripper_specs = ParallelPositionGripperSpecs(
                max_width=self._width_axis.max_value,
                min_width=self._width_axis.min_value,
                max_force=DEFAULT_MAX_FORCE,
                min_force=DEFAULT_MIN_FORCE,
                max_speed=self._width_axis.max_speed,
                min_speed=0.0,
            )
        super().__init__(gripper_specs)
        self._speed = gripper_specs.max_speed
        self._max_grasp_force = gripper_specs.max_force

    @staticmethod
    def discover(timeout: float = 5.0) -> list[HalberdAdvertisement]:
        """Scan for Halberd grippers in range (that are not connected to another client)."""
        return _discover(timeout=timeout)

    @classmethod
    def connect(
        cls,
        name: Optional[str] = None,
        address: Optional[str] = None,
        scan_timeout: float = 5.0,
        gripper_specs: Optional[ParallelPositionGripperSpecs] = None,
    ) -> "HalberdBLEGripper":
        """Connect to the gripper with the given user-assigned name (or address).

        With no filter, exactly one gripper must be in range; a name matching multiple
        grippers raises and lists the candidates.
        """
        return cls(_connect(name=name, address=address, scan_timeout=scan_timeout), gripper_specs=gripper_specs)

    @property
    def speed(self) -> float:
        """The last speed set from this client [m/s] (AGP has no speed read-back)."""
        return self._speed

    @speed.setter
    def speed(self, new_speed: float) -> None:
        clipped = float(np.clip(new_speed, self.gripper_specs.min_speed, self.gripper_specs.max_speed))
        self._client.set_speed(self._width_axis.axis_id, clipped)
        self._speed = clipped

    @property
    def max_grasp_force(self) -> float:
        """The last force set from this client [N] (AGP has no force read-back)."""
        return self._max_grasp_force

    @max_grasp_force.setter
    def max_grasp_force(self, new_force: float) -> None:
        clipped = float(np.clip(new_force, self.gripper_specs.min_force, self.gripper_specs.max_force))
        self._client.set_effort(self._width_axis.axis_id, clipped)
        self._max_grasp_force = clipped

    def get_current_width(self) -> float:
        """The current opening of the fingers in meters."""
        return self._client.get_state().positions[0]

    def move(self, width: float, speed: Optional[float] = None, force: Optional[float] = None) -> AwaitableAction:
        """Move the fingers to the desired opening [m]; speed/force persist for later moves."""
        if speed is not None:
            self.speed = speed
        if force is not None:
            self.max_grasp_force = force
        width = float(np.clip(width, self.gripper_specs.min_width, self.gripper_specs.max_width))
        return self._generic.move_axes({self._width_axis.axis_id: width})

    def open(self) -> AwaitableAction:
        """Open using the firmware's named pose, so custom geometries keep working."""
        return self._move_pose_with_fallback(OPEN_POSE, self.gripper_specs.max_width)

    def close(self) -> AwaitableAction:
        """Close using the firmware's named pose."""
        return self._move_pose_with_fallback(CLOSED_POSE, self.gripper_specs.min_width)

    def is_an_object_grasped(self) -> bool:
        return self._client.get_state().grasped

    @property
    def sensors(self) -> tuple[protocol.SensorDescriptor, ...]:
        """The sensor channels declared by the gripper firmware."""
        return self._generic.sensors

    @property
    def sensor_values(self) -> tuple[float, ...]:
        """Latest sensor snapshot, in ascending sensor-id order (matches :attr:`sensors`)."""
        return self._generic.sensor_values

    def get_sensor(self, name: str) -> float:
        """Latest value of the sensor with the given descriptor name (e.g. ``"force"``)."""
        return self._generic.get_sensor(name)

    def identify(self) -> None:
        """Blink the gripper's LED to find it physically."""
        self._generic.identify()

    def disconnect(self) -> None:
        self._client.disconnect()

    def _move_pose_with_fallback(self, pose_name: str, fallback_width: float) -> AwaitableAction:
        if pose_name in self._client.descriptor.poses:
            return self._generic.move_pose(pose_name)
        logger.debug(f"Gripper does not declare pose {pose_name!r}, moving to width {fallback_width} instead.")
        return self.move(fallback_width)


if __name__ == "__main__":
    """Manual hardware test: flash the parallel_gripper_servo example on a Halberd and run
    `python -m airo_robots.grippers.hardware.halberd_ble.gripper --name gripper-left`."""
    import click
    from airo_robots.grippers.hardware.manual_gripper_testing import manually_test_gripper_implementation

    @click.command()
    @click.option("--name", default=None, help="User-assigned gripper name (set in the sketch).")
    def test_halberd(name: Optional[str]) -> None:
        grippers = HalberdBLEGripper.discover()
        print(f"Discovered grippers: {grippers}")
        gripper = HalberdBLEGripper.connect(name=name)
        gripper.identify()
        manually_test_gripper_implementation(gripper, gripper.gripper_specs)

    test_halberd()


class GenericHalberdGripper:
    """Driver for any Halberd gripper, exotic kinematics included.

    Talks in the gripper's own terms: axes (as declared in its descriptor) and named
    poses. Use :class:`HalberdBLEGripper` instead when the gripper declares the
    ``parallel`` profile and you want the standard airo-robots gripper interface.
    """

    def __init__(self, client: HalberdClient) -> None:
        self._client = client

    @staticmethod
    def discover(timeout: float = 5.0) -> list[HalberdAdvertisement]:
        """Scan for Halberd grippers in range (that are not connected to another client)."""
        return _discover(timeout=timeout)

    @classmethod
    def connect(
        cls, name: Optional[str] = None, address: Optional[str] = None, scan_timeout: float = 5.0
    ) -> "GenericHalberdGripper":
        """Connect to the gripper with the given user-assigned name (or address)."""
        return cls(_connect(name=name, address=address, scan_timeout=scan_timeout))

    @property
    def client(self) -> HalberdClient:
        return self._client

    @property
    def descriptor(self) -> protocol.GripperDescriptor:
        return self._client.descriptor

    @property
    def axes(self) -> tuple[protocol.AxisDescriptor, ...]:
        return self._client.descriptor.axes

    @property
    def poses(self) -> tuple[str, ...]:
        return self._client.descriptor.poses

    @property
    def sensors(self) -> tuple[protocol.SensorDescriptor, ...]:
        """The sensor channels declared by the gripper firmware."""
        return self._client.descriptor.sensors

    @property
    def sensor_values(self) -> tuple[float, ...]:
        """Latest sensor snapshot, in ascending sensor-id order (matches :attr:`sensors`)."""
        return self._client.get_sensor_values()

    def get_sensor(self, name: str) -> float:
        """Latest value of the sensor with the given descriptor name (e.g. ``"force"``)."""
        for index, sensor in enumerate(self._client.descriptor.sensors):
            if sensor.name == name:
                return self._client.get_sensor_values()[index]
        declared = [sensor.name for sensor in self._client.descriptor.sensors]
        raise KeyError(f"Gripper declares no sensor named {name!r} (available: {declared}).")

    @property
    def positions(self) -> tuple[float, ...]:
        """Current position of every axis, in the units declared by the descriptor."""
        return self._client.get_state().positions

    def is_moving(self) -> bool:
        return self._client.get_state().moving

    def is_an_object_grasped(self) -> bool:
        return self._client.get_state().grasped

    def move_axes(self, targets: dict[int, float]) -> AwaitableAction:
        """Move one or more axes to absolute targets; returns when the firmware reports DONE."""
        seq = self._client.move_axes(targets)
        return AwaitableAction(lambda: self._client.is_command_terminal(seq), default_timeout=DEFAULT_MOVE_TIMEOUT)

    def move_pose(self, pose_name: str) -> AwaitableAction:
        """Move to a firmware-declared named pose (e.g. ``"open"``, ``"pinch"``)."""
        seq = self._client.move_pose(pose_name)
        return AwaitableAction(lambda: self._client.is_command_terminal(seq), default_timeout=DEFAULT_MOVE_TIMEOUT)

    def set_speed(self, axis_id: int, speed: float) -> None:
        self._client.set_speed(axis_id, speed)

    def set_effort(self, axis_id: int, effort: float) -> None:
        self._client.set_effort(axis_id, effort)

    def stop(self) -> None:
        """Preempt any in-flight motion."""
        self._client.stop()

    def identify(self) -> None:
        """Blink the gripper's LED to find it physically."""
        self._client.identify()

    def disconnect(self) -> None:
        self._client.disconnect()
