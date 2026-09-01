"""Pure encoding/decoding for the Airo Gripper Protocol (AGP) v1.

AGP is the BLE protocol spoken by Halberd-based grippers (see the HalberdGripper Arduino
library and its PROTOCOL.md in the Dwengo Blockly-for-Dwenguino repository). This module
contains only frame codecs and the descriptor model; it performs no I/O so it can be unit
tested without hardware.

Wire format summary (all values little-endian, f32 = IEEE-754 float):
- Command frame (central -> gripper):  ``[seq:u8][opcode:u8][payload...]``
- Event frame (gripper -> central):    ``[type:u8][seq:u8][payload...]``
- State frame (gripper -> central):    ``[flags:u8][axisCount:u8][position:f32 * axisCount]``
- Sensor frame (gripper -> central):   ``[sensorCount:u8][value:f32 * sensorCount]``
"""

import json
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

PROTOCOL_VERSION = 1

# AGP base UUID: 9B8Exxxx-5C3A-4F1B-B4A2-6C9D0A7E5D10
SERVICE_UUID = "9b8e0001-5c3a-4f1b-b4a2-6c9d0a7e5d10"
DESCRIPTOR_CHARACTERISTIC_UUID = "9b8e0002-5c3a-4f1b-b4a2-6c9d0a7e5d10"
COMMAND_CHARACTERISTIC_UUID = "9b8e0003-5c3a-4f1b-b4a2-6c9d0a7e5d10"
STATE_CHARACTERISTIC_UUID = "9b8e0004-5c3a-4f1b-b4a2-6c9d0a7e5d10"
EVENT_CHARACTERISTIC_UUID = "9b8e0005-5c3a-4f1b-b4a2-6c9d0a7e5d10"
SENSOR_CHARACTERISTIC_UUID = "9b8e0006-5c3a-4f1b-b4a2-6c9d0a7e5d10"

PARALLEL_PROFILE = "parallel"


class Opcode(IntEnum):
    """Command opcodes (central -> gripper)."""

    MOVE_AXES = 0x01
    MOVE_POSE = 0x02
    SET_SPEED = 0x03
    SET_EFFORT = 0x04
    STOP = 0x05
    PING = 0x06
    IDENTIFY = 0x07


class EventType(IntEnum):
    """Event frame types (gripper -> central)."""

    ACK = 0x01
    DONE = 0x02
    FAILED = 0x03
    GRASPED = 0x10
    RELEASED = 0x11


class FailureReason(IntEnum):
    """Reasons carried by FAILED events."""

    BAD_FRAME = 0x01
    PREEMPTED = 0x02
    UNKNOWN = 0x03
    REJECTED = 0x04
    OUT_OF_RANGE = 0x05


class ProtocolError(ValueError):
    """Raised when a frame or descriptor cannot be parsed."""


@dataclass(frozen=True)
class AxisDescriptor:
    """One actuated degree of freedom as declared by the gripper firmware."""

    axis_id: int
    min_value: float
    max_value: float
    max_speed: float
    unit: str = ""


@dataclass(frozen=True)
class SensorDescriptor:
    """One scalar sensor channel as declared by the gripper firmware.

    The protocol is agnostic about sensor semantics; ``name``/``unit``/``min``/``max``
    convey what the value means physically (e.g. force in N, proximity in m).
    """

    sensor_id: int
    name: str
    unit: str
    min_value: float
    max_value: float


@dataclass(frozen=True)
class GripperDescriptor:
    """Self-description read from the descriptor characteristic after connecting.

    The user-assigned ``name`` is the primary identity of a gripper; ``device_id`` (the
    nRF52 factory ID) is a stable tiebreaker when two grippers share a name.
    """

    protocol: int
    name: str
    device_id: str
    profile: str
    axes: tuple[AxisDescriptor, ...]
    poses: tuple[str, ...]
    sensors: tuple[SensorDescriptor, ...] = ()

    @classmethod
    def from_json(cls, raw: bytes) -> "GripperDescriptor":
        """Parse the descriptor characteristic value.

        Raises:
            ProtocolError: if the JSON is malformed or the protocol version is unsupported.
        """
        try:
            data = json.loads(raw.decode("utf-8"))
            protocol = int(data["protocol"])
            axes = tuple(
                AxisDescriptor(
                    axis_id=int(axis["id"]),
                    min_value=float(axis["min"]),
                    max_value=float(axis["max"]),
                    max_speed=float(axis["maxSpeed"]),
                    unit=str(axis.get("unit", "")),
                )
                for axis in data["axes"]
            )
            sensors = tuple(
                SensorDescriptor(
                    sensor_id=int(sensor["id"]),
                    name=str(sensor["name"]),
                    unit=str(sensor["unit"]),
                    min_value=float(sensor["min"]),
                    max_value=float(sensor["max"]),
                )
                for sensor in data.get("sensors", [])
            )
            descriptor = cls(
                protocol=protocol,
                name=str(data["name"]),
                device_id=str(data["deviceId"]),
                profile=str(data["profile"]),
                axes=axes,
                poses=tuple(str(pose) for pose in data.get("poses", [])),
                sensors=sensors,
            )
        except (ValueError, KeyError, TypeError) as error:
            raise ProtocolError(f"Malformed gripper descriptor: {raw!r}") from error
        if descriptor.protocol != PROTOCOL_VERSION:
            raise ProtocolError(
                f"Gripper speaks AGP v{descriptor.protocol}, this client implements v{PROTOCOL_VERSION}."
            )
        return descriptor


@dataclass(frozen=True)
class Event:
    """A decoded event frame."""

    type: EventType
    seq: int
    reason: Optional[FailureReason] = None


@dataclass(frozen=True)
class GripperState:
    """A decoded state snapshot (full state, idempotent)."""

    moving: bool
    grasped: bool
    positions: tuple[float, ...]


def encode_move_axes(seq: int, targets: dict[int, float]) -> bytes:
    """Encode a MOVE_AXES command moving each axis in ``targets`` to its value."""
    frame = struct.pack("<BBB", seq, Opcode.MOVE_AXES, len(targets))
    for axis_id, target in targets.items():
        frame += struct.pack("<Bf", axis_id, target)
    return frame


def encode_move_pose(seq: int, pose_name: str) -> bytes:
    """Encode a MOVE_POSE command targeting a firmware-declared named pose."""
    name_bytes = pose_name.encode("utf-8")
    return struct.pack("<BBB", seq, Opcode.MOVE_POSE, len(name_bytes)) + name_bytes


def encode_set_speed(seq: int, axis_id: int, speed: float) -> bytes:
    """Encode a SET_SPEED command for one axis."""
    return struct.pack("<BBBf", seq, Opcode.SET_SPEED, axis_id, speed)


def encode_set_effort(seq: int, axis_id: int, effort: float) -> bytes:
    """Encode a SET_EFFORT command for one axis."""
    return struct.pack("<BBBf", seq, Opcode.SET_EFFORT, axis_id, effort)


def encode_simple_command(seq: int, opcode: Opcode) -> bytes:
    """Encode a payload-less command (STOP, PING, IDENTIFY)."""
    return struct.pack("<BB", seq, opcode)


def decode_event(frame: bytes) -> Event:
    """Decode an event frame from the event characteristic.

    Raises:
        ProtocolError: if the frame is too short or carries an unknown type/reason.
    """
    if len(frame) < 2:
        raise ProtocolError(f"Event frame too short: {frame.hex()}")
    try:
        event_type = EventType(frame[0])
    except ValueError as error:
        raise ProtocolError(f"Unknown event type 0x{frame[0]:02x}") from error
    reason: Optional[FailureReason] = None
    if event_type == EventType.FAILED:
        if len(frame) < 3:
            raise ProtocolError(f"FAILED event without reason byte: {frame.hex()}")
        try:
            reason = FailureReason(frame[2])
        except ValueError as error:
            raise ProtocolError(f"Unknown failure reason 0x{frame[2]:02x}") from error
    return Event(type=event_type, seq=frame[1], reason=reason)


def decode_state(frame: bytes) -> GripperState:
    """Decode a state snapshot from the state characteristic.

    Raises:
        ProtocolError: if the frame is shorter than its declared axis count requires.
    """
    if len(frame) < 2:
        raise ProtocolError(f"State frame too short: {frame.hex()}")
    flags, axis_count = frame[0], frame[1]
    expected_length = 2 + axis_count * 4
    if len(frame) < expected_length:
        raise ProtocolError(f"State frame declares {axis_count} axes but is {len(frame)} bytes")
    positions = struct.unpack_from(f"<{axis_count}f", frame, 2)
    return GripperState(moving=bool(flags & 0x01), grasped=bool(flags & 0x02), positions=positions)


def decode_sensors(frame: bytes) -> tuple[float, ...]:
    """Decode a sensor snapshot from the sensor characteristic.

    Values appear in ascending sensor-id order, matching the descriptor's ``sensors`` array.

    Raises:
        ProtocolError: if the frame is shorter than its declared sensor count requires.
    """
    if len(frame) < 1:
        raise ProtocolError("Empty sensor frame")
    sensor_count = frame[0]
    expected_length = 1 + sensor_count * 4
    if len(frame) < expected_length:
        raise ProtocolError(f"Sensor frame declares {sensor_count} sensors but is {len(frame)} bytes")
    return struct.unpack_from(f"<{sensor_count}f", frame, 1)
