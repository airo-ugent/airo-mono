import struct

import pytest
from airo_robots.grippers.hardware.halberd_ble import protocol


def test_encode_move_axes() -> None:
    frame = protocol.encode_move_axes(7, {0: 0.05})
    assert frame[0] == 7
    assert frame[1] == protocol.Opcode.MOVE_AXES
    assert frame[2] == 1  # axis count
    assert frame[3] == 0  # axis id
    assert struct.unpack_from("<f", frame, 4)[0] == pytest.approx(0.05)


def test_encode_move_axes_multiple() -> None:
    frame = protocol.encode_move_axes(1, {0: 0.01, 2: 0.02})
    assert frame[2] == 2
    assert len(frame) == 3 + 2 * 5


def test_encode_move_pose() -> None:
    frame = protocol.encode_move_pose(3, "open")
    assert frame[0] == 3
    assert frame[1] == protocol.Opcode.MOVE_POSE
    assert frame[2] == 4
    assert frame[3:] == b"open"


def test_encode_set_speed_and_effort() -> None:
    speed_frame = protocol.encode_set_speed(1, 0, 0.15)
    assert speed_frame[1] == protocol.Opcode.SET_SPEED
    assert struct.unpack_from("<f", speed_frame, 3)[0] == pytest.approx(0.15)
    effort_frame = protocol.encode_set_effort(2, 0, 80.0)
    assert effort_frame[1] == protocol.Opcode.SET_EFFORT
    assert struct.unpack_from("<f", effort_frame, 3)[0] == pytest.approx(80.0)


def test_encode_simple_commands() -> None:
    for opcode in (protocol.Opcode.STOP, protocol.Opcode.PING, protocol.Opcode.IDENTIFY):
        frame = protocol.encode_simple_command(9, opcode)
        assert frame == bytes([9, opcode])


def test_decode_event_ack_done() -> None:
    assert protocol.decode_event(bytes([protocol.EventType.ACK, 5])) == protocol.Event(protocol.EventType.ACK, 5)
    assert protocol.decode_event(bytes([protocol.EventType.DONE, 5])) == protocol.Event(protocol.EventType.DONE, 5)


def test_decode_event_failed_with_reason() -> None:
    event = protocol.decode_event(bytes([protocol.EventType.FAILED, 5, protocol.FailureReason.PREEMPTED]))
    assert event.type == protocol.EventType.FAILED
    assert event.reason == protocol.FailureReason.PREEMPTED


def test_decode_event_rejects_malformed_frames() -> None:
    with pytest.raises(protocol.ProtocolError):
        protocol.decode_event(b"\x01")  # too short
    with pytest.raises(protocol.ProtocolError):
        protocol.decode_event(bytes([0xEE, 1]))  # unknown type
    with pytest.raises(protocol.ProtocolError):
        protocol.decode_event(bytes([protocol.EventType.FAILED, 1]))  # FAILED without reason


def test_decode_state() -> None:
    frame = bytes([0x03, 2]) + struct.pack("<2f", 0.01, 0.02)
    state = protocol.decode_state(frame)
    assert state.moving is True
    assert state.grasped is True
    assert state.positions == (pytest.approx(0.01), pytest.approx(0.02))


def test_decode_state_rejects_truncated_frame() -> None:
    with pytest.raises(protocol.ProtocolError):
        protocol.decode_state(bytes([0x00, 2]) + struct.pack("<f", 0.01))


def test_decode_sensors() -> None:
    frame = bytes([2]) + struct.pack("<2f", 12.5, 0.03)
    assert protocol.decode_sensors(frame) == (pytest.approx(12.5), pytest.approx(0.03))


def test_decode_sensors_empty() -> None:
    assert protocol.decode_sensors(bytes([0])) == ()


def test_decode_sensors_rejects_malformed_frames() -> None:
    with pytest.raises(protocol.ProtocolError):
        protocol.decode_sensors(b"")
    with pytest.raises(protocol.ProtocolError):
        protocol.decode_sensors(bytes([2]) + struct.pack("<f", 12.5))  # truncated


DESCRIPTOR_JSON = (
    b'{"protocol":1,"name":"gripper-left","deviceId":"E663E8A1C2D4F5B6","profile":"parallel",'
    b'"axes":[{"id":0,"unit":"m","min":0.0,"max":0.085,"maxSpeed":0.15}],"poses":["open","closed"],'
    b'"sensors":[{"id":0,"name":"force","unit":"N","min":0.0,"max":250.0}]}'
)


def test_descriptor_parsing() -> None:
    descriptor = protocol.GripperDescriptor.from_json(DESCRIPTOR_JSON)
    assert descriptor.name == "gripper-left"
    assert descriptor.device_id == "E663E8A1C2D4F5B6"
    assert descriptor.profile == "parallel"
    assert descriptor.axes == (protocol.AxisDescriptor(0, 0.0, 0.085, 0.15, unit="m"),)
    assert descriptor.poses == ("open", "closed")
    assert descriptor.sensors == (protocol.SensorDescriptor(0, "force", "N", 0.0, 250.0),)


def test_descriptor_without_sensors_or_axis_unit() -> None:
    # Firmware built before the sensor channel omits "sensors" and axis "unit".
    raw = (
        b'{"protocol":1,"name":"old","deviceId":"AA","profile":"parallel",'
        b'"axes":[{"id":0,"min":0.0,"max":0.085,"maxSpeed":0.15}],"poses":[]}'
    )
    descriptor = protocol.GripperDescriptor.from_json(raw)
    assert descriptor.sensors == ()
    assert descriptor.axes[0].unit == ""


def test_descriptor_rejects_wrong_protocol_version() -> None:
    with pytest.raises(protocol.ProtocolError):
        protocol.GripperDescriptor.from_json(DESCRIPTOR_JSON.replace(b'"protocol":1', b'"protocol":2'))


def test_descriptor_rejects_malformed_json() -> None:
    with pytest.raises(protocol.ProtocolError):
        protocol.GripperDescriptor.from_json(b"not json")
    with pytest.raises(protocol.ProtocolError):
        protocol.GripperDescriptor.from_json(b'{"protocol":1}')
