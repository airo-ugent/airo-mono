"""Tests for the Halberd BLE client and gripper drivers against an in-memory fake
transport that simulates the HalberdGripper firmware (ACK/DONE lifecycle, state
snapshots), so no Bluetooth hardware is needed."""

import struct
from typing import Callable, Optional

import pytest
from airo_robots.awaitable_action import ACTION_STATUS_ENUM
from airo_robots.grippers.hardware.halberd_ble import protocol
from airo_robots.grippers.hardware.halberd_ble.client import (
    AmbiguousGripperError,
    CommandStatus,
    HalberdAdvertisement,
    HalberdClient,
    HalberdConnectionError,
    select_advertisement,
)
from airo_robots.grippers.hardware.halberd_ble.gripper import GenericHalberdGripper, HalberdBLEGripper

PARALLEL_DESCRIPTOR = (
    b'{"protocol":1,"name":"gripper-left","deviceId":"E663E8A1C2D4F5B6","profile":"parallel",'
    b'"axes":[{"id":0,"unit":"m","min":0.0,"max":0.085,"maxSpeed":0.15}],"poses":["open","closed"],'
    b'"sensors":[{"id":0,"name":"force","unit":"N","min":0.0,"max":250.0}]}'
)

EXOTIC_DESCRIPTOR = (
    b'{"protocol":1,"name":"tentacle","deviceId":"AA00","profile":"generic",'
    b'"axes":[{"id":0,"min":0.0,"max":1.0,"maxSpeed":1.0},{"id":1,"min":-3.14,"max":3.14,"maxSpeed":2.0}],'
    b'"poses":["open","closed","curl"]}'
)


class FakeHalberdTransport:
    """In-memory HalberdTransport simulating the firmware.

    Motion commands (MOVE_AXES/MOVE_POSE) are ACKed immediately and stay in flight until
    the test calls ``complete_motion()``. Other commands complete immediately.
    """

    def __init__(self, descriptor_json: bytes = PARALLEL_DESCRIPTOR, initial_positions: tuple = (0.085,)) -> None:
        self.descriptor_json = descriptor_json
        self.positions = list(initial_positions)
        self.sent_frames: list[bytes] = []
        self.pending_motion_seq: Optional[int] = None
        self._event_listener: Callable[[bytes], None] = lambda frame: None
        self._state_listener: Callable[[bytes], None] = lambda frame: None
        self._sensor_listener: Callable[[bytes], None] = lambda frame: None
        self._disconnect_listener: Callable[[], None] = lambda: None

    # HalberdTransport interface -----------------------------------------

    def read_descriptor(self) -> bytes:
        return self.descriptor_json

    def write_command(self, frame: bytes) -> None:
        self.sent_frames.append(frame)
        seq, opcode = frame[0], frame[1]
        if opcode in (protocol.Opcode.MOVE_AXES, protocol.Opcode.MOVE_POSE):
            if self.pending_motion_seq is not None:
                self.emit_event(protocol.EventType.FAILED, self.pending_motion_seq, protocol.FailureReason.PREEMPTED)
            self.pending_motion_seq = seq
            self.emit_event(protocol.EventType.ACK, seq)
        else:
            self.emit_event(protocol.EventType.ACK, seq)
            self.emit_event(protocol.EventType.DONE, seq)

    def set_event_listener(self, listener: Callable[[bytes], None]) -> None:
        self._event_listener = listener

    def set_state_listener(self, listener: Callable[[bytes], None]) -> None:
        self._state_listener = listener
        self.emit_state()  # the firmware pushes a snapshot when a central connects

    def set_sensor_listener(self, listener: Callable[[bytes], None]) -> None:
        self._sensor_listener = listener

    def set_disconnect_listener(self, listener: Callable[[], None]) -> None:
        self._disconnect_listener = listener

    def disconnect(self) -> None:
        self._disconnect_listener()

    # test helpers --------------------------------------------------------

    def emit_event(self, event_type: protocol.EventType, seq: int, reason: Optional[int] = None) -> None:
        frame = bytes([event_type, seq]) + (bytes([reason]) if reason is not None else b"")
        self._event_listener(frame)

    def emit_state(self, moving: bool = False, grasped: bool = False) -> None:
        flags = (0x01 if moving else 0) | (0x02 if grasped else 0)
        frame = bytes([flags, len(self.positions)]) + struct.pack(f"<{len(self.positions)}f", *self.positions)
        self._state_listener(frame)

    def emit_sensors(self, values: tuple) -> None:
        frame = bytes([len(values)]) + struct.pack(f"<{len(values)}f", *values)
        self._sensor_listener(frame)

    def complete_motion(self, final_positions: Optional[tuple] = None) -> None:
        assert self.pending_motion_seq is not None, "no motion in flight"
        if final_positions is not None:
            self.positions = list(final_positions)
        self.emit_state()
        self.emit_event(protocol.EventType.DONE, self.pending_motion_seq)
        self.pending_motion_seq = None

    def fail_motion(self, reason: protocol.FailureReason) -> None:
        assert self.pending_motion_seq is not None, "no motion in flight"
        self.emit_event(protocol.EventType.FAILED, self.pending_motion_seq, reason)
        self.pending_motion_seq = None

    def drop_connection(self) -> None:
        self._disconnect_listener()


# ---------------------------------------------------------------------------
# advertisement selection (duplicate-name rule)
# ---------------------------------------------------------------------------

LEFT = HalberdAdvertisement(name="gripper-left", address="AA:BB", rssi=-50)
RIGHT = HalberdAdvertisement(name="gripper-right", address="CC:DD", rssi=-60)
LEFT_CLONE = HalberdAdvertisement(name="gripper-left", address="EE:FF", rssi=-70)


def test_select_by_name() -> None:
    assert select_advertisement([LEFT, RIGHT], name="gripper-left") == LEFT


def test_select_single_device_without_filter() -> None:
    assert select_advertisement([RIGHT]) == RIGHT


def test_select_fails_loudly_on_duplicate_names() -> None:
    with pytest.raises(AmbiguousGripperError) as exc_info:
        select_advertisement([LEFT, RIGHT, LEFT_CLONE], name="gripper-left")
    assert exc_info.value.candidates == [LEFT, LEFT_CLONE]


def test_select_fails_on_multiple_without_filter() -> None:
    with pytest.raises(AmbiguousGripperError):
        select_advertisement([LEFT, RIGHT])


def test_select_by_address_disambiguates() -> None:
    assert select_advertisement([LEFT, LEFT_CLONE], address="ee:ff") == LEFT_CLONE


def test_select_fails_when_nothing_matches() -> None:
    with pytest.raises(HalberdConnectionError):
        select_advertisement([LEFT], name="gripper-right")


# ---------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------


def test_client_reads_descriptor() -> None:
    client = HalberdClient(FakeHalberdTransport())
    assert client.descriptor.name == "gripper-left"
    assert client.descriptor.profile == "parallel"


def test_client_motion_lifecycle() -> None:
    transport = FakeHalberdTransport()
    client = HalberdClient(transport)
    seq = client.move_axes({0: 0.02})
    assert client.command_status(seq) == CommandStatus.ACKED
    assert not client.is_command_terminal(seq)
    transport.complete_motion(final_positions=(0.02,))
    assert client.is_command_terminal(seq)
    assert client.command_status(seq) == CommandStatus.DONE
    assert client.get_state().positions[0] == pytest.approx(0.02)


def test_client_failed_motion_is_terminal() -> None:
    transport = FakeHalberdTransport()
    client = HalberdClient(transport)
    seq = client.move_pose("open")
    transport.fail_motion(protocol.FailureReason.REJECTED)
    assert client.is_command_terminal(seq)
    assert client.command_status(seq) == CommandStatus.FAILED


def test_client_synchronous_setters_complete() -> None:
    transport = FakeHalberdTransport()
    client = HalberdClient(transport)
    client.set_speed(0, 0.1)
    client.set_effort(0, 50.0)
    client.ping()
    assert len(transport.sent_frames) == 3


def test_client_disconnect_fails_fast() -> None:
    transport = FakeHalberdTransport()
    client = HalberdClient(transport)
    seq = client.move_axes({0: 0.01})
    transport.drop_connection()
    assert client.is_command_terminal(seq)  # pending command released
    with pytest.raises(HalberdConnectionError):
        client.get_state()
    with pytest.raises(HalberdConnectionError):
        client.move_axes({0: 0.02})


def test_client_sensor_values() -> None:
    transport = FakeHalberdTransport()
    client = HalberdClient(transport)
    with pytest.raises(HalberdConnectionError):  # sensors declared, but no snapshot yet
        client.get_sensor_values()
    transport.emit_sensors((42.5,))
    assert client.get_sensor_values() == (pytest.approx(42.5),)


def test_client_sensor_values_empty_when_none_declared() -> None:
    transport = FakeHalberdTransport(descriptor_json=EXOTIC_DESCRIPTOR, initial_positions=(0.0, 0.0))
    client = HalberdClient(transport)
    assert client.get_sensor_values() == ()


# ---------------------------------------------------------------------------
# parallel gripper driver
# ---------------------------------------------------------------------------


def make_gripper() -> tuple[HalberdBLEGripper, FakeHalberdTransport]:
    transport = FakeHalberdTransport()
    return HalberdBLEGripper(HalberdClient(transport)), transport


def test_gripper_specs_from_descriptor() -> None:
    gripper, _ = make_gripper()
    assert gripper.gripper_specs.max_width == pytest.approx(0.085)
    assert gripper.gripper_specs.min_width == pytest.approx(0.0)
    assert gripper.gripper_specs.max_speed == pytest.approx(0.15)


def test_gripper_rejects_non_parallel_profile() -> None:
    with pytest.raises(ValueError):
        HalberdBLEGripper(HalberdClient(FakeHalberdTransport(descriptor_json=EXOTIC_DESCRIPTOR)))


def test_gripper_move_returns_awaitable_that_completes() -> None:
    gripper, transport = make_gripper()
    action = gripper.move(0.02)
    assert not action.is_action_done()
    transport.complete_motion(final_positions=(0.02,))
    assert action.wait(timeout=1.0) == ACTION_STATUS_ENUM.SUCCEEDED
    assert gripper.get_current_width() == pytest.approx(0.02)


def test_gripper_move_clips_width_to_specs() -> None:
    gripper, transport = make_gripper()
    gripper.move(1.0)  # way beyond max_width
    frame = transport.sent_frames[-1]
    target = struct.unpack_from("<f", frame, 4)[0]
    assert target == pytest.approx(0.085)


def test_gripper_open_close_use_named_poses() -> None:
    gripper, transport = make_gripper()
    gripper.open()
    assert transport.sent_frames[-1][1] == protocol.Opcode.MOVE_POSE
    assert transport.sent_frames[-1][3:] == b"open"
    transport.complete_motion()
    gripper.close()
    assert transport.sent_frames[-1][3:] == b"closed"


def test_gripper_move_with_speed_and_force() -> None:
    gripper, transport = make_gripper()
    gripper.move(0.03, speed=0.1, force=50.0)
    opcodes = [frame[1] for frame in transport.sent_frames]
    assert opcodes == [protocol.Opcode.SET_SPEED, protocol.Opcode.SET_EFFORT, protocol.Opcode.MOVE_AXES]
    assert gripper.speed == pytest.approx(0.1)
    assert gripper.max_grasp_force == pytest.approx(50.0)


def test_gripper_grasp_flag() -> None:
    gripper, transport = make_gripper()
    assert gripper.is_an_object_grasped() is False
    transport.emit_state(grasped=True)
    assert gripper.is_an_object_grasped() is True


def test_gripper_sensors() -> None:
    gripper, transport = make_gripper()
    assert [sensor.name for sensor in gripper.sensors] == ["force"]
    assert gripper.sensors[0].unit == "N"
    transport.emit_sensors((17.0,))
    assert gripper.sensor_values == (pytest.approx(17.0),)
    assert gripper.get_sensor("force") == pytest.approx(17.0)
    with pytest.raises(KeyError):
        gripper.get_sensor("pressure")


# ---------------------------------------------------------------------------
# generic (exotic) gripper driver
# ---------------------------------------------------------------------------


def test_generic_gripper_exposes_axes_and_poses() -> None:
    transport = FakeHalberdTransport(descriptor_json=EXOTIC_DESCRIPTOR, initial_positions=(0.0, 0.0))
    gripper = GenericHalberdGripper(HalberdClient(transport))
    assert len(gripper.axes) == 2
    assert "curl" in gripper.poses


def test_generic_gripper_multi_axis_move() -> None:
    transport = FakeHalberdTransport(descriptor_json=EXOTIC_DESCRIPTOR, initial_positions=(0.0, 0.0))
    gripper = GenericHalberdGripper(HalberdClient(transport))
    action = gripper.move_axes({0: 0.5, 1: 1.57})
    transport.complete_motion(final_positions=(0.5, 1.57))
    assert action.wait(timeout=1.0) == ACTION_STATUS_ENUM.SUCCEEDED
    assert gripper.positions == (pytest.approx(0.5), pytest.approx(1.57))


def test_generic_gripper_named_pose() -> None:
    transport = FakeHalberdTransport(descriptor_json=EXOTIC_DESCRIPTOR, initial_positions=(0.0, 0.0))
    gripper = GenericHalberdGripper(HalberdClient(transport))
    action = gripper.move_pose("curl")
    transport.complete_motion()
    assert action.wait(timeout=1.0) == ACTION_STATUS_ENUM.SUCCEEDED
