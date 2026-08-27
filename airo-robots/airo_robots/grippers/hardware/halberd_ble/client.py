"""BLE client for Halberd grippers speaking the Airo Gripper Protocol (AGP).

Split in two layers so the protocol logic is testable without Bluetooth hardware:
- :class:`HalberdTransport`: the byte-level seam (read descriptor, write commands, deliver
  notifications). :class:`BleakTransport` is the real implementation on top of ``bleak``;
  tests substitute an in-memory fake.
- :class:`HalberdClient`: command sequencing, ACK/DONE/FAILED bookkeeping and the state
  cache, written against the transport seam.

The gripper interface in airo-robots is synchronous while ``bleak`` is asyncio-only, so
``BleakTransport`` runs a private event loop on a daemon thread and marshals every call
onto it.

Install the BLE dependency with ``pip install "airo-robots[halberd]"``.
"""

import asyncio
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Coroutine, Optional, Protocol, TypeVar

from airo_robots.grippers.hardware.halberd_ble import protocol
from loguru import logger

try:
    from bleak import BleakClient, BleakScanner

    BLEAK_INSTALLED = True
except ImportError:
    BLEAK_INSTALLED = False

BLEAK_INSTALL_HINT = (
    'The Halberd BLE gripper requires the bleak package. Install it with `pip install "airo-robots[halberd]"`.'
)

T = TypeVar("T")


class HalberdTransport(Protocol):
    """Byte-level transport to one connected Halberd gripper."""

    def read_descriptor(self) -> bytes:
        """Read the raw descriptor characteristic value."""
        ...

    def write_command(self, frame: bytes) -> None:
        """Write one command frame to the command characteristic (write-with-response)."""
        ...

    def set_event_listener(self, listener: Callable[[bytes], None]) -> None:
        """Register the callback receiving raw event frames."""
        ...

    def set_state_listener(self, listener: Callable[[bytes], None]) -> None:
        """Register the callback receiving raw state frames."""
        ...

    def set_sensor_listener(self, listener: Callable[[bytes], None]) -> None:
        """Register the callback receiving raw sensor frames."""
        ...

    def set_disconnect_listener(self, listener: Callable[[], None]) -> None:
        """Register the callback invoked when the connection drops."""
        ...

    def disconnect(self) -> None:
        """Close the connection."""
        ...


class CommandStatus(Enum):
    PENDING = "pending"
    ACKED = "acked"
    DONE = "done"
    FAILED = "failed"


@dataclass
class _PendingCommand:
    status: CommandStatus = CommandStatus.PENDING
    reason: Optional[protocol.FailureReason] = None
    acked: threading.Event = None  # type: ignore[assignment]
    terminal: threading.Event = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.acked = threading.Event()
        self.terminal = threading.Event()


@dataclass(frozen=True)
class HalberdAdvertisement:
    """One gripper seen during discovery."""

    name: str
    address: str
    rssi: int


class HalberdConnectionError(ConnectionError):
    """Raised when a gripper cannot be (or is no longer) reached."""


class AmbiguousGripperError(HalberdConnectionError):
    """Raised when a discovery filter matches more than one gripper."""

    def __init__(self, candidates: list[HalberdAdvertisement]) -> None:
        listing = ", ".join(f"{adv.name} ({adv.address}, {adv.rssi} dBm)" for adv in candidates)
        super().__init__(
            f"Multiple Halberd grippers match: {listing}. Give each gripper a unique name in its sketch "
            f"(gripper.begin(name)) or connect by address."
        )
        self.candidates = candidates


def select_advertisement(
    advertisements: list[HalberdAdvertisement],
    name: Optional[str] = None,
    address: Optional[str] = None,
) -> HalberdAdvertisement:
    """Select exactly one gripper from discovery results, failing loudly on ambiguity.

    Args:
        advertisements: discovery results.
        name: the user-assigned gripper name to match (primary identity).
        address: BLE address to match (escape hatch, checked before name).

    Raises:
        HalberdConnectionError: if nothing matches.
        AmbiguousGripperError: if more than one gripper matches the filter.
    """
    if address is not None:
        candidates = [adv for adv in advertisements if adv.address.lower() == address.lower()]
    elif name is not None:
        candidates = [adv for adv in advertisements if adv.name == name]
    else:
        candidates = list(advertisements)

    if not candidates:
        filter_description = f"name={name!r}" if name else f"address={address!r}" if address else "no filter"
        raise HalberdConnectionError(
            f"No Halberd gripper found ({filter_description}). Is it powered, in range and not "
            f"already connected to another client?"
        )
    if len(candidates) > 1:
        raise AmbiguousGripperError(candidates)
    return candidates[0]


class HalberdClient:
    """AGP command/state bookkeeping on top of a :class:`HalberdTransport`.

    Reads and validates the descriptor on construction and tracks command lifecycles:
    every command frame gets an ACK, then motion commands later get exactly one
    DONE/FAILED. ``AwaitableAction`` termination conditions poll
    :meth:`is_command_terminal`.
    """

    def __init__(self, transport: HalberdTransport, first_state_timeout: float = 3.0) -> None:
        self._transport = transport
        self._lock = threading.Lock()
        self._seq = 0
        self._pending: dict[int, _PendingCommand] = {}
        self._state: Optional[protocol.GripperState] = None
        self._sensor_values: Optional[tuple[float, ...]] = None
        self._first_state = threading.Event()
        self._connected = True

        transport.set_event_listener(self._handle_event_frame)
        transport.set_state_listener(self._handle_state_frame)
        transport.set_sensor_listener(self._handle_sensor_frame)
        transport.set_disconnect_listener(self._handle_disconnect)

        self._descriptor = protocol.GripperDescriptor.from_json(transport.read_descriptor())

        # The firmware pushes a state snapshot right after a central connects.
        if not self._first_state.wait(timeout=first_state_timeout):
            logger.warning("No state snapshot received from gripper yet; positions unavailable until one arrives.")

    @property
    def descriptor(self) -> protocol.GripperDescriptor:
        return self._descriptor

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_state(self) -> protocol.GripperState:
        """The last received state snapshot.

        Raises:
            HalberdConnectionError: if disconnected or no snapshot was ever received.
        """
        self._assert_connected()
        if self._state is None:
            raise HalberdConnectionError("No state snapshot received from the gripper yet.")
        return self._state

    def get_sensor_values(self) -> tuple[float, ...]:
        """The last received sensor snapshot, in ascending sensor-id order.

        The firmware only notifies when a sensor value changes, so the first snapshot
        arrives after the sketch first reports a value different from the sensor minimum.

        Raises:
            HalberdConnectionError: if disconnected, or if the gripper declares sensors but
                no snapshot was received yet. Returns ``()`` when no sensors are declared.
        """
        self._assert_connected()
        if not self._descriptor.sensors:
            return ()
        if self._sensor_values is None:
            raise HalberdConnectionError("No sensor snapshot received from the gripper yet.")
        return self._sensor_values

    def move_axes(self, targets: dict[int, float], ack_timeout: float = 2.0) -> int:
        """Send MOVE_AXES; returns the command seq after the gripper ACKs it."""
        return self._send_and_wait_ack(lambda seq: protocol.encode_move_axes(seq, targets), ack_timeout)

    def move_pose(self, pose_name: str, ack_timeout: float = 2.0) -> int:
        """Send MOVE_POSE; returns the command seq after the gripper ACKs it."""
        return self._send_and_wait_ack(lambda seq: protocol.encode_move_pose(seq, pose_name), ack_timeout)

    def set_speed(self, axis_id: int, speed: float, timeout: float = 2.0) -> None:
        """Send SET_SPEED and wait for completion (setters are synchronous in airo-robots)."""
        seq = self._send_and_wait_ack(lambda s: protocol.encode_set_speed(s, axis_id, speed), timeout)
        self._wait_terminal(seq, timeout)

    def set_effort(self, axis_id: int, effort: float, timeout: float = 2.0) -> None:
        """Send SET_EFFORT and wait for completion."""
        seq = self._send_and_wait_ack(lambda s: protocol.encode_set_effort(s, axis_id, effort), timeout)
        self._wait_terminal(seq, timeout)

    def stop(self, timeout: float = 2.0) -> None:
        """Send STOP (preempts any in-flight motion) and wait for completion."""
        seq = self._send_and_wait_ack(lambda s: protocol.encode_simple_command(s, protocol.Opcode.STOP), timeout)
        self._wait_terminal(seq, timeout)

    def ping(self, timeout: float = 2.0) -> None:
        """Round-trip a PING; raises on timeout/disconnect."""
        seq = self._send_and_wait_ack(lambda s: protocol.encode_simple_command(s, protocol.Opcode.PING), timeout)
        self._wait_terminal(seq, timeout)

    def identify(self, timeout: float = 2.0) -> None:
        """Make the gripper blink its LED to identify it physically."""
        seq = self._send_and_wait_ack(lambda s: protocol.encode_simple_command(s, protocol.Opcode.IDENTIFY), timeout)
        self._wait_terminal(seq, timeout)

    def command_status(self, seq: int) -> CommandStatus:
        with self._lock:
            pending = self._pending.get(seq)
        if pending is None:
            raise KeyError(f"Unknown command seq {seq}")
        return pending.status

    def is_command_terminal(self, seq: int) -> bool:
        """True once the command reached DONE or FAILED (FAILED is logged when it arrives)."""
        if not self._connected:
            return True  # fail fast: don't let AwaitableActions spin until timeout
        return self.command_status(seq) in (CommandStatus.DONE, CommandStatus.FAILED)

    def disconnect(self) -> None:
        self._transport.disconnect()
        self._handle_disconnect()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _assert_connected(self) -> None:
        if not self._connected:
            raise HalberdConnectionError("The BLE connection to the gripper was lost.")

    def _next_seq(self) -> int:
        with self._lock:
            self._seq = (self._seq + 1) % 256
            self._pending[self._seq] = _PendingCommand()
            return self._seq

    def _send_and_wait_ack(self, encode: Callable[[int], bytes], ack_timeout: float) -> int:
        self._assert_connected()
        seq = self._next_seq()
        self._transport.write_command(encode(seq))
        pending = self._pending[seq]
        if not pending.acked.wait(timeout=ack_timeout):
            self._assert_connected()
            raise HalberdConnectionError(f"Gripper did not acknowledge command (seq {seq}) in {ack_timeout} s.")
        if pending.status == CommandStatus.FAILED:
            raise HalberdConnectionError(f"Gripper rejected command (seq {seq}): {pending.reason}.")
        return seq

    def _wait_terminal(self, seq: int, timeout: float) -> None:
        pending = self._pending[seq]
        if not pending.terminal.wait(timeout=timeout):
            self._assert_connected()
            raise HalberdConnectionError(f"Gripper did not complete command (seq {seq}) in {timeout} s.")
        if pending.status == CommandStatus.FAILED:
            raise HalberdConnectionError(f"Gripper command (seq {seq}) failed: {pending.reason}.")

    def _handle_event_frame(self, frame: bytes) -> None:
        try:
            event = protocol.decode_event(bytes(frame))
        except protocol.ProtocolError as error:
            logger.warning(f"Ignoring malformed event frame: {error}")
            return
        if event.type in (protocol.EventType.GRASPED, protocol.EventType.RELEASED):
            logger.info(
                f"Gripper reports object {'grasped' if event.type == protocol.EventType.GRASPED else 'released'}."
            )
            return
        with self._lock:
            pending = self._pending.get(event.seq)
        if pending is None:
            logger.debug(f"Event for unknown command seq {event.seq}: {event.type.name}")
            return
        if event.type == protocol.EventType.ACK:
            pending.status = CommandStatus.ACKED
            pending.acked.set()
        elif event.type == protocol.EventType.DONE:
            pending.status = CommandStatus.DONE
            pending.acked.set()
            pending.terminal.set()
        elif event.type == protocol.EventType.FAILED:
            pending.status = CommandStatus.FAILED
            pending.reason = event.reason
            logger.warning(f"Gripper command (seq {event.seq}) failed: {event.reason}.")
            pending.acked.set()
            pending.terminal.set()

    def _handle_state_frame(self, frame: bytes) -> None:
        try:
            self._state = protocol.decode_state(bytes(frame))
        except protocol.ProtocolError as error:
            logger.warning(f"Ignoring malformed state frame: {error}")
            return
        self._first_state.set()

    def _handle_sensor_frame(self, frame: bytes) -> None:
        try:
            self._sensor_values = protocol.decode_sensors(bytes(frame))
        except protocol.ProtocolError as error:
            logger.warning(f"Ignoring malformed sensor frame: {error}")

    def _handle_disconnect(self) -> None:
        if not self._connected:
            return
        self._connected = False
        # Release everything blocked on this connection so callers fail fast.
        with self._lock:
            pending_commands = list(self._pending.values())
        for pending in pending_commands:
            if not pending.terminal.is_set():
                pending.status = CommandStatus.FAILED
                pending.acked.set()
                pending.terminal.set()
        logger.warning("BLE connection to the gripper was closed.")


class BleakTransport:
    """:class:`HalberdTransport` implementation on top of ``bleak``.

    Runs a private asyncio event loop on a daemon thread; all public methods are
    synchronous and thread-safe.
    """

    def __init__(self, address: str, connect_timeout: float = 10.0) -> None:
        if not BLEAK_INSTALLED:
            raise ImportError(BLEAK_INSTALL_HINT)
        self._event_listener: Optional[Callable[[bytes], None]] = None
        self._state_listener: Optional[Callable[[bytes], None]] = None
        self._sensor_listener: Optional[Callable[[bytes], None]] = None
        self._disconnect_listener: Optional[Callable[[], None]] = None

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="halberd-ble", daemon=True)
        self._thread.start()

        self._client = BleakClient(address, disconnected_callback=self._on_bleak_disconnect)
        self._run(self._connect_and_subscribe(), timeout=connect_timeout + 5.0)

    def read_descriptor(self) -> bytes:
        return bytes(self._run(self._client.read_gatt_char(protocol.DESCRIPTOR_CHARACTERISTIC_UUID), timeout=5.0))

    def write_command(self, frame: bytes) -> None:
        self._run(
            self._client.write_gatt_char(protocol.COMMAND_CHARACTERISTIC_UUID, frame, response=True), timeout=5.0
        )

    def set_event_listener(self, listener: Callable[[bytes], None]) -> None:
        self._event_listener = listener

    def set_state_listener(self, listener: Callable[[bytes], None]) -> None:
        self._state_listener = listener

    def set_sensor_listener(self, listener: Callable[[bytes], None]) -> None:
        self._sensor_listener = listener

    def set_disconnect_listener(self, listener: Callable[[], None]) -> None:
        self._disconnect_listener = listener

    def disconnect(self) -> None:
        try:
            self._run(self._client.disconnect(), timeout=5.0)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)

    # ------------------------------------------------------------------
    # internals (loop thread)
    # ------------------------------------------------------------------

    def _run(self, coroutine: Coroutine[Any, Any, T], timeout: float) -> T:
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result(timeout=timeout)

    async def _connect_and_subscribe(self) -> None:
        await self._client.connect()
        await self._client.start_notify(protocol.EVENT_CHARACTERISTIC_UUID, self._on_event_notification)
        await self._client.start_notify(protocol.STATE_CHARACTERISTIC_UUID, self._on_state_notification)
        try:
            await self._client.start_notify(protocol.SENSOR_CHARACTERISTIC_UUID, self._on_sensor_notification)
        except Exception:  # noqa: BLE001 - firmware built before the sensor channel lacks the characteristic
            logger.debug("Gripper has no sensor characteristic (older firmware); sensor values unavailable.")

    def _on_event_notification(self, _characteristic: object, data: bytearray) -> None:
        if self._event_listener is not None:
            self._event_listener(bytes(data))

    def _on_state_notification(self, _characteristic: object, data: bytearray) -> None:
        if self._state_listener is not None:
            self._state_listener(bytes(data))

    def _on_sensor_notification(self, _characteristic: object, data: bytearray) -> None:
        if self._sensor_listener is not None:
            self._sensor_listener(bytes(data))

    def _on_bleak_disconnect(self, _client: object) -> None:
        if self._disconnect_listener is not None:
            self._disconnect_listener()


def discover(timeout: float = 5.0) -> list[HalberdAdvertisement]:
    """Scan for Halberd grippers advertising the AGP service.

    Only grippers that are not currently connected to another client advertise, so a busy
    gripper will not show up.
    """
    if not BLEAK_INSTALLED:
        raise ImportError(BLEAK_INSTALL_HINT)

    async def _scan() -> list[HalberdAdvertisement]:
        found = await BleakScanner.discover(timeout=timeout, return_adv=True, service_uuids=[protocol.SERVICE_UUID])
        return [
            HalberdAdvertisement(
                name=advertisement.local_name or device.name or "",
                address=device.address,
                rssi=advertisement.rssi,
            )
            for device, advertisement in found.values()
        ]

    return asyncio.run(_scan())


def connect(
    name: Optional[str] = None,
    address: Optional[str] = None,
    scan_timeout: float = 5.0,
) -> HalberdClient:
    """Discover and connect to exactly one Halberd gripper.

    The user-assigned gripper name (``gripper.begin(name)`` in the sketch) is the primary
    way to select a gripper. With no filter, exactly one gripper must be in range.

    Raises:
        HalberdConnectionError: if no gripper matches the filter.
        AmbiguousGripperError: if several match; give the grippers unique names.
    """
    advertisement = select_advertisement(discover(timeout=scan_timeout), name=name, address=address)
    logger.info(f"Connecting to Halberd gripper {advertisement.name!r} at {advertisement.address}.")
    return HalberdClient(BleakTransport(advertisement.address))
