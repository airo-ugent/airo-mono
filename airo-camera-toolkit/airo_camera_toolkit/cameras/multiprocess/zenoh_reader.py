"""Zenoh-based reader for inter-process frame buffer communication."""

import threading
import time
from typing import Generic

import zenoh
from airo_camera_toolkit.cameras.multiprocess.frame_data import Frame, FrameLayout
from loguru import logger

# Default time to wait for the first message before giving up.  Waiting forever
# turns a publisher that crashed, was never started, or that cannot be reached
# (e.g. multicast scouting blocked in a container) into a silent hang.
DEFAULT_FIRST_MESSAGE_TIMEOUT = 30.0


class WaitingForFirstMessageException(Exception):
    """Raised when no data has been received yet."""


class ZenohReader(Generic[Frame]):
    """Subscribes to a Zenoh key expression and deserializes the latest frame.

    The subscriber callback runs in a Zenoh-managed background thread and only
    stores a reference to the latest payload, so it keeps up with the publisher
    even when the consuming thread is busy: Zenoh queues samples without bound
    for a callback that falls behind. The payload is copied out of Zenoh (which
    cannot be avoided, since Zenoh payloads do not expose the buffer protocol)
    only when it is read, so frames that are never read are never copied.
    A :class:`threading.Lock` guards the latest payload so the calling thread
    can safely call :meth:`__call__` at any time.
    The constructor blocks until the first message on *key_expr* arrives.

    Call :meth:`stop` (or close the session) when done. Zenoh runs the callback on a non-daemon
    thread that Python joins *before* running ``atexit`` handlers, so a subscriber that is never
    undeclared keeps the interpreter from exiting.

    Args:
        session: An open :class:`zenoh.Session`.
        key_expr: Zenoh key expression to subscribe to.
        layout: The frame layout (e.g. from ``RGBFrameBuffer.layout()``) that
            defines the expected field shapes and dtypes used for
            deserialization.
        timeout: Maximum seconds to wait for the first message.  ``None``
            means wait indefinitely (use with care).  Defaults to
            :data:`DEFAULT_FIRST_MESSAGE_TIMEOUT`.
        warn_every: Log a warning every this many seconds while waiting.
    """

    def __init__(
        self,
        session: zenoh.Session,
        key_expr: str,
        layout: FrameLayout[Frame],
        timeout: float | None = DEFAULT_FIRST_MESSAGE_TIMEOUT,
        warn_every: int = 10,
    ) -> None:
        self._layout = layout
        self._latest_payload: zenoh.ZBytes | None = None
        # (frame count, bytes) of the payload last copied out by read(), so reading
        # the same message again does not copy it again.
        self._read_cache: tuple[int, bytes] | None = None
        self._frame_count: int = 0
        self._lock = threading.Lock()

        self._subscriber = session.declare_subscriber(key_expr, self._callback)
        try:
            self._wait_for_first_message(key_expr, timeout=timeout, warn_every=warn_every)
        except TimeoutError:
            self._subscriber.undeclare()
            raise

    @property
    def frame_count(self) -> int:
        """Number of messages received so far (monotonically increasing)."""
        with self._lock:
            return self._frame_count

    # ------------------------------------------------------------------
    # Internal helpers

    def _callback(self, sample: zenoh.Sample) -> None:
        # Keep this callback trivial: any work here lets Zenoh's per-subscriber
        # queue grow when frames arrive faster than the callback runs.
        with self._lock:
            self._latest_payload = sample.payload
            self._frame_count += 1

    def _wait_for_first_message(
        self,
        key_expr: str,
        timeout: float | None,
        warn_every: int,
    ) -> None:
        """Block until the first message is received."""
        t0 = time.time()
        warned = 0

        while True:
            with self._lock:
                if self._latest_payload is not None:
                    return

            elapsed = time.time() - t0

            if timeout is not None and elapsed >= timeout:
                raise TimeoutError(
                    f"ZenohReader '{key_expr}' timed out after {timeout}s waiting for the first message. "
                    "Is the publisher running and publishing on this key expression?"
                )

            if warn_every * (warned + 1) <= elapsed:
                warned += 1
                remaining = f"{timeout - elapsed:.0f}s remaining" if timeout is not None else "no timeout"
                logger.warning(f"ZenohReader '{key_expr}' has been waiting for {warned * warn_every}s ({remaining}).")

            time.sleep(0.01)

    # ------------------------------------------------------------------
    # Public API

    def read(self) -> tuple[Frame, int]:
        """Return the latest frame together with the message count it belongs to.

        The count is sampled under the same lock as the payload, so callers can
        use it to tell whether a later :attr:`frame_count` means a genuinely
        newer message has arrived.

        Returns:
            A ``(frame, frame_count)`` tuple.

        Raises:
            WaitingForFirstMessageException: If no data has been received yet
                (should not happen after ``__init__`` returns).
        """
        with self._lock:
            if self._latest_payload is None:
                raise WaitingForFirstMessageException("No data received yet — was the publisher started?")
            payload = self._latest_payload
            count = self._frame_count
        # Copy and deserialize outside the lock so the Zenoh callback is never blocked here.
        cache = self._read_cache
        if cache is not None and cache[0] == count:
            data = cache[1]
        else:
            data = payload.to_bytes()
            self._read_cache = (count, data)
        return self._layout.deserialize(data), count

    def __call__(self) -> Frame:
        """Return the most recently received frame, deserialized from bytes.

        Raises:
            WaitingForFirstMessageException: If no data has been received yet
                (should not happen after ``__init__`` returns).
        """
        return self.read()[0]

    def stop(self) -> None:
        """Undeclare the subscriber and release its resources."""
        try:
            self._subscriber.undeclare()
        except Exception as e:  # pragma: no cover - shutdown-only path
            logger.debug(f"ZenohReader: error while undeclaring subscriber: {e}")
