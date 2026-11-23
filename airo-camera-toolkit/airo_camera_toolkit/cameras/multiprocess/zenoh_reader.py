"""Zenoh-based reader for inter-process frame buffer communication."""

import threading
from typing import Callable, Generic

import zenoh
from airo_camera_toolkit.cameras.multiprocess.frame_data import Frame, FrameLayout
from loguru import logger

# Default time to wait for the first message before giving up.  Waiting forever
# turns a publisher that crashed, was never started, or that cannot be reached
# (e.g. multicast scouting blocked in a container) into a silent hang.
DEFAULT_FIRST_MESSAGE_TIMEOUT = 30.0
# Seconds between warnings while waiting for a first message.
_WARN_EVERY = 10.0


def wait_logging_progress(wait: Callable[[float | None], bool], key_expr: str, timeout: float | None) -> None:
    """Block on *wait* until it succeeds, logging a warning every ``_WARN_EVERY`` seconds.

    Args:
        wait: Blocks for at most the given number of seconds (``None``: indefinitely) and returns
            whether the awaited message arrived.
        key_expr: Key expression being waited on, for the log and error messages.
        timeout: Maximum total seconds to wait, or ``None`` to wait indefinitely.

    Raises:
        TimeoutError: If *wait* did not succeed within *timeout*.
    """
    waited = 0.0
    while True:
        chunk = _WARN_EVERY if timeout is None else min(_WARN_EVERY, timeout - waited)
        if wait(chunk):
            return
        waited += chunk
        if timeout is not None and waited >= timeout:
            raise TimeoutError(
                f"Timed out after {timeout}s waiting for the first message on '{key_expr}'. "
                "Is the publisher running and publishing on this key expression?"
            )
        remaining = f"{timeout - waited:.0f}s remaining" if timeout is not None else "no timeout"
        logger.warning(f"Still waiting for the first message on '{key_expr}' after {waited:.0f}s ({remaining}).")


def receive_one(
    session: zenoh.Session,
    key_expr: str,
    timeout: float | None = DEFAULT_FIRST_MESSAGE_TIMEOUT,
) -> bytes:
    """Wait for the next message on *key_expr* and return its payload.

    Args:
        session: An open :class:`zenoh.Session`.
        key_expr: Zenoh key expression to subscribe to.
        timeout: Maximum seconds to wait, or ``None`` to wait indefinitely.

    Raises:
        TimeoutError: If no message arrives within *timeout*.
    """
    payloads: list[zenoh.ZBytes] = []
    received = threading.Event()

    def callback(sample: zenoh.Sample) -> None:
        payloads.append(sample.payload)
        received.set()

    subscriber = session.declare_subscriber(key_expr, callback)
    try:
        wait_logging_progress(received.wait, key_expr, timeout)
    finally:
        subscriber.undeclare()
    return payloads[0].to_bytes()


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
    """

    def __init__(
        self,
        session: zenoh.Session,
        key_expr: str,
        layout: FrameLayout[Frame],
        timeout: float | None = DEFAULT_FIRST_MESSAGE_TIMEOUT,
    ) -> None:
        self._layout = layout
        self._key_expr = key_expr
        # Replaced by the first message, which the constructor waits for.
        self._latest_payload = zenoh.ZBytes(b"")
        # (frame count, bytes) of the payload last copied out by read(), so reading
        # the same message again does not copy it again.
        self._read_cache: tuple[int, bytes] | None = None
        self._message_count: int = 0
        self._lock = threading.Lock()
        # Notified by the callback whenever a message arrives.
        self._new_message = threading.Condition(self._lock)

        self._subscriber = session.declare_subscriber(key_expr, self._callback)
        try:
            wait_logging_progress(lambda t: self.wait_for_message(after=0, timeout=t), key_expr, timeout)
        except BaseException:
            # Also on e.g. KeyboardInterrupt: a subscriber left declared keeps the interpreter from exiting.
            self._subscriber.undeclare()
            raise

    @property
    def key_expr(self) -> str:
        """The key expression this reader subscribes to."""
        return self._key_expr

    # ------------------------------------------------------------------
    # Internal helpers

    def _callback(self, sample: zenoh.Sample) -> None:
        # Keep this callback trivial: any work here lets Zenoh's per-subscriber
        # queue grow when frames arrive faster than the callback runs.
        with self._new_message:
            self._latest_payload = sample.payload
            self._message_count += 1
            self._new_message.notify_all()

    # ------------------------------------------------------------------
    # Public API

    def wait_for_message(self, after: int, timeout: float | None) -> bool:
        """Block until a message newer than message count *after* has arrived.

        Args:
            after: A message count, e.g. as returned by :meth:`read`.
            timeout: Maximum seconds to wait, or ``None`` to wait indefinitely.

        Returns:
            Whether such a message arrived within *timeout*.
        """
        with self._new_message:
            return self._new_message.wait_for(lambda: self._message_count > after, timeout)

    def read(self) -> tuple[Frame, int]:
        """Return the latest frame together with its message count.

        The count is sampled under the same lock as the payload, so it can be passed to
        :meth:`wait_for_message` to wait for a genuinely newer message.

        Returns:
            A ``(frame, message_count)`` tuple.
        """
        with self._lock:
            payload = self._latest_payload
            count = self._message_count
        # Copy and deserialize outside the lock so the Zenoh callback is never blocked here.
        cache = self._read_cache
        if cache is not None and cache[0] == count:
            data = cache[1]
        else:
            data = payload.to_bytes()
            self._read_cache = (count, data)
        return self._layout.deserialize(data), count

    def stop(self) -> None:
        """Undeclare the subscriber and release its resources."""
        try:
            self._subscriber.undeclare()
        except Exception as e:  # pragma: no cover - shutdown-only path
            logger.debug(f"ZenohReader: error while undeclaring subscriber: {e}")
