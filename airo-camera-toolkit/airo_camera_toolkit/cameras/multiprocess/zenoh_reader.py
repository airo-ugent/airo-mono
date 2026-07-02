"""Zenoh-based reader for inter-process frame buffer communication."""

import atexit
import threading
import time
from typing import Any, Optional

import zenoh
from airo_camera_toolkit.cameras.multiprocess.frame_data import deserialize_frame
from loguru import logger


class WaitingForFirstMessageException(Exception):
    """Raised when no data has been received yet."""


class ZenohReader:
    """Subscribes to a Zenoh key expression and deserializes the latest frame.

    This is a drop-in replacement for ``airo_ipc``'s ``SMReader``.  The
    subscriber callback runs in a Zenoh-managed background thread; a
    :class:`threading.Lock` guards access to the latest received bytes so the
    calling thread can safely call :meth:`__call__` at any time.

    The constructor blocks until the first message arrives (mirroring
    ``SMReader``'s ``__wait_for_writer`` behaviour).

    Args:
        session: An open :class:`zenoh.Session`.
        key_expr: Zenoh key expression to subscribe to.
        template: A template instance (from ``FrameBuffer.template()``) that
            defines the expected field shapes and dtypes used for
            deserialization.
        timeout: Maximum seconds to wait for the first message.  ``None``
            means wait indefinitely.
        warn_every: Log a warning every this many seconds while waiting.
    """

    def __init__(
        self,
        session: zenoh.Session,
        key_expr: str,
        template: Any,
        timeout: Optional[float] = None,
        warn_every: int = 60,
    ) -> None:
        self._template = template
        self._latest_bytes: Optional[bytes] = None
        self._lock = threading.Lock()

        self._subscriber = session.declare_subscriber(key_expr, self._callback)
        self._wait_for_writer(key_expr, timeout=timeout, warn_every=warn_every)
        atexit.register(self.stop)

    # ------------------------------------------------------------------
    # Internal helpers

    def _callback(self, sample: zenoh.Sample) -> None:
        with self._lock:
            self._latest_bytes = sample.payload.to_bytes()

    def _wait_for_writer(
        self,
        key_expr: str,
        timeout: Optional[float],
        warn_every: int,
    ) -> None:
        """Block until the first message is received."""
        t0 = time.time()
        warned = 0

        while True:
            with self._lock:
                if self._latest_bytes is not None:
                    return

            elapsed = time.time() - t0

            if timeout is not None and elapsed >= timeout:
                raise RuntimeError(f"ZenohReader '{key_expr}' timed out after {timeout}s waiting for first message.")

            if warn_every * (warned + 1) <= elapsed:
                warned += 1
                remaining = f"{timeout - elapsed:.0f}s remaining" if timeout is not None else "no timeout"
                logger.warning(f"ZenohReader '{key_expr}' has been waiting for {warned * warn_every}s ({remaining}).")

            time.sleep(0.01)

    # ------------------------------------------------------------------
    # Public API

    def __call__(self) -> Any:
        """Return the most recently received frame, deserialized from bytes.

        Raises:
            WaitingForFirstMessageException: If no data has been received yet
                (should not happen after ``__init__`` returns).
        """
        with self._lock:
            if self._latest_bytes is None:
                raise WaitingForFirstMessageException("No data received yet — was the publisher started?")
            return deserialize_frame(self._template, self._latest_bytes)

    def stop(self) -> None:
        """Undeclare the subscriber and release its resources."""
        try:
            self._subscriber.undeclare()
        except Exception:
            pass
        atexit.unregister(self.stop)
