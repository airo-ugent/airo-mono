"""Zenoh-based writer for inter-process frame buffer communication."""

import atexit
import dataclasses
from typing import Any

import zenoh
from loguru import logger

# Number of in-flight SHM frames to keep resident in the pool.
# A larger pool means slow consumers are less likely to stall the publisher.
_POOL_FRAMES = 32
# Minimum SHM pool size regardless of frame size.
_MIN_POOL_BYTES = 64 * 1024 * 1024  # 64 MB


class ZenohWriter:
    """Publishes frame buffer dataclass instances over a Zenoh key expression.

    Each frame is allocated from a :class:`zenoh.shm.ShmProvider` pool and
    serialized into shared memory field-by-field.  The Zenoh transport then
    hands the SHM reference to subscribers without an additional copy, giving
    the same single-copy behaviour as ``airo_ipc``'s ``SMWriter``.

    Per-field copying (``arr.tobytes()`` followed by a SHM slice assignment)
    keeps each field's bytes hot in CPU cache between the two copies, which is
    measurably faster than first concatenating everything into a flat buffer.

    If the SHM pool is momentarily full (all buffers still in flight to slow
    consumers) the frame is dropped and a warning is logged.  This matches
    Zenoh's ``CongestionControl.DROP`` semantics used on the publisher.

    Args:
        session: An open :class:`zenoh.Session`.
        key_expr: Zenoh key expression (topic name) to publish on.
        template: A template instance (from ``FrameBuffer.template()``) whose
            field layout defines the wire format and the SHM pool allocation
            size.
    """

    def __init__(self, session: zenoh.Session, key_expr: str, template: Any) -> None:
        self._field_names = [f.name for f in dataclasses.fields(template)]
        self._frame_size = sum(getattr(template, name).nbytes for name in self._field_names)
        pool_size = max(self._frame_size * _POOL_FRAMES, _MIN_POOL_BYTES)
        self._provider = zenoh.shm.ShmProvider.default_backend(pool_size)
        self._publisher = session.declare_publisher(
            key_expr,
            congestion_control=zenoh.CongestionControl.DROP,
        )
        self._key_expr = key_expr
        atexit.register(self.stop)

    def __call__(self, msg: Any) -> None:
        """Serialize *msg* into an SHM buffer and publish it.

        Each numpy field is copied directly from the field's memory into the
        SHM buffer, keeping it in CPU cache between the two operations.

        Args:
            msg: A frame buffer dataclass instance (same type as the template).
        """
        try:
            buf = self._provider.alloc(self._frame_size, zenoh.shm.GarbageCollect())
        except zenoh.ZError:
            logger.warning(f"ZenohWriter: SHM pool full, dropping frame on '{self._key_expr}'")
            return
        offset = 0
        for name in self._field_names:
            field_bytes = getattr(msg, name).tobytes()
            n = len(field_bytes)
            buf[offset : offset + n] = field_bytes
            offset += n
        self._publisher.put(zenoh.ZBytes(buf))

    def stop(self) -> None:
        """Undeclare the publisher and release its resources."""
        try:
            self._publisher.undeclare()
        except Exception:
            pass
        atexit.unregister(self.stop)
