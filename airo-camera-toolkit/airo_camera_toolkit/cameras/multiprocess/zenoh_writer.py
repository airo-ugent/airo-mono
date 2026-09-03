"""Zenoh-based writer for inter-process frame buffer communication."""

import atexit
from typing import Any

import zenoh
from airo_camera_toolkit.cameras.multiprocess.frame_data import frame_field_specs, validate_frame
from loguru import logger

# Upper bound on the shared memory pool of a single writer.  The number of
# in-flight frames the pool can hold is derived from this and the frame size, so
# small (metadata) messages get a deep pool while multi-megabyte frames do not
# each reserve hundreds of megabytes of /dev/shm.
_MAX_POOL_BYTES = 128 * 1024 * 1024  # 128 MB
# Bounds on the derived pool depth.  A deeper pool means slow consumers are less
# likely to stall the publisher; two frames is the minimum for double buffering.
_MIN_POOL_FRAMES = 2
_MAX_POOL_FRAMES = 32
# The default (Talc) backend needs at least a page, and its bookkeeping costs
# roughly one allocation's worth of space, so pools are sized with one frame of
# slack and never go below this floor.
_MIN_POOL_BYTES = 1024 * 1024  # 1 MB


def _pool_size(frame_size: int, max_pool_bytes: int = _MAX_POOL_BYTES) -> int:
    """Return the SHM pool size in bytes for a writer publishing *frame_size* frames.

    Args:
        frame_size: Size of a single serialized frame in bytes.
        max_pool_bytes: Soft upper bound on the frames kept resident in the pool.

    Returns:
        The pool size to request from the SHM provider: room for the derived
        number of in-flight frames plus one frame of allocator overhead, and at
        least :data:`_MIN_POOL_BYTES`.
    """
    pool_frames = min(max(max_pool_bytes // frame_size, _MIN_POOL_FRAMES), _MAX_POOL_FRAMES)
    return max((pool_frames + 1) * frame_size, _MIN_POOL_BYTES)


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
        self._template = template
        self._field_specs = frame_field_specs(template)
        self._frame_size = sum(nbytes for _, _, _, nbytes in self._field_specs)
        self._provider = zenoh.shm.ShmProvider.default_backend(_pool_size(self._frame_size))
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

        Raises:
            TypeError: If *msg* is not of the template's type or has a
                non-array field.
            ValueError: If a field's dtype or shape differs from the template.
                The wire format is a bare byte concatenation, so publishing
                such a frame would silently corrupt it for every receiver.
        """
        # Validate before allocating: the receiver has no way to detect a field
        # whose dtype or shape drifted from the template.
        validate_frame(self._template, msg)

        try:
            buf = self._provider.alloc(self._frame_size, zenoh.shm.GarbageCollect())
        except zenoh.ZError:
            logger.warning(f"ZenohWriter: SHM pool full, dropping frame on '{self._key_expr}'")
            return
        # The validation above guarantees the field bytes sum to _frame_size, so
        # the buffer is written completely.  That matters because buffers come
        # from a recycled pool: an unfilled tail would publish bytes left behind
        # by an earlier frame.
        offset = 0
        for name, _, _, nbytes in self._field_specs:
            buf[offset : offset + nbytes] = getattr(msg, name).tobytes()
            offset += nbytes
        self._publisher.put(zenoh.ZBytes(buf))

    def stop(self) -> None:
        """Undeclare the publisher and release its resources."""
        try:
            self._publisher.undeclare()
        except Exception as e:  # pragma: no cover - shutdown-only path
            logger.debug(f"ZenohWriter '{self._key_expr}': error while undeclaring publisher: {e}")
        atexit.unregister(self.stop)
