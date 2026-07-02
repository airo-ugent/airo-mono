"""Zenoh-based writer for inter-process frame buffer communication."""

import atexit
from typing import Any

import zenoh
from airo_camera_toolkit.cameras.multiprocess.frame_data import serialize_frame


class ZenohWriter:
    """Publishes frame buffer dataclass instances over a Zenoh key expression.

    This is a drop-in replacement for ``airo_ipc``'s ``SMWriter``.  The payload
    is the concatenation of all numpy field bytes (see :func:`serialize_frame`).
    With a Zenoh session configured to use SHM transport the bytes are placed
    directly in shared memory, avoiding an extra copy between processes on the
    same host.

    Args:
        session: An open :class:`zenoh.Session`.
        key_expr: Zenoh key expression (topic name) to publish on.
        template: A template instance (from ``FrameBuffer.template()``) whose
            field layout defines the wire format.  Not used at runtime but kept
            for symmetry with :class:`ZenohReader`.
    """

    def __init__(self, session: zenoh.Session, key_expr: str, template: Any) -> None:
        self._publisher = session.declare_publisher(key_expr)
        self._template = template
        atexit.register(self.stop)

    def __call__(self, msg: Any) -> None:
        """Serialize *msg* and publish it.

        Args:
            msg: A frame buffer dataclass instance (same type as the template).
        """
        self._publisher.put(serialize_frame(msg))

    def stop(self) -> None:
        """Undeclare the publisher and release its resources."""
        try:
            self._publisher.undeclare()
        except Exception:
            pass
        atexit.unregister(self.stop)
