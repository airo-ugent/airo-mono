"""Unit tests for matching point clouds to frames by frame_id."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from airo_camera_toolkit.cameras.multiprocess.pointcloud_stream import read_matching_point_cloud


@dataclass
class _Message:
    frame_id: np.ndarray


class _Reader:
    """Holds only the latest message, like ZenohReader."""

    def __init__(self) -> None:
        self.count = 0
        self.latest: _Message | None = None

    def receive(self, frame_id: int) -> None:
        self.latest = _Message(np.array([frame_id], dtype=np.uint64))
        self.count += 1

    def read(self) -> tuple[Any, int]:
        return self.latest, self.count


class _Publisher:
    """Replays a scripted sequence of ``("frame" | "pointcloud", frame_id)`` publications into two readers.

    Publications in *delivered* have arrived before the receiver starts matching; the rest arrive
    only while the receiver waits for a newer message.
    """

    def __init__(self, delivered: list[tuple[str, int]], pending: list[tuple[str, int]]) -> None:
        self.readers = {"frame": _Reader(), "pointcloud": _Reader()}
        for kind, frame_id in delivered:
            self.readers[kind].receive(frame_id)
        self._pending = iter(pending)

    def wait(self, reader: _Reader, after: int) -> None:
        while reader.count <= after:
            try:
                kind, frame_id = next(self._pending)
            except StopIteration:
                raise TimeoutError("no newer message") from None
            self.readers[kind].receive(frame_id)

    def match(self) -> tuple[int, int]:
        """Match the latest frame to its point cloud; return their frame ids."""
        frame_reader = self.readers["frame"]
        frame, frame_count = frame_reader.read()
        frame, _, point_cloud = read_matching_point_cloud(
            frame_reader, frame, frame_count, self.readers["pointcloud"], self.wait  # type: ignore[arg-type]
        )
        return int(frame.frame_id.item()), int(point_cloud.frame_id.item())


def test_point_cloud_that_arrived_before_its_frame_is_used():
    publisher = _Publisher(delivered=[("pointcloud", 5), ("frame", 5)], pending=[])
    assert publisher.match() == (5, 5)


def test_waits_for_a_point_cloud_still_in_flight():
    publisher = _Publisher(delivered=[("pointcloud", 4), ("frame", 4), ("frame", 5)], pending=[("pointcloud", 5)])
    assert publisher.match() == (5, 5)


def test_moves_on_to_the_next_frame_when_its_point_cloud_was_replaced():
    """The next point cloud arrived before the receiver read this frame's: move on to the next frame."""
    publisher = _Publisher(
        delivered=[("pointcloud", 5), ("frame", 5), ("pointcloud", 6)], pending=[("frame", 6), ("pointcloud", 7)]
    )
    assert publisher.match() == (6, 6)


def test_recovers_from_a_publisher_restart():
    """A restarted publisher numbers its frames from 0 again, below the stale point cloud's frame id."""
    publisher = _Publisher(
        delivered=[("pointcloud", 100), ("frame", 100), ("frame", 0)], pending=[("pointcloud", 1), ("frame", 1)]
    )
    assert publisher.match() == (1, 1)


def test_times_out_when_no_matching_point_cloud_arrives():
    publisher = _Publisher(delivered=[("pointcloud", 4), ("frame", 5)], pending=[])
    with pytest.raises(TimeoutError):
        publisher.match()
