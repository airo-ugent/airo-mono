"""Latency benchmark for the multiprocess RGBD camera pipeline.

Measures end-to-end latency from frame capture (publisher side) to
frame availability (receiver side) for FullHD (1920×1080) RGBD images
published at 60 FPS using Zenoh with shared-memory transport.

Frame data size (RGBDFrameBuffer, no point cloud):
  RGB image:    1920 × 1080 × 3 × uint8  =   6.2 MB
  Depth image:  1920 × 1080 × 3 × uint8  =   6.2 MB
  Depth map:    1920 × 1080 × 1 × float32 =   8.3 MB
  Intrinsics:   3 × 3 × float64           =   0.0 MB
  ─────────────────────────────────────────────────
  Total per frame                          ≈  20.7 MB
  At 60 FPS                                ≈ 1.24 GB/s

Latency is measured as:
  latency = time_received − frame_timestamp

where ``frame_timestamp`` is set by the publisher immediately after
``camera.grab_images()`` returns, and ``time_received`` is recorded in
the receiver immediately after ``grab_images()`` returns.

Usage::

    python benchmarks/benchmark_multiprocess_rgbd_latency.py

Optional arguments::

    --duration   How many seconds to run (default: 10)
    --namespace  Zenoh key-expression namespace (default: bench_rgbd)
    --fps        Target publisher FPS (default: 60)
    --no-shm     Disable Zenoh shared-memory transport
"""

from __future__ import annotations

import argparse
import multiprocessing
import time
from typing import List

import numpy as np
from airo_camera_toolkit.cameras.multiprocess.multiprocess_rgbd_camera import (
    MultiprocessRGBDPublisher,
    MultiprocessRGBDReceiver,
)
from airo_camera_toolkit.interfaces import RGBDCamera
from airo_typing import (
    CameraIntrinsicsMatrixType,
    CameraResolutionType,
    NumpyDepthMapType,
    NumpyFloatImageType,
    NumpyIntImageType,
)

# ---------------------------------------------------------------------------
# FullHD constants
# ---------------------------------------------------------------------------

RESOLUTION = (1920, 1080)  # (width, height)
W, H = RESOLUTION

# Pre-allocate static image data so the mock camera avoids per-frame allocation
_RGB = np.zeros((H, W, 3), dtype=np.uint8)
_DEPTH_MAP = np.ones((H, W), dtype=np.float32) * 1.5
_DEPTH_IMAGE = np.zeros((H, W, 3), dtype=np.uint8)
_INTRINSICS = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]], dtype=np.float64)


# ---------------------------------------------------------------------------
# Mock camera
# ---------------------------------------------------------------------------


class MockFullHDRGBDCamera(RGBDCamera):
    """Mock FullHD RGBD camera that returns static pre-allocated arrays."""

    def __init__(self, fps: float = 60.0) -> None:
        self._fps = fps
        self._period = 1.0 / fps

    @property
    def resolution(self) -> CameraResolutionType:
        return RESOLUTION

    @property
    def fps(self) -> float:
        return self._fps

    def grab_images(self) -> None:
        time.sleep(self._period)

    def retrieve_rgb_image(self) -> NumpyFloatImageType:
        return _RGB.astype(np.float32) / 255.0

    def retrieve_rgb_image_as_int(self) -> NumpyIntImageType:
        return _RGB

    def intrinsics_matrix(self) -> CameraIntrinsicsMatrixType:
        return _INTRINSICS

    def retrieve_depth_map(self) -> NumpyDepthMapType:
        return _DEPTH_MAP

    def retrieve_depth_image(self) -> NumpyIntImageType:
        return _DEPTH_IMAGE


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(duration: float, namespace: str, fps: float) -> None:
    multiprocessing.set_start_method("spawn", force=True)

    frame_size_mb = (
        _RGB.nbytes
        + _DEPTH_MAP.nbytes
        + _DEPTH_IMAGE.nbytes
        + _INTRINSICS.nbytes
        + 8
        + 8  # frame_id  # frame_timestamp
    ) / (1024**2)

    print(f"\n{'─' * 60}")
    print("  Multiprocess RGBD Latency Benchmark")
    print(f"{'─' * 60}")
    print(f"  Resolution   : {W} × {H}")
    print(f"  Target FPS   : {fps:.0f}")
    print(f"  Frame size   : {frame_size_mb:.1f} MB")
    print(f"  Bandwidth    : {frame_size_mb * fps / 1024:.2f} GB/s (theoretical)")
    print(f"  Duration     : {duration:.0f} s")
    print(f"{'─' * 60}\n")

    publisher = MultiprocessRGBDPublisher(
        camera_cls=MockFullHDRGBDCamera,
        camera_kwargs={"fps": fps},
        shared_memory_namespace=namespace,
        enable_pointcloud=False,
    )
    publisher.start()

    print("Waiting for receiver to connect...")
    receiver = MultiprocessRGBDReceiver(namespace, enable_pointcloud=False)
    print("Connected. Collecting latency samples...\n")

    latencies_ms: List[float] = []
    t_start = time.time()
    frames_received = 0

    while time.time() - t_start < duration:
        receiver.grab_images()
        t_received = time.time()
        latency_ms = (t_received - receiver.get_current_timestamp()) * 1000.0
        latencies_ms.append(latency_ms)
        frames_received += 1

    elapsed = time.time() - t_start
    publisher.stop()
    publisher.join(timeout=5)

    # ---------------------------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------------------------
    a = np.array(latencies_ms)

    print(f"{'─' * 60}")
    print(f"  Results  ({frames_received} frames over {elapsed:.1f} s)")
    print(f"{'─' * 60}")
    print(f"  Achieved FPS  : {frames_received / elapsed:.1f}")
    print("  Latency (ms):")
    print(f"    mean        : {a.mean():.2f}")
    print(f"    std         : {a.std():.2f}")
    print(f"    min         : {a.min():.2f}")
    print(f"    median (p50): {np.percentile(a, 50):.2f}")
    print(f"    p95         : {np.percentile(a, 95):.2f}")
    print(f"    p99         : {np.percentile(a, 99):.2f}")
    print(f"    max         : {a.max():.2f}")
    print(f"{'─' * 60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration", type=float, default=10.0, help="Benchmark duration in seconds (default: 10)")
    parser.add_argument(
        "--namespace", type=str, default="bench_rgbd", help="Zenoh key-expression namespace (default: bench_rgbd)"
    )
    parser.add_argument("--fps", type=float, default=60.0, help="Target publisher FPS (default: 60)")
    args = parser.parse_args()

    run_benchmark(duration=args.duration, namespace=args.namespace, fps=args.fps)
