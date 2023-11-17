from typing import Dict, List, Tuple

import numpy as np
import pyrealsense2 as rs  # type: ignore

pipeline = rs.pipeline()

# Configure streams
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
color_sensor = device.first_color_sensor()
depth_sensor = device.first_depth_sensor()

# Print the device name and serial number
print(f"Device Name: {device.get_info(rs.camera_info.name)}")
print(f"Serial Number: {device.get_info(rs.camera_info.serial_number)}")

color_stream_profiles = color_sensor.get_stream_profiles()
depth_stream_profiles = depth_sensor.get_stream_profiles()


def calculate_fov(width: int, height: int, fx: float, fy: float) -> Tuple[float, float]:
    fov_x = np.rad2deg(2 * np.arctan2(width, 2 * fx))
    fov_y = np.rad2deg(2 * np.arctan2(height, 2 * fy))
    return (fov_x, fov_y)


def scan_profiles(stream_profiles: List[rs.stream_profile]) -> Tuple:  # type: ignore
    resolution_fps_combinations: Dict[Tuple[int, int], List[float]] = {}
    resolution_fov_combinations: Dict[Tuple[int, int], Tuple[float, float]] = {}

    for profile in stream_profiles:
        profile = profile.as_video_stream_profile()  # allows us to get width, height, etc.

        if profile.format() != rs.format.rgb8 and profile.format() != rs.format.z16:
            continue

        if profile.stream_type() == rs.stream.infrared:
            continue

        resolution = (profile.width(), profile.height())
        fps = profile.fps()

        intrinsics = profile.get_intrinsics()
        fov_H, fov_V = calculate_fov(intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy)

        if resolution not in resolution_fps_combinations:
            resolution_fps_combinations[resolution] = [fps]
            resolution_fov_combinations[resolution] = (fov_H, fov_V)
        else:
            resolution_fps_combinations[resolution].append(fps)

    # sort the fps lists in ascending order
    for resolution in resolution_fps_combinations:
        resolution_fps_combinations[resolution].sort()

    return resolution_fps_combinations, resolution_fov_combinations


def print_profile_info(resolution_fps_combinations: Dict, resolution_fov_combinations: Dict) -> None:
    for resolution in resolution_fps_combinations:
        fov_string = (
            f"({resolution_fov_combinations[resolution][0]:.1f}°, {resolution_fov_combinations[resolution][1]:.1f}°)"
        )
        print(f"{resolution}: {resolution_fps_combinations[resolution]} fps, {fov_string}")


color_framerates, color_fovs = scan_profiles(color_stream_profiles)
depth_framerates, depth_fovs = scan_profiles(depth_stream_profiles)

print("Available color resolutions, framerates and FoV:")
print_profile_info(color_framerates, color_fovs)
print("Available depth resolutions, framerates and FoV:")
print_profile_info(depth_framerates, depth_fovs)

color_resolution_max_fps = {resolution: color_framerates[resolution][-1] for resolution in color_framerates}
depth_resolution_max_fps = {resolution: depth_framerates[resolution][-1] for resolution in depth_framerates}

print("Checking aligned depth resolution (should equal color resolution):")
# Check the resolution of depth frames aligned to color frames
for color_resolution, color_fps in color_resolution_max_fps.items():
    for depth_resolution, depth_fps in depth_resolution_max_fps.items():
        config.enable_stream(rs.stream.color, color_resolution[0], color_resolution[1], rs.format.rgb8, color_fps)
        config.enable_stream(rs.stream.depth, depth_resolution[0], depth_resolution[1], rs.format.z16, depth_fps)

        pipeline.start(config)
        composite_frame = pipeline.wait_for_frames()
        color_frame = composite_frame.get_color_frame()
        depth_frame = composite_frame.get_depth_frame()
        aligned_depth_frame = rs.align(rs.stream.color).process(composite_frame).get_depth_frame()
        print(
            f"color_resolution = {color_resolution}, depth_resolution = {depth_resolution}, aligned_depth = ({aligned_depth_frame.get_width()}, {aligned_depth_frame.get_height()})"
        )
        pipeline.stop()
