# RealSense Installation

## 1. librealsense2 SDK
The first step is to install the librealsense2 SDK.
Follow the steps in this [librealsense2 installation guide](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md)

If the installation was successful you should be able to run the following command in a terminal:
```
realsense-viewer
```

## 2. pyrealsense2
Now we need to install the python bindings for the librealsense2 SDK.

For this, you can simply run the following command in your **conda environment**:
```
pip install pyrealsense2
```

### 3. airo_camera_toolkit
Now We will test whether our `airo_camera_toolkit can access the Realsense cameras.
In this directory run:
```
conda activate airo-mono
python realsense.py
```
Complete the prompts. If everything looks normal, congrats, you successfully completed the installation! :tada:

### 4. Camera details
Realsense cameras offer a lot of possible resolutions and framerates.
Use the [`realsense_scan_profiles.py`](./realsense_scan_profiles.py) script to print an outplut like this for your camera:

**D415**:
```python
Device Name: Intel RealSense D415
Serial Number: 925322060348
Available color resolutions, framerates and FoV:
(1920, 1080): [6, 15, 30] fps, (69.6°, 42.9°)
(1280, 720): [6, 15, 30] fps, (69.6°, 42.9°)
(960, 540): [6, 15, 30, 60] fps, (69.6°, 42.9°)
(848, 480): [6, 15, 30, 60] fps, (69.3°, 42.9°)
(640, 480): [6, 15, 30, 60] fps, (55.1°, 42.9°)
(640, 360): [6, 15, 30, 60] fps, (69.6°, 42.9°)
(424, 240): [6, 15, 30, 60] fps, (69.3°, 42.9°)
(320, 240): [6, 30, 60] fps, (55.1°, 42.9°)
(320, 180): [6, 30, 60] fps, (69.6°, 42.9°)
Available depth resolutions, framerates and FoV:
(1280, 720): [6, 15, 30] fps, (71.0°, 43.8°)
(848, 480): [6, 15, 30, 60, 90] fps, (70.7°, 43.8°)
(848, 100): [100] fps, (70.7°, 9.6°)
(640, 480): [6, 15, 30, 60, 90] fps, (56.3°, 43.8°)
(640, 360): [6, 15, 30, 60, 90] fps, (71.0°, 43.8°)
(480, 270): [6, 15, 30, 60, 90] fps, (71.0°, 43.8°)
(424, 240): [6, 15, 30, 60, 90] fps, (70.7°, 43.8°)
(256, 144): [90] fps, (16.3°, 9.2°)
```

**D435**:
```python
Device Name: Intel RealSense D435
Serial Number: 817612070315
Available color resolutions, framerates and FoV:
(1920, 1080): [6, 15, 30] fps, (69.0°, 42.2°)
(1280, 720): [6, 15, 30] fps, (69.0°, 42.2°)
(960, 540): [6, 15, 30, 60] fps, (69.0°, 42.2°)
(848, 480): [6, 15, 30, 60] fps, (68.7°, 42.2°)
(640, 480): [6, 15, 30, 60] fps, (54.5°, 42.2°)
(640, 360): [6, 15, 30, 60] fps, (69.0°, 42.2°)
(424, 240): [6, 15, 30, 60] fps, (68.7°, 42.2°)
(320, 240): [6, 30, 60] fps, (54.5°, 42.2°)
(320, 180): [6, 30, 60] fps, (69.0°, 42.2°)
Available depth resolutions, framerates and FoV:
(1280, 720): [6, 15, 30] fps, (88.9°, 57.8°)
(848, 480): [6, 15, 30, 60, 90] fps, (88.9°, 58.1°)
(848, 100): [100, 300] fps, (88.9°, 13.2°)
(640, 480): [6, 15, 30, 60, 90] fps, (78.6°, 63.1°)
(640, 360): [6, 15, 30, 60, 90] fps, (88.9°, 57.8°)
(480, 270): [6, 15, 30, 60, 90] fps, (88.9°, 57.8°)
(424, 240): [6, 15, 30, 60, 90] fps, (88.9°, 58.1°)
(256, 144): [90, 300] fps, (22.2°, 12.6°)
```

