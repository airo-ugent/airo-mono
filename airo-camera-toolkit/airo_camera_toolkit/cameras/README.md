
# Cameras
This subpackage contains implementations of the camera interface for the cameras we have at AIRO.

- ZED 2i
- Realsense D400 series

It also contains code to enable multiprocessed use of the camera streams: [multiprocessed camera](./multiprocess/)

There is also an implementation for generic RGB cameras using OpenCV `VideoCapture`: [generic OpenCV camera](./generic_opencv/)

## 1. Installation
Implementations usually require the installation of SDKs, drivers etc. to communicate with the camera.
This information can be found in `READMEs` for each camera:
* [ZED Installation](zed/installation.md)
* [RealSense Installation](realsense/realsense_installation.md)


## 2. Testing your hardware installation
Furthermore, there is code for testing the hardware implementations: `manual_test_hw.py`
But since this requires attaching a physical camera, these are 'user tests' which should be done manually by developers/users.
Each camera implementation can be run as a script and will execute the relevant tests, providing instructions on what to look out for.

For example, to test your ZED installation:
```
conda activate airo-mono
python3 zed2i.py
```