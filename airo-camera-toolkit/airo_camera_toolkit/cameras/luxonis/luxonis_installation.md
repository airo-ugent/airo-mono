# Luxonis Installation

The `Luxonis` class supports the Luxonis OAK-D stereo cameras (e.g. OAK-D, OAK-D Pro, OAK-D S2, OAK4-D) through the
[depthai](https://docs.luxonis.com/software-v3/) v3 library.

## 1. depthai
Unlike the ZED and RealSense SDKs, depthai is fully pip installable. Install it as an extra of `airo-camera-toolkit`
in your **environment**:
```
pip install "airo-camera-toolkit[luxonis]"
```
or directly with `pip install "depthai>=3.0"`.

## 2. udev rules (Linux only)
On Linux, USB devices need a udev rule to be accessible without root:
```
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```
Unplug and replug the camera afterwards. See the [Luxonis troubleshooting guide](https://docs.luxonis.com/hardware/platform/deploy/usb-deployment-guide/) for more information.

## 3. airo_camera_toolkit
Now we will test whether our `airo_camera_toolkit` can access the Luxonis cameras.
In this directory run:
```
python luxonis.py
```
Complete the prompts. If everything looks normal, congrats, you successfully completed the installation! :tada:

## 4. Camera details
* The RGB image comes from the center color camera, depth is computed on-device from the left/right mono cameras
  and is aligned to the color image.
* Select a specific camera with `serial_number`, which accepts the device ID (MxID), an IP address (PoE cameras) or a
  USB port name. Use `Luxonis.list_camera_serial_numbers()` to list the connected cameras.
* The available color resolutions depend on the color sensor: the IMX378 (most OAK-D / OAK-D Pro models) supports
  up to 4K, while the global shutter OV9782 supports at most 1280x800. The default `RESOLUTION_720` works on all of them;
  requesting an unsupported resolution raises an error listing the sensor's modes.
* The color width should be a multiple of 16 for on-device depth alignment (all built-in `RESOLUTION_*` aliases are).
* The `depth_preset` and `stereo_resolution` arguments trade off depth quality against on-device compute,
  see the [StereoDepth docs](https://docs.luxonis.com/software-v3/depthai/depthai-components/nodes/stereo_depth/).
