# ZED Installation
This file will get you started with the ZED2i cameras.

## 1. ZED SDK
The first step is to install the ZED SDK, before installing:
* Ensure you have **nvidia-drivers** installed, check with `nvidia-smi`. The installer will automatically install a CUDA 11.X version is you do not have it.
* During the setup, you will be asked whether the **Python API** should be installed. Make sure to **activate** the venv of conda env you want to use with the ZED cameras. (You can also answer `no` and install the Python package later.)
* You can say `no` to the AI-model stuff.

Now follow this [ZED Installation Guide](https://www.stereolabs.com/docs/installation/linux/).
You only need to complete the first part `Download and Install the ZED SDK`.

 > If you did not install the Python API during the SDK setup, you can install it now:
> ```
> conda activate airo-mono
> python /usr/local/zed/get_python_api.py
>```
>If this fails, install the dependencies from the [ZED Python API Installation](https://www.stereolabs.com/docs/app-development/python/install/).

After finishing the installation **reboot** your pc.

## 3. Testing the installation
### 3.1 ZED_Explorer
First plug in the USB cable to the ZED into your laptop.
Then we can try opening the `ZED_Explorer` by running the following command in a terminal:
```
/usr/local/bin/ZED_Explorer
```
This should open up a viewer that looks like this:
![ZED_Explorer](https://i.imgur.com/DGz6aSR.png)

### 3.2 ZED_Depth_Viewer
If the `ZED_Explorer` works as expected, you can try the `ZED_Depth_Viewer`:
```
/usr/local/bin/ZED_Depth_Viewer

```
This should like this:
![ZED_Depth_Viewer](https://i.imgur.com/SzamB6J.png)

### 3.3 airo_camera_toolkit
Now We will test whether our `airo_camera_toolkit can access the ZED cameras.
In this directory run:
```
conda activate airo-mono
python zed2i.py
```
Complete the prompts. If everything looks normal, congrats, you successfully completed the installation! :tada:
