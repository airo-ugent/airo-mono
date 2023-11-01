Hand-eye calibration
====================

The `calibration` submodule of the `airo-camera-toolkit` contains the code for hand-eye calibration (also called extrinsics calibration).
If you already know what this is, you can skip to the [quick start](#quick-start) section.

The goal of hand-eye calibration is to find the transformation between the camera and the robot.
There are two variants, depending on how the camera is mounted:
* `eye_in_hand`: the camera is mount on the wrist of the robot, the transform that we want is from TCP to camera.
* `eye_to_hand`: the camera is mounted on a fixed position in the robot's workspace, the transform that we want is from robot base to camera.

To compute this transform, we use a Charuco board, which's pose relative to the camera can be accurately detected from a single image.
In `eye_in_hand` mode the board is fixed in the workspace of the robot and we move the wrist camera around.
In `eye_to_hand` mode the board is grasped by the robot and moved around, and the camera is stationary.
Then using at least 3 samples of the form: `(tcp_pose_in_robot_base_frame, board_pose_in_camera_frame)` we can solve a non-linear system of equations to find the camera's pose.

Quick start
-----------

> :construction: **The  `hand_eye_calibration.py` script currently only supports ZED cameras and UR robots.** :construction:

**Starting the calibration**:

1. Place or grasp the Charuco board firmly (grasp pose can be chosen freely)
2. Put the robot in `Remote Control` mode
3. Then run the `hand_eye_calibration.py` script to start the calibration, e.g:


```shell
python hand_eye_calibration.py --mode eye_to_hand --robot_ip=10.42.0.163 --camera_serial_number=34670760
```

You will should be shown an OpenCV window with the camera feed and the detected Charuco board that looks like this:

![Calibration view](https://i.imgur.com/VuNfyGl.jpg)

The script will put the robot in Free Drive mode.
You can now move the robot around to collect samples.

**Collecting samples**:

1. Move the robot to a new position where the board is visible in the camera feed
2. Ensure the board pose detection is correct and stable
3. Press `s` to save the sample

The script should now have created a directory like `calibration_2023-11-01_11:46:41`, which we will call the `calibration_dir` from now on.
Inside the `data` folder in the `calibration_dir` you will find a backup of the samples you have collected so far.

4. Continue collecting samples until you have at least 3 samples, however more is usually better.

As soon as you have at least 3 samples, the script will try solving the calibration and store the results in the `calibration_dir` in a directory like `results_n=3`.
The script will also log the *residual error* of the solution for each solver available in OpenCV.
The residual error is a dimensionless measure of how well each solution fits the samples.
However it is important to note that low residual error is no guarantee that the solution is accurate.
For this reason, we also save images with the corresponding robot base frame visualized, e.g:

![robot base visualization](https://i.imgur.com/haIX5RQ.jpg)

**Finishing the calibration**:

If you are satisfied with the results, you can stop the script by pressing `q`.
The camera pose solutions are saved as json files in the `results` directory.
These can be loaded into Python and turned into 4x4 homogeneous matrices using the `Pose` class from `airo-dataset-tools`.

Additional tools
----------------

`collect_calibration_samples.py`: collect samples without running the calibration solvers.

`compute_calibration.py`: compute the calibration using saved samples.


Charuco board
-------------
By deafult we use a charuco board for hand-eye calibration.
 You can find the board in the `test/data` folder.
 To match the size of the markers to the desired size, the board should be printed on a 300mm x 220mm surface.
 Using Charuco boards is highly recommended as they are a lot more robust and precise than individual aruco markers, if you do use an aruco marker, make sure that the whitespace around the marker is at least 25% of the marker dimensions.