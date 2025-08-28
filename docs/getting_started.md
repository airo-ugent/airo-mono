# Getting started ✔

This is a brief getting started guide, providing an overview of the supported features in AIRO-mono, and some
recommended practices for research and development of robotics projects.

In this document, we will cover the following topics:

- Features supported by AIRO-mono and sister repositories
- For features not supported by AIRO-mono, which software we recommend using
- Recommended coding practices

By reading through this document, you will get an idea what is available in AIRO-mono and what is not,
and how to proceed if you need a feature that is not supported. This way, you can avoid reinventing the wheel.

## Supported features 👍

### Robots 🤖

We support the following manipulators:

- [UR-series cobots](https://www.universal-robots.com/) (based on the [
  `ur_rtde`](https://sdurobotics.gitlab.io/ur_rtde/) package)

We support the following grippers:

- [Robotiq 2F-85](https://robotiq.com/products/2f85-140-adaptive-robot-gripper)
- [Schunk EGK40](https://schunk.com/us/en/gripping-systems/parallel-gripper/egk/egk-40-mb-m-b/p/000000000001491762)

We support the following wheeled mobile platforms:

- [KELO](https://www.kelo-robotics.com/) (with the [`airo-tulip`](https://pypi.org/project/airo-tulip/) package)

Interaction with robots is performed with asynchronous communication. In many cases, you need to explicitly wait for
operations to complete.

For example, when moving a robot to a certain position, you need to wait for the robot to reach that position before
continuing with the next operation. The following will not work as expected:

```python
robot.move_to_joint_configuration(q1)
robot.move_to_joint_configuration(q2)
```

but this will:

```python
robot.move_to_joint_configuration(q1).wait()
robot.move_to_joint_configuration(q2).wait()
```

The asynchronous nature allows you to control gripper and arm simultaneously, for example:

```python
awaitable_robot = robot.move_to_joint_configuration(q1)  # Not blocking.
gripper.open().wait()  # Will already start executing while the arm is still moving. Blocking.
awaitable_robot.wait()  # Wait for the arm to stop moving.
```

See [the README](airo-robots/README.md) and [awaitable_action.py](airo-robots/airo_robots/awaitable_action.py) for more
information.

### Cameras 📸

We support the following cameras:

- [ZED2i](https://www.stereolabs.com/zed-2/) and [ZED Mini](https://www.stereolabs.com/zed-mini/) (other Zed cameras may also work)
- [Realsense D435](https://www.intelrealsense.com/depth-camera-d435/) (other Realsense cameras may also work)
- [USB Webcams via OpenCV](https://opencv.org/)

For RGB cameras, we support reading RGB images.
For RGB-D cameras, we support reading RGB images and depth images, as well as retrieving colored point clouds.

#### Multiprocessing

Often, you want to read camera images in a separate process, to control the robot and camera in parallel.
The `multiprocess` module in the `airo-camera-toolkit` package provides a way to do this:
see [the README](airo-camera-toolkit/README.md) for more information.

For other multiprocessing needs, e.g., custom sensors or logging, we recommend using [`airo-ipc`](https://pypi.org/project/airo-ipc/).
This is also the library underlying the `multiprocess` module in `airo-camera-toolkit`.

#### Image operations

When interfacing with RGB(-D) cameras, certain operations are often needed, such as projecting 3D points to pixels and
unprojecting pixels to 3D points.
These operations are provided in the `pinhole_operations` module in the `airo-camera-toolkit` package.
Image transformations, such as resizing, cropping, and rotating, are also provided in the `image_transforms` module.

#### Point clouds

Point clouds are often used in robotics for perception tasks. When you wish to crop point clouds, filter points with a
mask, or transform point clouds, you can use the `point_clouds` module in the `airo-camera-toolkit` package.

#### Camera calibration

Camera calibration is essential for many tasks in robotics. The `calibration` module in the `airo-camera-toolkit`
package provides functions to calibrate camera extrinsics matrices
using [ChArUcO boards](https://docs.opencv.org/3.4/df/d4a/tutorial_charuco_detection.html).

### Deep learning 🧠

Deep learning is often used in robotics for perception tasks. An important part of deep learning is data collection
and annotation. We recommend using [CVAT](https://www.cvat.ai/) for annotating images. The `airo-dataset-tools` package
provides conversion functions to convert CVAT annotations to different formats, to be used in deep learning pipelines.

### Algebra ➕

We provide a small set of algebraic operations in the `airo-spatial-algebra` package. These operations are useful for
robotics applications, such as transforming points and poses in 3D space.

### Teleoperation 🎮

The `airo-teleop` package provides a way to teleoperate robots using a joystick. This package is useful for testing
robot functionality and for collecting data.

### Motion planning and simulation 🛹

The AIRO-mono sister packages [airo-drake](https://pypi.org/project/airo-drake/),
[airo-planner](https://pypi.org/project/airo-planner/) and [airo-models](https://pypi.org/project/airo-models/) provide
tools for rendering robots in simulation and performing motion planning using [Drake](https://drake.mit.edu/)
and [OMPL](https://ompl.kavrakilab.org/).

See their respective READMEs for more information.

## Recommended software 💽

### Logging and visualization 📈

For logging of images, point clouds, sensor data... we recommend using [Rerun](https://rerun.io/).
This application allows for fast real-time logging of data, locally or remotely over a gRPC connection.
See their [documentation](https://rerun.io/docs/getting-started/what-is-rerun) for more information.

While Rerun was originally mainly intended for logging, it can also be used for visualization of data:
using the [Blueprint API](https://rerun.io/docs/concepts/blueprint), you can lay out the viewer as you see fit
and log data to create real-time visualizations. This was, e.g., done, for the [ITF World 2024 demo (0:27)](https://youtu.be/ThvECQgYLqQ?t=27).

### Deep learning 🧠

For deep learning, we recommend using [PyTorch](https://pytorch.org/). For annotating data, we recommend
using [CVAT](https://www.cvat.ai/). For logging, we recommend using [Weights and Biases](https://wandb.ai/site).

## Recommended coding practices ⌨

### Variable naming 📗

To express positions and poses in 3D space, use
the [Drake](https://drake.mit.edu/doxygen_cxx/group__multibody__spatial__pose.html) convention:

- `p_A_B` represents a position in frame `B` with respect to frame `A`.
- `X_A_B` represents a pose (position and orientation) of frame `B` with respect to frame `A`.

This makes it easy to reason about chains of operations, for example:

```python
X_World_PointOfInterest = X_World_Robot @ X_Robot_Tcp @ X_Tcp_Camera @ X_Camera_PointOfInterest
```

### Typing 🦉

Use type hints! This makes your code more readable and helps catch bugs early. For example, from [
`airo-spatial-algebra`](https://github.com/airo-ugent/airo-mono/blob/fb4097ec9477a45599a097a0189c3f317498841c/airo-spatial-algebra/airo_spatial_algebra/operations.py#L71):

```python
def transform_points(homogeneous_transform_matrix: HomogeneousMatrixType, points: Vectors3DType) -> Vectors3DType:
# ...
```

This function signature clearly states which types are expected and returned.

### Controllers 🧰

It can be sensible to structure your robotics application code with a controller-based approach.
Every controller is responsible for a specific task, such as moving the robot to a certain position or
performing a perception task. See [this issue](https://github.com/airo-ugent/airo-mono/issues/140)
for a discussion on how you can structure your controllers.