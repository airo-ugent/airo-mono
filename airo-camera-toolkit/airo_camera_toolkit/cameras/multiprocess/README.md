# Multiprocessing in the airo-camera-toolkit

Multiprocessing in the airo-camera-toolkit was born from a simple need:

> "We want to command robots and at the same time view a smooth camera feed.

There a few things that make this difficult:
* Robot commands can take long to execute (several seconds)
* Robot commands need to be responsive and high frequency (e.g. 500 Hz)
* Retrieving images is quite slow (a few milliseconds) at high resolutions or when using depth
* Images need to be retrieved from several camera's
* We also might want to log, visualize, save images or videos.
* Parallellism in a [single Python process is tricky due to the GIL](https://stackoverflow.com/questions/18114285/what-are-the-differences-between-the-threading-and-multiprocessing-modules).

To overcome these difficulties we used the Python [multiprocessing](https://docs.python.org/3/library/multiprocessing.html) module to create a solution where:
* Camera images can be retrieved, visualized, recorded, etc. without being blocked by user code (e.g. robot commands)
* Robot commands can be sent at high frequency without having to retrieve images inbetween

## Implementation
![Diagram](https://i.imgur.com/jEUOdZH.jpg)


Two classes are at the core of our solution:
* `MultiprocessRGBPublisher`: a class that write images from a camera to shared memory, from its own process.
* `MultiprocessRGBReceiver`: a class that reads images from shared memory, but hides this complexity from its users.

Note that the publisher is a subclass of `Process`, this way it can publish uninterrupted.
The receiver is subclass of `RGBCamera` which ensures that it follows the interface of a regular airo-camera-toolkit camera.

## Usage
See the  main function in [multiprocess_rgb_camera.py](./multiprocess_rgb_camera.py) for a simple example of how to use these classes with a ZED camera.
The main difference with the regular workflow is that instead of instantiating a `Zed2i` object, you now have to first create a `MultiprocessRGBPublisher` with the class and its kwargs, and then one or more `MultiprocessRGBReceiver`s.

> :information_source: Similar to how regular `RGBCamera`s behave, `MultiprocessRGBReceiver`s will block until a new image is available.

## Additional features
Logging and recording images and videos is computationally expensive.
This can interfere with robot controllers.
For this reason we provide two additional classes that can be used to log and record images and videos in parallel and in separate processes.
All they need to start working is the `namespace` of the camera publisher they should log or record.

### Rerun Loggers
The `MultiprocessRGBRerunLogger` logs RGB images to Rerun from its own process.
First start a `MultiprocessRGBPublisher` and then a Rerun viewer from a termimal:
```bash
python -m rerun --memory-limit 8GB
```
Finally create a `MultiprocessRGBRerunLogger` with the namespace of the publisher, as in the main function of [mutliprocess_rerun_logger.py](./multiprocess_rerun_logger.py).

A RGBD variant of this class is also available.

### Video Recording
To enable video recording install FFMPEG 6.0 and the python package [ffmpegcv](https://github.com/chenxinfeng4/ffmpegcv), this can be done via conda:

```yaml
dependencies:
  - ffmpeg=6.0.0
  - x265 # not 100% if this need to be installed separately
  - pip
  pip:
    - ffmpegcv
```
To start recording RGB videos from a `MultiprocessRGBPublisher` create a `MultiprocessRGBVideoRecorder` with the namespace of the publisher, and start it, as in the main function of [multiprocess_video_recorder.py](./multiprocess_video_recorder.py).