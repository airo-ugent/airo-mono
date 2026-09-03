# Multiprocessing in the airo-camera-toolkit

Multiprocessing in the airo-camera-toolkit was born from a simple need:

> We want to command robots and at the same time view a smooth camera feed.

There a few things that make this difficult:
* Robot commands can take long to execute (several seconds)
* Robot commands need to be responsive and high frequency (e.g. 500 Hz)
* Retrieving images is quite slow (a few milliseconds) at high resolutions or when using depth
* Images need to be retrieved from several camera's
* We also might want to log, visualize, save images or videos.
* Parallellism in a [single Python process is tricky due to the GIL](https://stackoverflow.com/questions/18114285/what-are-the-differences-between-the-threading-and-multiprocessing-modules).

To overcome these difficulties we use [Zenoh](https://zenoh.io/) to create a solution where:
* Camera images can be retrieved, visualized, recorded, etc. without being blocked by user code (e.g. robot commands)
* Robot commands can be sent at high frequency without having to retrieve images inbetween

## Implementation
Two classes are at the core of our solution:
* `MultiprocessRGBPublisher`: a class that write images from a camera to shared memory, from its own process.
* `MultiprocessRGBReceiver`: a class that reads images from shared memory, but hides this complexity from its users.

Note that the publisher is a subclass of `SpawnProcess`, this way it can publish uninterrupted.
The receiver is subclass of `RGBCamera` which ensures that it follows the interface of a regular airo-camera-toolkit camera.

## Read-only frames

The arrays a receiver returns (images, depth maps, intrinsics, point clouds) are read-only views into the frame it received; nothing is copied out of the payload.
This saves about 3 ms per FullHD RGBD frame, roughly a third of the end-to-end latency.

## Surviving a publisher restart

A receiver blocks in `grab_images()` until a frame newer than the one it holds arrives, but no longer than its `timeout` (30 s by default, `None` to wait forever).
A publisher that stopped, was restarted or became unreachable therefore surfaces as a `TimeoutError` instead of a hang:

```python
try:
    receiver.grab_images()
except TimeoutError:
    receiver.reconnect()   # opens a new session and re-reads resolution and fps
    receiver.grab_images()
```

`reconnect()` closes the Zenoh session and opens a new one, so it re-runs peer discovery and drops the shared memory mappings of the previous publisher.
It also re-reads the resolution and fps, which matters because the frame buffer template is derived from the resolution: frames from a publisher that came back at a different resolution do not match the old template and are rejected.
Arrays retrieved before the call keep pointing at their own payload and stay valid.

A receiver that is only stopped (not reconnected) raises `RuntimeError` from `grab_images()`; call `reconnect()` to use it again.

## Networking

Publishers and receivers communicate over Zenoh, which is configured to stay on the local host: peers are scouted over loopback only and sessions listen on loopback only.
This keeps the shared memory transport effective and prevents two machines on the same network that happen to use the same `shared_memory_namespace` from connecting to each other -- a receiver would otherwise silently consume another machine's frames over the network.

To publish on one machine and receive on another, run a [Zenoh router](https://zenoh.io/docs/getting-started/deployment/) (`zenohd`) and point both sides at it:

```bash
export AIRO_ZENOH_ROUTER=tcp/192.168.0.10:7447   # the host running zenohd
```

Multicast scouting is then disabled and peers discover each other through the router (gossip scouting) instead.
Note that frames to a peer on another host travel over the network (~1.2 GB/s for FullHD RGBD at 60 FPS), so shared memory only helps same-host peers.
Sessions still run in Zenoh's `peer` mode, so peers that turn out to be on the same host should link directly and keep using shared memory between them.

## Usage
See the  main function in [multiprocess_rgb_camera.py](./multiprocess_rgb_camera.py) for a simple example of how to use these classes with a ZED camera.
The main difference with the regular workflow is that instead of instantiating a `Zed` object, you now have to first create a `MultiprocessRGBPublisher` with the class and its kwargs, and then one or more `MultiprocessRGBReceiver`s.

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
Note that realtime video-encoding is computationally expensive, recording at 30 fps on laptops is not always possible.
The video recorder will try to keep up with the framerate, but will drop frames if it can't.
