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

To overcome these difficulties we used [airo-ipc](https://github.com/airo-ugent/airo-ipc) to create a solution where:
* Camera images can be retrieved, visualized, recorded, etc. without being blocked by user code (e.g. robot commands)
* Robot commands can be sent at high frequency without having to retrieve images inbetween

## Implementation
The implementation of multiprocessing in the airo-camera-toolkit decouples data from transport and provides a flexible way to construct new publisher/subscriber pairs. You need to know three core concepts:

- Buffers: contain the data.
- Schemas: contain the (de)serialization logic, and define which data is shared. Each schema is intrinsically linked to one buffer type. Schemas:
  - allocate buffers,
  - serialize data and write it to buffers,
  - read from buffers and deserialize data,
  - write deserialized data to specific fields in receivers.
- Mixins: provide implementations of camera interfaces for receivers. Each mixin is intrinsically linked to one schema, as the field(s) the mixin reads from should correspond to those the schema writes to.

More information on this follows below.

A **schema** represents a single type of data that can be transported over shared memory. Examples include RGBSchema, DepthSchema, and PointCloudSchema. Schemas define the structure of the data, how an empty buffer for that data is allocated, and how the buffer is filled using a camera instance.

Allocation is only done once, similarly to C's `malloc` (initialization of empty NumPy arrays), afterwards existing buffers are written to.

Each schema has an associated [POD](https://en.wikipedia.org/wiki/Passive_data_structure) **buffer** type, such as RGBFrameBuffer, DepthFrameBuffer, or PointCloudBuffer. Buffers are simple dataclasses that subclass BaseIdl (from `airo-ipc`) and contain NumPy arrays. They are responsible only for data layout; all transport logic lives elsewhere.

The **CameraPublisher** is a multiprocessing.Process that runs _grab_images() in a loop. In that same loop, it publishes all data defined by the schemas it was configured with. Which data is published is determined entirely by the list of schemas passed to the publisher.

On the receiving side, there is similarly a single **SharedMemoryReceiver** that implements the `Camera` interface. It is also configured with a list of schemas. In its _grab_images() method, it reads data for all configured schemas from shared memory and stores the latest buffers internally. Like the publisher, the receiver itself contains no RGB-, depth-, or point-cloud-specific logic.

```python
# [...]
def _grab_images(self) -> None:
        for s in self._schemas:
            s.read_into_receiver(self._readers[s](), self)
```

To expose the received data through the familiar camera interfaces, the design uses [**mixins**](https://en.wikipedia.org/wiki/Mixin). Each schema has a corresponding mixin, such as RGBMixin, DepthMixin, or PointCloudMixin. A mixin provides the methods needed to access a specific type of data by reading from the buffers populated by the SharedMemoryReceiver. Mixins contain no transport logic; they only expose data that is already available.

```python
class Mixin(ABC):
    pass

class RGBMixin(Mixin, RGBCamera):
    def _retrieve_rgb_image(self) -> NumpyFloatImageType:
        return ImageConverter.from_numpy_int_format(self._retrieve_rgb_image_as_int()).image_in_numpy_format

    def _retrieve_rgb_image_as_int(self) -> NumpyIntImageType:
        return self._rgb_frame.rgb
```

By combining SharedMemoryReceiver with the appropriate mixins, the existing multiprocess camera classes can be reconstructed with minimal code. For example, an RGB receiver can be implemented by combining SharedMemoryReceiver, CameraMixin, and RGBMixin, and by passing CameraSchema and RGBSchema to the receiver constructor. The schemas determine which data is read from shared memory, while the mixins determine which camera interface methods are available.

Extending this to RGB-D cameras is straightforward. A multiprocess RGB-D receiver is created by adding DepthSchema and DepthMixin. If point cloud support is desired, PointCloudSchema and PointCloudMixin can be added as well. Implementing a new camera variant is now largely a matter of selecting the appropriate schemas and mixins, rather than creating new publisher and receiver classes. This means that if you, for example, only want to transmit the depth map, or only the point cloud, this is trivial.

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