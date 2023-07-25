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
See [multiprocess_example.py](../../../docs/multiprocess_example.py) for a simple example of how to use these classes.
The main difference with the regular workflow is that instead of instantiating a `RGBCamera` subclass, you now have to first create a `MultiprocessRGBPublisher` and then one or more `MultiprocessRGBReceiver`s.

## Additional features
TODO implement and explain::
* Blocking MultiprocessRGBReceiver
* RerunLogger
* VideoRecorder .start() and .stop()