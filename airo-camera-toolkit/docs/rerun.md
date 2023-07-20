# Rerun

[Rerun](https://www.rerun.io/) is a great tool for **realtime data visualization**.
You simply start Rerun viewer, and then you can log many types of data from anywhere in your python script like so:
```python
import rerun

rerun.init("my_project") # required before logging

rerun.log_image("zed_top", image)
rerun.log_scalar("force_z", force[2])
...
```
See the [example notebook](./rerun-zed-example.ipynb) for more.

> :information_source: A note on starting the Rerun viewer: you can start it by calling `rerun.spawn()` from Python. However when starting Rerun like that, [there is no way to specify a memory limit](https://www.rerun.io/docs/howto/limit-ram). This quickly becomes a problem when logging images, so we recommend starting Rerun from a terminal:
>```
>python -m rerun --memory-limit 16GB
>```

Some limitations to be aware of:
* Rerun is for **one-way communication**. You cannot get user input, such of confirmation of clicks from Rerun back to your Pythn scripts.
If you want that, you can create OpenCV windows.
* Rerun is best for **non-persistent data**.
You can record everything sent to Rerun in a `.rrd` file, however it's not clear how easily get your data out of that format.
So you probably want to log some images or videos of your experiment to disk youself.


## Integration with our multiprocessing

We provide a `MultiprocessRerunRGBLogger` class that can be used to log a camera feed to Rerun when using a camera through the `MultiprocessRGBPublisher` class.
In its own process it simply checks the shared memory created by the publisher, and when it detects a new image, it logs it to Rerun.