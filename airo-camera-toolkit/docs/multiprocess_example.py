import cv2
from airo_camera_toolkit.cameras.multiprocess.multiprocess_rgb_camera import (
    MultiprocessRGBPublisher,
    MultiprocessRGBReceiver,
)
from airo_camera_toolkit.cameras.zed2i import Zed2i
from airo_camera_toolkit.utils import ImageConverter

# The publisher is passed a camera class and its kwargs, so it can be the (single) creator of the camera in its process.
publisher = MultiprocessRGBPublisher(
    Zed2i,
    camera_kwargs={
        "resolution": Zed2i.RESOLUTION_720,
        "fps": 15,
        "depth_mode": Zed2i.NONE_DEPTH_MODE,
    },
    shared_memory_namespace="zed_top",
)
publisher.start()

# The receiver only need to know the shared memory namespace of the publisher and the image resolution.
resolution = Zed2i.resolution_sizes[Zed2i.RESOLUTION_720]
camera = MultiprocessRGBReceiver("zed_top", *resolution)

# From this point on, the camera behaves like a regular airo-camera-toolkit camera
while True:
    image_float = camera.get_rgb_image()
    image = ImageConverter.from_numpy_format(image_float).image_in_opencv_format
    cv2.imshow("RGB Image", image)
    key = cv2.waitKey(10)
    if key == ord("q"):
        break
