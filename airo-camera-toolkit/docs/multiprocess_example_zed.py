import cv2
from airo_camera_toolkit.cameras.multiprocess.multiprocess_rgb_camera import (
    MultiprocessRGBPublisher,
    MultiprocessRGBReceiver,
)
from airo_camera_toolkit.cameras.zed2i import Zed2i
from airo_camera_toolkit.utils import ImageConverter

namespace = "rgb"

# The publisher is passed a camera class and its kwargs, so it can be the (single) creator of the camera in its process.
publisher = MultiprocessRGBPublisher(
    Zed2i,
    camera_kwargs={
        "resolution": Zed2i.RESOLUTION_720,
        "fps": 15,
        "depth_mode": Zed2i.NONE_DEPTH_MODE,
    },
    shared_memory_namespace=namespace,
)
publisher.start()

# The receiver only need to know the shared memory namespace of the publisher.
camera = MultiprocessRGBReceiver(namespace)

# From this point on, the camera behaves like a regular airo-camera-toolkit camera
while True:
    image_rgb = camera.get_rgb_image_as_int()
    image = ImageConverter.from_numpy_int_format(image_rgb).image_in_opencv_format
    cv2.imshow(namespace, image)
    key = cv2.waitKey(10)
    if key == ord("q"):
        break
