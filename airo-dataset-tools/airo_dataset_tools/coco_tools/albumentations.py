import albumentations as A
from PIL import Image
import numpy as np 


class PillowResize(A.Resize):
    """Use Pillow (instead of OpenCV) to resize the input to the given height and width.
    always uses Bicubic interpolation.

    PIllow instead of Opencv because opencv does not adapt the filter size to the scaling factor,
    which creates artifacts in the output for large downscaling factors. 
    cf. https://arxiv.org/pdf/2104.11222.pdf

    Args:
        height (int): desired height of the output.
        width (int): desired width of the output.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(self, height, width, interpolation=Image.BICUBIC, always_apply=False, p=1):
        super(PillowResize, self).__init__(height,width, always_apply=always_apply, p=p)
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def apply(self, img, **params):
        return np.array(Image.fromarray(img).resize((self.width, self.height), self.interpolation))

