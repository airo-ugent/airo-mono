from airo_dataset_tools.coco_tools.albumentations import PillowResize
import numpy as np 
import albumentations as A

def test_resize():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    resize = PillowResize(10,10)
    resized_img = resize(image=img)['image']
    assert resized_img.shape == (10,10,3)
