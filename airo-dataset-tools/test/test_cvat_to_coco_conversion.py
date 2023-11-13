from airo_dataset_tools.cvat_labeling.convert_cvat_to_coco import cvat_image_to_coco

from .test_cvat_images_load import COCO_CATEGORIES_PATH, CVAT_EXAMPLE_PATH


def test_conversion():
    coco_dict = cvat_image_to_coco(CVAT_EXAMPLE_PATH, COCO_CATEGORIES_PATH, add_bbox=True, add_segmentation=True)
    assert len(coco_dict["images"]) == 4
