import json
import os

import numpy as np
from airo_dataset_tools.data_parsers.coco import CocoInstancesDataset
from airo_dataset_tools.segmentation_mask_converter import BinarySegmentationMask


def test_encoded_rle_creation():
    mask = np.zeros((10, 10))
    mask[0, 0] = 1
    mask[1, 1:3] = 1
    mask[2, 2:4] = 1
    segmentation_mask = BinarySegmentationMask(mask)

    compressed_rle = segmentation_mask.as_compressed_rle
    assert isinstance(compressed_rle, dict)
    assert "counts" in compressed_rle.keys()
    assert "size" in compressed_rle.keys()
    assert compressed_rle["size"] == [10, 10]
    assert isinstance(compressed_rle["counts"], str)

    loaded_segmentation_mask = BinarySegmentationMask.from_coco_segmentation_mask(compressed_rle, 10, 10)
    assert np.array_equal(loaded_segmentation_mask.bitmap, mask)


def test_polygon_creation():
    mask = np.zeros((10, 10))
    mask[1, 1:3] = 1
    mask[2, 1:3] = 1
    segmentation_mask = BinarySegmentationMask(mask)

    polygon = segmentation_mask.as_polygon
    assert isinstance(polygon, list)
    assert isinstance(polygon[0], list)
    assert isinstance(polygon[0][0], float)
    assert len(polygon) == 1
    assert len(polygon[0]) == 8
    assert polygon[0] == [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0]


def test_coco_segmentation_loading():
    test_dir = os.path.dirname(os.path.realpath(__file__))
    annotations = os.path.join(test_dir, "test_data/instances_val2017_small.json")
    with open(annotations, "r") as file:
        data = json.load(file)
        coco_instances = CocoInstancesDataset(**data)
        for annotation in coco_instances.annotations:
            for image in coco_instances.images:
                if image.id == annotation.image_id:
                    width = image.width
                    height = image.height
            segmentation_mask = BinarySegmentationMask.from_coco_segmentation_mask(
                annotation.segmentation, width, height
            )
            assert isinstance(segmentation_mask.bitmap, np.ndarray)
            assert segmentation_mask.bitmap.shape == (height, width)


if __name__ == "__main__":
    test_encoded_rle_creation()
    test_coco_segmentation_loading()
