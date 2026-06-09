import json
import os

import albumentations as A
from airo_dataset_tools.coco_tools.transform_dataset import apply_transform_to_coco_dataset
from airo_dataset_tools.data_parsers.coco import CocoInstancesDataset


def test_apply_transform_to_coco_dataset_empty_segmentation(tmp_path):
    """Test that apply_transform_to_coco_dataset skips segmentation transform when segmentation is []."""
    test_dir = os.path.dirname(os.path.realpath(__file__))
    annotations_path = os.path.join(test_dir, "test_data/lego-battery-resized/annotations.json")
    image_dir = os.path.join(test_dir, "test_data/lego-battery-resized")

    with open(annotations_path, "r") as f:
        data = json.load(f)

    coco_dataset = CocoInstancesDataset(**data)
    coco_dataset.annotations[0].segmentation = []

    target_image_path = str(tmp_path / "transformed")
    os.makedirs(target_image_path, exist_ok=True)

    result = apply_transform_to_coco_dataset(
        [A.NoOp()],
        coco_dataset,
        image_dir,
        target_image_path,
    )

    assert result is not None
    assert len(result.annotations) == len(coco_dataset.annotations)
