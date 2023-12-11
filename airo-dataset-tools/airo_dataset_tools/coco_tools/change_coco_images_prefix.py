import json
from typing import Optional

from airo_dataset_tools.data_parsers.coco import CocoInstancesDataset


def change_coco_dataset_image_prefix(
    coco_dataset: CocoInstancesDataset, current_prefix: str, target_prefix: str
) -> CocoInstancesDataset:
    """Change the prefix of the image paths in a COCO dataset. Can be used to change the base directory of the images in a dataset.

    Args:
        coco_dataset: the dataset to modify
        current_prefix: the current prefix of the image paths in the dataset
        target_prefix: the target prefix of the image paths in the dataset

    e.g.

    if the images are currently relative to the image folder inside the dataset and you want to make them relative to the dataset folder:
    change_coco_image_base_dir(coco_dataset, "", "images/")
    """

    for image in coco_dataset.images:
        image.file_name = image.file_name.removeprefix(current_prefix)
        image.file_name = target_prefix + image.file_name

    return coco_dataset


def change_coco_json_image_prefix(
    coco_json_file: str, current_prefix: str, target_prefix: str, target_json_file: Optional[str]
) -> None:
    with open(coco_json_file, "r") as f:
        coco_dataset = CocoInstancesDataset(**json.load(f))

    coco_dataset = change_coco_dataset_image_prefix(coco_dataset, current_prefix, target_prefix)
    if target_json_file is None:
        target_json_file = coco_json_file
    with open(target_json_file, "w") as f:
        json.dump(coco_dataset.model_dump(), f)
