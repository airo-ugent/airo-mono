import json
import os
import pathlib

from airo_dataset_tools.coco_tools.merge_datasets import merge_coco_datasets
from airo_dataset_tools.coco_tools.split_dataset import split_and_save_coco_dataset
from airo_dataset_tools.data_parsers.coco import CocoKeypointsDataset


def test_split_and_merge():
    dataset_path = pathlib.Path(__file__).parent / "test_data" / "lego-battery-resized" / "annotations.json"
    split_and_save_coco_dataset(str(dataset_path), [0.6, 0.4])
    json1 = pathlib.Path(__file__).parent / "test_data" / "lego-battery-resized" / "annotations_train.json"
    json2 = pathlib.Path(__file__).parent / "test_data" / "lego-battery-resized" / "annotations_val.json"
    target_json = (
        pathlib.Path(__file__).parent / "test_data" / "lego-battery-resize-merged" / "annotations_merged.json"
    )
    merged_dataset = merge_coco_datasets(str(json1), str(json2), str(target_json))
    assert target_json.exists()
    merged_dataset = CocoKeypointsDataset(**json.load(target_json.open("r")))
    original_dataset = CocoKeypointsDataset(**json.load(dataset_path.open("r")))
    # some basic checks to see if the merge was successful
    assert len(merged_dataset.images) == len(original_dataset.images)
    assert len(merged_dataset.annotations) == len(original_dataset.annotations)
    assert len(merged_dataset.categories) == len(original_dataset.categories)

    os.remove(json1)
    os.remove(json2)
    os.remove(target_json)
