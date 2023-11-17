import json
import os

import pytest
from airo_dataset_tools.coco_tools.split_dataset import split_coco_dataset
from airo_dataset_tools.data_parsers.coco import CocoInstancesDataset


def test_keypoints_split():
    test_dir = os.path.dirname(os.path.realpath(__file__))
    annotations = os.path.join(test_dir, "test_data/person_keypoints_val2017_small.json")

    with open(annotations, "r") as file:
        data = json.load(file)
        coco_keypoints = CocoInstancesDataset(**data)
        datasets = split_coco_dataset(coco_keypoints, [0.5, 0.5], shuffle_before_splitting=False)
        assert len(datasets) == 2
        assert len(datasets[0].annotations) == 2
        assert len(datasets[1].annotations) == 3
        assert len(datasets[0].images) == 1
        assert len(datasets[1].images) == 1
        assert len(datasets[0].annotations[0].keypoints) > 0


def test_empty_annotations_split_raises_error():
    test_dir = os.path.dirname(os.path.realpath(__file__))
    annotations = os.path.join(test_dir, "test_data/person_keypoints_val2017_small.json")

    with open(annotations, "r") as file:
        data = json.load(file)
        coco_keypoints = CocoInstancesDataset(**data)
        with pytest.raises(ValueError):
            split_coco_dataset(coco_keypoints, [0.9, 0.1], shuffle_before_splitting=False)
