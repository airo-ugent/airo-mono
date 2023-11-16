"""Tests for loading datasets in COCO format.

Part of the tests use annotations from the official COCO dataset.
For this we use a small subset of the 2017 Val annotations files, the full versions can be found here:
https://cocodataset.org/#download

We do not use the captions, so we only consider these files:
- instances_val2017.json
- person_keypoints_val2017.json
"""

import json
import os

import pytest
from airo_dataset_tools.data_parsers.coco import (
    CocoCategory,
    CocoInstancesDataset,
    CocoKeypointCategory,
    CocoKeypointsDataset,
)


def test_coco_load_instances():
    """Test whether we can correctly load the partial instances_val2017 dataset."""
    test_dir = os.path.dirname(os.path.realpath(__file__))
    annotations = os.path.join(test_dir, "test_data/instances_val2017_small.json")

    with open(annotations, "r") as file:
        data = json.load(file)
        coco_instances = CocoInstancesDataset(**data)
        # Check a few known lengths to ensure that all elements were loaded
        assert len(coco_instances.images) == 3
        assert len(coco_instances.categories) == 80
        assert len(coco_instances.annotations) == 34

        assert isinstance(coco_instances.categories[0], CocoCategory)


def test_coco_load_keypoints():
    """Test whether we can correctly load the partial person_keypoints_val2017 dataset."""
    test_dir = os.path.dirname(os.path.realpath(__file__))
    annotations = os.path.join(test_dir, "test_data/person_keypoints_val2017_small.json")

    with open(annotations, "r") as file:
        data = json.load(file)
        coco_keypoints = CocoKeypointsDataset(**data)
        assert len(coco_keypoints.images) == 2
        assert len(coco_keypoints.categories) == 1
        assert len(coco_keypoints.annotations) == 5

        assert isinstance(coco_keypoints.categories[0], CocoKeypointCategory)


def test_coco_load_keypoints_as_instances_keeps_additional_fields():
    # to parse data that actually belongs to a subclass, useful for some coco tools where it does not matter what category the dataset is..
    test_dir = os.path.dirname(os.path.realpath(__file__))
    annotations = os.path.join(test_dir, "test_data/person_keypoints_val2017_small.json")

    with open(annotations, "r") as file:
        data = json.load(file)
        coco_keypoints = CocoInstancesDataset(**data)
        assert len(coco_keypoints.annotations[0].keypoints) > 0


def test_coco_load_instances_incorrectly():
    """Test whether an exception is raised when we try to load the regular instances as a keypoints dataset."""
    test_dir = os.path.dirname(os.path.realpath(__file__))
    annotations = os.path.join(test_dir, "test_data/instances_val2017_small.json")

    with open(annotations, "r") as file:
        data = json.load(file)
        with pytest.raises(Exception):
            CocoKeypointsDataset(**data)


def test_coco_load_instances_no_segmasks():
    """Test whether we can correctly load the partial instances_val2017 dataset without segmentations."""
    test_dir = os.path.dirname(os.path.realpath(__file__))
    annotations = os.path.join(test_dir, "test_data/person_keypoints_val2017_small_no_segmentations.json")

    with open(annotations, "r") as file:
        data = json.load(file)
        coco_instances = CocoKeypointsDataset(**data)
        # Check a few known lengths to ensure that all elements were loaded
        assert len(coco_instances.images) == 2


def test_coco_load_keypoints_no_bboxes():
    """Test whether we can correctly load the partial person_keypoints_val2017 dataset without bounding boxes."""
    test_dir = os.path.dirname(os.path.realpath(__file__))
    annotations = os.path.join(test_dir, "test_data/person_keypoints_val2017_small_no_bbox.json")

    with open(annotations, "r") as file:
        data = json.load(file)
        coco_keypoints = CocoKeypointsDataset(**data)
        assert len(coco_keypoints.images) == 2
        assert len(coco_keypoints.categories) == 1
        assert len(coco_keypoints.annotations) == 2

        assert isinstance(coco_keypoints.categories[0], CocoKeypointCategory)
