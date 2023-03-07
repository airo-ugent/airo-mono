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
from airo_dataset_tools.coco.coco_parser import CocoInstances, CocoKeypoints


def test_coco_load_instances():
    """Test whether we can correctly load the partial instances_val2017 dataset."""
    test_dir = os.path.dirname(os.path.realpath(__file__))
    annotations = os.path.join(test_dir, "test_data/instances_val2017_small.json")

    with open(annotations, "r") as file:
        data = json.load(file)
        coco_instances = CocoInstances(**data)
        # Check a few known lengths to ensure that all elements were loaded
        assert len(coco_instances.images) == 3
        assert len(coco_instances.categories) == 80
        assert len(coco_instances.annotations) == 34


def test_coco_load_keypoints():
    """Test whether we can correctly load the partial person_keypoints_val2017 dataset."""
    test_dir = os.path.dirname(os.path.realpath(__file__))
    annotations = os.path.join(test_dir, "test_data/person_keypoints_val2017_small.json")

    with open(annotations, "r") as file:
        data = json.load(file)
        coco_keypoints = CocoKeypoints(**data)
        assert len(coco_keypoints.images) == 2
        assert len(coco_keypoints.categories) == 1
        assert len(coco_keypoints.annotations) == 2


def test_coco_load_instances_incorrectly():
    """Test whether an exception is raised when we try to load the regular instances as a keypoints dataset."""
    test_dir = os.path.dirname(os.path.realpath(__file__))
    annotations = os.path.join(test_dir, "test_data/instances_val2017_small.json")

    with open(annotations, "r") as file:
        data = json.load(file)
        with pytest.raises(Exception):
            CocoKeypoints(**data)
