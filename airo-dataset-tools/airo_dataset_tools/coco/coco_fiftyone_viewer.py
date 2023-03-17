import os
from typing import List

import fiftyone as fo


def view_coco_dataset(labels_json_path: str, dataset_dir: str = None, label_types: List[str] = None):
    """visualize a coco dataset in fiftyone"""
    if dataset_dir is None:
        dataset_dir = os.path.dirname(labels_json_path)
    if label_types is None:
        label_types = ["detections", "segmentations", "keypoints"]
    else:
        assert all([label_type in ["detections", "segmentations", "keypoints"] for label_type in label_types])

    labels_json_path = os.path.realpath(labels_json_path)

    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        label_types=label_types,
        data_path=dataset_dir,
        dataset_dir=dataset_dir,
        labels_path=labels_json_path,
    )

    session = fo.launch_app(dataset)
    session.wait()
