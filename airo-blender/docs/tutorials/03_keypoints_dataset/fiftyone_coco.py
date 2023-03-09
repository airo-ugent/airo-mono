import argparse
import os

import fiftyone as fo

# Usage: python fiftyone_coco.py dataset0/annotations.json
parser = argparse.ArgumentParser()
parser.add_argument("labels_json_path", type=str)

args = parser.parse_args()
labels_json_path = os.path.realpath(args.labels_json_path)
dataset_dir = os.path.dirname(labels_json_path)

dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    label_types=["detections", "segmentations", "keypoints"],
    data_path=dataset_dir,
    dataset_dir=dataset_dir,
    labels_path=labels_json_path,
)

session = fo.launch_app(dataset)
session.wait()
