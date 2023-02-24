import os

import fiftyone as fo

dataset_dir = "/home/idlab185/airo-mono/airo-blender/docs/tutorials/03_keypoints_dataset"
labels_file = os.path.join(dataset_dir, "annotations.json")

dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    label_types=["detections", "segmentations", "keypoints"],
    data_path=dataset_dir,
    dataset_dir=dataset_dir,
    labels_path=labels_file,
)

session = fo.launch_app(dataset)
session.wait()
