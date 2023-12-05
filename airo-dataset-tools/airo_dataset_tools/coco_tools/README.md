# COCO

Tools for working with COCO datasets.
Most of the COCO tools are available in the CLI, which you can access by running `airo-dataset-tools --help` from the command line.

Overview of the functionality:
* [COCO](#dataset-loading) dataset loading
* COCO dataset creation (e.g. with synthetic data or CVAT)
* [CVAT labeling workflow](../../airo_dataset_tools/cvat_labeling/readme.md)
* FiftyOne visualisation (see CLI)
* Applying Albumentation transforms (e.g. resizing, flipping,...) to a COCO Keypoints dataset and its annotations, see [transform_dataset.py](transform_dataset.py)
* Converting COCO instances to YOLO format, see [coco_instances_to_yolo.py](coco_instances_to_yolo.py)

## Dataset loading
We provide two main dataset [classes](./airo_dataset_tools/data_parsers/coco.py) for working with COCO:
* `CocoInstancesDataset`: for COCO datasets without keypoints
* `CocoKeypointsDataset`: for COCO datasets with keypoints

You can read more about the COCO dataset format [here](https://cocodataset.org/#format-data).
It is our preferred format for computer vision datasets.

Loading a COCO dataset can be done as follows:
```python
from airo_dataset_tools.data_parsers.coco import CocoInstancesDataset

with open("./test/test_data.instances_val2017_small.json", "r") as file:
    dataset = CocoInstancesDataset.model_validate_json(file.read())

print(len(dataset.images))
print(len(dataset.annotations))
```

## Notes

[COCO](https://cocodataset.org/#format-data) is the preferred format for computer vision datasets. We strictly follow their data format with 2 exceptions:
* Segmentation masks are not required for Instance datasets,
* Bounding boxes nor segmentation masks are required for keypoint datasets.

This is to limit labeling effort for real-world datasets where you don't always need all annotation types.

Other formats will are added if they are needed for dataset creation (think the format of a labeling tool) or for consumption of the dataset (think the YOLO format for training an object detector). Besides datasets, we also provide tools for other persistent data such as camera intrinsics and extrinsics.

As always, we try to reuse existing tools/code as much as possible, but we have found that keypoints are by far not supported as well as segmentation or detection, so we had to write some custom tooling for working with keypoints.