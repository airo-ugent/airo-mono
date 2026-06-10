# COCO

Tools for working with COCO datasets.
Most of the COCO tools are available in the CLI, which you can access by running `airo-dataset-tools --help` from the command line.

Overview of the functionality:
* [COCO](#dataset-loading) dataset loading
* COCO dataset creation (e.g. with synthetic data or CVAT)
* [CVAT labeling workflow](../../airo_dataset_tools/cvat_labeling/readme.md)
* [FiftyOne visualisation](#fiftyone-visualisation)* [FiftyOne visualisation](see CLI, requires the `[fiftyone]` extra — see [installation note](../../README.md#fiftyone-installation))
* [Merging datasets](#merging-datasets) — combine two COCO datasets into one, see [merge_datasets.py](merge_datasets.py)
* [Splitting datasets](#splitting-datasets) — split a dataset into train/val/test subsets, see [split_dataset.py](split_dataset.py)
* [Applying transforms](#applying-transforms) — resize, flip, etc. via Albumentations, see [transform_dataset.py](transform_dataset.py)
* [Changing image prefixes](#changing-image-prefixes) — rebase image paths in a dataset, see [change_coco_images_prefix.py](change_coco_images_prefix.py)
* [Converting to YOLO format](#converting-to-yolo-format) — export instances as YOLO detection/segmentation labels, see [coco_instances_to_yolo.py](coco_instances_to_yolo.py)

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

## Merging datasets

`merge_coco_datasets(json_path_1, json_path_2, target_json_path)` merges two COCO datasets (annotations + images) into a single output directory. Images from both datasets are copied to the target directory, preserving any nested subdirectory structure. Duplicate images (same filename) are skipped. Annotation IDs are remapped to avoid conflicts.

```python
from airo_dataset_tools.coco_tools.merge_datasets import merge_coco_datasets

merge_coco_datasets("dataset_a/annotations.json", "dataset_b/annotations.json", "merged/annotations.json")
```

## Splitting datasets

`split_and_save_coco_dataset(json_path, ratios)` splits a dataset into subsets (e.g. train/val) by splitting the images according to the given ratios. All annotations for an image follow it into the same subset. The split files are saved alongside the original JSON.

```python
from airo_dataset_tools.coco_tools.split_dataset import split_and_save_coco_dataset

split_and_save_coco_dataset("dataset/annotations.json", [0.8, 0.2])
# saves dataset/annotations_train.json and dataset/annotations_val.json
```

## Applying transforms

`apply_transform_to_coco_dataset(transforms, dataset, image_dir, target_dir)` applies a list of [Albumentations](https://albumentations.ai/) transforms to all images and their annotations (bboxes, keypoints, segmentation masks). Useful for resizing or augmenting a dataset while keeping annotations in sync.

```python
import albumentations as A
from airo_dataset_tools.coco_tools.transform_dataset import apply_transform_to_coco_dataset

apply_transform_to_coco_dataset([A.Resize(640, 480)], dataset, "dataset/images", "resized/images")
```

## Changing image prefixes

`change_coco_dataset_image_prefix(dataset, current_prefix, target_prefix)` rewrites the `file_name` field of every image entry, replacing `current_prefix` with `target_prefix`. Useful when moving a dataset or changing whether paths are relative to the dataset root vs. the images subfolder.

```python
from airo_dataset_tools.coco_tools.change_coco_images_prefix import change_coco_dataset_image_prefix

dataset = change_coco_dataset_image_prefix(dataset, "", "images/")
```

## Converting to YOLO format

`create_yolo_dataset_from_coco_instances_dataset(json_path, target_dir, use_segmentation)` converts a COCO instances dataset to YOLO format. Set `use_segmentation=True` for segmentation labels, or `False` (default) for bounding-box detection labels.

```python
from airo_dataset_tools.coco_tools.coco_instances_to_yolo import create_yolo_dataset_from_coco_instances_dataset

create_yolo_dataset_from_coco_instances_dataset("dataset/annotations.json", "yolo_dataset/", use_segmentation=False)
```

## FiftyOne visualisation

Use the CLI to open a dataset in the [FiftyOne](https://voxel51.com/fiftyone/) interactive viewer:

```bash
airo-dataset-tools fiftyone-viewer --dataset-json dataset/annotations.json
```

## Notes

[COCO](https://cocodataset.org/#format-data) is the preferred format for computer vision datasets. We strictly follow their data format with 2 exceptions:
* Segmentation masks are not required for Instance datasets,
* Bounding boxes nor segmentation masks are required for keypoint datasets.

This is to limit labeling effort for real-world datasets where you don't always need all annotation types.

Other formats will are added if they are needed for dataset creation (think the format of a labeling tool) or for consumption of the dataset (think the YOLO format for training an object detector). Besides datasets, we also provide tools for other persistent data such as camera intrinsics and extrinsics.

As always, we try to reuse existing tools/code as much as possible, but we have found that keypoints are by far not supported as well as segmentation or detection, so we had to write some custom tooling for working with keypoints.