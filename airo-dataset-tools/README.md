# airo-dataset-tools
Package for working with datasets.

[COCO](https://cocodataset.org/#format-data) is the preferred format for computer vision datasets. We strictly follow their data format with 2 exceptions: segmentation masks are not required for Instance datasets, bounding boxes nor segmentation masks are required for keypoint datasets. This is to limit labeling effort for real-world datasets where you don't always need all annotation types.

Other formats will are added if they are needed for dataset creation (think the format of a labeling tool) or for consumption of the dataset (think the YOLO format for training an object detector). Besides datasets, we also provide tools for other persistent data such as camera intrinsics and extrinsics.

As always, we try to reuse existing tools/code as much as possible, but we have found that keypoints are by far not supported as well as segmentation or detection, so we had to write some custom tooling for working with keypoints.

## Data Parsers
The functionality is mainly provided in the form of [Pydantic](https://docs.pydantic.dev/) parsers, that can be used to load or create data(sets). The parsers can be found in the `data_parsers` folder.

Avalaible Data Parsers:
* [COCO Datasets](https://cocodataset.org/#format-data)
* [CVAT 1.1 Images annotations](https://opencv.github.io/cvat/docs/manual/advanced/xml_format/)
* [Pose format](docs/pose.md)
* [Camera instrinsics format](docs/camera_intrinsics.md)

## COCO dataset creation
We provide a [documented](airo_dataset_tools/cvat_labeling/readme.md) worklow for labeling real-world data with [CVAT]() and to create [COCO]() Keypoints or Instance datasets based on these annotations.

We also provide a number of tools for working with COCO datasets:
- visualisation using [FiftyOne](https://voxel51.com/)
- applying Albumentation transforms (e.g. resizing, flipping,...) to a COCO Keypoints dataset and its annotations, see [here](airo_dataset_tools/coco_tools/transform_dataset.py)
- converting COCO instances to YOLO format, see [here](airo_dataset_tools/coco_tools/coco_instances_to_yolo.py)
- combining COCO datasets (via datumaro)(TODO)

Most of the COCO tools are available in the CLI, which you can access by running `airo-dataset-tools --help` from the command line.


