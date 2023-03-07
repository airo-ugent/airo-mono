# airo-dataset-tools

Package for creating, saving and loading datasets

This functionality is mainly provided in the form of [Pydantic](https://docs.pydantic.dev/) parsers.

[COCO](https://cocodataset.org/#format-data) is the preferred format for computer vision datasets. Other formats will be added if they are needed for dataset creation (think the format of a labeling tool) or for consumption of the dataset (think the YOLO format for training an object detector).

Besides datasets, we also plan to provide parsers for other persistent data such as camera intrinsics and extrinsics.

