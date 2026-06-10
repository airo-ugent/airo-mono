# airo-dataset-tools
Tools for working with datasets.
They fall into two categories:

[**COCO related tools**](airo_dataset_tools/coco_tools/README.md):
* COCO dataset loading (and creation)
* FiftyOne visualisation (optional, see [installation note](#fiftyone-installation))
* Albumentation transforms
* COCO  to YOLO conversion.
* CVAT labeling workflow

[**Data formats**](airo_dataset_tools/data_parsers/README.md):
* 3D poses
* Camera instrinsics


## FiftyOne installation

FiftyOne is an optional dependency. Install it with:

```bash
pip install "airo-dataset-tools[fiftyone]"
```

> **Note:** FiftyOne installs `opencv-python-headless`, which conflicts with `opencv-contrib-python` (both write to the same `cv2` namespace). To ensure the contrib build takes precedence, reinstall it after:
> ```bash
> pip install --force-reinstall opencv-contrib-python==4.10.0.84
> ```

> [Pydantic](https://docs.pydantic.dev/latest/) is used heavily throughout this package.
It allows you to easily create Python objects that can be saved and loaded to and from JSON files.

[**CVAT labeling workflow & tools**](airo_dataset_tools/cvat_labeling/readme.md)
