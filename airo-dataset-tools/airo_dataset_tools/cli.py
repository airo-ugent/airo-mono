"""CLI interface for this package"""

import json
import os
from typing import List, Optional

import click
from airo_dataset_tools.coco_tools.change_coco_images_prefix import change_coco_json_image_prefix
from airo_dataset_tools.coco_tools.coco_instances_to_yolo import create_yolo_dataset_from_coco_instances_dataset
from airo_dataset_tools.coco_tools.fiftyone_viewer import view_coco_dataset
from airo_dataset_tools.coco_tools.merge_datasets import merge_coco_datasets
from airo_dataset_tools.coco_tools.split_dataset import split_and_save_coco_dataset
from airo_dataset_tools.coco_tools.transform_dataset import resize_coco_dataset
from airo_dataset_tools.cvat_labeling.convert_cvat_to_coco import cvat_image_to_coco


@click.group()
def cli() -> None:
    """CLI entrypoint for airo-dataset-tools"""


@cli.command(name="fiftyone-coco-viewer")  # no help, takes the docstring of the function.
@click.argument("annotations-json-path", type=click.Path(exists=True))
@click.option(
    "--dataset-dir",
    required=False,
    type=click.Path(exists=True),
    help="optional directory relative to which the image paths in the coco dataset are specified",
)
@click.option(
    "--label-types",
    "-l",
    multiple=True,
    type=click.Choice(["detections", "segmentations", "keypoints"]),
    help="add an argument for each label type you want to load (default: all)",
)
def view_coco_dataset_cli(
    annotations_json_path: str, dataset_dir: str, label_types: Optional[List[str]] = None
) -> None:
    """Explore COCO dataset with FiftyOne"""
    view_coco_dataset(annotations_json_path, dataset_dir, label_types)


@cli.command(name="convert-cvat-to-coco-keypoints")
@click.argument("cvat_xml_file", type=str, required=True)
@click.argument("coco-categories-json-file", type=str, required=True)
@click.option("--add-bbox", is_flag=True, default=False, help="include bounding box in coco annotations")
@click.option("--add-segmentation", is_flag=True, default=False, help="include segmentation in coco annotations")
def convert_cvat_to_coco_cli(
    cvat_xml_file: str, coco_categories_json_file: str, add_bbox: bool, add_segmentation: bool
) -> None:
    """Convert CVAT XML to COCO keypoints json according to specified coco categories"""
    coco = cvat_image_to_coco(
        cvat_xml_file, coco_categories_json_file, add_bbox=add_bbox, add_segmentation=add_segmentation
    )
    path = os.path.dirname(cvat_xml_file)
    filename = os.path.basename(cvat_xml_file)
    path = os.path.join(path, filename.split(".")[0] + ".json")
    with open(path, "w") as file:
        json.dump(coco, file)


@cli.command(name="resize-coco-dataset")
@click.argument("annotations-json-path", type=click.Path(exists=True))
@click.option("--width", type=int, required=True)
@click.option("--height", type=int, required=True)
@click.option("--target-dataset-dir", type=str, required=False)
def resize_coco_dataset_cli(
    annotations_json_path: str, width: int, height: int, target_dataset_dir: Optional[str]
) -> None:
    """Resize a COCO dataset. Will create a new directory with the resized dataset at the specified target_dataset_dir.
    Dataset is assumed to be
    /dir
        annotations.json # contains relative paths w.r.t. /dir
        ...
    """
    resize_coco_dataset(annotations_json_path, width, height, target_dataset_dir=target_dataset_dir)


@cli.command(name="coco-instances-to-yolo")
@click.option("--coco-json", type=str)
@click.option("--target-dir", type=str)
@click.option("--use-segmentation", is_flag=True)
def coco_intances_to_yolo(coco_json: str, target_dir: str, use_segmentation: bool) -> None:
    """Create a YOLO detections/segmentations dataset from a coco instances dataset"""
    print(f"converting coco instances dataset {coco_json} to yolo dataset {target_dir}")
    create_yolo_dataset_from_coco_instances_dataset(coco_json, target_dir, use_segmentation=use_segmentation)


@cli.command(name="split-coco-dataset")
@click.argument("json-path", type=click.Path(exists=True))
@click.option("--split-ratio", type=float, multiple=True, required=True)
@click.option("--shuffle-before-splitting", is_flag=True, default=True)
def split_coco_dataset_cli(json_path: str, split_ratio: List[float], shuffle_before_splitting: bool) -> None:
    """Split a COCO dataset into subsets according to the specified relative ratios and save them to disk.
    Images are split with their corresponding annotations. No guarantees on class balance or annotation ratios.

    If two ratios are specified, the dataset will be split into two subsets. these will be called train/val by default.
    If three ratios are specified, the dataset will be split into three subsets. these will be called train/val/test by default.

    e.g. split-coco-dataset <path> --split-ratio 0.8 --split-ratio 0.2
    """
    split_and_save_coco_dataset(json_path, split_ratio, shuffle_before_splitting=shuffle_before_splitting)


@cli.command(name="change-coco-images-prefix")
@click.argument("coco-json", type=click.Path(exists=True))
@click.option("--current-prefix", type=str, required=True)
@click.option("--new-prefix", type=str, required=True)
@click.option("--target-json-path", type=str)
def change_coco_images_prefix_cli(
    coco_json: str, current_prefix: str, new_prefix: str, target_json_path: Optional[str] = None
) -> None:
    """change the prefix of images in a coco dataset."""
    return change_coco_json_image_prefix(coco_json, current_prefix, new_prefix, target_json_path)


@cli.command(name="merge-coco-datasets")
@click.argument("coco-json-1", type=click.Path(exists=True))
@click.argument("coco-json-2", type=click.Path(exists=True))
@click.option(
    "--target-json-path",
    type=str,
    help="optional path to save the merged dataset to. If none is provided, a new directory will be created in the parent directory of coco-json-1 ",
)
def merge_coco_datasets_cli(coco_json_1: str, coco_json_2: str, target_json_path: Optional[str] = None) -> None:
    """merge two coco datasets into a single dataset."""
    if not target_json_path:
        target_json_path = os.path.join(os.path.dirname(coco_json_1), "merged")
        os.makedirs(target_json_path, exist_ok=True)
        target_json_path = os.path.join(target_json_path, "annotations.json")
    return merge_coco_datasets(coco_json_1, coco_json_2, target_json_path)


if __name__ == "__main__":
    cli()
