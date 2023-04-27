"""CLI interface for this package"""

import json
import os
from typing import List, Optional

import click
from airo_dataset_tools.cvat_labeling.convert_cvat_to_coco import cvat_image_to_coco
from airo_dataset_tools.fiftyone_viewer import view_coco_dataset


@click.group()
def cli() -> None:
    """CLI entrypoint for airo-dataset-tools"""


@cli.command(name="fiftyone-coco-viewer")  # no help, takes the docstring of the function.
@click.argument("annotations-json-path", type=click.Path(exists=True))
@click.option("--dataset-dir", required=False, type=click.Path(exists=True))
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
@click.option("--add_bbox", type=bool, default=True, help="include bounding box in coco annotations")
@click.option("--add_segmentation", type=bool, default=True, help="include segmentation in coco annotations")
def convert_cvat_to_coco_cli(cvat_xml_file: str, add_bbox: bool, add_segmentation: bool) -> None:
    """Convert CVAT XML to COCO keypoints json"""
    coco = cvat_image_to_coco(cvat_xml_file, add_bbox=add_bbox, add_segmentation=add_segmentation)
    path = os.path.dirname(cvat_xml_file)
    path = os.path.join(path, "coco.json")
    with open(path, "w") as file:
        json.dump(coco, file)
