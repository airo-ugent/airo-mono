"""CLI interface for this package"""

from typing import List, Optional

import click
from airo_dataset_tools.coco.coco_fiftyone_viewer import view_coco_dataset


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
    """Command for exploring COCO-formatted datasets with FiftyOne"""
    view_coco_dataset(annotations_json_path, dataset_dir, label_types)
