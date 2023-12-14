"""merge 2 coco datasets into one"""

import json
import pathlib
import shutil

import tqdm
from airo_dataset_tools.data_parsers.coco import CocoInstancesDataset


def merge_coco_annotations(dataset1: CocoInstancesDataset, dataset2: CocoInstancesDataset) -> CocoInstancesDataset:
    """merge 2 coco annotations schemas. Categories and Annotations are assumed to be unique. Images can be present in both datasets.

    Images will be checked for duplicates based on their file name.

    Annotation IDs will be changed to avoid conflicts and their image IDs will be updated if needed."""
    categories_1 = dataset1.categories
    categories_2 = dataset2.categories

    merged_categories = list(categories_1)
    for category in categories_2:
        if category.id not in [category.id for category in merged_categories]:
            merged_categories.append(category)
        else:
            same_id_category = merged_categories[[category.id for category in merged_categories].index(category.id)]
            if category != same_id_category:
                raise ValueError(
                    "Categories with the same ID are not equal: " + str(category) + " " + str(same_id_category)
                )

    merged_images = dataset1.images
    max_dataset_image_id = max([image.id for image in merged_images])
    merged_image_paths = [image.file_name for image in merged_images]

    dataset_2_image_id_mapping = {}
    for image in dataset2.images:
        if image.file_name not in merged_image_paths:
            # create a new ID to avoid collisions
            old_id = image.id
            image.id = max_dataset_image_id + 1
            max_dataset_image_id += 1
            dataset_2_image_id_mapping[old_id] = image.id
            merged_images.append(image)
            merged_image_paths.append(image.file_name)
        else:
            # dataset already contained this image, so we need to remap the annotations
            dataset_2_image_id_mapping[image.id] = merged_images[merged_image_paths.index(image.file_name)].id

    merged_annotations = list(dataset1.annotations)
    max_dataset_annotation_id = max([annotation.id for annotation in merged_annotations])
    for annotation in dataset2.annotations:
        if annotation.image_id in dataset_2_image_id_mapping.keys():
            annotation.image_id = dataset_2_image_id_mapping[annotation.image_id]
        annotation.id = max_dataset_annotation_id + 1
        max_dataset_annotation_id += 1
        merged_annotations.append(annotation)

    merged_dataset = CocoInstancesDataset(
        categories=merged_categories, images=merged_images, annotations=merged_annotations
    )
    return merged_dataset


def merge_coco_image_folders(dataset1_base_dir: str, dataset2_base_dir: str, target_dir: str) -> None:
    """merge 2 image folders into one. Images will be copied to the target folder.

    all base_dirs have following setup:

    base_dir
    --- images
    ------ image1.<>
    ------ image2.<>
    <>.json // annotations with path relative to base_dir
    """

    dataset1_base_dir_path = pathlib.Path(dataset1_base_dir)
    dataset2_base_dir_path = pathlib.Path(dataset2_base_dir)
    target_dir_path = pathlib.Path(target_dir)

    dataset1_image_paths = [image_path for image_path in dataset1_base_dir_path.iterdir()]
    dataset2_image_paths = [image_path for image_path in dataset2_base_dir_path.iterdir()]

    target_image_dir = target_dir_path / "images"
    target_image_dir.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm.tqdm(
        dataset1_image_paths, desc=f"copying images from {dataset1_base_dir_path.name} to {target_dir_path.name}"
    ):
        shutil.copy(image_path, target_image_dir / image_path.name)

    for image_path in tqdm.tqdm(
        dataset2_image_paths, desc=f"copying images from {dataset2_base_dir_path.name} to {target_dir_path.name}"
    ):
        if not (target_image_dir / image_path.name).exists():
            shutil.copy(image_path, target_image_dir / image_path.name)


def merge_coco_datasets(json_path_1: str, json_path_2: str, target_json_path: str) -> None:
    """merge 2 coco datasets into one. Categories and Annotations are assumed to be unique. Images can be present in both datasets.

    Images will be checked for duplicates based on their file name.

    Annotation IDs will be changed to avoid conflicts and their image IDs will be updated if needed."""

    image_path_1 = pathlib.Path(json_path_1).parent / "images"
    image_path_2 = pathlib.Path(json_path_2).parent / "images"

    merge_coco_image_folders(str(image_path_1), str(image_path_2), str(pathlib.Path(target_json_path).parent))

    dataset1 = None
    with open(json_path_1, "r") as f:
        dataset1 = CocoInstancesDataset(**json.load(f))

    dataset2 = None
    with open(json_path_2, "r") as f:
        dataset2 = CocoInstancesDataset(**json.load(f))

    merged_dataset = merge_coco_annotations(dataset1, dataset2)

    with open(target_json_path, "w") as f:
        json.dump(merged_dataset.model_dump(exclude_none=True), f)


if __name__ == "__main__":
    json1 = (
        "/home/tlips/Documents/airo-mono/airo-dataset-tools/test/test_data/lego-battery-resized/annotations_train.json"
    )
    json2 = (
        "/home/tlips/Documents/airo-mono/airo-dataset-tools/test/test_data/lego-battery-resized/annotations_val.json"
    )
    target_json = "/home/tlips/Documents/airo-mono/airo-dataset-tools/test/test_data/lego-battery-resize-merged/annotations_merged.json"
    merge_coco_datasets(json1, json2, target_json)
