import json
import os
import pathlib
import shutil

from airo_dataset_tools.coco_tools.merge_datasets import merge_coco_datasets
from airo_dataset_tools.coco_tools.split_dataset import split_and_save_coco_dataset
from airo_dataset_tools.data_parsers.coco import CocoInstancesDataset, CocoKeypointsDataset


def test_split_and_merge():
    dataset_path = pathlib.Path(__file__).parent / "test_data" / "lego-battery-resized" / "annotations.json"
    split_and_save_coco_dataset(str(dataset_path), [0.6, 0.4])
    json1 = pathlib.Path(__file__).parent / "test_data" / "lego-battery-resized" / "annotations_train.json"
    json2 = pathlib.Path(__file__).parent / "test_data" / "lego-battery-resized" / "annotations_val.json"
    target_json = (
        pathlib.Path(__file__).parent / "test_data" / "lego-battery-resize-merged" / "annotations_merged.json"
    )
    merged_dataset = merge_coco_datasets(str(json1), str(json2), str(target_json))
    assert target_json.exists()
    merged_dataset = CocoKeypointsDataset(**json.load(target_json.open("r")))
    original_dataset = CocoKeypointsDataset(**json.load(dataset_path.open("r")))
    # some basic checks to see if the merge was successful
    assert len(merged_dataset.images) == len(original_dataset.images)
    assert len(merged_dataset.annotations) == len(original_dataset.annotations)
    assert len(merged_dataset.categories) == len(original_dataset.categories)

    os.remove(json1)
    os.remove(json2)
    os.remove(target_json)


def test_split_and_merge_nested_images(tmp_path):
    """Test that merge_coco_datasets preserves nested subdirectory structure for images."""
    # Create dataset1 with an image in a nested subdirectory
    dataset1_dir = tmp_path / "dataset1"
    dataset1_subdir = dataset1_dir / "images" / "subdir"
    dataset1_subdir.mkdir(parents=True)

    # Copy a real image into the nested subdir
    source_image = (
        pathlib.Path(__file__).parent / "test_data" / "lego-battery-resized" / "images" / "IMG20231010083824.jpg"
    )
    nested_image_path = dataset1_subdir / "img.jpg"
    shutil.copy(source_image, nested_image_path)

    # Create dataset2 with a flat image
    dataset2_dir = tmp_path / "dataset2"
    dataset2_flat_image_dir = dataset2_dir / "images"
    dataset2_flat_image_dir.mkdir(parents=True)
    flat_image_path = dataset2_flat_image_dir / "img2.jpg"
    shutil.copy(source_image, flat_image_path)

    # Build minimal COCO annotation JSON files referencing these images
    dataset1_json = {
        "categories": [{"supercategory": "", "id": 1, "name": "battery", "keypoints": ["top"]}],
        "images": [{"id": 1, "width": 512, "height": 256, "file_name": "images/subdir/img.jpg"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "keypoints": [100.0, 50.0, 2.0],
                "num_keypoints": 1,
            }
        ],
    }
    dataset2_json = {
        "categories": [{"supercategory": "", "id": 1, "name": "battery", "keypoints": ["top"]}],
        "images": [{"id": 1, "width": 512, "height": 256, "file_name": "images/img2.jpg"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "keypoints": [200.0, 100.0, 2.0],
                "num_keypoints": 1,
            }
        ],
    }

    json_path_1 = dataset1_dir / "annotations.json"
    json_path_2 = dataset2_dir / "annotations.json"
    with open(json_path_1, "w") as f:
        json.dump(dataset1_json, f)
    with open(json_path_2, "w") as f:
        json.dump(dataset2_json, f)

    target_dir = tmp_path / "merged"
    target_json = target_dir / "annotations_merged.json"

    merge_coco_datasets(str(json_path_1), str(json_path_2), str(target_json))

    # Verify the merged JSON was created
    assert target_json.exists()

    # Verify the nested image structure was preserved in the target directory
    assert (
        target_dir / "images" / "subdir" / "img.jpg"
    ).exists(), "Nested image should be preserved at images/subdir/img.jpg"
    assert (target_dir / "images" / "img2.jpg").exists(), "Flat image should be preserved at images/img2.jpg"

    # Verify the merged annotation references both images
    merged = CocoInstancesDataset(**json.load(target_json.open("r")))
    assert len(merged.images) == 2
    assert len(merged.annotations) == 2
    file_names = {img.file_name for img in merged.images}
    assert "images/subdir/img.jpg" in file_names
    assert "images/img2.jpg" in file_names
