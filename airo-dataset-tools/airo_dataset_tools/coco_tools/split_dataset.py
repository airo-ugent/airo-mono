""" Split a COCO dataset into subsets"""
import json
import pathlib
import random
from typing import List

from airo_dataset_tools.data_parsers.coco import CocoInstanceAnnotation, CocoInstancesDataset, CocoKeypointsDataset


def split_coco_dataset(
    coco_dataset: CocoInstancesDataset, split_ratios: List[float], shuffle_before_splitting: bool = True
) -> List[CocoInstancesDataset]:
    """Split a COCO dataset into subsets by splitting the images according to the specified relative ratios.
    All annotations for an image will be placed in the same subset as the image.

    Note that this does not guarantee the ratio of the annotations OR an equal class balance in each subset.

    Ratios must sum to 1.0.
    """

    ratio_sum = sum(split_ratios)
    if abs(ratio_sum - 1.0) > 2e-2:
        raise ValueError(f"Ratios must sum to 1.0. Ratios sum to {ratio_sum}.")

    # split the images into 2 subsets (random or ordered)
    images = coco_dataset.images

    if shuffle_before_splitting:
        random.shuffle(images)

    image_splits = []
    split_sizes = [round(ratio * len(images)) for ratio in split_ratios]
    split_sizes[-1] = len(images) - sum(split_sizes[:-1])  # make sure the total number of images is correct
    print(f"Split sizes: {split_sizes}")

    for size in split_sizes:
        image_splits.append(images[:size])
        images = images[size:]

    image_id_to_split_id = {}
    for split_id, image_split in enumerate(image_splits):
        for image in image_split:
            image_id_to_split_id[image.id] = split_id

    # gather the annotations for each subset
    annotation_splits: List[List[CocoInstanceAnnotation]] = [[] for _ in range(len(split_ratios))]
    for annotation in coco_dataset.annotations:
        image_id = annotation.image_id
        split_id = image_id_to_split_id[image_id]
        # keep original image_ids and annotation_ids so that you could still reference the original dataset
        annotation_splits[split_id].append(annotation)

    # create a new COCO dataset for each subset
    dataset_type: type
    if isinstance(coco_dataset, CocoKeypointsDataset):
        dataset_type = CocoKeypointsDataset
    else:
        dataset_type = CocoInstancesDataset

    coco_dataset_splits: List[CocoInstancesDataset] = []
    for annotation_split, image_split in zip(annotation_splits, image_splits):
        coco_dataset_split = dataset_type(
            categories=coco_dataset.categories, images=image_split, annotations=annotation_split
        )
        coco_dataset_splits.append(coco_dataset_split)

    return coco_dataset_splits


def split_and_save_coco_dataset(
    coco_json_path: str, split_ratios: List[float], shuffle_before_splitting: bool = True
) -> None:
    """Split a COCO dataset into subsets according to the specified relative ratios and save them to disk.
    Images are split with their corresponding annotations. No guarantees on class balance or annotation ratios.

    Ratios must sum to 1.0.

    If two ratios are specified, the dataset will be split into two subsets. these will be called train/val by default.
    If three ratios are specified, the dataset will be split into three subsets. these will be called train/val/test by default.
    """
    split_names = ["train", "val", "test"]
    if len(split_ratios) > len(split_names):
        raise ValueError(f"Only {len(split_names)} splits are supported. {len(split_ratios)} splits were specified.")

    coco_dataset: CocoInstancesDataset
    with open(coco_json_path, "r") as f:
        try:
            coco_dataset = CocoKeypointsDataset(**json.load(f))
        except TypeError:
            print("Could not load as CocoKeypointsDataset. Trying CocoInstancesDataset")
            coco_dataset = CocoInstancesDataset(**json.load(f))
        finally:
            if not isinstance(coco_dataset, CocoKeypointsDataset) and not isinstance(
                coco_dataset, CocoInstancesDataset
            ):
                raise ValueError("Could not load as CocoKeypointsDataset or CocoInstancesDataset")

    coco_dataset_splits = split_coco_dataset(coco_dataset, split_ratios, shuffle_before_splitting)

    for split_id, coco_dataset_split in enumerate(coco_dataset_splits):

        file_name = coco_json_path.replace(".json", f"_{split_names[split_id]}.json")
        with open(file_name, "w") as f:
            json.dump(coco_dataset_split.dict(), f)


if __name__ == "__main__":
    json_path = pathlib.Path(__file__).parents[2] / "test" / "test_data" / "instances_val2017_small.json"
    split_and_save_coco_dataset(str(json_path), [0.8, 0.2])
