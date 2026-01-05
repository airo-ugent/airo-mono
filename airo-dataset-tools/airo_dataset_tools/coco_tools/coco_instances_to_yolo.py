import json
import pathlib
from collections import defaultdict

import cv2
import tqdm
from airo_dataset_tools.data_parsers.coco import CocoInstancesDataset
from airo_dataset_tools.segmentation_mask_converter import BinarySegmentationMask


def create_yolo_dataset_from_coco_instances_dataset(
    coco_dataset_json_path: str, target_directory: str, use_segmentation: bool = False
) -> None:
    """Converts a coco dataset to a yolo dataset, either segmentations or detections


    coco dataset is expected:
    dataset/
        images/
        <>.json # containing paths relative to /dataset

    yolo dataset format will be

    dataset/
        images/
        labels/
        <>.txt # paths as in coco dataset /images subdir.

        # ultralytics uses an additional <>.json, this has to be created manually..
        # we do export a .names file, as in the darknet yolo format.
        # each line contains a class name, the line number is the class id
        obj.names
    Args:
        coco_dataset_json_path: _description_
        target_directory: _description_
        use_segmentation: create a segmentation dataset instead of a detection dataset (requires segmentation annotations)


    quiz question: how many people would have written a similar converter in the past?...

    """
    _coco_dataset_json_path = pathlib.Path(coco_dataset_json_path)
    _target_directory = pathlib.Path(target_directory)

    # load the json file into the coco dataset object
    annotations = json.load(open(_coco_dataset_json_path, "r"))
    coco_dataset = CocoInstancesDataset(**annotations)

    target_dir = pathlib.Path(_target_directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    image_dir = target_dir / "images"
    label_dir = target_dir / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    # create the yolo dataset
    annotations = coco_dataset.annotations
    image_id_to_image = {image.id: image for image in coco_dataset.images}
    image_id_to_annotations = defaultdict(list)
    for annotation in annotations:
        image_id_to_annotations[annotation.image_id].append(annotation)

    category_id_to_category = {category.id: category for category in coco_dataset.categories}
    # sort by original COCO ID, but whereas COCO does not care about ID ranges, YOLO expects them to be in the range 0..N-1
    # so we sort them by ID and then use the index as the new ID
    yolo_category_index = list(sorted(category_id_to_category.values(), key=lambda category: category.id))

    for image_id, annotations in tqdm.tqdm(image_id_to_annotations.items()):
        coco_image = image_id_to_image[image_id]
        image_path = pathlib.Path(coco_image.file_name)
        if not image_path.is_absolute():
            image_path = _coco_dataset_json_path.parent / image_path

        relative_image_path = image_path.relative_to(_coco_dataset_json_path.parent)

        # ultralytics parser finds the 'latest' occurance of 'images' in the dataset path,
        #  so we need to replace any occurance of 'image' with 'img'
        #  https://github.com/ultralytics/ultralytics/issues/3581

        relative_image_path_str = str(relative_image_path).replace("image", "img")
        relative_image_path = pathlib.Path(relative_image_path_str)

        image = cv2.imread(str(image_path))
        height, width, _ = image.shape

        label_path = label_dir / f"{relative_image_path.with_suffix('')}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)

        with open(label_path, "w") as file:
            for annotation in annotations:
                category = category_id_to_category[annotation.category_id]
                yolo_id = yolo_category_index.index(category)
                if use_segmentation:
                    segmentation = annotation.segmentation
                    # convert to **single** polygon
                    segmentation = BinarySegmentationMask.from_coco_segmentation_mask(segmentation, width, height)
                    segmentation = segmentation.as_single_polygon

                    if segmentation is None:
                        # should actually never happen as each annotation is assumed to have a segmentation if you pass use_segmentation=True
                        # but we filter it for convenience to deal with edge cases
                        print(f"skipping annotation for image {image_path}, as it has no segmentation")
                        continue

                    file.write(f"{yolo_id}")
                    for (x, y) in zip(segmentation[0::2], segmentation[1::2]):
                        file.write(f" {x/width} {y/height}")
                    file.write("\n")

                else:
                    x, y, w, h = annotation.bbox
                    x_center = x + w / 2
                    y_center = y + h / 2
                    x_center /= width
                    y_center /= height
                    w /= width
                    h /= height
                    file.write(f"{yolo_id} {x_center} {y_center} {w} {h}\n")

        image_target_path = image_dir / relative_image_path
        image_target_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_target_path), image)

    # create the obj.names file
    with open(target_dir / "obj.names", "w") as file:
        # sort categories by id
        for category in yolo_category_index:
            file.write(f"{category.name}\n")


if __name__ == "__main__":
    """example usage"""
    import os

    path = pathlib.Path(__file__).parents[1] / "cvat_labeling" / "example" / "coco.json"

    coco_json_path = str(path)
    coco_dir = os.path.dirname(coco_json_path)
    coco_file_name = os.path.basename(coco_json_path)
    coco_target_dir = os.path.join(os.path.dirname(coco_dir), "yolo_dataset")
    os.makedirs(coco_target_dir, exist_ok=True)

    create_yolo_dataset_from_coco_instances_dataset(coco_json_path, coco_target_dir, use_segmentation=True)
