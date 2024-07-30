import json
import os
from typing import Any, Callable, List, Optional

import albumentations as A
import cv2
import numpy as np
import tqdm
from airo_dataset_tools.data_parsers.coco import (
    CocoImage,
    CocoInstanceAnnotation,
    CocoInstancesDataset,
    CocoKeypointAnnotation,
    CocoKeypointsDataset,
)
from airo_dataset_tools.segmentation_mask_converter import BinarySegmentationMask
from PIL import Image
from pydantic import ValidationError as PydanticValidationError


def apply_transform_to_coco_dataset(  # type: ignore # noqa: C901
    transforms: List[A.DualTransform],
    coco_dataset: CocoInstancesDataset,
    image_path: str,
    target_image_path: str,
    image_name_filter: Optional[Callable[[str], bool]] = None,
) -> CocoInstancesDataset:
    """Apply a sequence of albumentations transforms to a coco dataset, transforming images and keypoints, bounding boxes and segmentation masks if they are present.
    Present means that all annotations in the coco dataset have a bbox annotation.

    Args:
        transforms: _description_
        coco_dataset: _description_
        image_path: folder relative to which the image paths in the coco dataset are specified
        target_image_path: folder relative to which the image paths in the transformed coco dataset will be specified
        image_name_filter: optional filter for which images to transform based on their full path. Defaults to None.

    """
    # check if this is a keypoints dataset
    try:
        coco_dataset = CocoKeypointsDataset(**coco_dataset.model_dump(exclude_none=False))
        transform_keypoints = all(annotation.keypoints is not None for annotation in coco_dataset.annotations)

    except PydanticValidationError:
        transform_keypoints = False
    # check if bboxes and masks are present
    transform_bbox = all(annotation.bbox is not None for annotation in coco_dataset.annotations)
    transform_segmentation = all(annotation.segmentation is not None for annotation in coco_dataset.annotations)
    print(f"Transforming keypoints = {transform_keypoints}")
    print(f"Transforming bbox = {transform_bbox}")
    print(f"Transforming segmentation = {transform_segmentation}")

    keypoint_parameters = A.KeypointParams(format="xy", remove_invisible=False) if transform_keypoints else None
    bbox_parameters = A.BboxParams(format="coco", label_fields=["bbox_dummy_labels"]) if transform_bbox else None

    transform = A.Compose(
        transforms,
        keypoint_params=keypoint_parameters,
        bbox_params=bbox_parameters,
    )

    # create mappings between images & all corresponding annotations
    image_object_id_to_image_mapping = {image.id: image for image in coco_dataset.images}
    image_to_annotations_mapping: dict[CocoImage, List[CocoInstanceAnnotation]] = {
        coco_dataset.images[i]: [] for i in range(len(coco_dataset.images))
    }
    for annotation in coco_dataset.annotations:
        image_to_annotations_mapping[image_object_id_to_image_mapping[annotation.image_id]].append(annotation)

    for coco_image, annotations in tqdm.tqdm(image_to_annotations_mapping.items()):
        if image_name_filter is not None and image_name_filter(coco_image.file_name):
            print(f"skipping image {coco_image.file_name}")
            continue

        # load image
        pil_image = Image.open(os.path.join(image_path, coco_image.file_name)).convert(
            "RGB"
        )  # convert to RGB to avoid problems with PNG images
        image = np.array(pil_image)

        # combine annotations for all Annotation Instances related to the image
        # to transform them together with the image
        all_keypoints_xy = []
        all_bboxes = []
        all_masks = []
        for annotation in annotations:
            if transform_keypoints:
                assert isinstance(annotation, CocoKeypointAnnotation)
                all_keypoints_xy.extend(annotation.keypoints)
                # convert coco keypoints to list of (x,y) keypoints

            if transform_bbox:
                bbox = annotation.bbox
                assert bbox is not None  # for mypy
                # set bbox width, height to at least 1
                if bbox[3] < 1 or bbox[2] < 1:
                    # x_min must be < x_max for albumentations check
                    bbox = (0.0, 0.0, 1.0, 1.0)
                    print(
                        f"Invalid bbox for image {coco_image.file_name} and annotation {annotation.id}. Setting to [0.0, 0.0, 1.0, 1.0]"
                    )
                all_bboxes.append(bbox)

            if transform_segmentation:
                # convert segmentation to binary mask
                mask = annotation.segmentation
                assert mask is not None
                bitmap = BinarySegmentationMask.from_coco_segmentation_mask(
                    mask, coco_image.width, coco_image.height
                ).bitmap
                all_masks.append(bitmap)

        if transform_keypoints:
            all_keypoints_xy_nested = [all_keypoints_xy[i : i + 2] for i in range(0, len(all_keypoints_xy), 3)]

        arg_dict: dict[str, Any] = {
            "image": image,
        }
        if transform_keypoints:
            arg_dict["keypoints"] = all_keypoints_xy_nested
        if transform_bbox:
            arg_dict["bboxes"] = all_bboxes
            arg_dict["bbox_dummy_labels"] = [0 for _ in all_bboxes]
        if transform_segmentation:
            arg_dict["masks"] = all_masks
        transformed = transform(**arg_dict)

        # save transformed image
        transformed_image = transformed["image"]
        transformed_image = Image.fromarray(transformed_image)
        transformed_image_dir = os.path.join(target_image_path, os.path.dirname(coco_image.file_name))
        if not os.path.exists(transformed_image_dir):
            os.makedirs(transformed_image_dir)
        # specify quality to use for JPEG, (format is determined by file extension)
        # transformed_image.save(os.path.join(target_image_path, coco_image.file_name))
        # convert to BGR for opencv
        transformed_image_cv2 = cv2.cvtColor(np.array(transformed_image), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(target_image_path, coco_image.file_name), transformed_image_cv2)

        # change the metadata of the image coco object
        coco_image.width = transformed_image.width
        coco_image.height = transformed_image.height

        # store all modified annotations back into the coco dataset
        if transform_keypoints:
            all_transformed_keypoints_xy = transformed["keypoints"]
        if transform_bbox:
            all_transformed_bboxes = transformed["bboxes"]
        if transform_segmentation:
            all_transformed_masks = transformed["masks"]
        for annotation in annotations:
            if transform_keypoints:
                assert isinstance(annotation, CocoKeypointAnnotation)
                transformed_keypoints = all_transformed_keypoints_xy[: len(annotation.keypoints) // 3]
                all_transformed_keypoints_xy = all_transformed_keypoints_xy[len(annotation.keypoints) // 3 :]

                # set keypoints that are no longer in image to (0,0,0)
                for i, kp in enumerate(transformed_keypoints):
                    if 0 <= kp[0] < coco_image.width and 0 <= kp[1] < coco_image.height:
                        # add original visibility flag
                        transformed_keypoints[i] = [kp[0], kp[1], annotation.keypoints[i * 3 + 2]]
                    else:
                        transformed_keypoints[i] = [0.0, 0.0, 0]

                flattened_transformed_keypoints = [i for kp in transformed_keypoints for i in kp]
                annotation.keypoints = flattened_transformed_keypoints

            if transform_bbox:
                transformed_bbox = all_transformed_bboxes.pop(0)  # exactly one bbox per annotation
                annotation.bbox = transformed_bbox

            if transform_segmentation:
                transformed_segmentations = all_transformed_masks.pop(0)  # exactly one segmentation per annotation
                annotation.segmentation = BinarySegmentationMask(transformed_segmentations).as_polygon

    return coco_dataset


def resize_coco_dataset(
    annotations_json_path: str, width: int, height: int, target_dataset_dir: Optional[str] = None
) -> None:
    """Resize a COCO dataset. Will create a new directory at the `target_dataset_dir`with the resized dataset.
    Dataset is assumed to be
    /dir
        annotations.json # contains relative paths w.r.t. /dir
        ...
    """
    coco_dataset_dir = os.path.dirname(annotations_json_path)
    annotations_file_name = os.path.basename(annotations_json_path)
    dataset_parent_dir = os.path.dirname(coco_dataset_dir)
    if not target_dataset_dir:
        transformed_dataset_dir = os.path.join(
            dataset_parent_dir, f"{annotations_file_name.split('.')[0]}_resized_{width}x{height}"
        )
    else:
        transformed_dataset_dir = target_dataset_dir
    os.makedirs(transformed_dataset_dir, exist_ok=True)

    transforms = [A.Resize(height, width)]
    coco_json = json.load(open(annotations_json_path, "r"))
    coco_dataset = CocoInstancesDataset(**coco_json)
    transformed_dataset = apply_transform_to_coco_dataset(
        transforms, coco_dataset, coco_dataset_dir, transformed_dataset_dir
    )

    with open(os.path.join(transformed_dataset_dir, annotations_file_name), "w") as f:
        json.dump(transformed_dataset.model_dump(exclude_none=True), f)


if __name__ == "__main__":
    """example usage of the above function to resize all images in a coco dataset.
    Copy the following lines into your own codebase and modify as needed."""
    import pathlib

    path = pathlib.Path(__file__).parents[1] / "cvat_labeling" / "example" / "coco.json"

    coco_json_path = str(path)
    resize_coco_dataset(coco_json_path, 640, 480)
