import os
from typing import Callable, List

import albumentations as A
import numpy as np
import tqdm
from airo_dataset_tools.data_parsers.coco import CocoKeypointAnnotation, CocoKeypointsDataset
from airo_dataset_tools.segmentation_mask_converter import BinarySegmentationMask
from PIL import Image


def apply_transform_to_coco_dataset(
    transforms: List[A.DualTransform],
    coco_dataset: CocoKeypointsDataset,
    image_path: str,
    target_image_path: str,
    image_name_filter: Callable[[str], bool] = None,
) -> CocoKeypointsDataset:
    """Apply a sequence of albumentations transforms to a coco dataset, transforming images, keypoints, bounding boxes and segmentation masks.

    Args:
        transforms (List[A.DualTransform]): _description_
        coco_dataset (CocoKeypointsDataset): _description_
        image_path (str): folder relative to which the image paths in the coco dataset are specified
        target_image_path (str): folder relative to which the image paths in the transformed coco dataset will be specified
        image_name_filter (Callable[[str], bool], optional): optional filter for which images to transform based on their full path. Defaults to None.

    Returns:
        CocoKeypointsDataset: _description_
    """

    transform = A.Compose(
        transforms,
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        bbox_params=A.BboxParams(format="coco", label_fields=["bbox_dummy_labels"]),
    )

    # create mappings between images & all corresponding annotations
    image_object_id_to_image_mapping = {image.id: image for image in coco_dataset.images}
    image_to_annotations_mapping = {coco_dataset.images[i]: [] for i in range(len(coco_dataset.images))}
    for annotation in coco_dataset.annotations:
        image_to_annotations_mapping[image_object_id_to_image_mapping[annotation.image_id]].append(annotation)

    for coco_image, annotations in tqdm.tqdm(image_to_annotations_mapping.items()):
        if image_name_filter is not None and image_name_filter(coco_image.file_name):
            print(f"skipping image {coco_image.file_name}")
            continue

        # load image
        image = Image.open(os.path.join(image_path, coco_image.file_name)).convert(
            "RGB"
        )  # convert to RGB to avoid problems with PNG images
        image = np.array(image)
        print(image.shape)

        # combine annotations for all Annotation Instances related to the image
        # to transform them together with the image
        all_keypoints_xy = []
        all_bboxes = []
        all_masks = []
        for annotation in annotations:
            assert isinstance(annotation, CocoKeypointAnnotation)
            all_keypoints_xy.extend(annotation.keypoints)
            all_bboxes.append(annotation.bbox)
            # convert segmentation to binary mask
            mask = annotation.segmentation
            bitmap = BinarySegmentationMask.from_coco_segmentation_mask(
                mask, coco_image.width, coco_image.height
            ).bitmap
            print(bitmap.shape)
            all_masks.append(bitmap)

        # convert coco keypoints to list of (x,y) keypoints
        all_keypoints_xy = [all_keypoints_xy[i : i + 2] for i in range(0, len(all_keypoints_xy), 3)]

        transformed = transform(
            image=image,
            keypoints=all_keypoints_xy,
            bboxes=all_bboxes,
            masks=all_masks,
            bbox_dummy_labels=[0 for _ in all_bboxes],
        )

        # save transformed image
        transformed_image = transformed["image"]
        transformed_image = Image.fromarray(transformed_image)
        transformed_image_dir = os.path.join(target_image_path, os.path.dirname(coco_image.file_name))
        if not os.path.exists(transformed_image_dir):
            os.makedirs(transformed_image_dir)
        transformed_image.save(os.path.join(target_image_path, coco_image.file_name))

        # change the metadata of the image coco object
        coco_image.width = transformed_image.width
        coco_image.height = transformed_image.height

        # store all modified annotations back into the coco dataset
        all_transformed_keypoints_xy = transformed["keypoints"]
        all_transformed_bboxes = transformed["bboxes"]
        all_transformed_masks = transformed["masks"]
        for annotation in annotations:
            transformed_keypoints = all_transformed_keypoints_xy[: len(annotation.keypoints) // 3]
            all_transformed_keypoints_xy = all_transformed_keypoints_xy[len(annotation.keypoints) // 3 :]
            transformed_bbox = all_transformed_bboxes.pop(0)  # exactly one bbox per annotation
            transformed_segmentations = all_transformed_masks.pop(0)  # exactly one segmentation per annotation

            # set keypoints that are no longer in image to (0,0,0)
            for i, kp in enumerate(transformed_keypoints):
                if 0 <= kp[0] < coco_image.width and 0 <= kp[1] < coco_image.height:
                    # add original visibility flag
                    transformed_keypoints[i] = [kp[0], kp[1], annotation.keypoints[i * 3 + 2]]
                else:
                    transformed_keypoints[i] = [0.0, 0.0, 0]

            flattened_transformed_keypoints = [i for kp in transformed_keypoints for i in kp]
            annotation.keypoints = flattened_transformed_keypoints
            annotation.bbox = transformed_bbox
            Image.fromarray(transformed_segmentations).save(f"test_{annotation.id}.png")
            annotation.segmentation = BinarySegmentationMask(transformed_segmentations).as_polygon

    return coco_dataset


if __name__ == "__main__":
    """example usage of the above function to resize all images in a coco dataset.
    Copy the following lines into your own codebase and modify as needed."""
    import json
    import pathlib

    path = pathlib.Path(__file__).parents[1] / "cvat_labeling" / "example" / "coco.json"

    coco_json_path = str(path)
    coco_dir = os.path.dirname(coco_json_path)
    coco_file_name = os.path.basename(coco_json_path)
    coco_target_dir = os.path.join(os.path.dirname(coco_dir), "transformed")
    os.makedirs(coco_target_dir, exist_ok=True)

    transforms = [A.Resize(128, 128)]

    coco_json = json.load(open(coco_json_path, "r"))
    coco_dataset = CocoKeypointsDataset(**coco_json)
    transformed_dataset = apply_transform_to_coco_dataset(transforms, coco_dataset, coco_dir, coco_target_dir)

    transformed_dataset_dict = transformed_dataset.dict(exclude_none=True)
    with open(os.path.join(coco_target_dir, coco_file_name), "w") as f:
        json.dump(transformed_dataset_dict, f)
