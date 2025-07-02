""" module toconvert CVAT 1.1 images keypoint annotations to COCO Keypoint dataset"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import List, Tuple

import tqdm
from airo_dataset_tools.cvat_labeling.load_xml_to_dict import get_dict_from_xml
from airo_dataset_tools.data_parsers.coco import (
    CocoImage,
    CocoKeypointAnnotation,
    CocoKeypointCategory,
    CocoKeypointsDataset,
)
from airo_dataset_tools.data_parsers.cvat_images import CVATImagesParser, ImageItem, LabelItem, Point
from airo_dataset_tools.segmentation_mask_converter import BinarySegmentationMask


def cvat_image_to_coco(  # noqa: C901, too complex
    cvat_xml_path: str, coco_configuration_json_path: str, add_bbox: bool = True, add_segmentation: bool = True
) -> dict:
    """Function that converts an annotation XML in the CVAT 1.1 Image format to the COCO keypoints format.
    If you don't need keypoints, you can simply use CVAT to create a COCOinstances format and should not use this function!


    The xml should be obtained with the AIRO-cvat-keypoints flow:
    It requires the CVAT dataset to be created by using labels formatted as <category>.<semantic_type>, using the group_id to group multiple instances together.
    if only a single instance is present, the group id is set to 1 by default so you don't have to do this yourself.

    Args:
        cvat_xml_path (str): _description_
        coco_configuration_json_path (str): path to the COCO categories to use for annotating this dataset.
        add_bbox (bool): add bounding box annotations to the COCO dataset, requires all keypoint annotations to have a bbox annotation
        add_segmentation (bool): add segmentation annotations to the COCO dataset, requires all keypoint annotations to have a mask annotation. Bboxes will be created from the segmentation masks.

    Returns: COCO Keypoints dataset model as a dict
    """
    cvat_dict = get_dict_from_xml(cvat_xml_path)
    cvat_parsed = CVATImagesParser(**cvat_dict)

    # create a COCO Dataset Model
    coco_images: List[CocoImage] = []
    coco_annotations: List[CocoKeypointAnnotation] = []
    coco_categories: List[CocoKeypointCategory] = []

    # load the COCO categories from the configuration file
    with open(coco_configuration_json_path, "r") as file:
        coco_categories_config = json.load(file)
    for category_dict in coco_categories_config["categories"]:
        category = CocoKeypointCategory(**category_dict)
        coco_categories.append(category)

    _validate_coco_categories_are_in_cvat(
        cvat_parsed, coco_categories, add_bbox=add_bbox, add_segmentation=add_segmentation
    )

    annotation_id_counter = 1  # counter for the annotation ID

    # iterate over all cvat annotations (grouped per image)
    # and create the COCO Keypoint annotations
    for cvat_image in tqdm.tqdm(cvat_parsed.annotations.image):
        coco_image = CocoImage(
            file_name=cvat_image.name,
            height=int(cvat_image.height),
            width=int(cvat_image.width),
            id=int(cvat_image.id) + 1,
        )
        coco_images.append(coco_image)
        for category in coco_categories:
            n_image_category_instances = _get_n_category_instances_in_image(cvat_image, category.name)
            for instance_id in range(1, n_image_category_instances + 1):  # IDs start with 1
                instance_category_keypoints = []
                for semantic_type in category.keypoints:
                    keypoint = _get_semantic_type_keypoint_for_instance_from_cvat_image(
                        cvat_image, semantic_type, instance_id
                    )
                    instance_category_keypoints.extend(keypoint)

                coco_annotations.append(
                    CocoKeypointAnnotation(
                        category_id=category.id,
                        id=annotation_id_counter,
                        image_id=coco_image.id,
                        keypoints=instance_category_keypoints,
                        num_keypoints=sum([1 if flag > 0 else 0 for flag in instance_category_keypoints[2::3]]),
                    )
                )
                if add_bbox:
                    coco_annotations[-1].bbox = _get_bbox_for_instance_from_cvat_image(cvat_image, instance_id)

                if add_segmentation:
                    coco_annotations[-1].segmentation = _get_segmentation_for_instance_from_cvat_image(
                        cvat_image, instance_id
                    )
                    coco_annotations[-1].iscrowd = 0
                    mask = BinarySegmentationMask.from_coco_segmentation_mask(
                        coco_annotations[-1].segmentation, coco_image.width, coco_image.height
                    )
                    coco_annotations[-1].area = mask.area

                    if not add_bbox:
                        coco_annotations[-1].bbox = mask.bbox

                annotation_id_counter += 1

    coco_model = CocoKeypointsDataset(images=coco_images, annotations=coco_annotations, categories=coco_categories)
    return coco_model.model_dump(exclude_none=True)


####################
### helper functions
####################


def _validate_coco_categories_are_in_cvat(
    cvat_parsed: CVATImagesParser, coco_categories: List[CocoKeypointCategory], add_bbox: bool, add_segmentation: bool
) -> None:
    # gather the annotation from CVAT
    cvat_categories_dict = defaultdict(list)

    labels = cvat_parsed.annotations.meta.get_job_or_task().labels.label

    # Handle both single LabelItem and list of LabelItems
    if isinstance(labels, LabelItem):
        labels = [labels]

    for annotation_category in labels:
        assert isinstance(annotation_category, LabelItem)
        category_str, annotation_name = annotation_category.name.split(".")
        cvat_categories_dict[category_str].append(annotation_name)

    for category_str, semantic_types in cvat_categories_dict.items():
        if add_bbox:
            assert "bbox" in semantic_types, "bbox annotations are required"
        if add_segmentation:
            assert "mask" in semantic_types, "segmentation masks are required"
        # find the matching COCO category
        coco_category = None
        for coco_category in coco_categories:
            if coco_category.name == category_str:
                break
        if coco_category is not None:
            for category_keypoint in coco_category.keypoints:
                assert category_keypoint in semantic_types, f"semantic type {category_keypoint} not found"
        else:
            raise AssertionError(
                f"category {category_str} not found in coco categories. "
                f"Available categories: {[c.name for c in coco_categories]}"
            )


def _get_n_category_instances_in_image(cvat_image: ImageItem, category_name: str) -> int:
    """returns the number of instances for the specified category in the CVAT ImageItem.

    This is done by finding the maximum group_id for all annotations of the image.

    Edge cases include: no Points in the image or only 1 Point in the image.
    """
    if cvat_image.points is None:
        return 0
    if not isinstance(cvat_image.points, list):
        if _get_category_from_cvat_label(cvat_image.points.label) == category_name:
            assert cvat_image.points.group_id is not None, "group_id was None"
            return int(cvat_image.points.group_id)
        else:
            return 0
    max_group_id = 0
    for cvat_point in cvat_image.points:
        if _get_category_from_cvat_label(cvat_point.label) == category_name:
            assert cvat_point.group_id is not None, "group_id was None"
            max_group_id = max(max_group_id, int(cvat_point.group_id))
    return max_group_id


def _get_category_from_cvat_label(label: str) -> str:
    """cvat labels are formatted as <category>.<semantic_type>
    this function returns the category
    """
    split = label.split(".")
    assert len(split) == 2, " label was not formatted as category.semantic_type"
    return label.split(".")[0]


def _get_semantic_type_from_cvat_label(label: str) -> str:
    """cvat labels are formatted as <category>.<semantic_type>
    this function returns the semantic type
    """
    split = label.split(".")
    assert len(split) == 2, " label was not formatted as category.semantic_type"
    return label.split(".")[1]


def _get_bbox_for_instance_from_cvat_image(
    cvat_image: ImageItem, instance_id: int
) -> Tuple[float, float, float, float]:
    """returns the bbox for the instance in the cvat image."""
    instance_id_str = str(instance_id)
    if cvat_image.box is None:
        raise ValueError("bbox annotations are required for image {cvat_image.name}")
    if not isinstance(cvat_image.box, list):
        if instance_id_str == cvat_image.box.group_id:
            return (
                float(cvat_image.box.xtl),
                float(cvat_image.box.ytl),
                float(cvat_image.box.xbr) - float(cvat_image.box.xtl),
                float(cvat_image.box.ybr) - float(cvat_image.box.ytl),
            )
        else:
            raise ValueError("bbox annotations are required for image {cvat_image.name}")
    for bbox in cvat_image.box:
        if instance_id_str == bbox.group_id:
            return (
                float(bbox.xtl),
                float(bbox.ytl),
                float(bbox.xbr) - float(bbox.xtl),
                float(bbox.ybr) - float(bbox.ytl),
            )
    raise ValueError("bbox annotations are required for image {cvat_image.name}")


def _get_segmentation_for_instance_from_cvat_image(cvat_image: ImageItem, instance_id: int) -> List[List[float]]:
    """returns the segmentation polygon for the instance in the cvat image."""
    instance_id_str = str(instance_id)
    if cvat_image.polygon is None:
        raise ValueError(f"segmentation annotations are required for image {cvat_image.name}")
    if not isinstance(cvat_image.polygon, list):
        if instance_id_str == cvat_image.polygon.group_id:
            polygon_str = cvat_image.polygon.points
            polygon_str = polygon_str.replace(";", ",")
            return [[float(x) for x in polygon_str.split(",")]]
        else:
            raise ValueError("segmentation annotations are required for image {cvat_image.name}")
    for polygon in cvat_image.polygon:
        if instance_id_str == polygon.group_id:
            polygon_str = polygon.points
            polygon_str = polygon_str.replace(";", ",")
            return [[float(x) for x in polygon_str.split(",")]]
    raise ValueError("segmentation annotations are required for image {cvat_image.name}")


def _get_semantic_type_keypoint_for_instance_from_cvat_image(
    cvat_image: ImageItem, semantic_type: str, instance_id: int
) -> List[float]:
    """Finds the keypoint of the given semantic type for this instance in the image.
    Returns [0,0,0] if the keypoint is not annotated for this instance.

    Args:
        cvat_image (ImageItem): _description_
        semantic_type (str): _description_
        instance_id (int): _description_

    Returns:
        List: [x,y,visibility]
    """
    instance_id_str = str(instance_id)
    if cvat_image.points is None:
        return [0.0, 0.0, 0]
    if not isinstance(cvat_image.points, list):
        if (
            semantic_type == _get_semantic_type_from_cvat_label(cvat_image.points.label)
            and instance_id_str == cvat_image.points.group_id
        ):
            return _extract_coco_keypoint_from_cvat_point(cvat_image.points)
        else:
            return [0.0, 0.0, 0]
    for cvat_point in cvat_image.points:
        if (
            semantic_type == _get_semantic_type_from_cvat_label(cvat_point.label)
            and instance_id_str == cvat_point.group_id
        ):
            return _extract_coco_keypoint_from_cvat_point(cvat_point)
    return [0.0, 0.0, 0]


def _extract_coco_keypoint_from_cvat_point(cvat_point: Point) -> List:
    """extract keypoint in coco format (u,v,f) from cvat annotation point.
    Args:
        cvat_point (Point): _description_

    Returns:
        List: [u,v,f] where u,v are the coords scaled to the image resolution and f is the coco visibility flag.
        see the coco dataset format for more details.
    """
    u = float(cvat_point.points.split(",")[0])
    v = float(cvat_point.points.split(",")[1])
    f = (
        1 if cvat_point.occluded == "1" else 2
    )  # occluded = 1 means not visible, which is 1 in COCO; visible in COCO is 2
    return [u, v, f]


if __name__ == "__main__":
    """
    For development. See cli.py for the command line interface to this function that you can use to convert cvat annotations to coco.
    """

    import pathlib

    path = pathlib.Path(__file__).parent.absolute()
    cvat_xml_file = str(path / "example" / "annotations.xml")

    coco_categories_file = str(path / "example" / "coco_categories.json")

    coco = cvat_image_to_coco(cvat_xml_file, coco_categories_file, add_bbox=True, add_segmentation=False)
    with open("coco.json", "w") as file:
        json.dump(coco, file)
