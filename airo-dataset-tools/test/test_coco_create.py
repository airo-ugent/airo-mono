"""Tests for creating datasets in COCO format."""

from airo_dataset_tools.data_parsers.coco import (
    CocoCategory,
    CocoImage,
    CocoInfo,
    CocoInstanceAnnotation,
    CocoInstancesDataset,
    CocoKeypointAnnotation,
    CocoKeypointCategory,
    CocoKeypointsDataset,
    CocoLicense,
)


def coco_info_example() -> CocoInfo:
    return CocoInfo(
        description="Description of the dataset.",
        url="https://github.com/airo-ugent/airo-mono",
        version="0.1",
        year=2023,
        contributor="Victor-Louis De Gusseme",
        date_created="2023/03/06",
    )


def coco_image_example() -> CocoImage:
    return CocoImage(
        id=1,
        width=640,
        height=480,
        file_name="test_image.jpg",
        license=1,
        flicker_url="https://flicker.com/test_image.jpg",
        coco_url="https://coco.com/test_image.jpg",
        date_captured="2023/03/06",
    )


def coco_image_without_optional_fields_example() -> CocoImage:
    return CocoImage(
        id=1,
        width=640,
        height=480,
        file_name="test_image.jpg",
    )


def coco_category_example() -> CocoCategory:
    return CocoCategory(
        supercategory="cloth",
        id=99,
        name="towel",
    )


def coco_keypoint_category_example() -> CocoKeypointCategory:
    return CocoKeypointCategory(
        supercategory="cloth",
        id=99,
        name="towel",
        keypoints=["corner_1", "corner_2", "corner_3"],
    )


def coco_instance_annotation_example() -> CocoInstanceAnnotation:
    return CocoInstanceAnnotation(
        id=1,
        image_id=1,
        category_id=99,
        segmentation=[[0, 0, 50, 75, 100, 0]],
        area=375.0,
        bbox=(0, 0, 100, 75),
        iscrowd=0,
    )


def coco_instance_annotation_with_rle_example() -> CocoInstanceAnnotation:
    # TODO: make segmentation, area and bbox consistent
    return CocoInstanceAnnotation(
        id=2,
        image_id=1,
        category_id=99,
        segmentation={"counts": [0, 10, 1, 20], "size": [100, 75]},
        area=375.0,
        bbox=(0, 0, 100, 75),
        iscrowd=0,
    )


def coco_keypoint_annotation_example() -> CocoKeypointAnnotation:
    return CocoKeypointAnnotation(
        id=3,
        image_id=1,
        category_id=99,
        segmentation=[[0, 0, 50, 75, 100, 0]],
        area=375.0,
        bbox=(0, 0, 100, 75),
        iscrowd=0,
        keypoints=[0, 0, 1, 50, 75, 2, 100, 0, 0],
        num_keypoints=2,
    )


def coco_license_example() -> CocoLicense:
    return CocoLicense(
        id=1,
        name="Attribution-NonCommercial-ShareAlike License",
        url="http://creativecommons.org/licenses/by-nc-sa/2.0/",
    )


def test_coco_info():
    assert coco_info_example()


def test_coco_image():
    assert coco_image_example()


def test_coco_image_without_optional_fields():
    assert coco_image_without_optional_fields_example()


def test_coco_category():
    assert coco_category_example()


def test_coco_keypoint_category():
    assert coco_keypoint_category_example()


def test_coco_instance_annotation():
    assert coco_instance_annotation_example()


def test_coco_instance_annotation_with_rle():
    assert coco_instance_annotation_with_rle_example()


def test_coco_keypoint_annotation():
    assert coco_keypoint_annotation_example()


def test_coco_license():
    assert coco_license_example()


def test_coco_instances():
    coco_info = coco_info_example()
    coco_license = coco_license_example()
    coco_image = coco_image_example()
    coco_category = coco_category_example()
    coco_instance_annotation = coco_instance_annotation_example()
    coco_instance_annotation_with_rle = coco_instance_annotation_with_rle_example()

    coco_instances = CocoInstancesDataset(
        info=coco_info,
        licenses=[coco_license],
        images=[coco_image],
        categories=[coco_category],
        annotations=[coco_instance_annotation, coco_instance_annotation_with_rle],
    )
    assert coco_instances


def test_coco_keypoints():
    coco_info = coco_info_example()
    coco_license = coco_license_example()
    coco_image = coco_image_example()
    coco_keypoint_category = coco_keypoint_category_example()
    coco_keypoint_annotation = coco_keypoint_annotation_example()

    coco_keypoints = CocoKeypointsDataset(
        info=coco_info,
        licenses=[coco_license],
        images=[coco_image],
        categories=[coco_keypoint_category],
        annotations=[coco_keypoint_annotation],
    )
    assert coco_keypoints
