import pathlib

from airo_dataset_tools.cvat_labeling.load_xml_to_dict import get_dict_from_xml
from airo_dataset_tools.data_parsers.cvat_images import CVATImagesParser

path = pathlib.Path(__file__).parent.absolute()
CVAT_EXAMPLE_PATH = str(path.parent / "airo_dataset_tools" / "cvat_labeling" / "example" / "annotations.xml")
COCO_CATEGORIES_PATH = str(path.parent / "airo_dataset_tools" / "cvat_labeling" / "example" / "coco_categories.json")


def test_example_cvat_annotations_loading():
    cvat_dict = get_dict_from_xml(CVAT_EXAMPLE_PATH)
    cvat_keypoints_parser = CVATImagesParser(**cvat_dict)

    assert cvat_keypoints_parser.annotations.version == "1.1"
    assert len(cvat_keypoints_parser.annotations.image) == 4

    assert len(cvat_keypoints_parser.annotations.image[0].points) == 4  # 4 keypoints in first image
