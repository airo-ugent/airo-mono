import json
import pathlib

from airo_dataset_tools.data_parsers.cvat_images import CVATImagessParser


def test_example_cvat_annotations_loading():
    path = pathlib.Path(__file__).parent.absolute()
    cvat_dict = json.load(
        open(str(path.parent / "airo_dataset_tools" / "cvat_labeling" / "example" / "annotations.json"))
    )
    cvat_keypoints_parser = CVATImagessParser(**cvat_dict)

    assert cvat_keypoints_parser.annotations.version == "1.1"
    assert len(cvat_keypoints_parser.annotations.image) == 4

    assert len(cvat_keypoints_parser.annotations.image[0].points) == 4  # 4 keypoints in first image
