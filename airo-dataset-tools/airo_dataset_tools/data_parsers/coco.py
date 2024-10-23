"""parser for COCO dataset (both instance & keypoints) JSON files.

The main reference for this parser can be found here:
https://cocodataset.org/#format-data

Dataset loading example:

with open("path/to/annotations.json","r") as file:
    data = json.load(file) # dict
    parsed_data = CocoKeypoints(**data)

When there are no keypoints, you may use CocoInstances instead of CocoKeypoints.


Dataset creation example:

coco_image = CocoImage(
    id=1,
    width=100,
    height=100,
    file_name="path/to/image.jpg",
)
images = [coco_image]

# make more images, catergories and annotations here

coco_keypoints = CocoKeypoints(categories=categories, images=images, annotations=annotations)
with open("annotations.json", "w") as file:
    json.dump(coco_keypoints.model_dump(exclude_none=True), file, indent=4)

"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, field_validator, model_validator

# Used by CocoInfo and CocoImage
Datetime = Union[
    str, int
]  # COCO uses both YYYY-MM-DD and YYYY-MM-DD HH:MM:SS for datetime, CVAT places 0 for missing values
Url = str

# Used by CocoImage and CocoLicenses
LicenseID = int

# Used by CocoCategory and CocoInstanceAnnotation
CategoryID = int

# Used by CocoImage and CocoInstanceAnnotation
ImageID = int

# Used by CocoInstanceAnnotation
# Dict[int,int]  run length encoding(RLE) (of a pixel-mask), with keys "counts" and "size"
# where count contains the actual run length encoding (of a pixel-mask) [x1,l1,x2,l2,...]
# or a coco-encoded string that represents this list.
# see https://www.youtube.com/watch?v=h6s61a_pqfM for a video explanation of RLE
# see https://github.com/cocodataset/cocoapi/issues/386 for a discussion of RLE string encoding
# RLE is Usually only used for is_crowd segmentations.
RLEDict = Dict[str, Union[str, List]]

Polygon = List[float]  # list of vertices [x1, y1, x2, y2, ...]
Segmentation = Union[RLEDict, List[Polygon]]

# Used by the Annotations
Keypoints = List[float]  # list of keypoints [x1, y1, v1, x2, y2, v2, ...]
IsCrowd = int  # 0 or 1


class CocoInfo(BaseModel):
    year: Union[int, str]
    version: str
    description: str
    contributor: str
    url: Url
    date_created: Datetime


class CocoImage(BaseModel):
    id: ImageID
    width: int
    height: int
    file_name: str  # relative path to image
    license: Optional[LicenseID] = None
    flicker_url: Optional[Url] = None
    coco_url: Optional[Url] = None
    date_captured: Optional[Datetime] = None
    __hash__ = object.__hash__  # make hashable for use in set


class CocoCategory(BaseModel, extra="allow"):

    supercategory: str  # should be set to "name" for root category
    id: CategoryID
    name: str


class CocoKeypointCategory(CocoCategory):
    keypoints: List[str]
    skeleton: Optional[List[List[int]]] = None


class CocoInstanceAnnotation(BaseModel, extra="allow"):  # allow extra fields, to parse subclasses

    id: int  # unique id for the annotation
    image_id: ImageID
    category_id: CategoryID

    bbox: Optional[Tuple[float, float, float, float]] = None

    segmentation: Optional[Segmentation] = None
    area: Optional[float] = None
    iscrowd: Optional[IsCrowd] = None

    @field_validator("iscrowd")
    @classmethod
    def iscrowd_must_be_binary(cls, v: IsCrowd) -> IsCrowd:
        if v is None:
            return None
        assert v in [0, 1]
        return v


class CocoKeypointAnnotation(CocoInstanceAnnotation):
    keypoints: Keypoints
    num_keypoints: Optional[int] = None

    # make bbox optional
    bbox: Optional[Tuple[float, float, float, float]] = None

    @field_validator("keypoints")
    @classmethod
    def keypoints_must_be_multiple_of_three(cls, v: Keypoints) -> Keypoints:
        assert len(v) % 3 == 0, "keypoints list must be a multiple of 3"
        return v

    @field_validator("keypoints")
    @classmethod
    def keypoints_coordinates_must_be_in_pixel_space(cls, v: Keypoints) -> Keypoints:
        max_coordinate_value = 0.0
        for i in range(0, len(v), 3):
            max_coordinate_value = max(v[i], max_coordinate_value)
            max_coordinate_value = max(v[i + 2], max_coordinate_value)
        assert (
            max_coordinate_value > 1
        ), f"keypoints coordinates must be in pixel space, but max_coordinate is {max_coordinate_value}"
        return v

    @model_validator(mode="after")
    def num_keypoints_matches_amount_of_labeled_keypoints(self) -> "CocoKeypointAnnotation":
        labeled_keypoints = 0
        for v in self.keypoints[2::3]:
            if v > 0:
                labeled_keypoints += 1
        assert (
            labeled_keypoints == self.num_keypoints
        ), f"num_keypoints {self.num_keypoints} does not match number of labeled of keypoints {labeled_keypoints} for annotation {self.id}"
        return self


class CocoLicense(BaseModel):
    id: LicenseID
    name: str
    url: Url


class CocoInstancesDataset(BaseModel):
    info: Optional[CocoInfo] = None
    licenses: Optional[List[CocoLicense]] = None
    categories: Sequence[CocoCategory]
    images: List[CocoImage]
    annotations: Sequence[CocoInstanceAnnotation]

    @field_validator("annotations")
    @classmethod
    def annotations_list_cannot_be_empty(cls, v: List[CocoInstanceAnnotation]) -> List[CocoInstanceAnnotation]:
        assert len(v) > 0, "annotations list cannot be empty"
        return v

    # skip on failure becasue this validator requires the annotations list to be non-empty
    @model_validator(mode="after")
    def annotations_catergory_id_exist_in_categories(self) -> "CocoInstancesDataset":
        category_ids = set([category.id for category in self.categories])
        for annotation in self.annotations:
            assert (
                annotation.category_id in category_ids
            ), f"Annotation {annotation.id} has category_id {annotation.category_id} which does not exist in categories."
        return self


class CocoKeypointsDataset(CocoInstancesDataset):
    # Override the type of annotations.
    # annotations must be Sequence vs. list to allow this, see:
    # https://mypy.readthedocs.io/en/stable/common_issues.html#variance
    categories: Sequence[CocoKeypointCategory]
    annotations: Sequence[CocoKeypointAnnotation]

    # skip on failure becasue this validator requires the annotations list to be non-empty
    @model_validator(mode="after")
    def num_keypoints_matches_annotations(self) -> "CocoKeypointsDataset":
        category_dict = {category.id: category for category in self.categories}
        for annotation in self.annotations:
            assert len(annotation.keypoints) // 3 == len(
                category_dict[annotation.category_id].keypoints
            ), f"Number of keypoints for annotation {annotation.id} does not match number of keypoints in category."
        return self
