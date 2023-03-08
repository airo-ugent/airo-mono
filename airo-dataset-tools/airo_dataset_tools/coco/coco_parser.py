"""Custom parser for COCO dataset JSON files.

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
    json.dump(coco_keypoints.dict(exclude_none=True), file, indent=4)

"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, root_validator, validator

# Used by CocoInfo and CocoImage
Datetime = str  # COCO uses both YYYY-MM-DD and YYYY-MM-DD HH:MM:SS for datetime
Url = str

# Used by CocoImage and CocoLicenses
LicenseID = int

# Used by CocoCategory and CocoInstanceAnnotation
CategoryID = int

# Used by CocoImage and CocoInstanceAnnotation
ImageID = int

# Used by CocoInstanceAnnotation
RLEDict = Dict[str, List]  # Dict[int,int]  # run length encoding (of a pixel-mask), with keys "counts" and "size"
# where count contains the actual run length encoding (of a pixel-mask) [x1,l1,x2,l2,...]
Polygon = List[float]  # list of vertices [x1, y1, x2, y2, ...]
Segmentation = Union[RLEDict, List[Polygon]]

# Used by the Annotations
Keypoints = List[float]  # list of keypoints [x1, y1, v1, x2, y2, v2, ...]
IsCrowd = int  # 0 or 1


class CocoInfo(BaseModel):
    year: int
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
    license: Optional[LicenseID]
    flicker_url: Optional[Url]
    coco_url: Optional[Url]
    date_captured: Optional[Datetime]


class CocoCategory(BaseModel):
    supercategory: str  # should be set to "name" for root category
    id: CategoryID
    name: str


class CocoKeypointCategory(CocoCategory):
    keypoints: List[str]
    skeleton: Optional[List[List[int]]]


class CocoInstanceAnnotation(BaseModel):
    id: int  # unique id for the annotation
    image_id: ImageID
    category_id: CategoryID
    segmentation: Segmentation
    area: float
    bbox: Tuple[int, int, int, int]
    iscrowd: IsCrowd

    @validator("iscrowd")
    def iscrowd_must_be_binary(cls, v: IsCrowd) -> IsCrowd:
        assert v in [0, 1]
        return v


class CocoKeypointAnnotation(CocoInstanceAnnotation):
    keypoints: Keypoints
    num_keypoints: Optional[int]

    @validator("keypoints")
    def keypoints_must_be_multiple_of_three(cls, v: Keypoints) -> Keypoints:
        if len(v) % 3 != 0:
            raise ValueError("keypoints list length must be a multiple of 3")
        return v


class CocoLicense(BaseModel):
    id: LicenseID
    name: str
    url: Url


class CocoInstancesDataset(BaseModel):
    info: Optional[CocoInfo]
    licenses: Optional[List[CocoLicense]]
    categories: Sequence[CocoCategory]
    images: List[CocoImage]
    annotations: Sequence[CocoInstanceAnnotation]

    @root_validator
    def annotations_catergory_id_exist_in_categories(cls, values: dict) -> dict:
        category_ids = set([category.id for category in values["categories"]])
        for annotation in values["annotations"]:
            assert (
                annotation.category_id in category_ids
            ), f"Annotation {annotation.id} has category_id {annotation.category_id} which does not exist in categories."
        return values


class CocoKeypointsDataset(CocoInstancesDataset):
    # Override the type of annotations.
    # annotations must be Sequence vs. list to allow this, see:
    # https://mypy.readthedocs.io/en/stable/common_issues.html#variance
    categories: Sequence[CocoKeypointCategory]
    annotations: Sequence[CocoKeypointAnnotation]
