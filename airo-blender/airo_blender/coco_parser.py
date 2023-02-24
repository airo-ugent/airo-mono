from typing import Dict, List, Optional, Tuple, Union

from pydantic import BaseModel

"""Custom parser for COCO keypoints JSON"""

LicenseID = int
ImageID = int
CategoryID = int
AnnotationID = int
Segmentation = List[List[Union[float, int]]]
FileName = str
Relativepath = str
Url = str
IsCrowd = int
RLE = str  # run length encoding
# Polygon = List[Tuple[int, int]]
# Segmentation = Union[RLE, Polygon]  # not sure about this
Area = float
BoundingBox = Tuple[int, int, int, int]


class CocoInfo(BaseModel):
    description: str
    url: Url
    version: str
    year: int
    contributor: str
    date_created: str


class CocoLicenses(BaseModel):
    url: Url
    id: LicenseID
    name: str


class CocoImage(BaseModel):
    license: Optional[LicenseID]
    file_name: Relativepath
    height: int
    width: int
    id: ImageID


class CocoKeypointCategory(BaseModel):
    supercategory: str  # should be set to "name" for root category
    id: CategoryID
    name: str
    keypoints: List[str]
    skeleton: Optional[List[List[int]]]


class CocoKeypointAnnotation(BaseModel):
    id: AnnotationID
    image_id: ImageID
    category_id: CategoryID
    segmentation: Dict
    area: Area
    bbox: BoundingBox
    iscrowd: IsCrowd

    num_keypoints: Optional[int]
    keypoints: List[float]

    # TODO: add checks.
    # @validator("keypoints")
    # def check_amount_of_keypoints(cls, v, values, **kwargs):
    #     assert len(v) // 3 == values["num_keypoints"]


class CocoKeypoints(BaseModel):
    """Parser Class for COCO keypoints JSON

    Example:
    with open("path","r") as file:
        data = json.load(file) # dict
        parsed_data = COCOKeypoints(**data)
    """

    info: Optional[CocoInfo]
    licenses: Optional[List[CocoLicenses]]
    categories: List[CocoKeypointCategory]
    images: List[CocoImage]
    annotations: List[CocoKeypointAnnotation]
