from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np
from airo_dataset_tools.data_parsers.coco import Polygon, RLEDict, Segmentation
from pycocotools import mask


class BinarySegmentationMask:
    """Class that holds a binay segmentation mask and can convert between binary bitmask and/or the different COCO segmentation formats:
    - polygon: [list[list[float]]] containing [x,y] coordinates of the polygon(s)
    - uncompressed RLE: [dict] with keys "counts" and "size" where count contains the actual run length encoding[x1,l1,x2,l2,...]
    - compressed RLE: [dict] with keys "counts" and "size" where count contains a coco-encoded string that represents the run length encoding
    """

    def __init__(self, bitmap: np.ndarray):
        assert np.array_equal((bitmap == 1.0) * 1.0, bitmap), "bitmap must be 1 or 0 numpy array"
        bitmap = bitmap.astype(np.uint8)
        self.bitmap = bitmap

    @classmethod
    def from_coco_segmentation_mask(
        cls, segmentation: Segmentation, width: int, height: int
    ) -> BinarySegmentationMask:
        """Convert a coco segmentation mask to a bitmap. based on coco"""

        # convert to encoded RLE if required
        if isinstance(segmentation, list):
            # polygon [list[list[float]]]
            rles = mask.frPyObjects(segmentation, height, width)
            rle = mask.merge(rles)
        elif isinstance(segmentation, dict):
            if isinstance(segmentation["counts"], list):
                # uncompressed RLE
                rle = mask.frPyObjects(segmentation, height, width)
            else:
                # encoded RLE
                rle = segmentation

        # decode encoded RLE to bitmap
        else:
            raise ValueError("segmentation must be a valid coco segmentation mask")
        bitmap = mask.decode(rle)
        return BinarySegmentationMask(bitmap)

    @property
    def area(self) -> float:
        return mask.area(self.as_compressed_rle)

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """returns x,y,w,h of the enclosing bbox"""

        bbox: np.ndarray = mask.toBbox(self.as_compressed_rle)
        return (bbox[0], bbox[1], bbox[2], bbox[3])

    @property
    def as_polygon(self) -> List[Polygon]:
        # from https://github.com/cocodataset/cocoapi/issues/476#issuecomment-871804850
        contours, _ = cv2.findContours(self.bitmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        segmentation = []
        valid_poly = 0
        for contour in contours:
            # Valid polygons have >= 6 coordinates (3 points)
            if contour.size >= 6:
                segmentation.append(contour.astype(float).flatten().tolist())
                valid_poly += 1
        if valid_poly == 0:
            raise ValueError
        return segmentation

    @property
    def as_uncompressed_rle(self) -> RLEDict:
        raise NotImplementedError

    @property
    def as_compressed_rle(self) -> RLEDict:
        """Convert a bitmap to a compressed coco RLEDict"""
        b = np.asfortranarray(self.bitmap.astype(np.uint8))
        encoded_rle: RLEDict = mask.encode(b)
        assert isinstance(encoded_rle["counts"], bytes)
        encoded_rle["counts"] = encoded_rle["counts"].decode("utf-8")
        return encoded_rle


# if __name__ == "__main__":
#     mask = np.zeros((10, 10)).astype(np.uint8)
#     mask[1:3] = 1
#     print(mask)
#     print(BinarySegmentationMask(mask).as_polygon)
