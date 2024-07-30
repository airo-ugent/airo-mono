from __future__ import annotations

from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
from airo_dataset_tools.data_parsers.coco import Polygon, RLEDict, Segmentation
from pycocotools import mask


def merge_multi_segment(segments: List[List[Any]]) -> List[np.ndarray]:
    """

    code taken from: https://github.com/ultralytics/JSON2YOLO/blob/main/general_json2yolo.py#L330
    (AGPL licensed.)

    Merge multi segments to one list. Find the coordinates with min distance between each segment, then connect these
    coordinates with one thin line to merge all segments into one.

    This is useful to convert masks to yolo dataset, as yolo only supports one polygon for each object.

    Args:
        segments(List(List)): original segmentations in coco's json file.
            like [segmentation1, segmentation2,...],
            each segmentation is a list of coordinates.

    Returns:
        List(List): merged segments.
    """

    def min_index(arr1: np.ndarray, arr2: np.ndarray) -> Tuple[Any, ...]:
        """
        Find a pair of indexes with the shortest distance.

        Args:
            arr1: (N, 2).
            arr2: (M, 2).
        Return:
            a pair of indexes(tuple).
        """
        dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
        return np.unravel_index(np.argmin(dis, axis=None), dis.shape)

    s = []
    segments_arrays = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list: List[List[int]] = [[] for _ in range(len(segments_arrays))]

    # record the indexes with min distance between each segment
    for i in range(1, len(segments_arrays)):
        idx1, idx2 = min_index(segments_arrays[i - 1], segments_arrays[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # use two round to connect all the segments
    for k in range(2):
        # forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # middle segments have two indexes
                # reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments_arrays[i] = segments_arrays[i][::-1, :]

                segments_arrays[i] = np.roll(segments_arrays[i], -idx[0], axis=0)
                segments_arrays[i] = np.concatenate([segments_arrays[i], segments_arrays[i][:1]])
                # deal with the first segment and the last one
                if i in [0, len(idx_list) - 1]:
                    s.append(segments_arrays[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments_arrays[i][idx[0] : idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments_arrays[i][nidx:])

    s = np.concatenate(s, axis=0).reshape(-1).tolist()
    return s


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
        return float(mask.area(self.as_compressed_rle))

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """returns x,y,w,h of the enclosing bbox"""

        bbox: np.ndarray = mask.toBbox(self.as_compressed_rle)
        return (bbox[0], bbox[1], bbox[2], bbox[3])

    @property
    def as_polygon(self) -> Optional[List[Polygon]]:
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
            print("No valid polygons found in segmentation mask")
            return None
        return segmentation

    @property
    def as_single_polygon(self) -> Optional[Polygon | List[np.ndarray]]:
        """Convert a bitmap to a single polygon. If the bitmap contains multiple segments, they will be merged into one polygon."""
        poly = self.as_polygon
        if poly is None:
            return None

        if len(poly) == 1:
            return poly[0]

        poly_merged = merge_multi_segment(poly)
        return poly_merged

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


if __name__ == "__main__":
    m = np.zeros((10, 10)).astype(np.uint8)
    m[1:3] = 1
    m[6:8, 1:5] = 1
    print(m)
    print(BinarySegmentationMask(m).as_polygon)
    poly = BinarySegmentationMask(m).as_single_polygon
    # poly = merge_multi_segment(poly)
    print(poly)
    mask2 = BinarySegmentationMask.from_coco_segmentation_mask(poly, 10, 10)  # type: ignore # TODO fix this
    print(mask2.bitmap)
