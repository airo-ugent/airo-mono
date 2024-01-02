# Annotation Tool

![annotation tool](https://i.imgur.com/JWIzSLQ.png)

Simple OpenCV-based annotation tool for quick realtime annotation of images.
For real dataset annotation, we recommend using CVAT.
This tool is intented for creating small robot demos from clicked points.

## Usage

Create a new annotation spec dictionary, mapping annotation names to annotation types.
Calling the `get_manual_anntations()` function with it and an image will open an OpenCV window for clicking the necessary points.
The available annotations are `Keypoint`, `LineSegment`, `Line`, `BoundingBox` and `Polygon`.

```python
import cv2
from airo_camera_toolkit.annotation_tool import get_manual_annotations, Annotation

annotation_spec = {
   "keypoint": Annotation.Keypoint,
   "line_segment": Annotation.LineSegment,
   "line": Annotation.Line,
   "bounding_box": Annotation.BoundingBox,
   "polygon": Annotation.Polygon,
}

image = cv2.imread(image_path)
annotations = get_manual_annotations(image, annotation_spec)
```
Example output:
```python
{   'bounding_box': ((715, 423), (1408, 972)),
    'keypoint': (967, 261),
    'line': ((1736, 975), (-174, -449)),
    'line_segment': ((1102, 124), (1387, 325)),
    'polygon': [   (1037, 472),
                   (1183, 593),
                   (1284, 687),
                   (1341, 771),
                   (1336, 852),
                   (1333, 905),
                   (1313, 906),
                   (1241, 913),
                   (1227, 901),
                   (1235, 869),
                   (929, 903),
                   (861, 901),
                   (810, 851),
                   (753, 810),
                   (812, 730)]
}
```
