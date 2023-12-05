from enum import Enum
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
from airo_typing import OpenCVIntImageType


class Annotation(Enum):
    """The types of annotations that can be collected. To be used in the annotation_spec dict."""

    Keypoint = 0
    LineSegment = 1
    Line = 2
    BoundingBox = 3
    Polygon = 4


min_clicks_required = {
    Annotation.Keypoint: 1,
    Annotation.LineSegment: 2,
    Annotation.Line: 2,
    Annotation.BoundingBox: 2,
    Annotation.Polygon: 3,
}

# Colors for drawing
BGRColorType = Tuple[int, int, int]
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
CYAN = (255, 255, 0)


class AnnotationWindow:
    def __init__(self, image: OpenCVIntImageType, annotation_spec: dict[str, Annotation]):
        self.original_image = image
        self.annotation_spec = annotation_spec
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        _, width, _ = self.original_image.shape
        self.configure_scaling(width)
        self.polygon_close_distance = 10

        # State variables
        self.current_mouse_position: Tuple[int, int] = (0, 0)
        self.clicked_points: List[Tuple[int, int]] = []

    def mouse_callback(self, event: Any, x: int, y: int, flags: Any, parm: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_points.append((x, y))
        elif event == cv2.EVENT_MOUSEMOVE:
            self.current_mouse_position = x, y

    def configure_scaling(self, image_width: int) -> None:
        if image_width > 1280:
            self.font_scale_banner = 1.5
            self.font_scale_annotation = 1.0
            self.text_offset = 8
            self.thickness = 2
            self.line_type = cv2.LINE_AA
            self.keypoint_size = 5
            self.keypoint_thickness = 2
        elif image_width > 720:
            self.font_scale_banner = 0.9
            self.font_scale_annotation = 0.7
            self.text_offset = 6
            self.thickness = 2
            self.line_type = cv2.LINE_8
            self.keypoint_size = 3
            self.keypoint_thickness = 2
        else:
            self.font_scale_banner = 0.329
            self.font_scale_annotation = 0.329
            self.text_offset = 4
            self.thickness = 1
            self.line_type = cv2.LINE_8
            self.keypoint_size = 2
            self.keypoint_thickness = 1

    def put_banner_text(self, image: OpenCVIntImageType, text: str, color: BGRColorType = GREEN) -> None:
        _, height = cv2.getTextSize(text, self.font, self.font_scale_banner, self.thickness)[0]
        origin = (self.text_offset, height + self.text_offset)
        cv2.putText(image, text, origin, self.font, self.font_scale_banner, color, self.thickness, self.line_type)

    def put_annotation_text(
        self,
        image: OpenCVIntImageType,
        text: str,
        origin: Tuple[int, int],
        color: BGRColorType = GREEN,
    ) -> None:
        cv2.putText(image, text, origin, self.font, self.font_scale_annotation, color, self.thickness, self.line_type)

    def draw_keypoint(
        self, image: OpenCVIntImageType, point: Tuple[int, int], name: str, color: BGRColorType = GREEN
    ) -> None:
        text_location = (point[0] + self.text_offset, point[1] - self.text_offset)
        cv2.circle(image, point, self.keypoint_size, color, self.keypoint_thickness, self.line_type)
        self.put_annotation_text(image, name, text_location, color)

    def draw_line_segment(
        self,
        image: OpenCVIntImageType,
        point0: Tuple[int, int],
        point1: Tuple[int, int],
        name: str,
        color: BGRColorType = GREEN,
    ) -> None:
        cv2.line(image, point0, point1, color, self.thickness, self.line_type)
        self.draw_keypoint(image, point0, f"{name}", color)
        self.draw_keypoint(image, point1, "", color)

    def draw_polygon(
        self,
        image: OpenCVIntImageType,
        points: List[Tuple[int, int]],
        name: str,
        color: BGRColorType = GREEN,
        closed: bool = False,
    ) -> None:
        for i, point in enumerate(points):
            text = name if i == 0 else ""
            self.draw_keypoint(image, point, text, color)
        cv2.polylines(
            image, [np.array(points)], isClosed=closed, color=color, thickness=self.thickness, lineType=self.line_type
        )

    def draw_line(
        self,
        image: OpenCVIntImageType,
        point0: Tuple[int, int],
        direction: Tuple[int, int],
        name: str,
        color: BGRColorType = GREEN,
    ) -> None:
        self.draw_keypoint(image, point0, name, color)

        point = np.array(point0)
        direction_array = np.array(direction)
        direction_norm = np.linalg.norm(direction_array)
        if direction_norm == 0:
            return
        direction_scaled = (5000 / direction_norm) * direction_array

        line_start = point - direction_scaled
        line_end = point + direction_scaled
        line_start = tuple(line_start.astype(int))
        line_end = tuple(line_end.astype(int))
        cv2.line(image, line_start, line_end, color, self.thickness, self.line_type)  # Arrow is currently off screen

    def draw_bounding_box(
        self,
        image: OpenCVIntImageType,
        point0: Tuple[int, int],
        point1: Tuple[int, int],
        name: str,
        color: BGRColorType = GREEN,
    ) -> None:
        cv2.rectangle(image, point0, point1, color, self.thickness, self.line_type)
        self.draw_keypoint(image, point0, name, color)
        self.draw_keypoint(image, point1, "", color)

    def draw_annotation_in_progress(  # noqa: C901
        self,
        image: OpenCVIntImageType,
        annotation_name: str,
        annotation_type: Annotation,
        color: BGRColorType = GREEN,
    ) -> None:
        """Draws the annotation that is currently being collected. This function is more complex than the oen for
        finished annotations because it needs to confer more information. For example, we color polygons yellow if they
        can be closed.

        Args:
            image: The image upon which the annotation will be drawn.
            annotation_name: The name of the annotation.
            annotation_type: The type of annotation.
            color: Color for the drawn text, lines and points.
        """
        mouse_position = self.current_mouse_position
        clicked_points = self.clicked_points

        if len(clicked_points) == 0:
            self.draw_keypoint(image, mouse_position, annotation_name, color)

        if annotation_type == Annotation.Keypoint:
            banner_text = f"Click to annotate keypoint: {annotation_name}"
            self.draw_keypoint(image, mouse_position, annotation_name, color)

        if annotation_type == Annotation.LineSegment:
            banner_text = f"Click to annotate line segment: {annotation_name}"
            if len(clicked_points) == 1:
                self.draw_line_segment(image, clicked_points[0], mouse_position, annotation_name, color)

        if annotation_type == Annotation.Line:
            banner_text = f"Click to annotate line: {annotation_name}"
            if len(clicked_points) == 1:
                direction_array = np.array(mouse_position) - np.array(clicked_points[0])
                direction = direction_array[0], direction_array[1]
                self.draw_line(image, clicked_points[0], direction, annotation_name, color)

        if annotation_type == Annotation.BoundingBox:
            banner_text = f"Click to annotate bounding box: {annotation_name}"
            if len(clicked_points) == 0:
                self.draw_line(image, mouse_position, (0, 1), "", color)
                self.draw_line(image, mouse_position, (1, 0), "", color)
            elif len(clicked_points) == 1:
                self.draw_bounding_box(image, clicked_points[0], mouse_position, annotation_name, color)

        if annotation_type == Annotation.Polygon:
            if len(clicked_points) < 3:
                banner_text = f"Click to annotate polygon: {annotation_name}"
            else:
                banner_text = f"Click first point to finish polygon: {annotation_name}"
                distance = np.linalg.norm(np.array(mouse_position) - np.array(clicked_points[0]))
                if distance < self.polygon_close_distance:
                    color = YELLOW

            if len(clicked_points) > 0:
                self.draw_polygon(image, [*clicked_points, mouse_position], annotation_name, color)

        self.put_banner_text(image, banner_text)

    def draw_finished_annotations(
        self, image: OpenCVIntImageType, annotations: dict, color: BGRColorType = CYAN
    ) -> None:
        for name, annotation in annotations.items():
            annotation_type = self.annotation_spec[name]
            if annotation_type == Annotation.Keypoint:
                self.draw_keypoint(image, annotation, name, color)
            elif annotation_type == Annotation.LineSegment:
                self.draw_line_segment(image, annotation[0], annotation[1], name, color)
            elif annotation_type == Annotation.Line:
                self.draw_line(image, annotation[0], annotation[1], name, color)
            elif annotation_type == Annotation.BoundingBox:
                self.draw_bounding_box(image, annotation[0], annotation[1], name, color)
            elif annotation_type == Annotation.Polygon:
                self.draw_polygon(image, annotation, name, color, closed=True)

    def process_clicked_points(self, annotation_type: Annotation) -> Optional[tuple]:
        """Process the clicked points into the annotations dictionary.

        Args:
            annotation_type: The type of annotation currently being processed.

        Returns:
            annotation: The annotation to be added to the annotations dictionary. None if not enough points have been clicked.
        """
        if len(self.clicked_points) < min_clicks_required[annotation_type]:
            return None

        annotation: Any  # Lets mypy know that annotation can be several types
        if annotation_type == Annotation.Keypoint:
            annotation = self.clicked_points[0]
        elif annotation_type == Annotation.LineSegment:
            annotation = self.clicked_points[0], self.clicked_points[1]
        elif annotation_type == Annotation.Line:
            direction = tuple(np.array(self.clicked_points[1]) - np.array(self.clicked_points[0]))
            annotation = self.clicked_points[0], direction
        elif annotation_type == Annotation.BoundingBox:
            annotation = self.clicked_points[0], self.clicked_points[1]
        elif annotation_type == Annotation.Polygon:
            distance = np.linalg.norm(np.array(self.clicked_points[-1]) - np.array(self.clicked_points[0]))
            if distance >= self.polygon_close_distance:
                return None
            annotation = self.clicked_points[:-1]  # don't include last point

        self.clicked_points = []
        return annotation

    def collect_annotations(self) -> Optional[dict[str, Any]]:
        annotation_names = list(self.annotation_spec.keys())
        annotation_types = list(self.annotation_spec.values())
        num_annotations = len(annotation_names)
        annotations = {}
        current_id = 0

        window_name = "Annotation window"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        cv2.setMouseCallback(window_name, self.mouse_callback)  # type: ignore[attr-defined] # cv2 4.8 misses attribute

        while True:
            image = self.original_image.copy()
            if current_id >= num_annotations:
                self.put_banner_text(image, "All annotations collected. Press 'Enter' to confirm.")
            else:
                current_name = annotation_names[current_id]
                current_type = annotation_types[current_id]
                annotation = self.process_clicked_points(current_type)
                if annotation is None:
                    self.draw_annotation_in_progress(image, current_name, current_type)
                else:
                    annotations[current_name] = annotation
                    current_id += 1

            self.draw_finished_annotations(image, annotations)

            cv2.imshow(window_name, image)
            key = cv2.waitKey(10)
            if key == 8:  # Backspace key
                if current_id == 0:
                    continue
                if len(self.clicked_points) > 0:
                    self.clicked_points.pop()
                else:
                    current_id -= 1
                    del annotations[annotation_names[current_id]]
            elif key == ord("q"):
                cv2.destroyAllWindows()
                return None
            elif key == 13 and current_id >= num_annotations:  # Enter key
                cv2.destroyAllWindows()
                return annotations


def get_manual_annotations(
    image: OpenCVIntImageType, annotation_spec: dict[str, Annotation]
) -> Optional[dict[str, Any]]:
    return AnnotationWindow(image, annotation_spec).collect_annotations()


if __name__ == "__main__":
    import pprint

    import click

    @click.command()  # no help, takes the docstring of the function.
    @click.argument("image_path", type=click.Path(exists=True))
    def test_annotation_types(image_path: str) -> None:
        annotation_spec = {
            "keypoint": Annotation.Keypoint,
            "line_segment": Annotation.LineSegment,
            "line": Annotation.Line,
            "bounding_box": Annotation.BoundingBox,
            "polygon": Annotation.Polygon,
        }

        image = cv2.imread(image_path)
        annotations = get_manual_annotations(image, annotation_spec)

        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(annotations)

    test_annotation_types()
