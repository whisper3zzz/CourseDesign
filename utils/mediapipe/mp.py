import math
from typing import Tuple, Union
import mediapipe as mp
import numpy as np
import cv2


class MediaPipe:
    def __init__(self):
        self.model_path = 'utils/mediapipe/blaze_face_short_range.tflite'

        self.BaseOptions = mp.tasks.BaseOptions
        self.FaceDetector = mp.tasks.vision.FaceDetector
        self.FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        self.MARGIN = 10  # pixels
        self.ROW_SIZE = 10  # pixels
        self.FONT_SIZE = 1
        self.FONT_THICKNESS = 1
        self.TEXT_COLOR = (255, 0, 0)  # red

        self.options = self.FaceDetectorOptions(
            base_options=self.BaseOptions(
                model_asset_path=self.model_path,
                delegate=self.BaseOptions.Delegate.CPU,
            ),
            running_mode=self.VisionRunningMode.IMAGE,
            min_detection_confidence=0.35,
            min_suppression_threshold=0.25,
        )
        self.detector = None

    def ensure_detector(self):
        if self.detector is None:
            self.detector = self.FaceDetector.create_from_options(self.options)
        return self.detector

    def detect_face(self, frame):
        detector = self.ensure_detector()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return detector.detect(mp_image)

    def close(self):
        if getattr(self, "detector", None) is not None:
            self.detector.close()
            self.detector = None

    def crop_face(self, frame, face_detector_result):
        face = None
        if len(face_detector_result.detections) > 0:
            detection = face_detector_result.detections[0]
            bbox = detection.bounding_box
            start_point = [bbox.origin_x - 10, bbox.origin_y - 10]
            end_point = [bbox.origin_x + bbox.width + 20, bbox.origin_y + bbox.height + 20]
            face_image = frame[start_point[1]:end_point[1], start_point[0]:end_point[0]]
            face = cv2.resize(face_image, (112, 112))
        return face

    def visualize(self,
                  image,
                  detection_result
                  ) -> np.ndarray:
        """Draws bounding boxes and keypoints on the input image and return it.
        Args:
          image: The input RGB image.
          detection_result: The list of all "Detection" entities to be visualize.
        Returns:
          Image with bounding boxes.
        """
        annotated_image = image.copy()
        height, width, _ = image.shape

        for detection in detection_result.detections:
            # Draw bounding_box
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(annotated_image, start_point, end_point, self.TEXT_COLOR, 3)

            # Draw keypoints
            for keypoint in detection.keypoints:
                keypoint_px = self._normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                                    width, height)
                color, thickness, radius = (0, 255, 0), 2, 2
                cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

            # Draw label and score
            category = detection.categories[0]
            category_name = category.category_name
            category_name = '' if category_name is None else category_name
            probability = round(category.score, 2)
            result_text = category_name + ' (' + str(probability) + ')'
            text_location = (self.MARGIN + bbox.origin_x,
                             self.MARGIN + self.ROW_SIZE + bbox.origin_y)
            cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                        self.FONT_SIZE, self.TEXT_COLOR, self.FONT_THICKNESS)

        return annotated_image

    def transform_result(self, detection_result):
        faces = []
        for detection in detection_result.detections:
            x = detection.bounding_box.origin_x
            y = detection.bounding_box.origin_y
            w = detection.bounding_box.width
            h = detection.bounding_box.height
            faces.append((x, y, w, h))
        return faces

    def _normalized_to_pixel_coordinates(self,
                                         normalized_x: float, normalized_y: float, image_width: int,
                                         image_height: int) -> Union[None, Tuple[int, int]]:
        """Converts normalized value pair to pixel coordinates."""

        # Checks if the float value is between 0 and 1.
        def is_valid_normalized_value(value: float) -> bool:
            return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                              math.isclose(1, value))

        if not (is_valid_normalized_value(normalized_x) and
                is_valid_normalized_value(normalized_y)):
            # TODO: Draw coordinates even if it's outside of the image bounds.
            return None
        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)
        return x_px, y_px
