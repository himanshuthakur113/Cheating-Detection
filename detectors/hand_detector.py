import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import mp_hand_model

class HandDetector:
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path = mp_hand_model)
        options = vision.HandLandmarkerOptions(
                    base_options = base_options,
                    running_mode = vision.RunningMode.VIDEO,
                    num_hands = 2,
                    min_hand_detection_confidence = 0.5,
                    min_hand_presence_confidence  = 0.5,
                    min_tracking_confidence       = 0.5,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame, timestamp_ms):

        rgb = mp.Image(
                image_format = mp.ImageFormat.SRGB,
                data = frame
        )

        result = self.detector.detect_for_video(
                rgb,
                timestamp_ms
        )

        return result

    def get_wrist(self, hand_result):
        """Returns (x, y) of first detected wrist in normalised coords, or None."""
        if hand_result.hand_landmarks:
            wrist = hand_result.hand_landmarks[0][0]   # wrist = index 0
            return (wrist.x, wrist.y)
        return None