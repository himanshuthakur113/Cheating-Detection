import mediapipe as mp
from mediapipe.tasks import python 
from mediapipe.tasks.python import vision

from config import mp_face_model

class FaceDetector:
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path = mp_face_model)
        options = vision.FaceLandmarkerOptions(
                    base_options = base_options,
                    running_mode = vision.RunningMode.VIDEO,
                     num_faces = 1
        )          
        self.detector = vision.FaceLandmarker.create_from_options(options)

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


