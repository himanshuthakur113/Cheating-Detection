import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision  #mediapipe has three main modules: vision,audio and text we are using vision

from config import shakal_ka_model

class FaceCheck:
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path = shakal_ka_model)

        options = vision.FaceLandmarkerOptions(
            base_options = base_options,
            output_face_blendshapes = True,
            num_faces = 1,
            running_mode = vision.RunningMode.VIDEO)

        self.shakal_model = vision.FaceLandmarker.create_from_options(options)

    def gdmx(self,image,timestamp):
        return self.shakal_model.detect_for_video(image,timestamp)