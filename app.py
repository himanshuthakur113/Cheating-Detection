import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def draw_landmarks_on_image(image, landmarks, color=(0, 255, 0)):
    height, width, _ = image.shape
    for landmark in landmarks:
        px = int(landmark.x * width)
        py = int(landmark.y * height)
        cv2.circle(image, (px, py), 1, color, -1)

def draw_iris_tracking(image, face_landmarks):
    height, width, _ = image.shape
    iris_indices = [468, 469, 470, 471, 472, 473, 474, 475, 476, 477]
    for idx in iris_indices:
        lm = face_landmarks[idx]
        px, py = int(lm.x * width), int(lm.y * height)
        cv2.circle(image, (px, py), 2, (0, 0, 255), -1)

base_face_options = python.BaseOptions(model_asset_path='face_landmarker.task')
face_options = vision.FaceLandmarkerOptions(
    base_options=base_face_options,
    output_face_blendshapes=True,
    num_faces=1,
    running_mode=vision.RunningMode.VIDEO)

base_hand_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
hand_options = vision.HandLandmarkerOptions(
    base_options=base_hand_options,
    num_hands=2,
    running_mode=vision.RunningMode.VIDEO)

face_landmarker = vision.FaceLandmarker.create_from_options(face_options)
hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

    face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
    hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

    if face_result.face_landmarks:
        for face_lms in face_result.face_landmarks:
            draw_landmarks_on_image(frame, face_lms, color=(100, 100, 100)) # Gray face mesh
            draw_iris_tracking(frame, face_lms)

    if hand_result.hand_landmarks:
        for hand_lms in hand_result.hand_landmarks:
            draw_landmarks_on_image(frame, hand_lms, color=(0, 255, 0))

    cv2.imshow('Face & Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_landmarker.close()
hand_landmarker.close()