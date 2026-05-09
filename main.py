import cv2
import mediapipe as mp
import numpy as np
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from check import get_head_direction

cap = cv2.VideoCapture(0)

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

IRIS_INDICES = set([468,469,470,471,472,473,474,475,476,477])

frame_count = 0
SKIP_FRAMES = 2

last_face_result = None
last_hand_result = None


def draw_face_all(image, landmarks):
    h, w = image.shape[:2]

    for idx, lm in enumerate(landmarks):
        x = int(lm.x * w)
        y = int(lm.y * h)

        if idx in IRIS_INDICES:
            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
        else:
            cv2.circle(image, (x, y), 1, (100, 100, 100), -1)

def draw_landmarks_on_image(image, landmarks):
    h, w = image.shape[:2]
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    timestamp_ms = int(time.time() * 1000)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    #this is smart skipping frames to increase performance
    if frame_count % SKIP_FRAMES == 0:
        last_face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
        last_hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

    face_result = last_face_result
    hand_result = last_hand_result

    if face_result and face_result.face_landmarks:
        for face_lms in face_result.face_landmarks:
            draw_face_all(frame, face_lms)
        
            direction = get_head_direction(face_lms)

            cv2.putText(frame, f'Head: {direction}',
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255) if direction != "CENTER" else (0, 255, 0),
                    2)

    if hand_result and hand_result.hand_landmarks:
        for hand_lms in hand_result.hand_landmarks:
            draw_landmarks_on_image(frame, hand_lms)

    #this is to display the FPS 
    curr_time = time.time()
    fps = 1/(curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {int(fps)}',(10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    cv2.imshow('Face & Hand Tracking', frame)

    #this is to end the process
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



    
cap.release()
cv2.destroyAllWindows()