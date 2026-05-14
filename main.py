import cv2
import mediapipe as mp
import time

from detectors.face_detector import FaceDetector
from visual.facevisual import draw_face
from head_pose import get_head_pose
from x import update
from y import update2

cap = cv2.VideoCapture(0)

face = FaceDetector()     #this is the object created from mediapipe face_landmarker.task and have option set

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        print("unable to read the frame")
        break

    timestamp_ms = int(time.time() * 1000)

    faces_result = face.detect(frame, timestamp_ms)   #now that object has a detect method which gives result like landmarks

    if faces_result.face_landmarks:                  #if there are landmarks on that result
        landmarks = faces_result.face_landmarks[0]   #a single frame can have multiple faces each face has 468 landmarks
        draw_face(frame, landmarks)                  #then we will draw those landmarks on the frame    
        text = get_head_pose(landmarks, frame)       #getting direction
        cv2.putText(frame, text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2) #printing direction
        #update(text)
        update2(text)

    cv2.imshow("Cheating Divyansh", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()