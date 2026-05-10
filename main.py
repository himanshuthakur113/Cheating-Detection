import cv2
import mediapipe as mp
import time

from detectors import facecheck

from visual import facevisual

from check import get_head_direction

from scoring.sus_eng import SusEng

cap = cv2.VideoCapture(0)

face = facecheck.FaceCheck()
frame_count = 0
SKIP_FRAMES = 2

last_face_result = None

prev_time = time.time()
eng = SusEng()

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
        last_face_result = face.gdmx(mp_image, timestamp_ms)
       
    face_result = last_face_result

    if face_result and face_result.face_landmarks:
        for face_lms in face_result.face_landmarks:
            facevisual.draw_face(frame, face_lms)
        
            direction = get_head_direction(face_lms)
            if direction == "LEFT":
                eng.add_score("look_left")
            elif direction == "RIGHT":
                eng.add_score("look_right")
            elif direction == "UP":
                eng.add_score("look_up")
            elif direction == "DOWN":
                eng.add_score("look_down")

            cv2.putText(frame, f'Head: {direction}',
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255) if direction != "CENTER" else (0, 255, 0),
                    2)

    score = eng.get_score()
    cv2.putText(frame,f'Score: {score}',(10,110), cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255))
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