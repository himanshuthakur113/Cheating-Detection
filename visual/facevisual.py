import cv2
def draw_face(image, landmarks):
    h, w = image.shape[:2]

    for landmark in landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)

        cv2.circle(image, (x,y), 1, (100,100,100), -1)
        