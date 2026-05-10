import cv2
def draw_face(image, landmarks):
    h, w = image.shape[:2]

    for i,j in enumerate(landmarks):
        x = int(j.x * w)
        y = int(j.y * h)

        cv2.circle(image, (x,y), 1, (100,100,100), -1)
        