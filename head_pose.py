import cv2
import numpy as np 
import math 

def get_head_pose(lm,frame):
    h, w = frame.shape[:2]
    nose = 1
    chin = 152
    left_eye = 33
    right_eye = 263
    left_mouth = 61
    right_mouth = 291

    face_2d = []
    face_3d = []

    points = [nose, chin, left_eye, right_eye, left_mouth, right_mouth]

    for i in points:
        x = lm[i].x * w
        y = lm[i].y * h
        
        face_2d.append([x,y])
        face_3d.append([x,y,lm[i].z])

    face_2d = np.array(face_2d, dtype = np.float64)
    face_3d = np.array(face_3d, dtype = np.float64)

    cam_matrix = np.array([[w,0,w/2],
                          [0,w,h/2],
                          [0,0,1]])
    
    dist_matrix = np.zeros((4,1), dtype = np.float64)

    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

    rmat, jac = cv2.Rodrigues(rot_vec)
    angles, mtxR, mtxQ, qx, qy, qz = cv2.RQDecomp3x3(rmat)

    pitch = angles[0] * 360
    yaw = angles[1] * 360
    roll = angles[2] * 360

    if yaw < -10:
        text = "L"
    elif yaw > 10:
        text = "R"
    elif pitch < -10:
        text = "D"
    elif pitch > 10:
        text = "U"
    else:
        text = "F"

    return text

                        


