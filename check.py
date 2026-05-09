def get_head_direction(face_landmarks):
    nose = face_landmarks[1]
    
    left_face = face_landmarks[234]
    right_face = face_landmarks[454]

    left_dist = abs(nose.x - left_face.x)
    right_dist = abs(nose.x - right_face.x)

    ratio = left_dist/(right_dist + 1e-6)

    if ratio > 1.5:
        return 'Right'
    elif ratio < 0.067:
        return 'Left'
    else:
        return 'Center'

