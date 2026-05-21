"""this file will contain the configuration"""

SKIP_FRAMES = 2

mp_face_model = "model/models/face_landmarker.task"
mp_hand_model = "model/models/hand_landmarker.task"

# paths to trained model and scaler
MODEL_PATH  = "model/attention_model.pkl"
SCALER_PATH = "model/scaler.pkl"

# head pose thresholds (degrees) — tuned to match train.py labels
YAW_THRESHOLD   = 10    # left/right — matches head_pose.py
PITCH_THRESHOLD = 10    # down

# wrist velocity threshold — above this = actively writing
WRITING_VEL_THRESHOLD = 0.008

# seconds a suspicious pose must be held before escalating watch → flag
SUSTAINED_FLAG_SECS = 8.0