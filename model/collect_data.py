"""
collect_data.py
---------------
Behavioral training data collection pipeline for student attention/writing detection.

Labels:
    0 → looking at screen (attentive)
    1 → looking left or right
    2 → looking down, hand still
    3 → looking down, hand moving (writing)

Controls:
    0-3 → set active label
    q   → quit

Dependencies:
    pip install mediapipe opencv-python numpy pandas

Model files expected at:
    models/face_landmarker.task
    models/hand_landmarker.task

Download from:
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
"""

import csv
import math
import os
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# ── MediaPipe Tasks imports ────────────────────────────────────────────────────
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_DIR = Path("models")
FACE_MODEL = MODEL_DIR / "face_landmarker.task"
HAND_MODEL = MODEL_DIR / "hand_landmarker.task"
OUTPUT_CSV = "attention_training_data.csv"
CSV_HEADERS = ["timestamp", "frame_id", "yaw", "pitch", "roll", "ear", "wrist_velocity", "label"]

LABEL_NAMES = {
    0: "Attentive / Screen",
    1: "Looking Left/Right",
    2: "Looking Down (Still)",
    3: "Looking Down (Writing)",
}

# 3-D face model points (generic, in mm) matching landmark indices used below
# Indices for FaceLandmarker (478-point model):
#   1   = nose tip
#   152 = chin
#   263 = left eye outer corner
#   33  = right eye outer corner
#   287 = mouth left corner
#   57  = mouth right corner
FACE_3D_MODEL = np.array([
    [0.0,    0.0,    0.0],      # nose tip
    [0.0,   -63.6,  -12.5],     # chin
    [-43.3,  32.7,  -26.0],     # left eye outer corner
    [43.3,   32.7,  -26.0],     # right eye outer corner
    [-28.9, -28.9,  -24.1],     # mouth left corner
    [28.9,  -28.9,  -24.1],     # mouth right corner
], dtype=np.float64)

FACE_LANDMARK_INDICES = [1, 152, 263, 33, 287, 57]

# EAR landmark indices (FaceLandmarker 478-point model)
# Left eye:  p1=362 p2=385 p3=387 p4=263 p5=373 p6=380
# Right eye: p1=33  p2=160 p3=158 p4=133 p5=153 p6=144
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]

# Velocity smoothing window
VELOCITY_WINDOW = 5


# ── Model initialisation ───────────────────────────────────────────────────────

def initialize_models():
    """Load FaceLandmarker and HandLandmarker from .task files (VIDEO mode)."""
    if not FACE_MODEL.exists():
        raise FileNotFoundError(
            f"Face model not found: {FACE_MODEL}\n"
            "Download from: https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
        )
    if not HAND_MODEL.exists():
        raise FileNotFoundError(
            f"Hand model not found: {HAND_MODEL}\n"
            "Download from: https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        )

    face_options = FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(FACE_MODEL)),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,      # set True to enable blendshapes
        output_facial_transformation_matrixes=False,
    )

    hand_options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    face_landmarker = FaceLandmarker.create_from_options(face_options)
    hand_landmarker = HandLandmarker.create_from_options(hand_options)
    return face_landmarker, hand_landmarker


# ── Feature extraction ─────────────────────────────────────────────────────────

def calculate_head_pose(landmarks, frame_shape):
    """
    Estimate head yaw / pitch / roll (degrees) from FaceLandmarker landmarks
    using cv2.solvePnP with a generic 3-D face model.

    Returns (yaw, pitch, roll) as floats; (0.0, 0.0, 0.0) on failure.
    """
    if landmarks is None:
        return 0.0, 0.0, 0.0
    try:
        h, w = frame_shape[:2]
        image_points = np.array(
            [[landmarks[i].x * w, landmarks[i].y * h] for i in FACE_LANDMARK_INDICES],
            dtype=np.float64,
        )

        focal_length = w
        camera_matrix = np.array(
            [[focal_length, 0, w / 2],
             [0, focal_length, h / 2],
             [0, 0, 1]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, _ = cv2.solvePnP(
            FACE_3D_MODEL, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return 0.0, 0.0, 0.0

        rmat, _ = cv2.Rodrigues(rvec)
        # Decompose rotation matrix to Euler angles
        sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            roll  = math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))
            pitch = math.degrees(math.atan2(-rmat[2, 0], sy))
            yaw   = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))
        else:
            roll  = math.degrees(math.atan2(-rmat[1, 2], rmat[1, 1]))
            pitch = math.degrees(math.atan2(-rmat[2, 0], sy))
            yaw   = 0.0

        return (
            round(float(yaw),   2),
            round(float(pitch), 2),
            round(float(roll),  2),
        )
    except Exception:
        return 0.0, 0.0, 0.0


def _eye_aspect_ratio(landmarks, eye_indices):
    """Compute EAR for a single eye given landmark indices."""
    p = [landmarks[i] for i in eye_indices]
    # Vertical distances
    v1 = math.dist((p[1].x, p[1].y), (p[5].x, p[5].y))
    v2 = math.dist((p[2].x, p[2].y), (p[4].x, p[4].y))
    # Horizontal distance
    h  = math.dist((p[0].x, p[0].y), (p[3].x, p[3].y))
    return (v1 + v2) / (2.0 * h + 1e-6)


def calculate_ear(face_landmarks):
    """
    Compute average Eye Aspect Ratio from FaceLandmarker output.
    Returns a single float; 0.0 if no face detected.
    """
    if face_landmarks is None:
        return 0.0
    try:
        left_ear  = _eye_aspect_ratio(face_landmarks, LEFT_EYE_IDX)
        right_ear = _eye_aspect_ratio(face_landmarks, RIGHT_EYE_IDX)
        return round((left_ear + right_ear) / 2.0, 4)
    except Exception:
        return 0.0


def calculate_wrist_velocity(current_wrist, prev_wrist):
    """
    Euclidean distance between consecutive normalised wrist positions.
    Returns 0.0 when either wrist is None.
    """
    if current_wrist is None or prev_wrist is None:
        return 0.0
    try:
        dx = current_wrist[0] - prev_wrist[0]
        dy = current_wrist[1] - prev_wrist[1]
        return round(math.sqrt(dx * dx + dy * dy), 6)
    except Exception:
        return 0.0


# ── Drawing ────────────────────────────────────────────────────────────────────

def draw_landmarks(frame, face_result, hand_result):
    """Draw face mesh and hand skeleton on the frame (in-place)."""
    h, w = frame.shape[:2]

    # Face landmarks (thin dots)
    if face_result and face_result.face_landmarks:
        for lm_list in face_result.face_landmarks:
            for lm in lm_list:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)

    # Hand skeleton
    # Hardcoded hand connections (21 landmarks, MediaPipe standard topology)
    HAND_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),          # thumb
        (0,5),(5,6),(6,7),(7,8),          # index
        (5,9),(9,10),(10,11),(11,12),     # middle
        (9,13),(13,14),(14,15),(15,16),   # ring
        (13,17),(17,18),(18,19),(19,20),  # pinky
        (0,17),                           # palm base
    ]
    if hand_result and hand_result.hand_landmarks:
        connections = HAND_CONNECTIONS
        for lm_list in hand_result.hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in lm_list]
            for a, b in connections:
                cv2.line(frame, pts[a], pts[b], (255, 165, 0), 1)
            for pt in pts:
                cv2.circle(frame, pt, 3, (0, 140, 255), -1)


def draw_overlay(frame, label, yaw, pitch, roll, ear, wrist_vel, frames_saved):
    """Render HUD text on the frame."""
    overlay_data = [
        (f"LABEL: {label} ({LABEL_NAMES.get(label, '?')})", (0, 255, 255)),
        (f"Yaw: {yaw:.1f}",          (200, 200, 200)),
        (f"Pitch: {pitch:.1f}",      (200, 200, 200)),
        (f"Roll: {roll:.1f}",        (200, 200, 200)),
        (f"EAR: {ear:.3f}",          (200, 200, 200)),
        (f"Wrist Vel: {wrist_vel:.4f}", (200, 200, 200)),
        (f"Frames Saved: {frames_saved}", (100, 255, 100)),
    ]
    for i, (text, color) in enumerate(overlay_data):
        cv2.putText(frame, text, (10, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

    # Controls reminder at bottom
    cv2.putText(frame, "Keys: 0=Attentive 1=Left/Right 2=DownStill 3=Writing q=Quit",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


# ── CSV storage ────────────────────────────────────────────────────────────────

def _ensure_csv(path):
    """Create CSV with headers if it does not exist."""
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)


def save_to_csv(features: dict, path: str = OUTPUT_CSV):
    """Append one row to the CSV and flush immediately."""
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writerow(features)


# ── Main ───────────────────────────────────────────────────────────────────────

def print_startup_instructions():
    print("\n" + "=" * 60)
    print("  Behavioral Data Collector — MediaPipe Tasks Vision API")
    print("=" * 60)
    print("  CONTROLS")
    print("    0  →  Attentive / looking at screen")
    print("    1  →  Looking left or right")
    print("    2  →  Looking down, hand still")
    print("    3  →  Looking down, writing")
    print("    q  →  Quit and save")
    print()
    print("  COLLECTION TIPS")
    print("    • Collect ~3 minutes (≈540 frames @3fps) per label")
    print("    • Move naturally, include posture changes")
    print("    • Blink normally; vary lighting conditions")
    print("    • For label 3, vary writing speed")
    print("    • Target: 500–1000 frames per class")
    print()
    print(f"  Output → {OUTPUT_CSV}")
    print("=" * 60 + "\n")


def main():
    print_startup_instructions()

    # Ensure models directory and CSV exist
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_csv(OUTPUT_CSV)

    # Initialise models
    print("Loading models …", end=" ", flush=True)
    face_landmarker, hand_landmarker = initialize_models()
    print("OK\n")

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Check device index or permissions.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    current_label   = 0
    frame_id        = 0
    frames_saved    = 0
    prev_wrist      = None
    velocity_buffer = deque(maxlen=VELOCITY_WINDOW)

    print("Webcam open. Press a label key (0-3) to start collecting.\n")

    while True:
        ret, bgr_frame = cap.read()
        if not ret:
            print("Frame capture failed — retrying …")
            time.sleep(0.05)
            continue

        frame_id   += 1
        timestamp   = time.time()
        rgb_frame   = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image    = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        ts_ms       = int(timestamp * 1000)   # milliseconds for VIDEO mode

        # ── Inference ─────────────────────────────────────────────────────────
        face_result = face_landmarker.detect_for_video(mp_image, ts_ms)
        hand_result = hand_landmarker.detect_for_video(mp_image, ts_ms)

        # ── Extract face landmarks ─────────────────────────────────────────────
        face_lm = None
        if face_result.face_landmarks:
            face_lm = face_result.face_landmarks[0]   # first detected face

        # ── Extract wrist position ─────────────────────────────────────────────
        current_wrist = None
        if hand_result.hand_landmarks:
            wrist_lm      = hand_result.hand_landmarks[0][0]  # wrist = index 0
            current_wrist = (wrist_lm.x, wrist_lm.y)

        # ── Compute features ───────────────────────────────────────────────────
        yaw, pitch, roll = calculate_head_pose(face_lm, bgr_frame.shape)
        ear              = calculate_ear(face_lm)
        raw_velocity     = calculate_wrist_velocity(current_wrist, prev_wrist)

        # Smooth wrist velocity
        velocity_buffer.append(raw_velocity)
        wrist_velocity = round(sum(velocity_buffer) / len(velocity_buffer), 6)

        prev_wrist = current_wrist

        # ── Draw ───────────────────────────────────────────────────────────────
        draw_landmarks(bgr_frame, face_result, hand_result)
        draw_overlay(bgr_frame, current_label, yaw, pitch, roll,
                     ear, wrist_velocity, frames_saved)

        cv2.imshow("Behavioral Data Collector", bgr_frame)

        # ── Key handling ───────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print(f"\nQuitting. Total frames saved: {frames_saved}")
            break
        elif key in (ord("0"), ord("1"), ord("2"), ord("3")):
            new_label = int(chr(key))
            if new_label != current_label:
                print(f"  Label changed → {new_label}: {LABEL_NAMES[new_label]}")
            current_label = new_label

        # ── Save row ───────────────────────────────────────────────────────────
        # Validate features; skip NaN/Inf
        feature_values = [yaw, pitch, roll, ear, wrist_velocity]
        if any(not math.isfinite(v) for v in feature_values):
            continue

        save_to_csv({
            "timestamp":     round(timestamp, 4),
            "frame_id":      frame_id,
            "yaw":           yaw,
            "pitch":         pitch,
            "roll":          roll,
            "ear":           ear,
            "wrist_velocity": wrist_velocity,
            "label":         current_label,
        })
        frames_saved += 1

    # ── Cleanup ────────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()
    hand_landmarker.close()
    print(f"Data saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()