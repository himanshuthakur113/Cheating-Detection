"""
detector.py
-----------
Core fusion engine for the cheating detection system.

Combines:
    - MediaPipe FaceLandmarker  (head pose via head_pose.py)
    - MediaPipe HandLandmarker  (wrist velocity)
    - Trained SVM + scaler      (behavioural classification)
    - Rule-based fusion         (multi-signal → final verdict)
    - Browser events            (tab switch, window blur, etc.)

Verdict levels:
    "safe"  → normal exam behaviour
    "watch" → ambiguous, monitor for longer
    "flag"  → high-confidence suspicious behaviour

Usage:
    from detector import CheatingDetector
    det = CheatingDetector()
    result = det.analyze(frame_bgr, timestamp_ms, student_id="s01")
"""

import math
import time
import base64
from collections import defaultdict, deque

import cv2
import numpy as np
import joblib
import mediapipe as mp

from detectors.face_detector import FaceDetector
from detectors.hand_detector import HandDetector
from head_pose import get_head_pose
from config import (
    YAW_THRESHOLD,
    PITCH_THRESHOLD,
    WRITING_VEL_THRESHOLD,
    SUSTAINED_FLAG_SECS,
    MODEL_PATH,
    SCALER_PATH,
)


# ── Feature helpers ────────────────────────────────────────────────────────────

LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]

VELOCITY_WINDOW = 5   # frames to smooth wrist velocity over


def _eye_aspect_ratio(landmarks, eye_indices):
    """EAR for one eye — same formula as collect_data.py."""
    p = [landmarks[i] for i in eye_indices]
    v1 = math.dist((p[1].x, p[1].y), (p[5].x, p[5].y))
    v2 = math.dist((p[2].x, p[2].y), (p[4].x, p[4].y))
    h  = math.dist((p[0].x, p[0].y), (p[3].x, p[3].y))
    return (v1 + v2) / (2.0 * h + 1e-6)


def _calculate_ear(face_lm):
    if face_lm is None:
        return 0.0
    try:
        left  = _eye_aspect_ratio(face_lm, LEFT_EYE_IDX)
        right = _eye_aspect_ratio(face_lm, RIGHT_EYE_IDX)
        return round((left + right) / 2.0, 4)
    except Exception:
        return 0.0


def _wrist_velocity(current, prev):
    if current is None or prev is None:
        return 0.0
    dx = current[0] - prev[0]
    dy = current[1] - prev[1]
    return round(math.sqrt(dx * dx + dy * dy), 6)


def _get_angles(face_lm, frame):
    """
    Returns (yaw, pitch, roll) using the same logic as head_pose.py
    but as floats instead of direction text, by duplicating the math.
    head_pose.get_head_pose returns only direction text so we redo here.
    """
    if face_lm is None:
        return 0.0, 0.0, 0.0
    try:
        h, w = frame.shape[:2]
        nose       = 1
        chin       = 152
        left_eye   = 33
        right_eye  = 263
        left_mouth = 61
        right_mouth= 291

        face_2d, face_3d = [], []
        for i in [nose, chin, left_eye, right_eye, left_mouth, right_mouth]:
            x = face_lm[i].x * w
            y = face_lm[i].y * h
            face_2d.append([x, y])
            face_3d.append([x, y, face_lm[i].z])

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        cam_matrix  = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]])
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        success, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
        if not success:
            return 0.0, 0.0, 0.0

        rmat, _  = cv2.Rodrigues(rot_vec)
        angles, *_ = cv2.RQDecomp3x3(rmat)

        pitch = angles[0] * 360
        yaw   = angles[1] * 360
        roll  = angles[2] * 360
        return round(float(yaw), 2), round(float(pitch), 2), round(float(roll), 2)
    except Exception:
        return 0.0, 0.0, 0.0


# ── Fusion rules ───────────────────────────────────────────────────────────────

def _apply_fusion(yaw, pitch, wrist_vel, hand_visible,
                  num_faces, svm_label, svm_proba,
                  sustained_secs, browser_event):
    """
    Combine all signals into a single verdict dict.
    Same threshold values as config.py and head_pose.py (yaw < -10, > 10, pitch < -10).
    """

    # browser events — highest confidence, skip all vision checks
    BROWSER_FLAGS = {
        "tab_switch"  : "switched browser tab",
        "window_blur" : "switched to another window",
        "devtools"    : "DevTools / inspect opened",
        "copy_paste"  : "copy / paste shortcut detected",
        "right_click" : "right-click detected",
    }
    if browser_event in BROWSER_FLAGS:
        return {
            "verdict":    "flag",
            "reason":     BROWSER_FLAGS[browser_event],
            "confidence": 1.0,
            "score":      100,
        }

    score   = 0
    verdict = "safe"
    reason  = "normal behaviour"

    # multiple faces in frame
    if num_faces > 1:
        return {
            "verdict":    "flag",
            "reason":     "multiple faces detected in frame",
            "confidence": 0.95,
            "score":      90,
        }

    # head turned sideways — matches head_pose.py thresholds (>10 / <-10)
    if yaw < -YAW_THRESHOLD or yaw > YAW_THRESHOLD:
        direction = "right" if yaw > 0 else "left"
        score    += 40
        reason    = f"head turned {direction}"
        verdict   = "flag" if sustained_secs > 3.0 else "watch"

    # head down — check hand to distinguish writing from reading notes
    elif pitch < -PITCH_THRESHOLD:
        if hand_visible and wrist_vel > WRITING_VEL_THRESHOLD:
            score  += 5
            verdict = "safe"
            reason  = "looking down — writing on paper (ok)"
        elif hand_visible and wrist_vel <= WRITING_VEL_THRESHOLD:
            score  += 20
            verdict = "watch"
            reason  = "looking down, hand still — monitoring"
            if sustained_secs > SUSTAINED_FLAG_SECS:
                score  += 30
                verdict = "flag"
                reason  = f"looking down, hand still for {sustained_secs:.0f}s"
        else:
            score  += 25
            verdict = "watch"
            reason  = "looking down — no hand visible"
            if sustained_secs > SUSTAINED_FLAG_SECS:
                score  += 25
                verdict = "flag"
                reason  = f"looking down for {sustained_secs:.0f}s — possible notes"

    # SVM caught something the rules missed
    elif svm_label in (1, 2) and svm_proba > 0.75:
        score  += int(svm_proba * 30)
        reason  = f"model flagged suspicious gaze ({svm_proba:.0%})"
        verdict = "flag" if svm_proba > 0.90 else "watch"

    confidence = round(min(1.0, score / 100.0), 2)
    return {"verdict": verdict, "reason": reason,
            "confidence": confidence, "score": score}


# ── Main detector class ────────────────────────────────────────────────────────

class CheatingDetector:
    """
    Stateful detector — one instance per Flask app.
    Maintains per-student wrist history and suspicious-pose timing.
    """

    def __init__(self):
        print("[detector] loading face and hand models ...")
        self.face = FaceDetector()
        self.hand = HandDetector()

        print("[detector] loading SVM and scaler ...")
        try:
            self.model  = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            print("[detector] SVM ready")
        except FileNotFoundError as e:
            print(f"[detector] WARNING: {e}")
            print("[detector] running without SVM — rule-based only")
            self.model  = None
            self.scaler = None

        # per-student state
        self._prev_wrist:  dict = defaultdict(lambda: None)
        self._vel_buffer:  dict = defaultdict(lambda: deque(maxlen=VELOCITY_WINDOW))
        self._flag_start:  dict = defaultdict(lambda: None)
        self._alert_count: dict = defaultdict(int)

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyze(self, frame_bgr, timestamp_ms, student_id="student", browser_event=None):
        """
        Analyze one BGR frame.

        Parameters
        ----------
        frame_bgr    : np.ndarray  BGR image from OpenCV / decoded JPEG
        timestamp_ms : int         monotonic ms timestamp (required by MediaPipe VIDEO mode)
        student_id   : str         keeps state separate per student
        browser_event: str | None  one of: tab_switch, window_blur, devtools, copy_paste

        Returns
        -------
        dict with: verdict, reason, confidence, score, alert_count,
                   student_id, details, timestamp
        """

        # browser events need no vision
        if browser_event:
            fusion = _apply_fusion(0, 0, 0, False, 1, 0, 0.0, 0.0, browser_event)
            self._maybe_count_alert(student_id, fusion["verdict"])
            return self._build(fusion, {}, student_id)

        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        face_result = self.face.detect(rgb_frame, timestamp_ms)
        hand_result = self.hand.detect(rgb_frame, timestamp_ms)

        num_faces = len(face_result.face_landmarks) if face_result.face_landmarks else 0

        # no face — flag after a few seconds
        if num_faces == 0:
            self._update_timer(student_id, suspicious=True)
            secs = self._sustained(student_id)
            fusion = {
                "verdict":    "flag" if secs > 5 else "watch",
                "reason":     "no face detected",
                "confidence": round(min(1.0, secs / 10), 2),
                "score":      min(100, int(secs * 10)),
            }
            self._maybe_count_alert(student_id, fusion["verdict"])
            return self._build(fusion, {}, student_id)

        face_lm = face_result.face_landmarks[0]

        # wrist position + smoothed velocity
        current_wrist = self.hand.get_wrist(hand_result)
        raw_vel       = _wrist_velocity(current_wrist, self._prev_wrist[student_id])
        self._vel_buffer[student_id].append(raw_vel)
        wrist_vel = sum(self._vel_buffer[student_id]) / len(self._vel_buffer[student_id])
        self._prev_wrist[student_id] = current_wrist

        hand_visible = current_wrist is not None

        # angles (float version of head_pose.py logic)
        yaw, pitch, roll = _get_angles(face_lm, frame_bgr)

        # ear
        ear = _calculate_ear(face_lm)

        # SVM inference — feature order must match collect_data.py / train.py
        # columns: yaw, pitch, roll, ear, wrist_velocity
        svm_label, svm_proba = 0, 0.0
        if self.model and self.scaler:
            try:
                import pandas as pd
                feat        = pd.DataFrame(
                    [[yaw, pitch, roll, ear, wrist_vel]],
                    columns=["yaw", "pitch", "roll", "ear", "wrist_velocity"]
                )
                feat_scaled = self.scaler.transform(feat)
                proba       = self.model.predict_proba(feat_scaled)[0]
                svm_label   = int(np.argmax(proba))
                svm_proba   = float(np.max(proba))
            except Exception as e:
                print(f"[detector] SVM error: {e}")

        # sustained timing
        is_suspicious = (abs(yaw) > YAW_THRESHOLD or
                         pitch < -PITCH_THRESHOLD or
                         (svm_label in (1, 2) and svm_proba > 0.75))
        self._update_timer(student_id, suspicious=is_suspicious)
        sustained = self._sustained(student_id)

        fusion = _apply_fusion(
            yaw, pitch, wrist_vel, hand_visible,
            num_faces, svm_label, svm_proba,
            sustained, browser_event
        )

        details = {
            "yaw":          yaw,
            "pitch":        pitch,
            "roll":         roll,
            "ear":          ear,
            "wrist_vel":    round(wrist_vel, 5),
            "hand_visible": hand_visible,
            "svm_label":    svm_label,
            "svm_proba":    round(svm_proba, 3),
            "sustained_s":  round(sustained, 1),
            "num_faces":    num_faces,
        }

        self._maybe_count_alert(student_id, fusion["verdict"])
        return self._build(fusion, details, student_id)

    def analyze_base64(self, b64_string, timestamp_ms, student_id="student", browser_event=None):
        """Convenience wrapper — accepts a base64 JPEG string from the frontend."""
        if "," in b64_string:
            b64_string = b64_string.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_string)
        arr       = np.frombuffer(img_bytes, np.uint8)
        frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return self._build({"verdict": "watch", "reason": "could not decode image",
                                "confidence": 0.0, "score": 0}, {}, student_id)
        return self.analyze(frame_bgr, timestamp_ms, student_id, browser_event)

    def reset_student(self, student_id):
        """Call this at the start of a new exam session."""
        self._prev_wrist[student_id]  = None
        self._vel_buffer[student_id]  = deque(maxlen=VELOCITY_WINDOW)
        self._flag_start[student_id]  = None
        self._alert_count[student_id] = 0

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _update_timer(self, student_id, suspicious):
        if suspicious:
            if self._flag_start[student_id] is None:
                self._flag_start[student_id] = time.time()
        else:
            self._flag_start[student_id] = None

    def _sustained(self, student_id):
        start = self._flag_start[student_id]
        return 0.0 if start is None else time.time() - start

    def _maybe_count_alert(self, student_id, verdict):
        if verdict != "safe":
            self._alert_count[student_id] += 1

    def _build(self, fusion, details, student_id):
        return {
            "verdict":     fusion["verdict"],
            "reason":      fusion["reason"],
            "confidence":  fusion["confidence"],
            "score":       fusion["score"],
            "alert_count": self._alert_count[student_id],
            "student_id":  student_id,
            "details":     details,
            "timestamp":   round(time.time(), 3),
        }