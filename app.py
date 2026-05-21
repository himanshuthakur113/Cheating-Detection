"""
app.py
------
Flask API server for the cheating detection system.

Endpoints:
    POST /analyze        → analyze a webcam frame (base64 JPEG)
    POST /browser-event  → log a browser-level event (tab switch etc.)
    GET  /logs           → return all session alerts
    POST /reset          → clear a student's session
    GET  /health         → sanity check

Run:
    python app.py
"""

import json
import time
from collections import defaultdict

from flask import Flask, request, jsonify
from flask_cors import CORS

from detector import CheatingDetector

app = Flask(__name__)
CORS(app)   # allow frontend on any port to call us

# one detector instance shared across all requests (holds MediaPipe models)
detector = CheatingDetector()

# in-memory alert log  { student_id: [ result_dict, ... ] }
alert_log = defaultdict(list)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": round(time.time(), 3)})


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Expects JSON:
    {
        "frame":        "<base64 JPEG string>",
        "student_id":   "student_01",        // optional
        "timestamp_ms": 1234567890123        // optional, defaults to now
    }
    Returns the verdict dict from CheatingDetector.
    """
    data = request.get_json(silent=True)
    if not data or "frame" not in data:
        return jsonify({"error": "missing 'frame' field"}), 400

    student_id   = data.get("student_id", "student")
    timestamp_ms = int(data.get("timestamp_ms", time.time() * 1000))

    result = detector.analyze_base64(
        data["frame"],
        timestamp_ms,
        student_id=student_id,
    )

    # store in log if not safe
    if result["verdict"] != "safe":
        alert_log[student_id].append(result)

    return jsonify(result)


@app.route("/browser-event", methods=["POST"])
def browser_event():
    """
    Expects JSON:
    {
        "event":      "tab_switch",   // tab_switch | window_blur | devtools | copy_paste | right_click
        "student_id": "student_01"
    }
    """
    data = request.get_json(silent=True)
    if not data or "event" not in data:
        return jsonify({"error": "missing 'event' field"}), 400

    student_id = data.get("student_id", "student")
    event      = data["event"]

    result = detector.analyze(
        __blank_frame(),
        int(time.time() * 1000),
        student_id=student_id,
        browser_event=event,
    )

    alert_log[student_id].append(result)
    return jsonify(result)


@app.route("/logs", methods=["GET"])
def logs():
    """
    Returns all alerts.
    Optional query param: ?student_id=student_01
    """
    student_id = request.args.get("student_id")
    if student_id:
        return jsonify(alert_log.get(student_id, []))

    # return summary across all students for the dashboard
    summary = []
    for sid, events in alert_log.items():
        if not events:
            continue
        latest   = events[-1]
        flagged  = [e for e in events if e["verdict"] == "flag"]
        score    = sum(e["score"] for e in events)
        risk     = "high" if score > 50 else "medium" if score > 20 else "low"
        summary.append({
            "student_id":   sid,
            "alert_count":  len(events),
            "flag_count":   len(flagged),
            "risk_level":   risk,
            "total_score":  score,
            "last_verdict": latest["verdict"],
            "last_reason":  latest["reason"],
            "last_time":    latest["timestamp"],
            "events":       events,
        })

    return jsonify(summary)


@app.route("/reset", methods=["POST"])
def reset():
    """
    Expects JSON: { "student_id": "student_01" }
    Clears session state for that student.
    """
    data       = request.get_json(silent=True) or {}
    student_id = data.get("student_id", "student")
    detector.reset_student(student_id)
    alert_log[student_id] = []
    return jsonify({"status": "reset", "student_id": student_id})


# ── Helpers ────────────────────────────────────────────────────────────────────

def __blank_frame():
    """A tiny blank BGR frame — used for browser events that need no vision."""
    import numpy as np
    return np.zeros((4, 4, 3), dtype=np.uint8)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n[app] Starting Cheating Detection API ...")
    print("[app] Endpoints:")
    print("       POST http://localhost:5000/analyze")
    print("       POST http://localhost:5000/browser-event")
    print("       GET  http://localhost:5000/logs")
    print("       POST http://localhost:5000/reset")
    print("       GET  http://localhost:5000/health\n")
    app.run(host="0.0.0.0", port=5000, debug=False)