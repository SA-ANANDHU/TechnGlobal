"""
app.py
------
Flask API + dashboard server for the Face Recognition Attendance System.

Endpoints
---------
GET  /api/health              -> service health check
POST /api/recognize            -> recognize a face from an uploaded image (does NOT mark attendance)
POST /api/mark-attendance      -> recognize a face AND mark attendance if confident enough
GET  /api/attendance           -> attendance records (optional ?date=YYYY-MM-DD&student_id=STUxxx)
GET  /api/attendance/summary   -> present/absent counts for a date (defaults to today)
GET  /api/students             -> full student roster
GET  /api/model-info           -> deployed model metadata + test-set metrics
GET  /                         -> serves the dashboard frontend
"""

import base64
import json
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from attendance_db import get_attendance, get_attendance_summary, get_students, init_db
from recognize import recognize_and_mark, recognize_face

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"
RESULTS_DIR = ROOT / "results"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")
CORS(app)

init_db()  # ensure tables exist and roster is loaded on startup


def _decode_image(data: dict) -> np.ndarray:
    """
    Accept either a base64 data URL (from a <canvas>/webcam capture) or raw
    base64 bytes, and decode to a grayscale OpenCV image.
    """
    b64 = data.get("image", "")
    if "," in b64:  # strip a data URL prefix like "data:image/png;base64,"
        b64 = b64.split(",", 1)[1]
    img_bytes = base64.b64decode(b64)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    return img


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "face-attendance-api"}), 200


@app.route("/api/recognize", methods=["POST"])
def recognize():
    data = request.get_json(silent=True) or {}
    if "image" not in data:
        return jsonify({"error": "Field 'image' (base64) is required."}), 400
    try:
        img = _decode_image(data)
        if img is None:
            return jsonify({"error": "Could not decode image."}), 400
        result = recognize_face(img)
        return jsonify(result), 200
    except Exception as exc:
        return jsonify({"error": f"Recognition failed: {str(exc)}"}), 500


@app.route("/api/mark-attendance", methods=["POST"])
def mark():
    data = request.get_json(silent=True) or {}
    if "image" not in data:
        return jsonify({"error": "Field 'image' (base64) is required."}), 400
    try:
        img = _decode_image(data)
        if img is None:
            return jsonify({"error": "Could not decode image."}), 400
        result = recognize_and_mark(img)
        return jsonify(result), 200
    except Exception as exc:
        return jsonify({"error": f"Attendance marking failed: {str(exc)}"}), 500


@app.route("/api/attendance", methods=["GET"])
def attendance():
    date_filter = request.args.get("date")
    student_id = request.args.get("student_id")
    records = get_attendance(date_filter=date_filter, student_id=student_id)
    return jsonify({"count": len(records), "records": records}), 200


@app.route("/api/attendance/summary", methods=["GET"])
def attendance_summary():
    date_filter = request.args.get("date")
    return jsonify(get_attendance_summary(date_filter=date_filter)), 200


@app.route("/api/students", methods=["GET"])
def students():
    return jsonify({"count": len(get_students()), "students": get_students()}), 200


@app.route("/api/model-info", methods=["GET"])
def model_info():
    summary_path = RESULTS_DIR / "metrics_summary.json"
    if not summary_path.exists():
        return jsonify({"error": "Model metrics not found. Run src/train.py first."}), 404
    with open(summary_path) as f:
        summary = json.load(f)
    best = summary.get("best_pipeline")
    return jsonify({
        "best_pipeline": best,
        "n_students": summary.get("n_students"),
        "train_size": summary.get("train_size"),
        "test_size": summary.get("test_size"),
        "metrics": summary.get("metrics", {}).get(best, {}),
    }), 200


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed."}), 405


if __name__ == "__main__":
    import os
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=5001, debug=debug_mode, use_reloader=False)
