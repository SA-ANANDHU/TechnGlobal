"""
recognize.py
------------
Loads the trained embedder + classifier, and given a new face image (as
would come from a classroom camera), runs the full pipeline:
  detect -> crop -> preprocess -> embed -> classify -> mark attendance

This is the "attendance marking system integrating recognition and
timestamp" step from the project brief.
"""

import json
from pathlib import Path

import cv2
import joblib
import numpy as np

from attendance_db import get_students, init_db, mark_attendance
from face_detection import detect_and_crop_largest_face, detect_faces
from preprocessing import preprocess_pipeline

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

CONFIDENCE_THRESHOLD = 0.55  # below this, treat as "unrecognized" rather than a low-confidence guess

_embedder = None
_classifier = None
_label_map = None


def _load_artifacts():
    global _embedder, _classifier, _label_map
    if _embedder is None:
        _embedder = joblib.load(MODELS_DIR / "embedder.joblib")
        _classifier = joblib.load(MODELS_DIR / "classifier.joblib")
        with open(MODELS_DIR / "label_map.json") as f:
            _label_map = json.load(f)
    return _embedder, _classifier, _label_map


def recognize_face(gray_img: np.ndarray) -> dict:
    """
    Run the full recognition pipeline on a single grayscale image.
    Returns a dict with detection status, predicted student, and confidence.
    Does NOT mark attendance — call mark_attendance_from_image for that.
    """
    embedder, classifier, label_map = _load_artifacts()

    faces = detect_faces(gray_img)
    if not faces:
        return {"face_detected": False, "message": "No face detected in the image."}

    cropped = detect_and_crop_largest_face(gray_img)
    processed = preprocess_pipeline(cropped)

    E = embedder.transform(np.array([processed]))
    student_id = str(classifier.predict(E)[0])

    confidence = None
    if hasattr(classifier, "predict_proba"):
        proba = classifier.predict_proba(E)[0]
        classes = list(classifier.classes_)
        confidence = float(proba[classes.index(student_id)])

    student_info = label_map.get(student_id, {"name": "Unknown", "roll_no": "N/A"})

    result = {
        "face_detected": True,
        "num_faces_detected": len(faces),
        "student_id": student_id,
        "name": student_info["name"],
        "roll_no": student_info["roll_no"],
        "confidence": round(confidence, 4) if confidence is not None else None,
        "recognized": confidence is None or confidence >= CONFIDENCE_THRESHOLD,
    }
    return result


def recognize_and_mark(gray_img: np.ndarray) -> dict:
    """Recognize a face and, if confident enough, mark attendance."""
    result = recognize_face(gray_img)
    if not result["face_detected"]:
        return result
    if not result["recognized"]:
        result["attendance"] = {"status": "rejected_low_confidence"}
        return result

    attendance_result = mark_attendance(result["student_id"], result["name"], result["confidence"])
    result["attendance"] = attendance_result
    return result


if __name__ == "__main__":
    init_db()
    # Demo: run recognition on a few held-out test images
    data_dir = ROOT / "data" / "raw_faces"
    demo_students = ["STU001", "STU010", "STU025"]

    print("=" * 70)
    print("TESTING FACE RECOGNITION + ATTENDANCE MARKING")
    print("=" * 70)
    for sid in demo_students:
        img_path = data_dir / sid / "img_09.png"  # a held-out-style sample
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        result = recognize_and_mark(img)
        print(f"\nInput: {img_path}")
        print(f"  Predicted: {result.get('name')} ({result.get('student_id')})  "
              f"confidence={result.get('confidence')}")
        print(f"  Attendance: {result.get('attendance')}")
