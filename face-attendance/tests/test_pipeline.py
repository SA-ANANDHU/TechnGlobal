"""
test_pipeline.py
-----------------
Unit tests covering preprocessing, detection, embeddings, and the
attendance database. Run with: python3 -m pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from attendance_db import (
    already_marked_today,
    get_attendance,
    get_students,
    init_db,
    mark_attendance,
)
from embeddings import EigenfaceEmbedder, LBPHEmbedder
from face_detection import detect_and_crop_largest_face, detect_faces
from preprocessing import normalize_image, preprocess_pipeline, resize_image

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_IMG_PATH = ROOT / "data" / "raw_faces" / "STU001" / "img_00.png"


def _load_sample():
    import cv2
    return cv2.imread(str(SAMPLE_IMG_PATH), cv2.IMREAD_GRAYSCALE)


def test_resize_image_shape():
    img = np.zeros((64, 64), dtype=np.uint8)
    resized = resize_image(img, (128, 128))
    assert resized.shape == (128, 128)


def test_normalize_image_range():
    img = (np.random.rand(64, 64) * 255).astype(np.uint8)
    normalized = normalize_image(img)
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0


def test_preprocess_pipeline_output_shape():
    img = _load_sample()
    processed = preprocess_pipeline(img)
    assert processed.shape == (128, 128)
    assert processed.dtype == np.float32


def test_face_detection_finds_a_face():
    img = _load_sample()
    faces = detect_faces(img)
    assert len(faces) >= 1
    x, y, w, h = faces[0]
    assert w > 0 and h > 0


def test_detect_and_crop_returns_valid_image():
    img = _load_sample()
    cropped = detect_and_crop_largest_face(img)
    assert cropped.size > 0


def test_eigenface_embedder_shape():
    rng = np.random.default_rng(0)
    imgs = rng.random((10, 128, 128)).astype(np.float32)
    embedder = EigenfaceEmbedder(n_components=5)
    emb = embedder.fit_transform(imgs)
    assert emb.shape == (10, 5)


def test_lbph_embedder_shape():
    rng = np.random.default_rng(0)
    imgs = rng.random((4, 128, 128)).astype(np.float32)
    embedder = LBPHEmbedder(grid_x=4, grid_y=4)
    emb = embedder.transform(imgs)
    assert emb.shape[0] == 4
    assert emb.shape[1] == 4 * 4 * 256  # grid cells x histogram bins


def test_attendance_db_roundtrip():
    init_db()
    students = get_students()
    assert len(students) == 40

    test_id = students[0]["student_id"]
    test_name = students[0]["name"]

    # clean any leftover record from a previous test run
    from attendance_db import get_conn
    with get_conn() as conn:
        conn.execute("DELETE FROM attendance WHERE student_id = ?", (test_id,))

    assert not already_marked_today(test_id)
    result = mark_attendance(test_id, test_name, 0.9)
    assert result["status"] == "marked"
    assert already_marked_today(test_id)

    # second mark same day should be rejected as duplicate
    result2 = mark_attendance(test_id, test_name, 0.9)
    assert result2["status"] == "already_marked"

    records = get_attendance(student_id=test_id)
    assert len(records) == 1

    # cleanup
    with get_conn() as conn:
        conn.execute("DELETE FROM attendance WHERE student_id = ?", (test_id,))


if __name__ == "__main__":
    import subprocess
    subprocess.run(["python3", "-m", "pytest", __file__, "-v"])
