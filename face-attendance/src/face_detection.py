"""
face_detection.py
------------------
Face detection using OpenCV's Haar Cascade classifier, as specified in the
project brief ("Use face detection models (Haar Cascade / MTCNN)").

Haar Cascade was chosen over MTCNN because it ships inside opencv-python's
own package data (`cv2.data.haarcascades`) with no extra model download
required, which matters in a sandboxed build environment. Swapping in MTCNN
(e.g. via the `mtcnn` or `facenet-pytorch` package) is a drop-in change:
implement `detect_faces()` with the same return signature and nothing else
in the pipeline needs to change.
"""

from pathlib import Path

import cv2
import numpy as np

_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(_CASCADE_PATH)

if _face_cascade.empty():
    raise RuntimeError(f"Failed to load Haar Cascade from {_CASCADE_PATH}")


def detect_faces(gray_img: np.ndarray) -> list:
    """
    Detect faces in a grayscale image.
    Returns a list of (x, y, w, h) bounding boxes (in ORIGINAL image
    coordinates), largest face first.

    Two preprocessing steps make Haar Cascade reliable across very
    different image sources:
      1. Padding + upscaling small/tightly-cropped images. Haar Cascade's
         frontal-face features expect some background context around the
         face (forehead-to-chin proportion relative to surroundings) —
         without it, tightly pre-cropped thumbnails (like this project's
         dataset photos) are missed entirely. Full, uncropped camera
         frames don't need this but padding is harmless for them.
      2. Histogram equalization, which compensates for low contrast /
         uneven lighting — directly relevant to the "different lighting"
         robustness requirement in the brief.
    """
    orig_h, orig_w = gray_img.shape[:2]
    pad = max(10, int(min(orig_h, orig_w) * 0.3))
    padded = cv2.copyMakeBorder(gray_img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    target_dim = 300
    scale = target_dim / max(padded.shape[:2])
    scaled = cv2.resize(padded, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    eq = cv2.equalizeHist(scaled.astype(np.uint8))

    min_size = max(30, int(target_dim * 0.2))
    raw_faces = _face_cascade.detectMultiScale(
        eq, scaleFactor=1.05, minNeighbors=4, minSize=(min_size, min_size)
    )
    raw_faces = sorted(raw_faces, key=lambda f: f[2] * f[3], reverse=True)

    # Map detections back to the original (unpadded, unscaled) coordinate space
    faces = []
    for (x, y, w, h) in raw_faces:
        ox = int(x / scale) - pad
        oy = int(y / scale) - pad
        ow = int(w / scale)
        oh = int(h / scale)
        # clip to original image bounds
        ox, oy = max(0, ox), max(0, oy)
        ow = min(ow, orig_w - ox)
        oh = min(oh, orig_h - oy)
        faces.append((ox, oy, ow, oh))
    return faces


def detect_and_crop_largest_face(gray_img: np.ndarray, padding: float = 0.15) -> np.ndarray:
    """
    Detect the largest face in the image and return a cropped version with
    a small padding margin. Falls back to the full image if no face is
    detected (common with the tightly-cropped Olivetti dataset images,
    where a small Haar Cascade margin miss shouldn't break the pipeline).
    """
    faces = detect_faces(gray_img)
    if not faces:
        return gray_img  # graceful fallback

    x, y, w, h = faces[0]
    pad_w, pad_h = int(w * padding), int(h * padding)
    H, W = gray_img.shape[:2]
    x0, y0 = max(0, x - pad_w), max(0, y - pad_h)
    x1, y1 = min(W, x + w + pad_w), min(H, y + h + pad_h)
    return gray_img[y0:y1, x0:x1]


def draw_detections(bgr_img: np.ndarray, faces: list) -> np.ndarray:
    """Draw bounding boxes on a copy of the image (for debugging / demo output)."""
    out = bgr_img.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return out


if __name__ == "__main__":
    # Smoke test on one of the prepared student images
    sample_path = Path(__file__).resolve().parent.parent / "data" / "raw_faces" / "STU001" / "img_00.png"
    img = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)
    faces = detect_faces(img)
    print(f"Image: {sample_path.name}  shape={img.shape}  faces detected={len(faces)}")
    if faces:
        print("Largest face bbox (x, y, w, h):", faces[0])
    cropped = detect_and_crop_largest_face(img)
    print("Cropped shape:", cropped.shape)
