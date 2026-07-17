"""
preprocessing.py
-----------------
Image preprocessing utilities: resize, normalize, and augment face images
before feature extraction, as specified in the project brief.
"""

import cv2
import numpy as np

STANDARD_SIZE = (128, 128)  # (width, height) — all faces are resized to this before feature extraction


def resize_image(img: np.ndarray, size=STANDARD_SIZE) -> np.ndarray:
    """Resize a grayscale image to a standard size using area interpolation (best for shrinking)."""
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Scale pixel intensities to [0, 1] and apply histogram equalization.
    Histogram equalization specifically helps face recognition be more robust
    to lighting variation, which is one of the axes this project is tested on.
    """
    equalized = cv2.equalizeHist(img.astype(np.uint8))
    return equalized.astype(np.float32) / 255.0


def augment_image(img_uint8: np.ndarray, rng: np.random.Generator) -> list:
    """
    Generate a small set of augmented variants of a face image to increase
    training diversity, simulating slightly different capture conditions
    (as real classroom cameras would produce):
      - horizontal flip
      - small random rotation (+/- 10 degrees)
      - brightness jitter
    Returns a list of uint8 images (does NOT include the original).
    """
    h, w = img_uint8.shape
    variants = []

    # 1. Horizontal flip
    variants.append(cv2.flip(img_uint8, 1))

    # 2. Small rotation
    angle = rng.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(img_uint8, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    variants.append(rotated)

    # 3. Brightness jitter
    factor = rng.uniform(0.7, 1.3)
    brightened = np.clip(img_uint8.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    variants.append(brightened)

    return variants


def preprocess_pipeline(img_uint8: np.ndarray) -> np.ndarray:
    """Full pipeline for a single face crop: resize -> normalize. Returns float32 [0,1] array."""
    resized = resize_image(img_uint8, STANDARD_SIZE)
    normalized = normalize_image(resized)
    return normalized


if __name__ == "__main__":
    # Quick smoke test using a synthetic gradient "face"
    sample = np.tile(np.linspace(0, 255, 64), (64, 1)).astype(np.uint8)
    processed = preprocess_pipeline(sample)
    print("Input shape:", sample.shape, "Output shape:", processed.shape,
          "Output range:", processed.min(), "-", processed.max())
    rng = np.random.default_rng(42)
    augmented = augment_image(sample, rng)
    print(f"Generated {len(augmented)} augmented variants")
