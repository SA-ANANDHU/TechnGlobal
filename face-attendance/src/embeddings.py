"""
embeddings.py
-------------
Facial embedding extraction.

Why not FaceNet / Dlib (as the brief suggests)?
------------------------------------------------
Both require downloading large pretrained weight files from hosts that
aren't reachable from this sandboxed build environment (dlib's models are
hosted on dlib.net as .bz2 files; Keras FaceNet weights are typically
distributed via Google Drive / GitHub Releases, not plain repo files).
Rather than silently skip this step, this project implements two classical,
fully self-contained embedding techniques that were the direct predecessors
to deep-learned face embeddings and are still taught alongside them:

  1. Eigenfaces (PCA) — projects each face into a low-dimensional space
     built from the directions of maximum variance across the training
     faces. This is the historically first successful automatic face
     recognition technique (Turk & Pentland, 1991) and is functionally
     an "embedding": a fixed-length vector per face that a classifier
     can be trained on, exactly as the brief's pipeline expects.
  2. LBPH (Local Binary Pattern Histograms) — a texture-based descriptor
     that's robust to monotonic lighting changes, computed via OpenCV's
     built-in `cv2.face` module (ships with opencv-contrib, no download
     needed).

Both are benchmarked in train.py; either can be swapped for a deep
embedding model later by implementing the same `.transform()` interface
(e.g. wrapping `facenet-pytorch`'s InceptionResnetV1) without touching
the rest of the pipeline.
"""

from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


class EigenfaceEmbedder:
    """PCA-based facial embedding ("Eigenfaces")."""

    def __init__(self, n_components: int = 60, random_state: int = 42):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, whiten=True, random_state=random_state)
        self.image_shape = None

    def fit(self, images: np.ndarray):
        """images: (N, H, W) float32 array in [0, 1]."""
        self.image_shape = images.shape[1:]
        flat = images.reshape(len(images), -1)
        self.pca.fit(flat)
        return self

    def transform(self, images: np.ndarray) -> np.ndarray:
        flat = images.reshape(len(images), -1)
        return self.pca.transform(flat)

    def fit_transform(self, images: np.ndarray) -> np.ndarray:
        self.fit(images)
        return self.transform(images)

    def explained_variance_ratio(self) -> float:
        return float(self.pca.explained_variance_ratio_.sum())

    def save(self, path: Path):
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path):
        return joblib.load(path)


class LBPHEmbedder:
    """
    Wraps OpenCV's LBPH descriptor computation to produce a fixed-length
    histogram feature vector per face (rather than using LBPH's built-in
    classifier, so it can plug into the same SVM/KNN classifier step as
    the Eigenfaces embedding for a fair comparison).
    """

    def __init__(self, radius=1, neighbors=8, grid_x=8, grid_y=8):
        self.radius = radius
        self.neighbors = neighbors
        self.grid_x = grid_x
        self.grid_y = grid_y

    @staticmethod
    def _lbp_image(img: np.ndarray, radius=1, neighbors=8) -> np.ndarray:
        """Compute a simple circular LBP image (uint8) from a grayscale face."""
        img = img.astype(np.float32)
        h, w = img.shape
        lbp = np.zeros((h, w), dtype=np.uint8)
        angles = [2 * np.pi * k / neighbors for k in range(neighbors)]
        offsets = [(radius * np.cos(a), -radius * np.sin(a)) for a in angles]

        padded = cv2.copyMakeBorder(img, radius, radius, radius, radius, cv2.BORDER_REFLECT)
        for k, (dx, dy) in enumerate(offsets):
            shifted = cv2.warpAffine(
                padded, np.float32([[1, 0, -dx], [0, 1, -dy]]),
                (padded.shape[1], padded.shape[0]), flags=cv2.INTER_LINEAR
            )
            shifted = shifted[radius:radius + h, radius:radius + w]
            lbp |= ((shifted >= img).astype(np.uint8)) << k
        return lbp

    def _histogram_for_image(self, img: np.ndarray) -> np.ndarray:
        lbp = self._lbp_image(img, self.radius, self.neighbors)
        h, w = lbp.shape
        cell_h, cell_w = h // self.grid_y, w // self.grid_x
        hist = []
        n_bins = 2 ** self.neighbors
        for gy in range(self.grid_y):
            for gx in range(self.grid_x):
                cell = lbp[gy * cell_h:(gy + 1) * cell_h, gx * cell_w:(gx + 1) * cell_w]
                h_, _ = np.histogram(cell, bins=n_bins, range=(0, n_bins))
                hist.append(h_.astype(np.float32) / (h_.sum() + 1e-6))
        return np.concatenate(hist)

    def fit(self, images: np.ndarray):
        return self  # stateless — no fitting required

    def transform(self, images: np.ndarray) -> np.ndarray:
        return np.array([self._histogram_for_image((img * 255).astype(np.uint8)) for img in images])

    def fit_transform(self, images: np.ndarray) -> np.ndarray:
        return self.transform(images)

    def save(self, path: Path):
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path):
        return joblib.load(path)


if __name__ == "__main__":
    # Smoke test with random data shaped like our preprocessed faces
    rng = np.random.default_rng(0)
    fake_faces = rng.random((20, 128, 128)).astype(np.float32)

    eig = EigenfaceEmbedder(n_components=10)
    emb = eig.fit_transform(fake_faces)
    print("Eigenfaces embedding shape:", emb.shape,
          f"(explains {eig.explained_variance_ratio()*100:.1f}% variance)")

    lbph = LBPHEmbedder(grid_x=4, grid_y=4)
    emb2 = lbph.fit_transform(fake_faces)
    print("LBPH embedding shape:", emb2.shape)
