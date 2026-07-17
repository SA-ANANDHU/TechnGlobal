"""
train.py
--------
End-to-end training pipeline for the Face Recognition Attendance System.

Steps (matching the project's 10-point guidance):
  1. Load student face images (data/raw_faces/<student_id>/*.png)
  2. Detect faces with Haar Cascade, crop to the detected region
  3. Preprocess: resize, normalize (histogram equalization)
  4. Augment the training split (flip / rotate / brightness jitter)
  5. Extract embeddings: Eigenfaces (PCA) and LBPH, benchmarked against each other
  6. Train SVM and KNN classifiers on each embedding (4 combinations total)
  7. Tune hyperparameters via GridSearchCV
  8. Evaluate with accuracy, precision, recall, F1, confusion matrix
  9. Persist the best pipeline (embedder + classifier + label map) to disk
 10. Save plots/metrics to results/ for documentation
"""

import json
import time
import warnings
from pathlib import Path

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from embeddings import EigenfaceEmbedder, LBPHEmbedder
from face_detection import detect_and_crop_largest_face
from preprocessing import augment_image, preprocess_pipeline

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_FACES_DIR = DATA_DIR / "raw_faces"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def load_and_prepare_dataset():
    """
    Load every student image, run face detection + crop, then preprocess.
    Returns X (float32 images, H x W), y (student_id strings), and the roster.
    """
    roster = pd.read_csv(DATA_DIR / "students.csv")
    student_dirs = sorted(RAW_FACES_DIR.iterdir())

    images, labels = [], []
    detection_hits = 0
    total = 0

    for student_dir in student_dirs:
        student_id = student_dir.name
        for img_path in sorted(student_dir.glob("*.png")):
            total += 1
            raw = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            cropped = detect_and_crop_largest_face(raw)
            if cropped.shape != raw.shape:
                detection_hits += 1
            processed = preprocess_pipeline(cropped)
            images.append(processed)
            labels.append(student_id)

    print(f"Loaded {total} images across {len(student_dirs)} students "
          f"(face detector adjusted crop on {detection_hits}/{total} images)")
    return np.array(images, dtype=np.float32), np.array(labels), roster


def augment_training_set(X_train, y_train, rng):
    """Expand the training split with augmented variants (flip/rotate/brightness)."""
    aug_images, aug_labels = [], []
    for img, label in zip(X_train, y_train):
        img_uint8 = (img * 255).astype(np.uint8)
        for variant in augment_image(img_uint8, rng):
            aug_images.append(variant.astype(np.float32) / 255.0)
            aug_labels.append(label)

    X_aug = np.concatenate([X_train, np.array(aug_images, dtype=np.float32)], axis=0)
    y_aug = np.concatenate([y_train, np.array(aug_labels)], axis=0)
    return X_aug, y_aug


def get_classifier_grids():
    # NOTE: probability=False during the grid search — SVC's probability=True
    # runs an internal 5-fold Platt-scaling CV on every single .fit() call,
    # which nested inside GridSearchCV's own CV loop multiplies runtime by ~5x
    # for no benefit during search. We re-fit the winning config with
    # probability=True once, after the best hyperparameters are already known.
    return {
        "SVM": (
            SVC(probability=False, random_state=RANDOM_STATE),
            {"C": [1, 5, 10], "kernel": ["linear", "rbf"]},
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"n_neighbors": [1, 3, 5], "weights": ["uniform", "distance"]},
        ),
    }


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, average="macro", zero_division=0),
        "recall": recall_score(y_test, preds, average="macro", zero_division=0),
        "f1_score": f1_score(y_test, preds, average="macro", zero_division=0),
    }
    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.4f}")
    return metrics, preds


def plot_confusion(name, y_test, preds, labels, path):
    cm = confusion_matrix(y_test, preds, labels=labels)
    plt.figure(figsize=(11, 9))
    sns.heatmap(cm, cmap="Blues", cbar=True, square=True,
                xticklabels=False, yticklabels=False)
    plt.title(f"Confusion Matrix — {name}  (40 students)")
    plt.xlabel("Predicted student")
    plt.ylabel("Actual student")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_eigenfaces(pca_model, path, n=10):
    """Visualize the top principal components ("eigenfaces") — a classic
    sanity-check plot for PCA-based face recognition."""
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    h, w = pca_model.image_shape
    for i, ax in enumerate(axes.flat):
        if i < n:
            ax.imshow(pca_model.pca.components_[i].reshape(h, w), cmap="gray")
            ax.set_title(f"Eigenface {i+1}", fontsize=9)
        ax.axis("off")
    plt.suptitle("Top 10 Eigenfaces (principal components of student face variation)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_model_comparison(all_metrics: dict, path):
    df = pd.DataFrame(all_metrics).T[["accuracy", "precision", "recall", "f1_score"]]
    ax = df.plot(kind="bar", figsize=(10, 5), rot=20)
    ax.set_title("Embedding + Classifier Comparison — Face Attendance System")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def main():
    t0 = time.time()
    rng = np.random.default_rng(RANDOM_STATE)
    print("=" * 70)
    print("FACE RECOGNITION ATTENDANCE SYSTEM — TRAINING PIPELINE")
    print("=" * 70)

    # 1-3. Load, detect, preprocess
    X, y, roster = load_and_prepare_dataset()
    print(f"Dataset shape: {X.shape}, {len(np.unique(y))} unique students")

    # Stratified split: 7 train / 3 test images per student (preserves per-class balance,
    # and the held-out images naturally cover different lighting/expression/angle —
    # this is exactly the brief's "test under different lighting and angles" step).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")

    # 4. Augment training data only (never touch the test set)
    X_train_aug, y_train_aug = augment_training_set(X_train, y_train, rng)
    print(f"After augmentation: {len(X_train_aug)} training images "
          f"({len(X_train_aug) - len(X_train)} synthetic variants added)")

    # 5. Embeddings
    embedders = {
        "Eigenfaces (PCA)": EigenfaceEmbedder(n_components=80, random_state=RANDOM_STATE),
        "LBPH": LBPHEmbedder(grid_x=6, grid_y=6),
    }

    all_metrics = {}
    fitted_pipelines = {}  # (embedding_name, clf_name) -> (embedder, classifier)

    for emb_name, embedder in embedders.items():
        print(f"\n{'='*50}\nEmbedding: {emb_name}\n{'='*50}")
        E_train = embedder.fit_transform(X_train_aug)
        E_test = embedder.transform(X_test)
        if hasattr(embedder, "explained_variance_ratio"):
            print(f"  PCA explains {embedder.explained_variance_ratio()*100:.1f}% of variance "
                  f"with {embedder.n_components} components")

        for clf_name, (clf, grid) in get_classifier_grids().items():
            combo_name = f"{emb_name} + {clf_name}"
            print(f"\nTuning {combo_name} (5-fold GridSearchCV)...", flush=True)
            t_fit = time.time()
            gs = GridSearchCV(clf, grid, scoring="f1_macro", cv=5, n_jobs=1)
            gs.fit(E_train, y_train_aug)
            print(f"  ({time.time()-t_fit:.1f}s)", flush=True)
            print(f"  Best params: {gs.best_params_}")
            best_clf = gs.best_estimator_

            # Re-fit SVM with probability=True (needed for confidence scores
            # at inference time) now that the best hyperparameters are known —
            # this one-time fit is much cheaper than doing it for every
            # candidate during the grid search.
            if clf_name == "SVM":
                best_clf = SVC(probability=True, random_state=RANDOM_STATE, **gs.best_params_)
                best_clf.fit(E_train, y_train_aug)

            metrics, preds = evaluate(combo_name, best_clf, E_test, y_test)
            metrics["best_params"] = gs.best_params_
            all_metrics[combo_name] = metrics
            fitted_pipelines[combo_name] = (embedder, best_clf, preds)

    # Pick best overall combination by macro F1
    best_combo = max(all_metrics, key=lambda k: all_metrics[k]["f1_score"])
    best_embedder, best_clf, best_preds = fitted_pipelines[best_combo]
    print(f"\n*** Best pipeline: {best_combo} "
          f"(F1={all_metrics[best_combo]['f1_score']:.4f}) ***")

    # Plots
    labels_sorted = sorted(np.unique(y))
    plot_confusion(best_combo, y_test, best_preds, labels_sorted,
                    RESULTS_DIR / "confusion_matrix_best_model.png")
    plot_model_comparison(all_metrics, RESULTS_DIR / "model_comparison.png")
    if isinstance(best_embedder, EigenfaceEmbedder):
        plot_eigenfaces(best_embedder, RESULTS_DIR / "eigenfaces.png")
    else:
        # still plot eigenfaces for documentation even if LBPH wins overall
        eig_for_plot = embedders["Eigenfaces (PCA)"]
        plot_eigenfaces(eig_for_plot, RESULTS_DIR / "eigenfaces.png")

    # Persist best pipeline
    joblib.dump(best_embedder, MODELS_DIR / "embedder.joblib")
    joblib.dump(best_clf, MODELS_DIR / "classifier.joblib")
    label_map = roster.set_index("student_id")[["name", "roll_no"]].to_dict(orient="index")
    with open(MODELS_DIR / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)
    with open(MODELS_DIR / "pipeline_info.json", "w") as f:
        json.dump({"embedding": best_combo.split(" + ")[0], "classifier": best_combo.split(" + ")[1]}, f)
    print(f"\nSaved best pipeline ({best_combo}) to {MODELS_DIR}")

    # Save full metrics summary
    summary = {
        "best_pipeline": best_combo,
        "n_students": len(labels_sorted),
        "train_size": len(X_train),
        "train_size_after_augmentation": len(X_train_aug),
        "test_size": len(X_test),
        "metrics": all_metrics,
        "training_time_seconds": round(time.time() - t0, 2),
    }
    with open(RESULTS_DIR / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved: {RESULTS_DIR / 'metrics_summary.json'}")

    print(f"\nTotal pipeline time: {time.time()-t0:.1f}s")
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
