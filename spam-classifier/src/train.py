"""
train.py
--------
End-to-end training pipeline for the Spam Email/SMS Classifier.

Steps (matching the project's 10-point guidance):
  1. Load labeled dataset
  2. Preprocess text (see preprocessing.py)
  3. Vectorize with TF-IDF (CountVectorizer also benchmarked for comparison)
  4. Train/test split (stratified, 80/20)
  5. Train Naive Bayes, Logistic Regression, and SVM
  6. Tune hyperparameters via GridSearchCV
  7. Evaluate with accuracy, precision, recall, F1, ROC-AUC, confusion matrix
  8. Persist the best model + vectorizer to disk for the API to serve
  9. Save all metrics/plots to results/ for documentation
"""

import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from preprocessing import preprocess

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "spam_dataset.csv"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["message", "label"]).drop_duplicates(subset=["message"])
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def build_features(df: pd.DataFrame):
    print("Preprocessing text (clean -> tokenize -> stopword removal -> lemmatize)...")
    df["clean_message"] = df["message"].apply(preprocess)
    return df


def compare_vectorizers(X_train, X_test, y_train, y_test):
    """Quick benchmark: TF-IDF vs CountVectorizer with a default MultinomialNB."""
    results = {}
    for name, vec in [
        ("TF-IDF", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("CountVectorizer", CountVectorizer(max_features=5000, ngram_range=(1, 2))),
    ]:
        Xtr = vec.fit_transform(X_train)
        Xte = vec.transform(X_test)
        clf = MultinomialNB()
        clf.fit(Xtr, y_train)
        preds = clf.predict(Xte)
        results[name] = {
            "accuracy": accuracy_score(y_test, preds),
            "f1": f1_score(y_test, preds),
        }
    print("Vectorizer comparison (MultinomialNB baseline):")
    for k, v in results.items():
        print(f"  {k:16s} accuracy={v['accuracy']:.4f}  f1={v['f1']:.4f}")
    return results


def get_models_and_grids():
    """Model + hyperparameter grid definitions for GridSearchCV."""
    return {
        "Naive Bayes": (
            MultinomialNB(),
            {"alpha": [0.1, 0.5, 1.0, 2.0]},
        ),
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            {"C": [0.1, 1, 5, 10], "class_weight": [None, "balanced"]},
        ),
        "SVM": (
            SVC(probability=True, random_state=RANDOM_STATE),
            {"C": [0.5, 1, 5], "kernel": ["linear", "rbf"]},
        ),
    }


def evaluate_model(name, model, X_test, y_test):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1_score": f1_score(y_test, preds),
    }
    if proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, proba)

    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.4f}")
    print(classification_report(y_test, preds, target_names=["ham", "spam"]))

    return metrics, preds, proba


def plot_confusion_matrices(fitted_models, X_test, y_test):
    fig, axes = plt.subplots(1, len(fitted_models), figsize=(5 * len(fitted_models), 4))
    if len(fitted_models) == 1:
        axes = [axes]
    for ax, (name, model) in zip(axes, fitted_models.items()):
        cm = confusion_matrix(y_test, model.predict(X_test))
        ConfusionMatrixDisplay(cm, display_labels=["ham", "spam"]).plot(
            ax=ax, colorbar=False, cmap="Blues"
        )
        ax.set_title(name)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrices.png", dpi=150)
    plt.close()
    print(f"Saved: {RESULTS_DIR / 'confusion_matrices.png'}")


def plot_roc_curves(fitted_models, X_test, y_test):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in fitted_models.items():
        if hasattr(model, "predict_proba"):
            RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("ROC Curves — Spam Classifier Models")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "roc_curves.png", dpi=150)
    plt.close()
    print(f"Saved: {RESULTS_DIR / 'roc_curves.png'}")


def plot_metric_comparison(all_metrics: dict):
    df = pd.DataFrame(all_metrics).T[["accuracy", "precision", "recall", "f1_score"]]
    ax = df.plot(kind="bar", figsize=(9, 5), rot=0)
    ax.set_title("Model Comparison — Spam Classifier")
    ax.set_ylabel("Score")
    ax.set_ylim(0.8, 1.0)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: {RESULTS_DIR / 'model_comparison.png'}")


def plot_class_distribution(df: pd.DataFrame):
    plt.figure(figsize=(5, 4))
    sns.countplot(x="label", data=df, hue="label", palette=["#2ecc71", "#e74c3c"], legend=False)
    plt.title("Class Distribution — Ham vs Spam")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "class_distribution.png", dpi=150)
    plt.close()
    print(f"Saved: {RESULTS_DIR / 'class_distribution.png'}")


def main():
    t0 = time.time()
    print("=" * 70)
    print("SPAM EMAIL/SMS CLASSIFIER — TRAINING PIPELINE")
    print("=" * 70)

    # 1. Load
    df = load_data()
    print(f"\nDataset loaded: {len(df)} messages "
          f"({(df.label == 'ham').sum()} ham / {(df.label == 'spam').sum()} spam)")
    plot_class_distribution(df)

    # 2. Preprocess
    df = build_features(df)

    # 3+4. Split (on cleaned text; stratified to preserve class ratio)
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["clean_message"], df["label_num"],
        test_size=0.2, random_state=RANDOM_STATE, stratify=df["label_num"]
    )
    print(f"\nTrain size: {len(X_train_text)}  |  Test size: {len(X_test_text)}")

    compare_vectorizers(X_train_text, X_test_text, y_train, y_test)

    # Final vectorizer choice: TF-IDF with uni+bi-grams (best trade-off, standard for text spam detection)
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # 5+6. Train each model with grid search
    fitted_models = {}
    all_metrics = {}
    best_name, best_f1, best_model = None, -1, None

    for name, (model, grid) in get_models_and_grids().items():
        print(f"\nTuning {name} via GridSearchCV (5-fold)...")
        gs = GridSearchCV(model, grid, scoring="f1", cv=5, n_jobs=-1)
        gs.fit(X_train, y_train)
        print(f"  Best params: {gs.best_params_}")
        best_est = gs.best_estimator_
        fitted_models[name] = best_est

        metrics, preds, proba = evaluate_model(name, best_est, X_test, y_test)
        metrics["best_params"] = gs.best_params_
        all_metrics[name] = metrics

        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_name = name
            best_model = best_est

    # 7. Evaluation artifacts
    plot_confusion_matrices(fitted_models, X_test, y_test)
    plot_roc_curves(fitted_models, X_test, y_test)
    plot_metric_comparison(all_metrics)

    # 8. Persist best model
    joblib.dump(best_model, MODELS_DIR / "spam_classifier.joblib")
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.joblib")
    print(f"\nBest model: {best_name} (F1={best_f1:.4f}) -> saved to {MODELS_DIR}")

    # 9. Save metrics summary
    summary = {
        "best_model": best_name,
        "dataset_size": len(df),
        "train_size": len(X_train_text),
        "test_size": len(X_test_text),
        "metrics": all_metrics,
        "training_time_seconds": round(time.time() - t0, 2),
    }
    with open(RESULTS_DIR / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved: {RESULTS_DIR / 'metrics_summary.json'}")

    print(f"\nTotal pipeline time: {time.time() - t0:.1f}s")
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
