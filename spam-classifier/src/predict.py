"""
predict.py
----------
Load the trained model + vectorizer and classify new, unseen messages.
Run directly for a demo, or `from predict import predict_message` to use
it elsewhere (e.g. the Flask API).
"""

from pathlib import Path

import joblib

from preprocessing import preprocess

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

_model = None
_vectorizer = None


def _load_artifacts():
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        _model = joblib.load(MODELS_DIR / "spam_classifier.joblib")
        _vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    return _model, _vectorizer


def predict_message(message: str) -> dict:
    """Return label + confidence for a single raw message string."""
    model, vectorizer = _load_artifacts()
    cleaned = preprocess(message)
    X = vectorizer.transform([cleaned])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None

    result = {
        "message": message,
        "cleaned_message": cleaned,
        "prediction": "spam" if pred == 1 else "ham",
    }
    if proba is not None:
        result["confidence"] = round(float(proba[pred]), 4)
        result["spam_probability"] = round(float(proba[1]), 4)
    return result


DEMO_SAMPLES = [
    "Congratulations! You've WON a $1000 Walmart gift card. Click here to claim now!!!",
    "Hey, are we still on for lunch tomorrow at 1pm?",
    "URGENT: Your account has been suspended. Verify your details immediately at http://bit.ly/verify123",
    "Can you send me the notes from today's ML class?",
    "FREE entry into our $250 weekly competition just text WIN to 80086 NOW",
    "Mom, I'll be home by 8, don't wait for dinner",
    "You have been selected for a cash prize of Rs. 50,000! Call 09876543210 to claim.",
    "Reminder: your dentist appointment is on Friday at 10am.",
]


if __name__ == "__main__":
    print("=" * 70)
    print("TESTING CLASSIFIER ON NEW / UNSEEN SAMPLES")
    print("=" * 70)
    for msg in DEMO_SAMPLES:
        result = predict_message(msg)
        tag = "🚫 SPAM" if result["prediction"] == "spam" else "✅ HAM "
        print(f"\n{tag}  (confidence: {result['confidence']*100:.1f}%)")
        print(f"  Message: {result['message']}")
