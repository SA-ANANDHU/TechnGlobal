"""
test_pipeline.py
-----------------
Basic unit tests for preprocessing and prediction.
Run with: python3 -m pytest tests/ -v   (from project root)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preprocessing import clean_text, engineer_features, preprocess
from predict import predict_message


def test_clean_text_lowercases():
    assert clean_text("HELLO World") == "hello world"


def test_clean_text_removes_urls():
    result = clean_text("Visit http://example.com now")
    assert "http" not in result
    assert "example" not in result


def test_clean_text_removes_punctuation():
    result = clean_text("Free!!! Win $$$ now...")
    assert "!" not in result
    assert "$" not in result


def test_preprocess_removes_stopwords():
    result = preprocess("this is a message about the weather")
    for stopword in ["this", "is", "a", "about", "the"]:
        assert stopword not in result.split()


def test_engineer_features_currency():
    feats = engineer_features("Win $500 now")
    assert feats["has_currency"] == 1


def test_engineer_features_digits():
    feats = engineer_features("Call 12345")
    assert feats["digit_count"] == 5


def test_predict_returns_expected_keys():
    result = predict_message("Hello, how are you today?")
    assert "prediction" in result
    assert result["prediction"] in ("ham", "spam")
    assert "confidence" in result


def test_predict_obvious_spam():
    result = predict_message("FREE FREE FREE WIN CASH PRIZE NOW call 08001234567 claim urgent")
    assert result["prediction"] == "spam"


def test_predict_obvious_ham():
    result = predict_message("Hi mom, just landed. Will call you when I get home.")
    assert result["prediction"] == "ham"


if __name__ == "__main__":
    import subprocess
    subprocess.run(["python3", "-m", "pytest", __file__, "-v"])
