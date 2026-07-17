"""
preprocessing.py
-----------------
Text preprocessing utilities for the Spam Email/SMS Classifier project.

Pipeline implemented (per project spec):
    1. Lowercasing
    2. Punctuation / special character removal
    3. Tokenization
    4. Stopword removal
    5. Lemmatization
Each step is documented so the methodology can be reproduced or audited.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ---------------------------------------------------------------------------
# One-time NLTK resource download. Wrapped in try/except so re-runs are fast
# and the module works offline once resources are cached.
# ---------------------------------------------------------------------------
_REQUIRED_NLTK_RESOURCES = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
]

for path, pkg in _REQUIRED_NLTK_RESOURCES:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg, quiet=True)

_STOPWORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Lowercase the text and strip URLs, email addresses, punctuation and
    digits, collapsing extra whitespace. Digits are removed because raw
    numbers (phone numbers, prize amounts) don't generalize as features,
    but their *presence* is captured separately as an engineered feature
    (see `engineer_features`).
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"\S+@\S+", " ", text)                    # emails
    text = re.sub(r"[^a-z\s]", " ", text)                   # punctuation & digits
    text = re.sub(r"\s+", " ", text).strip()                # collapse whitespace
    return text


def tokenize_and_lemmatize(text: str) -> list:
    """Tokenize, drop stopwords/short tokens, and lemmatize what remains."""
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in _STOPWORDS and len(t) > 2]
    tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]
    return tokens


def preprocess(text: str) -> str:
    """Full pipeline: clean -> tokenize -> remove stopwords -> lemmatize -> rejoin."""
    cleaned = clean_text(text)
    tokens = tokenize_and_lemmatize(cleaned)
    return " ".join(tokens)


def engineer_features(raw_text: str) -> dict:
    """
    A handful of lightweight hand-crafted features that are known to be
    strong spam signals but get washed out by pure bag-of-words models:
      - message length (characters)
      - count of digits (phone numbers / amounts / codes)
      - count of uppercase words ("WIN", "FREE")
      - presence of currency symbols
      - exclamation mark count
    These are concatenated with the TF-IDF matrix at training time.
    """
    return {
        "char_count": len(raw_text),
        "digit_count": sum(c.isdigit() for c in raw_text),
        "upper_word_count": sum(1 for w in raw_text.split() if w.isupper() and len(w) > 1),
        "has_currency": int(bool(re.search(r"[$£€]|\brs\b", raw_text.lower()))),
        "exclamation_count": raw_text.count("!"),
    }


if __name__ == "__main__":
    sample = "WIN a FREE prize!!! Call 08000930705 now www.claim-prize.com"
    print("Raw       :", sample)
    print("Cleaned   :", clean_text(sample))
    print("Processed :", preprocess(sample))
    print("Features  :", engineer_features(sample))
