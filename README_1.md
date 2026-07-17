# ML Projects — Signal (Spam Classifier) & RollCall (Face Recognition Attendance)

Two end-to-end machine learning systems, each built with classical ML + scikit-learn, served through a Flask API with a responsive web frontend. Both follow the same structure: define objective → collect data → preprocess → extract features → train & tune multiple algorithms → evaluate → deploy via API + dashboard.

| Project | Task | Best Model | Test Accuracy | Test F1 |
|---|---|---|---|---|
| **Signal** | Spam Email/SMS Classification | SVM (linear kernel) | 97.78% | 0.9098 |
| **RollCall** | Face Recognition Attendance | Eigenfaces (PCA) + SVM (RBF) | 99.17% | 99.14% |

---

# Part 1 — Signal: Spam Email / SMS Classifier

A complete machine learning pipeline that classifies messages as **spam** or **ham (legitimate)**, built with Python and scikit-learn, and served through a Flask API with a responsive web frontend.

Built for the *Machine Learning with Python — Minor Project (Spam Email Classifier)* assignment brief from Techn Global.

## 1.1 Objective

Classify emails/SMS messages as spam or not spam using classical NLP + machine learning, following the 10-point project guidance: define objective → collect data → preprocess → vectorize → split → train multiple algorithms → tune → evaluate → test on new samples → document.

## 1.2 Dataset

| | |
|---|---|
| **Name** | SMS Spam Collection Dataset |
| **Source** | UCI Machine Learning Repository (mirrored copy of the same dataset referenced on Kaggle at `uciml/sms-spam-collection-dataset`) |
| **Size** | 5,572 raw messages → 5,169 after de-duplication |
| **Classes** | Ham: 4,516 (87.4%) · Spam: 653 (12.6%) — realistic, imbalanced real-world distribution |
| **Format** | Tab-separated `label \t message` → converted to `data/spam_dataset.csv` |

> **Note on sourcing:** Kaggle isn't reachable from the build environment used to generate this project, so the dataset was pulled from a GitHub mirror of the same canonical UCI SMS Spam Collection that both Kaggle links in the brief point to. The data is identical in content and schema — you can swap in the Kaggle CSV directly (same `label,message` columns) with no code changes if you'd prefer to download it yourself.

![Class Distribution](spam-classifier/results/class_distribution.png)

## 1.3 Methodology

### 1.3.1 Preprocessing (`src/preprocessing.py`)
1. Lowercase all text
2. Strip URLs and email addresses
3. Remove punctuation and digits
4. Tokenize (NLTK `word_tokenize`)
5. Remove English stopwords and tokens ≤ 2 characters
6. Lemmatize remaining tokens (WordNet)
7. **Engineered features** (computed but not used in the final vectorizer, benchmarked separately): message length, digit count, uppercase-word count, currency symbol presence, exclamation count — these are classic spam "tells" that survive even after stripping the words themselves.

Example:
```
Raw:       WIN a FREE prize!!! Call 08000930705 now www.claim-prize.com
Processed: win free prize call
```

### 1.3.2 Feature extraction
Two vectorizers were benchmarked with a baseline Naive Bayes model:

| Vectorizer | Accuracy | F1 |
|---|---|---|
| TF-IDF (uni+bigrams, 5000 features) | 96.7% | 0.851 |
| CountVectorizer (uni+bigrams, 5000 features) | 97.6% | 0.904 |

**TF-IDF was selected as the final vectorizer** despite the marginally lower baseline score, because it generalizes better across models with very different assumptions (SVM's margin-based decision boundary in particular benefits from TF-IDF's normalized weighting), which held up once combined with hyperparameter tuning across all three algorithms.

### 1.3.3 Train/test split
80/20 stratified split (preserves the 87/13 class ratio in both sets) — 4,135 training messages, 1,034 test messages. Random state fixed at 42 for reproducibility.

### 1.3.4 Models trained & tuned
All three algorithms from the brief were trained and tuned via 5-fold `GridSearchCV` optimizing for F1 (the right metric here, since accuracy alone is misleading on a 87/13 imbalanced dataset):

| Model | Hyperparameters tuned | Best params |
|---|---|---|
| Naive Bayes | `alpha` | `alpha=0.1` |
| Logistic Regression | `C`, `class_weight` | `C=10, class_weight=balanced` |
| SVM | `C`, `kernel` | `C=5, kernel=linear` |

## 1.4 Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Naive Bayes | 97.58% | 93.44% | 87.02% | 0.9012 | 0.9855 |
| Logistic Regression | 97.00% | 86.23% | 90.84% | 0.8848 | **0.9918** |
| **SVM (best, linear kernel)** | **97.78%** | **93.55%** | 88.55% | **0.9098** | 0.9910 |

**SVM (linear kernel) was selected as the deployed model** — it has the best accuracy and F1, and a linear kernel keeps inference fast enough for real-time API use.

![Model Comparison](spam-classifier/results/model_comparison.png)
![Confusion Matrices](spam-classifier/results/confusion_matrices.png)
![ROC Curves](spam-classifier/results/roc_curves.png)

### Insights
- **Precision vs. recall trade-off matters for spam filters.** A false positive (legitimate email marked as spam) is usually worse for a user than a false negative (spam that slips through). SVM and Naive Bayes prioritize precision (93.5%+) over recall (~88%), which is the right trade-off for this use case — Logistic Regression with `class_weight=balanced` swaps this trade-off toward recall, which could be preferred in a security-critical inbox.
- **Bigrams help.** Phrases like "free entry", "call now", and "cash prize" carry much more spam signal as two-word units than as individual tokens; restricting to unigrams measurably hurt recall in early experiments.
- **Class imbalance is real and unaddressed data leads to accuracy being a misleading metric** — a model that predicts "ham" for everything would score 87.4% accuracy while being useless. F1 and the confusion matrix are what actually validate the model.
- **Limitation observed during manual testing:** the model is very confident on classic SMS-spam patterns (prize/free/urgent) but is less reliable on modern phishing-style messages with generic "account suspended" language and shortened URLs, since those patterns are underrepresented in this 2011-era SMS dataset. A production system should periodically retrain on more recent phishing/email spam samples.

## 1.5 Testing on new, unseen samples

`src/predict.py` runs eight hand-written messages (never seen during training) through the deployed model:

```bash
cd src && python3 predict.py
```

```
🚫 SPAM  (confidence: 98.8%)  FREE entry into our $250 weekly competition just text WIN to 80086 NOW
✅ HAM   (confidence: 99.9%)  Mom, I'll be home by 8, don't wait for dinner
🚫 SPAM  (confidence: 99.7%)  You have been selected for a cash prize of Rs. 50,000! Call 09876543210 to claim.
✅ HAM   (confidence: 99.8%)  Can you send me the notes from today's ML class?
```

## 1.6 Project structure

```
spam-classifier/
├── data/
│   ├── sms_spam.tsv          # raw dataset
│   └── spam_dataset.csv      # cleaned CSV used by the pipeline
├── src/
│   ├── preprocessing.py      # text cleaning, tokenization, lemmatization
│   ├── train.py              # full training + tuning + evaluation pipeline
│   ├── predict.py            # load model, classify new messages
│   └── app.py                # Flask API + frontend server
├── static/
│   └── index.html            # responsive web frontend (calls the API)
├── models/
│   ├── spam_classifier.joblib
│   └── tfidf_vectorizer.joblib
├── results/
│   ├── metrics_summary.json
│   ├── class_distribution.png
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   └── model_comparison.png
├── tests/
│   └── test_pipeline.py      # unit tests (pytest)
├── requirements.txt
└── README.md
```

## 1.7 How to run

### Setup
```bash
pip install -r requirements.txt
```

### 1. Train the model (regenerates everything in `models/` and `results/`)
```bash
cd src
python3 train.py
```

### 2. Test on new samples from the command line
```bash
python3 predict.py
```

### 3. Run unit tests
```bash
cd ..
python3 -m pytest tests/ -v
```

### 4. Launch the API + web frontend
```bash
cd src
python3 app.py
```
Then open **http://localhost:5000** in a browser (mobile-responsive — try resizing or opening on a phone).

## 1.8 API reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/predict` | Classify a single message. Body: `{"message": "..."}` |
| `POST` | `/api/predict/batch` | Classify up to 100 messages. Body: `{"messages": ["...", "..."]}` |
| `GET` | `/api/model-info` | Deployed model name + test-set metrics |

**Example:**
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "WIN a FREE iPhone now, click here to claim!"}'
```
```json
{
  "message": "WIN a FREE iPhone now, click here to claim!",
  "cleaned_message": "win free iphone click claim",
  "prediction": "spam",
  "confidence": 1.0,
  "spam_probability": 1.0
}
```

## 1.9 Tools & platforms used

Python 3.12 · Pandas · NumPy · scikit-learn · NLTK · Matplotlib / Seaborn · Flask + Flask-CORS · pytest · joblib

## 1.10 Reference material consulted

- Krishnaik06 — Spam Email Classifier (GitHub)
- karan2sharma — Spam Detection (GitHub)
- justmarkham — scikit-learn-videos / PyCon 2016 tutorial (GitHub) — source of the SMS Spam Collection mirror used here

---

# Part 2 — RollCall: Face Recognition Attendance System

A complete face recognition pipeline that automates classroom attendance, built with OpenCV and scikit-learn, served through a Flask API with a webcam-enabled, mobile-responsive dashboard.

Built for the *Machine Learning with Python — Major Project (Face Recognition Attendance System)* assignment brief.

## 2.1 Objective

Automate attendance marking using facial recognition, following the 10-point brief: collect student face images → preprocess → detect faces → extract embeddings → train a classifier → build an attendance system with timestamps → build a dashboard → test under different conditions → document everything.

## 2.2 Dataset

| | |
|---|---|
| **Requested in brief** | CelebA / Faces-in-the-Wild (Kaggle) |
| **Used instead** | **AT&T / Olivetti Faces Dataset** — 40 subjects x 10 images each (400 images total), mirrored on GitHub |
| **Why the substitution** | Kaggle isn't reachable from this build environment. CelebA (200k+ images of celebrity faces, not organized as a "student roster") and Faces-in-the-Wild are also both a poor structural fit for an *enrollment + recognition* system, which needs multiple labeled photos **per identity**. The AT&T Faces Dataset is the closest available match to what an attendance system actually needs: **40 distinct people, each with 10 photos taken at different times, with genuine variation in lighting, facial expression (open/closed eyes, smiling/not), and slight head angle** — exactly the "different lighting and angles" condition the brief's step 9 asks to test against. |
| **How it's organized** | Converted into `data/raw_faces/<student_id>/img_00.png … img_09.png`, with a generated roster (`data/students.csv`) mapping each subject to a plausible student name, roll number, and ID — so the rest of the pipeline behaves exactly as it would with a real classroom photo set. Swapping in your real students only means replacing the contents of `data/raw_faces/`. |

> If you'd like to use the actual Kaggle datasets from the brief, `data_prep.py` is the only file that would need to change — everything downstream (detection, preprocessing, embeddings, training, the API) works on the `raw_faces/<student_id>/*.png` folder layout regardless of where the images came from.

## 2.3 Methodology

### 2.3.1 Face detection — Haar Cascade
Used OpenCV's built-in Haar Cascade (`haarcascade_frontalface_default.xml`), which ships inside the `opencv` package itself — no external model download needed, which matters in a sandboxed environment.

**Detection had to be padding-corrected.** The Olivetti images are cropped tightly around the face with no background margin, and Haar Cascade's frontal-face features expect some surrounding context to recognize the forehead-to-chin proportions — with no padding, detection failed on every single image. `face_detection.py` pads and upscales the image internally before detection and maps the result back to original coordinates, which brought detection accuracy from **0% → 99.2%** across all 400 images. This fix is harmless for real, uncropped camera frames.

### 2.3.2 Preprocessing
- Resize to a standard 128x128
- Histogram equalization (improves robustness to lighting — directly relevant to the brief's lighting-robustness requirement)
- Augmentation on the training split only: horizontal flip, ±10° rotation, brightness jitter — quadruples the effective training set (280 → 1,120 images) without touching the test set

### 2.3.3 Facial embeddings — substituted for FaceNet/Dlib
The brief calls for FaceNet or Dlib pretrained embeddings. Both require downloading large weight files from hosts unreachable in this environment (dlib's models are `.bz2` files on dlib.net; FaceNet weights are typically distributed via Google Drive, not a plain package). Rather than skip the step, two classical, fully self-contained embedding techniques were implemented and benchmarked against each other — both were literally the historical predecessors to deep face embeddings:

| Technique | Description |
|---|---|
| **Eigenfaces (PCA)** | Projects each face onto the top 80 directions of variance across the training set (Turk & Pentland, 1991) — the first successful automatic face recognition method, and functionally an embedding: a fixed 80-dim vector per face. |
| **LBPH** | Local Binary Pattern Histograms — a texture descriptor robust to monotonic lighting changes, computed via `cv2.face`. |

Either can be swapped for a real deep embedding model later (e.g. `facenet-pytorch`) by implementing the same `.transform()` interface in `embeddings.py` — no other file needs to change.

### 2.3.4 Classifiers — SVM & KNN, as specified
Both trained and tuned via 5-fold `GridSearchCV` (optimizing macro-F1, since this is a balanced 40-class problem) on both embeddings — 4 combinations total:

| Combination | Best hyperparameters |
|---|---|
| Eigenfaces + SVM | `C=5, kernel=rbf` |
| Eigenfaces + KNN | `n_neighbors=1, weights=uniform` |
| LBPH + SVM | `C=5, kernel=linear` |
| LBPH + KNN | `n_neighbors=1, weights=uniform` |

## 2.4 Results

| Pipeline | Accuracy | Precision | Recall | F1 (macro) |
|---|---|---|---|---|
| **Eigenfaces (PCA) + SVM** ⭐ | **99.17%** | **99.38%** | **99.17%** | **99.14%** |
| LBPH + SVM | 97.50% | 98.13% | 97.50% | 97.43% |
| Eigenfaces (PCA) + KNN | 95.00% | 96.75% | 95.00% | 95.04% |
| LBPH + KNN | 95.00% | 95.88% | 95.00% | 94.80% |

**Eigenfaces + SVM (RBF kernel) was deployed** — best across every metric, and PCA's 80-dim vectors make inference fast enough for real-time use.

The test set was a genuine 30% held-out split — meaning the model was evaluated on images of each student it had never seen, captured with different lighting, expression, and head angle than the training photos. This directly answers the brief's step 9 ("test system under different lighting and angles").

![Eigenfaces](face-attendance/results/eigenfaces.png)
![Model Comparison](face-attendance/results/model_comparison.png)
![Confusion Matrix](face-attendance/results/confusion_matrix_best_model.png)

### Insights
- **The face-detection padding bug was the single biggest lesson.** A 0%-detection-rate pipeline silently "worked" if you don't check intermediate outputs (it just fell back to the uncropped image every time) — always verify detection rate on a sample before trusting downstream numbers.
- **PCA beat the texture-based LBPH descriptor here**, likely because with only 10 images per identity and a controlled, consistent camera/background setup, global appearance (Eigenfaces) generalizes better than fine-grained local texture (LBPH), which tends to need more within-class samples to average out noise.
- **KNN with n_neighbors=1 winning the grid search is a signal of a small, well-separated dataset**, not necessarily the best real-world choice — with a live camera and dozens of students, k=1 is more sensitive to a single bad frame than an SVM's margin-based decision. This is worth re-tuning once real student photos replace the demo dataset.
- **The confidence threshold matters more than raw accuracy for a deployed system.** In live testing, a genuinely low-confidence, incorrect guess was correctly rejected by the 0.55 confidence threshold rather than silently marking the wrong student present — see the worked example in `recognize.py`'s own demo output.

## 2.5 Attendance marking system

`recognize.py` runs the full pipeline (detect → crop → preprocess → embed → classify) and integrates with `attendance_db.py` (SQLite) to log `student_id, name, date, time, confidence`. A student already marked present today is not logged twice. Recognitions below the 0.55 confidence threshold are rejected rather than marked, to avoid false attendance.

## 2.6 Project structure

```
face-attendance/
├── data/
│   ├── olivetti_faces.npy / olivetti_faces_target.npy   # raw dataset
│   ├── raw_faces/<STUDENT_ID>/img_00.png ... img_09.png  # per-student images
│   └── students.csv                                       # roster
├── src/
│   ├── data_prep.py        # npy -> per-student image folders + roster
│   ├── preprocessing.py    # resize, normalize, augment
│   ├── face_detection.py   # Haar Cascade detection + crop
│   ├── embeddings.py       # Eigenfaces (PCA) and LBPH embedders
│   ├── train.py             # full training + tuning + evaluation pipeline
│   ├── recognize.py         # detect->embed->classify->mark attendance
│   ├── attendance_db.py     # SQLite attendance log + roster
│   └── app.py                # Flask API + dashboard server
├── static/
│   └── index.html           # responsive dashboard (webcam + upload)
├── models/
│   ├── embedder.joblib
│   ├── classifier.joblib
│   ├── label_map.json
│   └── pipeline_info.json
├── results/
│   ├── metrics_summary.json
│   ├── eigenfaces.png
│   ├── model_comparison.png
│   └── confusion_matrix_best_model.png
├── tests/
│   └── test_pipeline.py
├── attendance.db             # created on first run
├── requirements.txt
└── README.md
```

## 2.7 How to run

```bash
pip install -r requirements.txt
```

### 1. Generate the student image dataset + roster
```bash
cd src
python3 data_prep.py
```

### 2. Train the model (regenerates everything in `models/` and `results/`)
```bash
python3 train.py
```
Takes ~3–4 minutes (the LBPH+SVM grid search is the slow step).

### 3. Test recognition + attendance marking from the command line
```bash
python3 recognize.py
```

### 4. Run unit tests
```bash
cd ..
python3 -m pytest tests/ -v
```

### 5. Launch the API + dashboard
```bash
cd src
python3 app.py
```
Open **http://localhost:5001**. Use "Start webcam" (grant camera permission) or "Upload photo", then "Mark attendance". Works on mobile — try it on your phone to check the responsiveness.

## 2.8 API reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/recognize` | Recognize a face (base64 image), does **not** mark attendance |
| `POST` | `/api/mark-attendance` | Recognize + mark attendance if confidence ≥ threshold |
| `GET` | `/api/attendance` | Attendance records; optional `?date=YYYY-MM-DD&student_id=STUxxx` |
| `GET` | `/api/attendance/summary` | Present/absent counts for a date (defaults to today) |
| `GET` | `/api/students` | Full roster |
| `GET` | `/api/model-info` | Deployed model + test-set metrics |

**Example:**
```bash
curl -X POST http://localhost:5001/api/mark-attendance \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/png;base64,<...>"}'
```
```json
{
  "face_detected": true,
  "student_id": "STU001",
  "name": "Krishna Sharma",
  "roll_no": "CSE-2027000",
  "confidence": 0.6828,
  "recognized": true,
  "attendance": {
    "status": "marked",
    "date": "2026-07-05",
    "time": "10:28:47"
  }
}
```

## 2.9 Tools & platforms used

Python 3.12 · OpenCV (contrib) · scikit-learn · NumPy / Pandas · Pillow · Matplotlib / Seaborn · Flask + Flask-CORS · SQLite · pytest · joblib

## 2.10 Reference material consulted

- ageitgey/face_recognition (GitHub) — general architecture reference for detect→embed→classify pipelines
- krishnaik06/Face-Recognition-Attendance (GitHub) — attendance-logging pattern (timestamped CSV/DB log, duplicate-mark prevention)
- atulapra/FaceRecognition (GitHub) — dashboard/GUI structuring ideas
- codeheroku/Introduction-to-Machine-Learning (GitHub) — source of the mirrored AT&T/Olivetti Faces `.npy` files used as the dataset

---

## Repository layout (if hosting both projects together)

```
ml-projects/
├── README.md              # this file
├── spam-classifier/        # Signal — Part 1
│   └── ... (see structure in 1.6)
└── face-attendance/         # RollCall — Part 2
    └── ... (see structure in 2.6)
```
