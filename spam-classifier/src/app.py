"""
app.py
------
Flask API that serves the trained spam classifier.

Endpoints
---------
GET  /api/health          -> service health check
POST /api/predict         -> classify a single message
POST /api/predict/batch   -> classify a list of messages
GET  /api/model-info      -> metadata about the deployed model
GET  /                    -> serves the demo web frontend (static/index.html)
"""

import json
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from predict import predict_message

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"
RESULTS_DIR = ROOT / "results"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")
CORS(app)  # allow the frontend (and any external client) to call the API


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "spam-classifier-api"}), 200


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Field 'message' is required and cannot be empty."}), 400
    if len(message) > 5000:
        return jsonify({"error": "Message too long (max 5000 characters)."}), 400

    try:
        result = predict_message(message)
        return jsonify(result), 200
    except Exception as exc:  # keep the API resilient to unexpected input
        return jsonify({"error": f"Prediction failed: {str(exc)}"}), 500


@app.route("/api/predict/batch", methods=["POST"])
def predict_batch():
    data = request.get_json(silent=True) or {}
    messages = data.get("messages", [])

    if not isinstance(messages, list) or not messages:
        return jsonify({"error": "Field 'messages' must be a non-empty list of strings."}), 400
    if len(messages) > 100:
        return jsonify({"error": "Batch size limited to 100 messages per request."}), 400

    try:
        results = [predict_message(str(m)) for m in messages]
        return jsonify({"count": len(results), "results": results}), 200
    except Exception as exc:
        return jsonify({"error": f"Batch prediction failed: {str(exc)}"}), 500


@app.route("/api/model-info", methods=["GET"])
def model_info():
    summary_path = RESULTS_DIR / "metrics_summary.json"
    if not summary_path.exists():
        return jsonify({"error": "Model metrics not found. Run src/train.py first."}), 404
    with open(summary_path) as f:
        summary = json.load(f)
    return jsonify({
        "best_model": summary.get("best_model"),
        "dataset_size": summary.get("dataset_size"),
        "train_size": summary.get("train_size"),
        "test_size": summary.get("test_size"),
        "metrics": summary.get("metrics", {}).get(summary.get("best_model"), {}),
    }), 200


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed."}), 405


if __name__ == "__main__":
    import os
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=5000, debug=debug_mode, use_reloader=False)
