from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
from ultralytics import YOLO
import joblib
import cv2
import os

from predict_posture import predict_posture

BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

pose_model = YOLO(os.path.join(BASE_DIR, "yolov8m-pose.pt"))
classifier = joblib.load(os.path.join(BASE_DIR, "posture_classifier.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
label_enc = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))

app = Flask(__name__)


@app.post("/analyze")
def analyze():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_DIR, filename)
    file.save(save_path)

    img = cv2.imread(save_path)
    if img is None:
        return jsonify({"error": "Could not read image"}), 400

    result = predict_posture(img, pose_model, classifier, scaler, label_enc)
    if result.get("error"):
        return jsonify({"error": result["error"]}), 500

    label = result["label"]
    confidence = result["confidence"]

    if label == "good":
        recommendation = "Posture looks good. Maintain current habits."
    else:
        recommendation = "Poor posture detected. Follow the suggested exercises."

    return jsonify(
        {
            "label": label,
            "confidence": confidence,
            "recommendation": recommendation,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "filename": filename,
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

