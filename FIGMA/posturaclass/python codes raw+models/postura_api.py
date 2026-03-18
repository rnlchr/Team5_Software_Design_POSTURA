from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
from ultralytics import YOLO
import joblib
import cv2
import numpy as np
import torch
import os

BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Paths to trained models (update if your .pkl files live elsewhere)
MODELS_DIR = r"C:\Users\Ron\Desktop\postura\models"
YOLO_WEIGHTS = os.path.join(BASE_DIR, "yolov8m-pose.pt")

DEVICE = 0 if torch.cuda.is_available() else "cpu"
CONF_THRESH = 0.3

pose_model = YOLO(YOLO_WEIGHTS)
classifier = joblib.load(os.path.join(MODELS_DIR, "posture_classifier.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
label_enc = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

app = Flask(__name__)


def angle_between(p1, vertex, p2):
    v1 = np.array(p1) - np.array(vertex)
    v2 = np.array(p2) - np.array(vertex)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(
        np.degrees(
            np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        )
    )


def angle_from_vertical(p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return float(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6)))


def extract_features(keypoints):
    kp, confs = keypoints, keypoints[:, 2]
    if any(confs[i] < CONF_THRESH for i in [0, 5, 6, 11, 12]):
        return None

    def get(i):
        return kp[i, :2] if confs[i] >= CONF_THRESH else None

    ls, rs = get(5), get(6)
    lh, rh = get(11), get(12)
    nose = get(0)

    mid_shoulder = (ls + rs) / 2
    mid_hip = (lh + rh) / 2
    torso_h = np.linalg.norm(mid_shoulder - mid_hip) + 1e-6

    norm_kps = []
    for i in range(13):
        if confs[i] >= CONF_THRESH:
            rel = (kp[i, :2] - mid_hip) / torso_h
            norm_kps.extend([rel[0], rel[1]])
        else:
            norm_kps.extend([0.0, 0.0])

    angles = [
        angle_from_vertical(mid_hip, mid_shoulder),
        angle_from_vertical(mid_shoulder, nose) if nose is not None else 0.0,
        float((nose[0] - mid_shoulder[0]) / torso_h)
        if nose is not None
        else 0.0,
        float((ls[1] - rs[1]) / torso_h),
        float((lh[1] - rh[1]) / torso_h),
    ]

    le_kp, lw = get(7), get(9)
    angles.append(
        angle_between(ls, le_kp, lw) if le_kp is not None and lw is not None else 0.0
    )
    re_kp, rw = get(8), get(10)
    angles.append(
        angle_between(rs, re_kp, rw) if re_kp is not None and rw is not None else 0.0
    )
    angles.append(float((mid_shoulder[0] - mid_hip[0]) / torso_h))

    return np.array(norm_kps + angles, dtype=np.float32)


def predict_posture(image_input):
    """
    image_input: file path (str) OR numpy array (BGR from cv2)
    Returns: {'label': 'good'/'bad', 'confidence': 0.92, 'error': None}
    """
    img = cv2.imread(image_input) if isinstance(image_input, str) else image_input
    if img is None:
        return {"label": None, "confidence": None, "error": "Could not read image"}

    results = pose_model(img, device=DEVICE, verbose=False)
    if (
        not results
        or results[0].keypoints is None
        or results[0].keypoints.data.shape[0] == 0
    ):
        return {"label": None, "confidence": None, "error": "No person detected"}

    kps_data = results[0].keypoints.data
    best_idx = kps_data[:, :, 2].mean(dim=1).argmax().item()
    kps_np = kps_data[best_idx].cpu().numpy()

    feat = extract_features(kps_np)
    if feat is None:
        return {"label": None, "confidence": None, "error": "Keypoints not visible enough"}

    feat_scaled = scaler.transform(feat.reshape(1, -1))
    pred_enc = classifier.predict(feat_scaled)[0]
    proba = classifier.predict_proba(feat_scaled)[0]
    label = label_enc.inverse_transform([pred_enc])[0]
    confidence = float(proba[pred_enc])

    return {"label": label, "confidence": confidence, "error": None}


@app.post("/analyze")
def analyze():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_DIR, filename)
    file.save(save_path)

    result = predict_posture(save_path)
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

