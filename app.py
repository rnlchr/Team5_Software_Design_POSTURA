"""
app.py  —  Postura Flask Backend
=================================
Serves the existing HTML pages and exposes /analyze endpoint
for posture classification using YOLOv8 + trained classifiers.

Folder structure expected:
  postura/
  ├── app.py                  ← this file
  ├── models/
  │   ├── yolov8m/
  │   │   ├── posture_classifier.pkl
  │   │   ├── scaler.pkl
  │   │   ├── label_encoder.pkl
  │   │   └── feature_selector.pkl
  │   └── yolov8l/
  │       └── (same structure)
  ├── FIGMA/                  ← your existing HTML/CSS/JS files
  │   ├── index.html
  │   ├── home.html
  │   ├── results.html
  │   └── ...
  └── yolov8m-pose.pt
      yolov8l-pose.pt
"""

import os
import cv2
import numpy as np
import joblib
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
import torch

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
STATIC_DIR  = os.path.join(BASE_DIR, "FIGMA")   # your HTML files
DEVICE      = 0 if torch.cuda.is_available() else "cpu"
CONF_THRESH = 0.2

# Which variant to use for the website (m = faster, l = slightly more accurate)
# Change to "yolov8l" if you want the large model
ACTIVE_VARIANT = "yolov8m"
# ─────────────────────────────────────────────

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")
CORS(app)  # allows the frontend JS to call the API


# ── Recommendations ───────────────────────────────────────────────────
RECOMMENDATIONS = {
    "good": [
        "Great posture! Keep it up. Remember to take a short break every 30 minutes.",
        "Excellent posture! Your spine is well-aligned. Stay hydrated and keep moving.",
        "Your posture looks great! Consider a standing desk break every hour.",
    ],
    "bad": [
        "Try sitting up straight with your back against the chair. Your ears should be aligned over your shoulders.",
        "Consider adjusting your monitor height so you're not leaning forward. Your screen should be at eye level.",
        "Try the 20-20-20 rule: every 20 minutes, look at something 20 feet away for 20 seconds, and reset your posture.",
        "Place your feet flat on the floor and keep your knees at a 90-degree angle to improve your posture.",
    ],
    "error": "We couldn't fully analyze your posture. Please try a side-profile photo with your full upper body visible.",
}


# ── Feature Engineering (must match train_posture.py) ────────────────
def angle_between(p1, vertex, p2):
    v1 = np.array(p1) - np.array(vertex)
    v2 = np.array(p2) - np.array(vertex)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))))

def angle_from_vertical(p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return float(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6)))

def signed_ratio(a, b, denom):
    return float((a - b) / denom)

def extract_features(keypoints):
    kp, confs = keypoints, keypoints[:, 2]
    if any(confs[i] < CONF_THRESH for i in [5, 6, 11, 12]):
        return None

    def get(i):
        return kp[i, :2] if confs[i] >= CONF_THRESH else None

    nose  = get(0)
    le_   = get(1);  re_  = get(2)
    lear  = get(3);  rear = get(4)
    ls    = get(5);  rs   = get(6)
    lelb  = get(7);  relb = get(8)
    lwr   = get(9);  rwr  = get(10)
    lh    = get(11); rh   = get(12)

    mid_shoulder = (ls + rs) / 2
    mid_hip      = (lh + rh) / 2
    torso_h      = np.linalg.norm(mid_shoulder - mid_hip) + 1e-6
    shoulder_w   = np.linalg.norm(ls - rs) + 1e-6

    feats = []
    upper_idx = {
        0: "nose", 1: "l_eye", 2: "r_eye", 3: "l_ear", 4: "r_ear",
        5: "l_shoulder", 6: "r_shoulder", 7: "l_elbow", 8: "r_elbow",
        9: "l_wrist", 10: "r_wrist", 11: "l_hip", 12: "r_hip",
    }
    for i in upper_idx:
        if confs[i] >= CONF_THRESH:
            rel = (kp[i, :2] - mid_hip) / torso_h
            feats.extend([rel[0], rel[1]])
        else:
            feats.extend([0.0, 0.0])

    feats.append(angle_from_vertical(mid_hip, mid_shoulder))
    feats.append(signed_ratio(mid_shoulder[0], mid_hip[0], torso_h))

    if nose is not None:
        feats.append(angle_from_vertical(mid_shoulder, nose))
        feats.append(signed_ratio(nose[0], mid_shoulder[0], torso_h))
        feats.append(signed_ratio(nose[1], mid_shoulder[1], torso_h))
    else:
        feats.extend([0.0, 0.0, 0.0])

    if lear is not None and rear is not None:
        mid_ear = (lear + rear) / 2
        feats.append(angle_from_vertical(mid_shoulder, mid_ear))
        feats.append(signed_ratio(lear[1], rear[1], torso_h))
    else:
        feats.extend([0.0, 0.0])

    feats.append(signed_ratio(ls[1], rs[1], torso_h))
    feats.append(signed_ratio(ls[0], rs[0], shoulder_w))
    feats.append(signed_ratio(lh[1], rh[1], torso_h))

    feats.append(angle_between(ls, lelb, lwr) if lelb is not None and lwr is not None else 0.0)
    feats.append(angle_between(rs, relb, rwr) if relb is not None and rwr is not None else 0.0)
    feats.append(angle_from_vertical(ls, lelb) if lelb is not None else 0.0)
    feats.append(angle_from_vertical(rs, relb) if relb is not None else 0.0)
    feats.append(signed_ratio(lwr[1], ls[1], torso_h) if lwr is not None else 0.0)
    feats.append(signed_ratio(rwr[1], rs[1], torso_h) if rwr is not None else 0.0)

    sym_pairs = [(5, 6), (7, 8), (9, 10), (11, 12)]
    sym_scores = []
    for l_idx, r_idx in sym_pairs:
        if confs[l_idx] >= CONF_THRESH and confs[r_idx] >= CONF_THRESH:
            sym_scores.append(abs(kp[l_idx, 1] - kp[r_idx, 1]) / torso_h)
    feats.append(float(np.mean(sym_scores)) if sym_scores else 0.0)

    return np.array(feats, dtype=np.float32)


# ── Load Models ───────────────────────────────────────────────────────
def load_models(variant):
    d = os.path.join(MODELS_DIR, variant)
    print(f"Loading {variant} models from {d}...")
    return {
        "yolo":     YOLO(f"yolov8{variant[-1]}-pose.pt"),
        "clf":      joblib.load(os.path.join(d, "posture_classifier.pkl")),
        "scaler":   joblib.load(os.path.join(d, "scaler.pkl")),
        "le":       joblib.load(os.path.join(d, "label_encoder.pkl")),
        "selector": joblib.load(os.path.join(d, "feature_selector.pkl")),
    }

print(f"Device: {'GPU — ' + torch.cuda.get_device_name(0) if DEVICE == 0 else 'CPU'}")
MODEL = load_models(ACTIVE_VARIANT)
print(f"✅ Model ready ({ACTIVE_VARIANT})\n")


# ── Core Inference ────────────────────────────────────────────────────
def analyze_image(img_bytes):
    """
    img_bytes : raw image bytes from request
    Returns   : dict with label, confidence, recommendation, date
    """
    # Decode image
    nparr = np.frombuffer(img_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Could not decode image. Please upload a valid JPG or PNG."}

    # Run YOLO pose
    results = MODEL["yolo"](img, device=DEVICE, verbose=False)
    if not results or results[0].keypoints is None or results[0].keypoints.data.shape[0] == 0:
        return {"error": "No person detected. Please ensure your full upper body is visible."}

    kps_data = results[0].keypoints.data

    # Pick largest bounding box = most prominent person
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 1:
        xyxy  = boxes.xyxy.cpu().numpy()
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        best_idx = int(np.argmax(areas))
    else:
        best_idx = 0

    kps_np = kps_data[best_idx].cpu().numpy()
    feat   = extract_features(kps_np)

    if feat is None:
        return {
            "error": "Could not detect enough keypoints. Try a clearer side-profile photo with your head, shoulders, and hips visible."
        }

    # Classify
    feat_scaled   = MODEL["scaler"].transform(feat.reshape(1, -1))
    feat_selected = MODEL["selector"].transform(feat_scaled)
    pred_enc      = MODEL["clf"].predict(feat_selected)[0]
    proba         = MODEL["clf"].predict_proba(feat_selected)[0]
    label         = MODEL["le"].inverse_transform([pred_enc])[0]
    confidence    = float(proba[pred_enc])

    # Pick a recommendation
    recs = RECOMMENDATIONS.get(label, RECOMMENDATIONS["error"])
    recommendation = recs[hash(str(img.shape)) % len(recs)]  # consistent per image

    return {
        "label":          label,
        "confidence":     round(confidence, 4),
        "recommendation": recommendation,
        "date":           datetime.now().strftime("%Y-%m-%d %H:%M"),
        "error":          None,
    }


# ── Routes ────────────────────────────────────────────────────────────

# Serve HTML pages
@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)

# Main analysis endpoint — called by script.js
@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file      = request.files["image"]
    img_bytes = file.read()

    if not img_bytes:
        return jsonify({"error": "Empty file received."}), 400

    result = analyze_image(img_bytes)

    if result.get("error"):
        return jsonify({"error": result["error"]}), 422

    return jsonify(result), 200


# ── Health check (useful for debugging) ──────────────────────────────
@app.route("/health")
def health():
    return jsonify({
        "status":  "ok",
        "device":  torch.cuda.get_device_name(0) if DEVICE == 0 else "cpu",
        "variant": ACTIVE_VARIANT,
        "model":   type(MODEL["clf"]).__name__,
    })


if __name__ == "__main__":
    print("=" * 50)
    print("  Postura Flask Server")
    print("=" * 50)
    print(f"  Open: http://127.0.0.1:5000")
    print(f"  Model: {ACTIVE_VARIANT}")
    print("=" * 50)
    app.run(debug=True, host="127.0.0.1", port=5000)