"""
app.py  —  Postura Flask Backend (Render-ready)
================================================
- Detects if running on Render vs locally
- Downloads YOLO model automatically if not present
- Uses /data disk on Render for SQLite database
- Falls back to local paths when running locally
"""

import os
import cv2
import numpy as np
import joblib
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin,
    login_user, logout_user, login_required, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from ultralytics import YOLO
import torch

# ─────────────────────────────────────────────
# ENVIRONMENT DETECTION
# ─────────────────────────────────────────────
IS_RENDER   = os.environ.get("RENDER", False)
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = "/data" if IS_RENDER else BASE_DIR
MODELS_DIR  = os.path.join(BASE_DIR, "models")
STATIC_DIR  = os.path.join(BASE_DIR, "FIGMA")
DB_PATH     = os.path.join(DATA_DIR, "postura.db")
DEVICE      = "cpu"  # Render free tier has no GPU
CONF_THRESH = 0.2
ACTIVE_VARIANT = "yolov8m"

print(f"Running on: {'Render' if IS_RENDER else 'Local'}")
print(f"Device: {DEVICE}")
print(f"DB path: {DB_PATH}")

# ─────────────────────────────────────────────
# FLASK SETUP
# ─────────────────────────────────────────────
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")
app.config["SECRET_KEY"]                     = os.environ.get("SECRET_KEY", "postura-local-dev-key")
app.config["SQLALCHEMY_DATABASE_URI"]        = f"sqlite:///{DB_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

CORS(app, supports_credentials=True)
db            = SQLAlchemy(app)
login_manager = LoginManager(app)


# ═════════════════════════════════════════════
# DATABASE MODELS
# ═════════════════════════════════════════════
class User(UserMixin, db.Model):
    __tablename__ = "users"
    id         = db.Column(db.Integer, primary_key=True)
    username   = db.Column(db.String(80), unique=True, nullable=False)
    email      = db.Column(db.String(120), unique=True, nullable=False)
    password   = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    records    = db.relationship("Record", backref="user", lazy=True, cascade="all, delete-orphan")


class Record(db.Model):
    __tablename__  = "records"
    id             = db.Column(db.Integer, primary_key=True)
    user_id        = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    label          = db.Column(db.String(10), nullable=False)
    confidence     = db.Column(db.Float, nullable=False)
    recommendation = db.Column(db.Text, nullable=False)
    date           = db.Column(db.String(20), nullable=False)
    created_at     = db.Column(db.DateTime, default=datetime.utcnow)


# ═════════════════════════════════════════════
# RECOMMENDATIONS
# ═════════════════════════════════════════════
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
}


# ═════════════════════════════════════════════
# FEATURE ENGINEERING
# ═════════════════════════════════════════════
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
    nose = get(0)
    lear = get(3);  rear = get(4)
    ls   = get(5);  rs   = get(6)
    lelb = get(7);  relb = get(8)
    lwr  = get(9);  rwr  = get(10)
    lh   = get(11); rh   = get(12)
    mid_shoulder = (ls + rs) / 2
    mid_hip      = (lh + rh) / 2
    torso_h      = np.linalg.norm(mid_shoulder - mid_hip) + 1e-6
    shoulder_w   = np.linalg.norm(ls - rs) + 1e-6
    feats = []
    for i in range(13):
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
    sym_pairs = [(5,6),(7,8),(9,10),(11,12)]
    sym_scores = []
    for l_idx, r_idx in sym_pairs:
        if confs[l_idx] >= CONF_THRESH and confs[r_idx] >= CONF_THRESH:
            sym_scores.append(abs(kp[l_idx,1] - kp[r_idx,1]) / torso_h)
    feats.append(float(np.mean(sym_scores)) if sym_scores else 0.0)
    return np.array(feats, dtype=np.float32)


# ═════════════════════════════════════════════
# LOAD POSTURE MODELS
# ═════════════════════════════════════════════
def load_models(variant):
    d         = os.path.join(MODELS_DIR, variant)
    yolo_name = f"yolov8{variant[-1]}-pose.pt"

    # YOLO downloads automatically if not present
    print(f"Loading YOLO ({yolo_name})...")
    yolo_model = YOLO(yolo_name)

    print(f"Loading classifiers from {d}...")
    return {
        "yolo":     yolo_model,
        "clf":      joblib.load(os.path.join(d, "posture_classifier.pkl")),
        "scaler":   joblib.load(os.path.join(d, "scaler.pkl")),
        "le":       joblib.load(os.path.join(d, "label_encoder.pkl")),
        "selector": joblib.load(os.path.join(d, "feature_selector.pkl")),
    }

MODEL = load_models(ACTIVE_VARIANT)
print(f"✅ Model ready ({ACTIVE_VARIANT})\n")


# FLASK-LOGIN
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@login_manager.unauthorized_handler
def unauthorized():
    return jsonify({"error": "Login required."}), 401


# MAIN INTERFACE FUNCTIONS
def analyze_image(img_bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Could not decode image."}

    results = MODEL["yolo"](img, device=DEVICE, verbose=False)
    if not results or results[0].keypoints is None or results[0].keypoints.data.shape[0] == 0:
        return {"error": "No person detected. Please ensure your full upper body is visible."}

    kps_data = results[0].keypoints.data
    boxes    = results[0].boxes
    if boxes is not None and len(boxes) > 1:
        xyxy     = boxes.xyxy.cpu().numpy()
        areas    = (xyxy[:,2] - xyxy[:,0]) * (xyxy[:,3] - xyxy[:,1])
        best_idx = int(np.argmax(areas))
    else:
        best_idx = 0

    kps_np = kps_data[best_idx].cpu().numpy()
    feat   = extract_features(kps_np)
    if feat is None:
        return {"error": "Could not detect enough keypoints. Try a clearer side-profile photo."}

    feat_scaled   = MODEL["scaler"].transform(feat.reshape(1, -1))
    feat_selected = MODEL["selector"].transform(feat_scaled)
    pred_enc      = MODEL["clf"].predict(feat_selected)[0]
    proba         = MODEL["clf"].predict_proba(feat_selected)[0]
    label         = MODEL["le"].inverse_transform([pred_enc])[0]
    confidence    = float(proba[pred_enc])
    recs          = RECOMMENDATIONS.get(label, RECOMMENDATIONS["bad"])
    recommendation = recs[hash(str(img.shape)) % len(recs)]

    return {
        "label":          label,
        "confidence":     round(confidence, 4),
        "recommendation": recommendation,
        "date":           datetime.now().strftime("%Y-%m-%d %H:%M"),
        "error":          None,
    }



# AUTHENTICATION ROUTES
@app.route("/api/register", methods=["POST"])
def register():
    data     = request.get_json()
    username = (data.get("username") or "").strip()
    email    = (data.get("email") or "").strip().lower()
    password = (data.get("password") or "").strip()
    if not username or not email or not password:
        return jsonify({"error": "All fields are required."}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email already registered."}), 409
    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username already taken."}), 409
    user = User(username=username, email=email,
                password=generate_password_hash(password))
    db.session.add(user)
    db.session.commit()
    return jsonify({"message": "Account created successfully."}), 201


@app.route("/api/login", methods=["POST"])
def login():
    data     = request.get_json()
    email    = (data.get("email") or "").strip().lower()
    password = (data.get("password") or "").strip()
    user     = User.query.filter_by(email=email).first()
    if not user or not check_password_hash(user.password, password):
        return jsonify({"error": "Invalid email or password."}), 401
    login_user(user, remember=True)
    return jsonify({
        "message":  "Logged in successfully.",
        "username": user.username,
        "email":    user.email,
    }), 200


@app.route("/api/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    return jsonify({"message": "Logged out."}), 200


@app.route("/api/me", methods=["GET"])
def me():
    if current_user.is_authenticated:
        return jsonify({"logged_in": True, "username": current_user.username})
    return jsonify({"logged_in": False})


# ANALYSIS ROUTE

@app.route("/analyze", methods=["POST"])
@login_required
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
    record = Record(
        user_id        = current_user.id,
        label          = result["label"],
        confidence     = result["confidence"],
        recommendation = result["recommendation"],
        date           = result["date"],
    )
    db.session.add(record)
    db.session.commit()
    return jsonify(result), 200


# RECORDS ROUTES
@app.route("/api/records", methods=["GET"])
@login_required
def get_records():
    page     = request.args.get("page", 1, type=int)
    per_page = 5
    total    = Record.query.filter_by(user_id=current_user.id).count()
    records  = (
        Record.query
        .filter_by(user_id=current_user.id)
        .order_by(Record.created_at.desc())
        .offset((page - 1) * per_page)
        .limit(per_page)
        .all()
    )
    return jsonify({
        "records": [{
            "id":             r.id,
            "label":          r.label,
            "confidence":     r.confidence,
            "recommendation": r.recommendation,
            "date":           r.date,
        } for r in records],
        "total":       total,
        "page":        page,
        "per_page":    per_page,
        "total_pages": max(1, (total + per_page - 1) // per_page),
    }), 200


@app.route("/api/records/<int:record_id>", methods=["DELETE"])
@login_required
def delete_record(record_id):
    record = Record.query.filter_by(id=record_id, user_id=current_user.id).first()
    if not record:
        return jsonify({"error": "Record not found."}), 404
    db.session.delete(record)
    db.session.commit()
    return jsonify({"message": "Record deleted."}), 200


# STATIC ROUTES
@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)

@app.route("/health")
def health():
    return jsonify({
        "status":   "ok",
        "render":   bool(IS_RENDER),
        "variant":  ACTIVE_VARIANT,
        "model":    type(MODEL["clf"]).__name__,
    })

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        print("✅ Database ready")

    port = int(os.environ.get("PORT", 5000))
    print("=" * 50)
    print(f"  Postura running on port {port}")
    print("=" * 50)
    app.run(host="0.0.0.0", port=port, debug=not IS_RENDER)