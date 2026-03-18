"""
train_posture.py
Local training script for Posture Recognition
Dataset: C:\\Users\\Ron\\Desktop\\SOFT DES PROTO\\data
"""

import os
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

import torch
from ultralytics import YOLO

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

DATA_ROOT  = r"C:\Users\Ron\Desktop\postura\data"
MODELS_OUT = r"C:\Users\Ron\Desktop\postura\models"
YOLO_MODEL = "yolov8m-pose.pt"   
CONF_THRESH = 0.3                
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEVICE = 0 if torch.cuda.is_available() else "cpu"



# ── 1. Load YOLOv8 Pose ──────────────────────────────────────────────
pose_model = YOLO(YOLO_MODEL)
print(f"✅ Loaded {YOLO_MODEL}\n")


# ── 2. Feature Engineering ───────────────────────────────────────────
def angle_between(p1, vertex, p2):
    v1 = np.array(p1) - np.array(vertex)
    v2 = np.array(p2) - np.array(vertex)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))


def angle_from_vertical(p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6))


def extract_features(keypoints, conf_threshold=CONF_THRESH):
    """
    keypoints: np.array (17, 3) — [x, y, confidence]
    Returns feature vector or None if pose unusable.
    """
    kp, confs = keypoints, keypoints[:, 2]

    # Require core upper-body joints
    if any(confs[i] < conf_threshold for i in [0, 5, 6, 11, 12]):
        return None

    def get(i):
        return kp[i, :2] if confs[i] >= conf_threshold else None

    ls, rs = get(5), get(6)   # shoulders
    lh, rh = get(11), get(12) # hips
    nose   = get(0)

    mid_shoulder = (ls + rs) / 2
    mid_hip      = (lh + rh) / 2
    torso_h      = np.linalg.norm(mid_shoulder - mid_hip) + 1e-6

    # Normalized upper-body keypoints (relative to hip center)
    norm_kps = []
    for i in range(13):  # indices 0-12 = upper body
        if confs[i] >= conf_threshold:
            rel = (kp[i, :2] - mid_hip) / torso_h
            norm_kps.extend([rel[0], rel[1]])
        else:
            norm_kps.extend([0.0, 0.0])

    # Angle features
    angles = [
        angle_from_vertical(mid_hip, mid_shoulder),           # spine tilt
        angle_from_vertical(mid_shoulder, nose) if nose is not None else 0.0,  # neck tilt
        float((nose[0] - mid_shoulder[0]) / torso_h) if nose is not None else 0.0,  # head forward
        float((ls[1] - rs[1]) / torso_h),                     # shoulder symmetry
        float((lh[1] - rh[1]) / torso_h),                     # hip symmetry
    ]

    # Elbow angles
    le, lw = get(7), get(9)
    angles.append(angle_between(ls, le, lw) if le is not None and lw is not None else 0.0)
    re, rw = get(8), get(10)
    angles.append(angle_between(rs, re, rw) if re is not None and rw is not None else 0.0)

    # Torso lean (forward/back)
    angles.append(float((mid_shoulder[0] - mid_hip[0]) / torso_h))

    return np.array(norm_kps + angles, dtype=np.float32)


# ── 3. Collect Image Paths ───────────────────────────────────────────
def collect_images(root):
    samples = []
    for img_path in Path(root).rglob("*"):
        if img_path.suffix.lower() not in IMG_EXTENSIONS:
            continue
        label = None
        for parent in img_path.parents:
            if parent.name.lower() == "good":
                label = "good"; break
            elif parent.name.lower() == "bad":
                label = "bad"; break
        if label:
            samples.append((str(img_path), label))
    return samples


samples = collect_images(DATA_ROOT)
good_n = sum(1 for _, l in samples if l == "good")
bad_n  = sum(1 for _, l in samples if l == "bad")
print(f"📁 Dataset: {len(samples)} images  (good={good_n}, bad={bad_n})\n")

if len(samples) == 0:
    raise RuntimeError(f"No images found at: {DATA_ROOT}\nCheck the path and folder structure.")


# ── 4. Extract Features via YOLO ─────────────────────────────────────
X, y = [], []
skipped = {"no_detection": 0, "low_conf": 0, "read_err": 0}

for img_path, label in tqdm(samples, desc="Extracting features"):
    img = cv2.imread(img_path)
    if img is None:
        skipped["read_err"] += 1
        continue

    results = pose_model(img, device=DEVICE, verbose=False)

    if not results or results[0].keypoints is None:
        skipped["no_detection"] += 1
        continue

    kps_data = results[0].keypoints.data
    if kps_data.shape[0] == 0:
        skipped["no_detection"] += 1
        continue

    best_idx = kps_data[:, :, 2].mean(dim=1).argmax().item()
    kps_np   = kps_data[best_idx].cpu().numpy()

    feat = extract_features(kps_np)
    if feat is None:
        skipped["low_conf"] += 1
        continue

    X.append(feat)
    y.append(label)

X = np.array(X)
y = np.array(y)

print(f"\n✅ Usable samples: {len(X)}")
print(f"   Skipped: {sum(skipped.values())} "
      f"(no_detection={skipped['no_detection']}, "
      f"low_conf={skipped['low_conf']}, "
      f"read_err={skipped['read_err']})")
print(f"   Feature vector size: {X.shape[1]}")

if len(X) < 20:
    raise RuntimeError("Too few usable samples. Check image quality.")


# ── 5. Train / Test Split ────────────────────────────────────────────
le_enc = LabelEncoder()
y_enc  = le_enc.fit_transform(y)
print(f"\nLabel mapping: {dict(zip(le_enc.classes_, le_enc.transform(le_enc.classes_)))}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}\n")

scaler = StandardScaler()
Xtr = scaler.fit_transform(X_train)
Xte = scaler.transform(X_test)


# ── 6. Train & Evaluate Models ───────────────────────────────────────
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42
    ),
    "MLP": MLPClassifier(
        hidden_layer_sizes=(128, 64, 32), activation="relu",
        max_iter=500, early_stopping=True, random_state=42
    ),
}

results = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(Xtr, y_train)
    preds = model.predict(Xte)
    acc   = accuracy_score(y_test, preds)
    cv    = cross_val_score(model, Xtr, y_train, cv=5, scoring="accuracy")
    results[name] = {"model": model, "acc": acc, "cv": cv, "preds": preds}
    print(f"  Test acc: {acc:.4f}  |  CV: {cv.mean():.4f} ± {cv.std():.4f}")
    print(classification_report(y_test, preds, target_names=le_enc.classes_))


# ── 7. Confusion Matrix Plot ─────────────────────────────────────────
fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["preds"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le_enc.classes_, yticklabels=le_enc.classes_, ax=ax)
    ax.set_title(f"{name}\nAcc: {res['acc']:.3f} | CV: {res['cv'].mean():.3f}")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
os.makedirs(MODELS_OUT, exist_ok=True)
plot_path = os.path.join(MODELS_OUT, "confusion_matrices.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n📊 Confusion matrices saved → {plot_path}")


# ── 8. Save Best Model ───────────────────────────────────────────────
best_name  = max(results, key=lambda k: results[k]["cv"].mean())
best_model = results[best_name]["model"]
print(f"\n🏆 Best model: {best_name} (CV acc: {results[best_name]['cv'].mean():.4f})")

joblib.dump(best_model, os.path.join(MODELS_OUT, "posture_classifier.pkl"))
joblib.dump(scaler,     os.path.join(MODELS_OUT, "scaler.pkl"))
joblib.dump(le_enc,     os.path.join(MODELS_OUT, "label_encoder.pkl"))

print(f"\n✅ Models saved to: {MODELS_OUT}")
print("   posture_classifier.pkl")
print("   scaler.pkl")
print("   label_encoder.pkl")


# ── 9. Inference Function (for your web app) ─────────────────────────
def predict_posture(image_input):
    """
    image_input: file path (str) OR numpy array from cv2/webcam
    Returns: {'label': 'good'/'bad', 'confidence': 0.92, 'error': None}
    """
    img = cv2.imread(image_input) if isinstance(image_input, str) else image_input
    if img is None:
        return {"label": None, "confidence": None, "error": "Could not read image"}

    results = pose_model(img, device=DEVICE, verbose=False)
    if not results or results[0].keypoints is None or results[0].keypoints.data.shape[0] == 0:
        return {"label": None, "confidence": None, "error": "No person detected"}

    kps_data = results[0].keypoints.data
    best_idx = kps_data[:, :, 2].mean(dim=1).argmax().item()
    kps_np   = kps_data[best_idx].cpu().numpy()

    feat = extract_features(kps_np)
    if feat is None:
        return {"label": None, "confidence": None, "error": "Keypoints not visible enough"}

    feat_scaled = scaler.transform(feat.reshape(1, -1))
    pred_enc    = best_model.predict(feat_scaled)[0]
    proba       = best_model.predict_proba(feat_scaled)[0]
    label       = le_enc.inverse_transform([pred_enc])[0]
    confidence  = float(proba[pred_enc])

    return {"label": label, "confidence": confidence, "error": None}


# Quick test
test_path, test_label = samples[0]
print(f"\n🔍 Quick test on: {Path(test_path).name}  (true label: {test_label})")
print(f"   Result: {predict_posture(test_path)}")

print("\n🎉 Training complete!")