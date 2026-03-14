"""
test_model.py
Test your saved posture model on a single image or a folder of images.
"""

import cv2
import numpy as np
import joblib
from pathlib import Path
from ultralytics import YOLO
import torch

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODELS_DIR = r"C:\Users\Ron\Desktop\postura\models"
YOLO_MODEL = "yolov8n-pose.pt"
DEVICE     = 0 if torch.cuda.is_available() else "cpu"
CONF_THRESH = 0.3

# ── Load models ───────────────────────────────
print("Loading models...")
pose_model = YOLO(YOLO_MODEL)
classifier = joblib.load(f"{MODELS_DIR}\\posture_classifier.pkl")
scaler     = joblib.load(f"{MODELS_DIR}\\scaler.pkl")
label_enc  = joblib.load(f"{MODELS_DIR}\\label_encoder.pkl")
print(f"✅ Models loaded | Device: {'GPU - ' + torch.cuda.get_device_name(0) if DEVICE == 0 else 'CPU'}")
print(f"   Classifier type: {type(classifier).__name__}")
print(f"   Classes: {list(label_enc.classes_)}\n")


# ── Feature extraction (must match train_posture.py) ──────────────────
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

def extract_features(keypoints):
    kp, confs = keypoints, keypoints[:, 2]
    if any(confs[i] < CONF_THRESH for i in [0, 5, 6, 11, 12]):
        return None
    def get(i):
        return kp[i, :2] if confs[i] >= CONF_THRESH else None
    ls, rs = get(5), get(6)
    lh, rh = get(11), get(12)
    nose   = get(0)
    mid_shoulder = (ls + rs) / 2
    mid_hip      = (lh + rh) / 2
    torso_h      = np.linalg.norm(mid_shoulder - mid_hip) + 1e-6
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
        float((nose[0] - mid_shoulder[0]) / torso_h) if nose is not None else 0.0,
        float((ls[1] - rs[1]) / torso_h),
        float((lh[1] - rh[1]) / torso_h),
    ]
    le_kp, lw = get(7), get(9)
    angles.append(angle_between(ls, le_kp, lw) if le_kp is not None and lw is not None else 0.0)
    re_kp, rw = get(8), get(10)
    angles.append(angle_between(rs, re_kp, rw) if re_kp is not None and rw is not None else 0.0)
    angles.append(float((mid_shoulder[0] - mid_hip[0]) / torso_h))
    return np.array(norm_kps + angles, dtype=np.float32)


# ── Predict function ──────────────────────────
def predict_posture(image_input):
    """
    image_input: file path (str) OR numpy array (BGR from cv2)
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
    pred_enc    = classifier.predict(feat_scaled)[0]
    proba       = classifier.predict_proba(feat_scaled)[0]
    label       = label_enc.inverse_transform([pred_enc])[0]
    confidence  = float(proba[pred_enc])

    return {"label": label, "confidence": confidence, "error": None}


# ── Test on a single image ────────────────────
def test_single(image_path):
    print(f"Testing: {Path(image_path).name}")
    result = predict_posture(image_path)

    if result["error"]:
        print(f"  ❌ Error: {result['error']}")
        return

    label = result["label"]
    conf  = result["confidence"] * 100
    icon  = "✅" if label == "good" else "⚠️"
    print(f"  {icon} Posture: {label.upper()}  ({conf:.1f}% confidence)")

    # Show annotated image
    img = cv2.imread(image_path)
    results = pose_model(img, device=DEVICE, verbose=False)
    annotated = results[0].plot()

    # Add label overlay
    color = (0, 200, 0) if label == "good" else (0, 0, 220)
    text  = f"{label.upper()} ({conf:.1f}%)"
    cv2.putText(annotated, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    out_path = str(Path(image_path).parent / f"result_{Path(image_path).name}")
    cv2.imwrite(out_path, annotated)
    print(f"  📸 Annotated image saved: {out_path}\n")


# ── Test on a whole folder ────────────────────
def test_folder(folder_path):
    IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images  = [p for p in Path(folder_path).rglob("*") if p.suffix.lower() in IMG_EXT]
    print(f"Testing {len(images)} images in: {folder_path}\n")

    correct, total, errors = 0, 0, 0
    for img_path in images:
        # Infer true label from folder name
        true_label = None
        for parent in img_path.parents:
            if parent.name.lower() == "good":
                true_label = "good"; break
            elif parent.name.lower() == "bad":
                true_label = "bad"; break

        result = predict_posture(str(img_path))
        if result["error"]:
            errors += 1
            continue

        pred  = result["label"]
        conf  = result["confidence"] * 100
        match = "✅" if pred == true_label else "❌"
        if pred == true_label:
            correct += 1
        total += 1
        print(f"  {match} {img_path.name:<30} true={true_label:<4}  pred={pred:<4}  conf={conf:.1f}%")

    print(f"\n{'='*50}")
    print(f"Accuracy: {correct}/{total} = {correct/total*100:.1f}%" if total > 0 else "No results")
    print(f"Skipped (no detection): {errors}")


# ─────────────────────────────────────────────
# MAIN — edit this section to test your images
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Usage: python test_model.py path\to\image.jpg
        # Usage: python test_model.py path\to\folder
        path = sys.argv[1]
        if Path(path).is_dir():
            test_folder(path)
        else:
            test_single(path)
    else:
        # ── Default: test on the dataset folder ──
        DATA_ROOT = r"C:\Users\Ron\Desktop\postura\data"
        print("No path provided — running on full dataset as validation.\n")
        test_folder(DATA_ROOT)