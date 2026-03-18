"""
test_model.py
Test your saved posture models (yolov8m and yolov8l variants).
Now includes feature_selector.pkl in the inference pipeline.

Usage:
  # Test a single image with both models:
  python test_model.py path\\to\\image.jpg

  # Test a whole folder with both models:
  python test_model.py path\\to\\folder

  # Run on full dataset (default, no args):
  python test_model.py
"""

import sys
import cv2
import numpy as np
import joblib
from pathlib import Path
from ultralytics import YOLO
import torch

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODELS_DIR  = r"C:\Users\Ron\Desktop\postura\models"
CONF_THRESH = 0.2
IMG_EXT     = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEVICE      = 0 if torch.cuda.is_available() else "cpu"

VARIANTS = {
    "yolov8m": {
        "yolo":      "yolov8m-pose.pt",
        "models_dir": f"{MODELS_DIR}\\yolov8m",
    },
    "yolov8l": {
        "yolo":      "yolov8l-pose.pt",
        "models_dir": f"{MODELS_DIR}\\yolov8l",
    },
}
# ─────────────────────────────────────────────


# ── Feature engineering (must match train_posture.py) ────────────────
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


# ── Load all models ───────────────────────────────────────────────────
def load_variant(variant_name, config):
    d = config["models_dir"]
    print(f"  Loading {variant_name}... ", end="")
    try:
        clf      = joblib.load(f"{d}\\posture_classifier.pkl")
        scaler   = joblib.load(f"{d}\\scaler.pkl")
        le       = joblib.load(f"{d}\\label_encoder.pkl")
        selector = joblib.load(f"{d}\\feature_selector.pkl")
        yolo     = YOLO(config["yolo"])
        print(f"✅  ({type(clf).__name__})")
        return {"yolo": yolo, "clf": clf, "scaler": scaler, "le": le, "selector": selector}
    except FileNotFoundError as e:
        print(f"❌  Missing file: {e}")
        return None


print("=" * 60)
print("  Posture Model Tester")
print("=" * 60)
print(f"Device: {'GPU — ' + torch.cuda.get_device_name(0) if DEVICE == 0 else 'CPU'}\n")
print("Loading models:")

loaded = {}
for name, cfg in VARIANTS.items():
    result = load_variant(name, cfg)
    if result:
        loaded[name] = result

if not loaded:
    raise RuntimeError("No models loaded. Run train_posture.py first.")

print(f"\n{len(loaded)} model(s) ready: {list(loaded.keys())}\n")


# ── Core prediction function ──────────────────────────────────────────
def predict(image_input, variant_name):
    """
    image_input : file path (str) OR numpy BGR array
    variant_name: 'yolov8m' or 'yolov8l'
    Returns: {'label': 'good'/'bad', 'confidence': 0.92, 'error': None}
    """
    v = loaded[variant_name]
    img = cv2.imread(image_input) if isinstance(image_input, str) else image_input
    if img is None:
        return {"label": None, "confidence": None, "error": "Could not read image"}

    results = v["yolo"](img, device=DEVICE, verbose=False)
    if not results or results[0].keypoints is None or results[0].keypoints.data.shape[0] == 0:
        return {"label": None, "confidence": None, "error": "No person detected"}

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
        return {"label": None, "confidence": None, "error": "Keypoints not visible enough"}

    # Pipeline: scale → select features → classify
    feat_scaled    = v["scaler"].transform(feat.reshape(1, -1))
    feat_selected  = v["selector"].transform(feat_scaled)
    pred_enc       = v["clf"].predict(feat_selected)[0]
    proba          = v["clf"].predict_proba(feat_selected)[0]
    label          = v["le"].inverse_transform([pred_enc])[0]
    confidence     = float(proba[pred_enc])

    return {"label": label, "confidence": confidence, "error": None}


# ── Test single image ─────────────────────────────────────────────────
def test_single(image_path):
    print(f"Image: {Path(image_path).name}")
    print(f"{'─'*45}")

    for name in loaded:
        result = predict(image_path, name)
        if result["error"]:
            print(f"  [{name}]  ❌ {result['error']}")
        else:
            icon  = "✅" if result["label"] == "good" else "⚠️"
            conf  = result["confidence"] * 100
            print(f"  [{name}]  {icon} {result['label'].upper():<5}  ({conf:.1f}% confidence)")

    # Save annotated image from first available model
    first = next(iter(loaded.values()))
    img = cv2.imread(image_path)
    if img is not None:
        res       = first["yolo"](img, device=DEVICE, verbose=False)
        annotated = res[0].plot()
        result    = predict(image_path, next(iter(loaded)))
        if result["label"]:
            color = (0, 200, 0) if result["label"] == "good" else (0, 0, 220)
            text  = f"{result['label'].upper()} ({result['confidence']*100:.1f}%)"
            cv2.putText(annotated, text, (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)
        out = str(Path(image_path).parent / f"result_{Path(image_path).name}")
        cv2.imwrite(out, annotated)
        print(f"\n  📸 Annotated saved: {out}")
    print()


# ── Test folder ───────────────────────────────────────────────────────
def test_folder(folder_path):
    images = [p for p in Path(folder_path).rglob("*") if p.suffix.lower() in IMG_EXT]
    print(f"Testing {len(images)} images in: {folder_path}\n")

    stats = {name: {"correct": 0, "total": 0, "errors": 0} for name in loaded}

    for img_path in images:
        true_label = None
        for parent in img_path.parents:
            if parent.name.lower() == "good":
                true_label = "good"; break
            elif parent.name.lower() == "bad":
                true_label = "bad"; break

        row = f"  {img_path.name:<35}"
        if true_label:
            row += f"true={true_label:<5}"

        for name in loaded:
            result = predict(str(img_path), name)
            if result["error"]:
                row += f"  [{name}] ERR"
                stats[name]["errors"] += 1
            else:
                pred  = result["label"]
                conf  = result["confidence"] * 100
                if true_label:
                    match = "✅" if pred == true_label else "❌"
                    if pred == true_label:
                        stats[name]["correct"] += 1
                    stats[name]["total"] += 1
                else:
                    match = "→"
                row += f"  [{name}] {match}{pred:<5}({conf:.0f}%)"

        print(row)

    # Summary
    print(f"\n{'='*60}")
    print("ACCURACY SUMMARY")
    print(f"{'─'*60}")
    for name, s in stats.items():
        if s["total"] > 0:
            acc = s["correct"] / s["total"] * 100
            print(f"  {name:<12}  {s['correct']}/{s['total']} correct = {acc:.1f}%  "
                  f"(errors/skipped: {s['errors']})")
        else:
            print(f"  {name:<12}  No labeled images found (errors: {s['errors']})")


# ── Main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if Path(path).is_dir():
            test_folder(path)
        else:
            test_single(path)
    else:
        # Default: run on full dataset
        DATA_ROOT = r"C:\Users\Ron\Desktop\postura\data"
        print(f"No path provided — running on full dataset: {DATA_ROOT}\n")
        test_folder(DATA_ROOT)