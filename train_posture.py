"""
train_posture.py  —  Full Improved Pipeline
============================================
Improvements included:
  1. Lower confidence threshold (0.3 → 0.2) — recovers more images
  2. Expanded angle features — more descriptive keypoint relationships
  3. Feature selection — drops zero-importance/noisy features
  4. Separate old/ folder as independent validation set
  5. SMOTE oversampling — fixes good/bad class imbalance
  6. GridSearchCV hyperparameter tuning — finds best model params
  7. Voting Classifier ensemble — combines RF + GB + MLP
  8. Probability calibration — fixes overconfident confidence scores
  9. ROC-AUC + per-source evaluation — more honest metrics
 10. Trains with BOTH yolov8m-pose and yolov8l-pose
     → saves to models/yolov8m/ and models/yolov8l/
"""

import os
import cv2
import numpy as np
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

import torch
from ultralytics import YOLO

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
)
from sklearn.feature_selection import SelectFromModel
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────
DATA_ROOT      = r"C:\Users\Ron\Desktop\postura\data"
MODELS_OUT     = r"C:\Users\Ron\Desktop\postura\models"
CONF_THRESH    = 0.2          # lowered from 0.3 → recovers more images
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEVICE         = 0 if torch.cuda.is_available() else "cpu"
YOLO_VARIANTS  = ["yolov8m-pose.pt", "yolov8l-pose.pt"]
RANDOM_STATE   = 42
# ─────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  Posture Recognition — Full Improved Dual Training")
print("=" * 65)
print(f"Device : {'GPU — ' + torch.cuda.get_device_name(0) if DEVICE == 0 else 'CPU'}")
print(f"Data   : {DATA_ROOT}")
print(f"Models : {YOLO_VARIANTS}")
print(f"Conf   : {CONF_THRESH} (lowered for better image recovery)")
print()


# ═════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING  (expanded)
# ═════════════════════════════════════════════════════════════════════
FEATURE_NAMES = []  # populated during first extract call

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


def extract_features(keypoints, conf_threshold=CONF_THRESH, record_names=False):
    """
    keypoints : np.array (17, 3)  [x, y, confidence]
    Returns feature vector (np.float32) or None if pose unusable.
    """
    kp, confs = keypoints, keypoints[:, 2]

    # Require core upper-body joints
    if any(confs[i] < conf_threshold for i in [5, 6, 11, 12]):
        return None

    def get(i):
        return kp[i, :2] if confs[i] >= conf_threshold else None

    nose  = get(0)
    le_   = get(1);  re_  = get(2)   # eyes
    lear  = get(3);  rear = get(4)   # ears
    ls    = get(5);  rs   = get(6)   # shoulders
    lelb  = get(7);  relb = get(8)   # elbows
    lwr   = get(9);  rwr  = get(10)  # wrists
    lh    = get(11); rh   = get(12)  # hips

    mid_shoulder = (ls + rs) / 2
    mid_hip      = (lh + rh) / 2
    torso_h      = np.linalg.norm(mid_shoulder - mid_hip) + 1e-6
    shoulder_w   = np.linalg.norm(ls - rs) + 1e-6

    feats = []
    names = []

    # ── A. Normalized upper-body keypoints (relative to hip centre) ───
    upper_idx = {
        0: "nose", 1: "l_eye", 2: "r_eye", 3: "l_ear", 4: "r_ear",
        5: "l_shoulder", 6: "r_shoulder", 7: "l_elbow", 8: "r_elbow",
        9: "l_wrist", 10: "r_wrist", 11: "l_hip", 12: "r_hip",
    }
    for i, label in upper_idx.items():
        if confs[i] >= conf_threshold:
            rel = (kp[i, :2] - mid_hip) / torso_h
            feats.extend([rel[0], rel[1]])
        else:
            feats.extend([0.0, 0.0])
        names.extend([f"kp_{label}_x", f"kp_{label}_y"])

    # ── B. Spine & torso angles ────────────────────────────────────────
    feats.append(angle_from_vertical(mid_hip, mid_shoulder))
    names.append("spine_tilt")

    torso_lean = signed_ratio(mid_shoulder[0], mid_hip[0], torso_h)
    feats.append(torso_lean)
    names.append("torso_lean_fwd")

    # ── C. Head / neck angles ──────────────────────────────────────────
    if nose is not None:
        feats.append(angle_from_vertical(mid_shoulder, nose))
        feats.append(signed_ratio(nose[0], mid_shoulder[0], torso_h))  # fwd
        feats.append(signed_ratio(nose[1], mid_shoulder[1], torso_h))  # up/down
    else:
        feats.extend([0.0, 0.0, 0.0])
    names.extend(["neck_tilt", "head_fwd_offset", "head_vert_offset"])

    # Ear-to-shoulder angle (head tilt left/right)
    if lear is not None and rear is not None:
        mid_ear = (lear + rear) / 2
        feats.append(angle_from_vertical(mid_shoulder, mid_ear))
        feats.append(signed_ratio(lear[1], rear[1], torso_h))  # ear level sym
    else:
        feats.extend([0.0, 0.0])
    names.extend(["ear_tilt", "ear_level_sym"])

    # ── D. Shoulder symmetry ───────────────────────────────────────────
    feats.append(signed_ratio(ls[1], rs[1], torso_h))   # vertical sym
    feats.append(signed_ratio(ls[0], rs[0], shoulder_w)) # width norm
    names.extend(["shoulder_vert_sym", "shoulder_width_ratio"])

    # ── E. Hip symmetry ───────────────────────────────────────────────
    feats.append(signed_ratio(lh[1], rh[1], torso_h))
    names.append("hip_vert_sym")

    # ── F. Elbow angles ───────────────────────────────────────────────
    feats.append(angle_between(ls, lelb, lwr) if lelb is not None and lwr is not None else 0.0)
    feats.append(angle_between(rs, relb, rwr) if relb is not None and rwr is not None else 0.0)
    names.extend(["l_elbow_angle", "r_elbow_angle"])

    # Shoulder-to-elbow angle from vertical
    feats.append(angle_from_vertical(ls, lelb) if lelb is not None else 0.0)
    feats.append(angle_from_vertical(rs, relb) if relb is not None else 0.0)
    names.extend(["l_upper_arm_angle", "r_upper_arm_angle"])

    # ── G. Wrist relative to shoulder (desk arm position) ────────────
    feats.append(signed_ratio(lwr[1], ls[1], torso_h) if lwr is not None else 0.0)
    feats.append(signed_ratio(rwr[1], rs[1], torso_h) if rwr is not None else 0.0)
    names.extend(["l_wrist_rel_shoulder", "r_wrist_rel_shoulder"])

    # ── H. Left/right body symmetry score ────────────────────────────
    # Average difference between mirrored keypoints
    sym_pairs = [(5, 6), (7, 8), (9, 10), (11, 12)]
    sym_scores = []
    for l_idx, r_idx in sym_pairs:
        if confs[l_idx] >= conf_threshold and confs[r_idx] >= conf_threshold:
            diff = abs(kp[l_idx, 1] - kp[r_idx, 1]) / torso_h
            sym_scores.append(diff)
    feats.append(float(np.mean(sym_scores)) if sym_scores else 0.0)
    names.append("lr_symmetry_score")

    if record_names and not FEATURE_NAMES:
        FEATURE_NAMES.extend(names)

    return np.array(feats, dtype=np.float32)


# ═════════════════════════════════════════════════════════════════════
# DATASET COLLECTION  (separates old/ as independent validation)
# ═════════════════════════════════════════════════════════════════════
def collect_images(root):
    """
    Collects ALL images (old/ and new) into one pool.
    Label is determined by nearest 'good' or 'bad' folder in the path.
    old/ is no longer a separate validation set — everything is merged
    so blurry/clean images are distributed across train and test splits.
    """
    all_samples = []

    for img_path in Path(root).rglob("*"):
        if img_path.suffix.lower() not in IMG_EXTENSIONS:
            continue

        label = None
        for parent in img_path.parents:
            pname = parent.name.lower()
            if pname == "good":
                label = "good"
                break
            elif pname == "bad":
                label = "bad"
                break

        if label is not None:
            all_samples.append((str(img_path), label))

    return all_samples, []   # empty list keeps rest of code compatible


train_samples, old_samples = collect_images(DATA_ROOT)

def count(s): return sum(1 for _,l in s if l=="good"), sum(1 for _,l in s if l=="bad")
tg, tb = count(train_samples)

print(f"Total images     : {len(train_samples)}  (good={tg}, bad={tb})")
print(f"Strategy         : merged old+new, random 80/20 stratified split")
print()

if len(train_samples) == 0:
    raise RuntimeError(f"No images found at: {DATA_ROOT}")


# ═════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION HELPER
# ═════════════════════════════════════════════════════════════════════
def run_extraction(pose_model, samples, desc, record_names=False):
    X, y = [], []
    skipped = {"no_detection": 0, "low_conf": 0, "read_err": 0}

    for img_path, label in tqdm(samples, desc=desc):
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

        # Pick the most prominent person — largest bounding box area
        # (more robust than confidence for busy backgrounds with multiple people)
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 1:
            xyxy  = boxes.xyxy.cpu().numpy()  # (N, 4)
            areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            best_idx = int(np.argmax(areas))
        else:
            best_idx = 0
        kps_np = kps_data[best_idx].cpu().numpy()

        feat = extract_features(kps_np, record_names=record_names)
        record_names = False  # only record on first successful extraction

        if feat is None:
            skipped["low_conf"] += 1
            continue

        X.append(feat)
        y.append(label)

    return np.array(X), np.array(y), skipped


# ═════════════════════════════════════════════════════════════════════
# MAIN TRAINING PIPELINE  (called once per YOLO variant)
# ═════════════════════════════════════════════════════════════════════
def train_pipeline(yolo_model_name):
    variant  = yolo_model_name.replace("yolov8", "").replace("-pose.pt", "")
    save_dir = os.path.join(MODELS_OUT, f"yolov8{variant}")
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "═" * 65)
    print(f"  PIPELINE — {yolo_model_name}")
    print("═" * 65)

    pose_model = YOLO(yolo_model_name)
    print(f"✅ Loaded {yolo_model_name}\n")

    # ── 1. Extract features ───────────────────────────────────────────
    X_train_raw, y_train_raw, sk_train = run_extraction(
        pose_model, train_samples, f"Train extract [{variant}]", record_names=True
    )
    X_old_raw,   y_old_raw,   sk_old   = run_extraction(
        pose_model, old_samples,   f"Old/  extract [{variant}]"
    )

    print(f"\nTrain set  — usable: {len(X_train_raw)}, skipped: {sum(sk_train.values())} "
          f"(low_conf={sk_train['low_conf']}, no_det={sk_train['no_detection']})")
    print(f"Old/ valid — usable: {len(X_old_raw)},  skipped: {sum(sk_old.values())} "
          f"(low_conf={sk_old['low_conf']}, no_det={sk_old['no_detection']})")
    print(f"Feature vector size: {X_train_raw.shape[1]}")

    # ── 2. Encode labels ──────────────────────────────────────────────
    le_enc = LabelEncoder()
    y_train_enc = le_enc.fit_transform(y_train_raw)
    y_old_enc   = le_enc.transform(y_old_raw) if len(y_old_raw) > 0 else np.array([])
    print(f"Label map: {dict(zip(le_enc.classes_, le_enc.transform(le_enc.classes_)))}")
    print(f"Class dist — good: {(y_train_raw=='good').sum()}, bad: {(y_train_raw=='bad').sum()}")

    # ── 3. Train/test split ───────────────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_train_raw, y_train_enc,
        test_size=0.2, random_state=RANDOM_STATE, stratify=y_train_enc
    )

    # ── 4. Scale ──────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    X_old_s = scaler.transform(X_old_raw) if len(X_old_raw) > 0 else np.array([])

    # ── 5. SMOTE — balance classes in training set ────────────────────
    print(f"\nBefore SMOTE — train: {len(X_tr_s)} "
          f"(good={( y_tr==le_enc.transform(['good'])[0] ).sum()}, "
          f"bad={( y_tr==le_enc.transform(['bad'])[0] ).sum()})")

    smote = SMOTE(random_state=RANDOM_STATE)
    X_tr_bal, y_tr_bal = smote.fit_resample(X_tr_s, y_tr)

    print(f"After  SMOTE — train: {len(X_tr_bal)} "
          f"(good={( y_tr_bal==le_enc.transform(['good'])[0] ).sum()}, "
          f"bad={( y_tr_bal==le_enc.transform(['bad'])[0] ).sum()})")

    # ── 6. Feature selection — drop near-zero importance features ────
    print("\nRunning feature selection...")
    selector_base = RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
    )
    selector_base.fit(X_tr_bal, y_tr_bal)

    selector = SelectFromModel(selector_base, threshold="0.5*mean", prefit=True)
    X_tr_sel  = selector.transform(X_tr_bal)
    X_te_sel  = selector.transform(X_te_s)
    X_old_sel = selector.transform(X_old_s) if len(X_old_s) > 0 else np.array([])

    selected_mask  = selector.get_support()
    n_selected     = selected_mask.sum()
    selected_names = [FEATURE_NAMES[i] for i in range(len(FEATURE_NAMES)) if selected_mask[i]] \
                     if FEATURE_NAMES else [f"feat_{i}" for i in range(n_selected)]

    print(f"Features kept: {n_selected} / {X_tr_bal.shape[1]}")
    print(f"Selected: {selected_names[:10]}{'...' if len(selected_names) > 10 else ''}")

    # ── 7. GridSearchCV — tune each base model ────────────────────────
    print("\nGridSearchCV tuning (this may take a few minutes)...")
    cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    # Random Forest (n_jobs=1 to avoid paging file / memory errors on Windows)
    rf_grid = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1),
        param_grid={
            "n_estimators": [100, 200, 300],
            "max_depth":    [None, 10, 20],
            "min_samples_split": [2, 5],
        },
        cv=cv_strategy, scoring="roc_auc", n_jobs=1, verbose=0
    )
    rf_grid.fit(X_tr_sel, y_tr_bal)
    best_rf = rf_grid.best_estimator_
    print(f"  RF   best params : {rf_grid.best_params_}  (AUC={rf_grid.best_score_:.4f})")

    # Gradient Boosting
    gb_grid = GridSearchCV(
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        param_grid={
            "n_estimators":  [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth":     [3, 4, 5],
        },
        cv=cv_strategy, scoring="roc_auc", n_jobs=1, verbose=0
    )
    gb_grid.fit(X_tr_sel, y_tr_bal)
    best_gb = gb_grid.best_estimator_
    print(f"  GB   best params : {gb_grid.best_params_}  (AUC={gb_grid.best_score_:.4f})")

    # MLP
    mlp_grid = GridSearchCV(
        MLPClassifier(early_stopping=True, random_state=RANDOM_STATE, max_iter=500),
        param_grid={
            "hidden_layer_sizes": [(64, 32), (128, 64), (128, 64, 32)],
            "alpha":              [0.0001, 0.001, 0.01],
        },
        cv=cv_strategy, scoring="roc_auc", n_jobs=1, verbose=0
    )
    mlp_grid.fit(X_tr_sel, y_tr_bal)
    best_mlp = mlp_grid.best_estimator_
    print(f"  MLP  best params : {mlp_grid.best_params_}  (AUC={mlp_grid.best_score_:.4f})")

    # ── 8. Voting Ensemble ────────────────────────────────────────────
    ensemble = VotingClassifier(
        estimators=[("rf", best_rf), ("gb", best_gb), ("mlp", best_mlp)],
        voting="soft",   # uses predicted probabilities
        n_jobs=1
    )
    ensemble.fit(X_tr_sel, y_tr_bal)
    print("\n✅ Voting ensemble trained")

    # ── 9. Probability Calibration ────────────────────────────────────
    calibrated = CalibratedClassifierCV(ensemble, method="isotonic", cv=5)
    calibrated.fit(X_tr_sel, y_tr_bal)
    print("✅ Probability calibration applied")

    # ── 10. Evaluate all models ───────────────────────────────────────
    all_models = {
        "RandomForest":      best_rf,
        "GradientBoosting":  best_gb,
        "MLP":               best_mlp,
        "Ensemble":          ensemble,
        "Ensemble_Calibrated": calibrated,
    }

    print(f"\n{'─'*65}")
    print(f"{'Model':<25} {'Test Acc':>10} {'ROC-AUC':>10} {'CV-AUC (10f)':>14}")
    print(f"{'─'*65}")

    eval_results = {}
    for name, model in all_models.items():
        preds  = model.predict(X_te_sel)
        probas = model.predict_proba(X_te_sel)[:, 1]
        acc    = accuracy_score(y_te, preds)
        auc    = roc_auc_score(y_te, probas)

        cv_auc = cross_val_score(
            model, X_tr_sel, y_tr_bal,
            cv=cv_strategy, scoring="roc_auc", n_jobs=-1
        )
        eval_results[name] = {
            "model": model, "acc": acc, "auc": auc,
            "cv_auc": cv_auc, "preds": preds
        }
        print(f"  {name:<23} {acc:>10.4f} {auc:>10.4f} {cv_auc.mean():>10.4f}±{cv_auc.std():.4f}")

    print(f"{'─'*65}")

    # Per-source: classification report on old/ validation set
    if len(X_old_sel) > 0:
        print(f"\n── Old/ folder independent validation ──")
        for name in ["Ensemble_Calibrated", "Ensemble"]:
            preds_old = all_models[name].predict(X_old_sel)
            acc_old   = accuracy_score(y_old_enc, preds_old)
            print(f"  {name}: {acc_old:.4f}")
            print(classification_report(y_old_enc, preds_old, target_names=le_enc.classes_))

    # ── 11. Confusion matrices ────────────────────────────────────────
    fig, axes = plt.subplots(1, len(all_models), figsize=(6 * len(all_models), 5))
    fig.suptitle(f"YOLOv8{variant}-pose — All Models", fontsize=13, fontweight="bold")
    for ax, (name, res) in zip(axes, eval_results.items()):
        cm = confusion_matrix(y_te, res["preds"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=le_enc.classes_, yticklabels=le_enc.classes_, ax=ax)
        ax.set_title(f"{name}\nAcc={res['acc']:.3f} AUC={res['auc']:.3f}")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    plot_path = os.path.join(save_dir, "confusion_matrices.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    # ── 12. Feature importance radar chart ────────────────────────────
    importances = best_rf.feature_importances_
    feat_labels = selected_names

    angles_plot = np.linspace(0, 2 * np.pi, len(importances), endpoint=False).tolist()
    values      = importances.tolist()
    angles_plot += angles_plot[:1]
    values      += values[:1]

    fig2, ax2 = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax2.plot(angles_plot, values, linewidth=2)
    ax2.fill(angles_plot, values, alpha=0.25)
    ax2.set_xticks(angles_plot[:-1])
    ax2.set_xticklabels(feat_labels, size=7)
    ax2.set_title(f"Feature Importance — YOLOv8{variant} (after selection)",
                  fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    radar_path = os.path.join(save_dir, "feature_importance_radar.png")
    plt.savefig(radar_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n📊 Confusion matrices → {plot_path}")
    print(f"📊 Feature radar      → {radar_path}")

    # ── 13. Save best model (calibrated ensemble) ─────────────────────
    best_name  = max(eval_results, key=lambda k: eval_results[k]["cv_auc"].mean())
    best_model = eval_results[best_name]["model"]
    print(f"\n🏆 Best model: {best_name} "
          f"(CV-AUC: {eval_results[best_name]['cv_auc'].mean():.4f})")

    joblib.dump(best_model,  os.path.join(save_dir, "posture_classifier.pkl"))
    joblib.dump(scaler,      os.path.join(save_dir, "scaler.pkl"))
    joblib.dump(le_enc,      os.path.join(save_dir, "label_encoder.pkl"))
    joblib.dump(selector,    os.path.join(save_dir, "feature_selector.pkl"))

    print(f"✅ Saved to: {save_dir}")
    print(f"   posture_classifier.pkl  ({best_name})")
    print(f"   scaler.pkl")
    print(f"   label_encoder.pkl")
    print(f"   feature_selector.pkl")

    # Cleanup GPU memory
    del pose_model
    torch.cuda.empty_cache()

    return {
        "variant":        f"yolov8{variant}",
        "best_model":     best_name,
        "best_cv_auc":    eval_results[best_name]["cv_auc"].mean(),
        "best_test_acc":  eval_results[best_name]["acc"],
        "best_test_auc":  eval_results[best_name]["auc"],
        "usable_train":   len(X_train_raw),
        "usable_old":     len(X_old_raw),
        "n_features":     n_selected,
    }


# ═════════════════════════════════════════════════════════════════════
# RUN BOTH PIPELINES
# ═════════════════════════════════════════════════════════════════════
summary = []
for yolo_variant in YOLO_VARIANTS:
    info = train_pipeline(yolo_variant)
    summary.append(info)

# ── Final summary table ───────────────────────────────────────────────
print("\n" + "═" * 65)
print("  FINAL SUMMARY")
print("═" * 65)
print(f"{'Variant':<12} {'Best Model':<25} {'Test Acc':>10} {'Test AUC':>10} {'CV-AUC':>10} {'Feats':>7}")
print("─" * 65)
for s in summary:
    print(f"{s['variant']:<12} {s['best_model']:<25} "
          f"{s['best_test_acc']:>10.4f} {s['best_test_auc']:>10.4f} "
          f"{s['best_cv_auc']:>10.4f} {s['n_features']:>7}")

print(f"\n📁 All models saved under: {MODELS_OUT}")
print("   models/")
print("   ├── yolov8m/")
print("   │   ├── posture_classifier.pkl")
print("   │   ├── scaler.pkl")
print("   │   ├── label_encoder.pkl")
print("   │   ├── feature_selector.pkl")
print("   │   ├── confusion_matrices.png")
print("   │   └── feature_importance_radar.png")
print("   └── yolov8l/")
print("       └── (same structure)")
print("\n🎉 Full improved dual training complete!")