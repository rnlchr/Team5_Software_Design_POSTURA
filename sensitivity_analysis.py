"""
sensitivity_analysis.py
Visualizes feature importance for both yolov8m and yolov8l models.
Generates:
  - Radar chart (sensitivity analysis) per variant
  - Bar chart of top features per variant
  - Side-by-side comparison of both variants
"""

import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODELS_DIR = r"C:\Users\Ron\Desktop\postura\models"
VARIANTS   = ["yolov8m", "yolov8l"]
OUT_DIR    = r"C:\Users\Ron\Desktop\postura\models\analysis"
# ─────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 55)
print("  Sensitivity / Feature Importance Analysis")
print("=" * 55)


# ── Load models and extract importances ───────────────────────────────
variant_data = {}

for variant in VARIANTS:
    model_dir = os.path.join(MODELS_DIR, variant)
    try:
        clf      = joblib.load(f"{model_dir}\\posture_classifier.pkl")
        selector = joblib.load(f"{model_dir}\\feature_selector.pkl")
        scaler   = joblib.load(f"{model_dir}\\scaler.pkl")
    except FileNotFoundError as e:
        print(f"❌ Could not load {variant}: {e}")
        continue

    # Get feature names from selector
    # selector was fit on the full 43-feature space — get which were selected
    selected_mask = selector.get_support()

    # Full feature names (must match train_posture.py order)
    upper_idx_names = [
        "nose", "l_eye", "r_eye", "l_ear", "r_ear",
        "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
        "l_wrist", "r_wrist", "l_hip", "r_hip",
    ]
    all_feature_names = []
    for name in upper_idx_names:
        all_feature_names.extend([f"kp_{name}_x", f"kp_{name}_y"])
    all_feature_names += [
        "spine_tilt", "torso_lean_fwd",
        "neck_tilt", "head_fwd_offset", "head_vert_offset",
        "ear_tilt", "ear_level_sym",
        "shoulder_vert_sym", "shoulder_width_ratio",
        "hip_vert_sym",
        "l_elbow_angle", "r_elbow_angle",
        "l_upper_arm_angle", "r_upper_arm_angle",
        "l_wrist_rel_shoulder", "r_wrist_rel_shoulder",
        "lr_symmetry_score",
    ]

    selected_names = [all_feature_names[i] for i in range(len(all_feature_names))
                      if i < len(selected_mask) and selected_mask[i]]

    # Extract feature importances
    clf_name = type(clf).__name__

    if hasattr(clf, "feature_importances_"):
        # RandomForest, GradientBoosting
        importances = clf.feature_importances_
        importance_type = "Gini Importance"

    elif hasattr(clf, "estimators_") and hasattr(clf.estimators_[0], "feature_importances_"):
        # VotingClassifier — average across estimators that have importances
        imps = []
        for est_name, est in clf.estimators_:
            if hasattr(est, "feature_importances_"):
                imps.append(est.feature_importances_)
        importances = np.mean(imps, axis=0) if imps else None
        importance_type = "Avg Gini (Ensemble)"

    elif hasattr(clf, "base_estimator") and hasattr(clf.base_estimator, "feature_importances_"):
        # CalibratedClassifierCV wrapping an ensemble
        importances = clf.base_estimator.feature_importances_
        importance_type = "Gini (Calibrated)"

    else:
        # Try to dig into calibrated classifier
        importances = None
        if hasattr(clf, "calibrated_classifiers_"):
            for cal_clf in clf.calibrated_classifiers_:
                base = cal_clf.estimator
                if hasattr(base, "feature_importances_"):
                    importances = base.feature_importances_
                    importance_type = "Gini (from Calibrated)"
                    break
                elif hasattr(base, "estimators_"):
                    imps = []
                    for _, e in base.estimators_:
                        if hasattr(e, "feature_importances_"):
                            imps.append(e.feature_importances_)
                    if imps:
                        importances = np.mean(imps, axis=0)
                        importance_type = "Avg Gini (Calibrated Ensemble)"
                        break

    if importances is None:
        print(f"⚠️  {variant}: Cannot extract importances from {clf_name} — skipping")
        continue

    # Pad or trim importances to match selected_names length
    n = min(len(importances), len(selected_names))
    importances = importances[:n]
    selected_names = selected_names[:n]

    # Normalize to sum to 1
    importances = importances / (importances.sum() + 1e-10)

    variant_data[variant] = {
        "importances":  importances,
        "names":        selected_names,
        "clf_name":     clf_name,
        "imp_type":     importance_type,
        "n_features":   n,
    }

    print(f"\n✅ {variant} — {clf_name} ({importance_type})")
    print(f"   Features selected: {n}")
    print(f"   Top 5 features:")
    top5_idx = np.argsort(importances)[::-1][:5]
    for i, idx in enumerate(top5_idx):
        print(f"     {i+1}. {selected_names[idx]:<30} {importances[idx]:.4f}")


if not variant_data:
    raise RuntimeError("No model data could be loaded. Run train_posture.py first.")


# ═════════════════════════════════════════════
# PLOT 1: Radar charts per variant
# ═════════════════════════════════════════════
colors = {"yolov8m": "#2196F3", "yolov8l": "#FF9800"}

for variant, data in variant_data.items():
    imps  = data["importances"]
    names = data["names"]
    n     = len(imps)
    color = colors.get(variant, "#4CAF50")

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values = imps.tolist()
    angles += angles[:1]
    values += values[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    ax.plot(angles, values, color=color, linewidth=2.5)
    ax.fill(angles, values, color=color, alpha=0.2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(names, size=8, color="black")
    ax.set_yticklabels([])

    # Highlight top 3 features
    top3 = np.argsort(imps)[::-1][:3]
    for idx in top3:
        ax.get_xticklabels()[idx].set_color("red")
        ax.get_xticklabels()[idx].set_fontweight("bold")

    ax.set_title(
        f"Feature Importance Radar — {variant}\n"
        f"({data['clf_name']} | {data['imp_type']} | {n} features)\n"
        f"Red labels = top 3 most important",
        fontsize=12, fontweight="bold", pad=25
    )

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"radar_{variant}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n📊 Radar saved: {path}")


# ═════════════════════════════════════════════
# PLOT 2: Bar charts — top 15 features per variant
# ═════════════════════════════════════════════
n_variants = len(variant_data)
fig, axes = plt.subplots(1, n_variants, figsize=(10 * n_variants, 8))
if n_variants == 1:
    axes = [axes]

for ax, (variant, data) in zip(axes, variant_data.items()):
    imps  = data["importances"]
    names = data["names"]
    color = colors.get(variant, "#4CAF50")

    # Top 15
    top_n   = min(15, len(imps))
    top_idx = np.argsort(imps)[::-1][:top_n]
    top_imp = imps[top_idx]
    top_nam = [names[i] for i in top_idx]

    bar_colors = [color if i > 0 else "#F44336" for i in range(top_n)]
    bar_colors[0] = "#F44336"   # most important = red

    bars = ax.barh(range(top_n), top_imp[::-1], color=bar_colors[::-1], edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_nam[::-1], fontsize=10)
    ax.set_xlabel("Normalized Importance", fontsize=11)
    ax.set_title(
        f"Top {top_n} Features — {variant}\n({data['clf_name']})",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlim(0, max(top_imp) * 1.15)

    # Value labels
    for bar, val in zip(bars, top_imp[::-1]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)

    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.suptitle("Feature Importance — Top Features per Model Variant",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
path = os.path.join(OUT_DIR, "bar_top_features.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.show()
print(f"📊 Bar chart saved: {path}")


# ═════════════════════════════════════════════
# PLOT 3: Side-by-side comparison (if both variants loaded)
# ═════════════════════════════════════════════
if len(variant_data) == 2:
    v_names = list(variant_data.keys())
    d0, d1  = variant_data[v_names[0]], variant_data[v_names[1]]

    # Find common features
    common = list(set(d0["names"]) & set(d1["names"]))
    common.sort()

    if common:
        imp0 = np.array([d0["importances"][d0["names"].index(f)] for f in common])
        imp1 = np.array([d1["importances"][d1["names"].index(f)] for f in common])

        x     = np.arange(len(common))
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(14, len(common) * 0.6), 6))
        bars0 = ax.bar(x - width/2, imp0, width, label=v_names[0],
                       color=colors[v_names[0]], alpha=0.85, edgecolor="white")
        bars1 = ax.bar(x + width/2, imp1, width, label=v_names[1],
                       color=colors[v_names[1]], alpha=0.85, edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels(common, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Normalized Importance")
        ax.set_title("Feature Importance Comparison — yolov8m vs yolov8l\n"
                     "(common selected features only)",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        path = os.path.join(OUT_DIR, "comparison_m_vs_l.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"📊 Comparison chart saved: {path}")

        # Print agreement score
        corr = np.corrcoef(imp0, imp1)[0, 1]
        print(f"\n📈 Model agreement (Pearson correlation): {corr:.4f}")
        if corr > 0.9:
            print("   ✅ Models strongly agree on feature importance — good sign!")
        elif corr > 0.7:
            print("   ⚠️  Models moderately agree — some difference in what they learned")
        else:
            print("   ❌ Models disagree significantly — consider using the ensemble")

print(f"\n✅ All charts saved to: {OUT_DIR}")
print("   radar_yolov8m.png")
print("   radar_yolov8l.png")
print("   bar_top_features.png")
print("   comparison_m_vs_l.png")