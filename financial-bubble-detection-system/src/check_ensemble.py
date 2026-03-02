"""
DIAGNOSTIC SCRIPT
==================
Run this to check whether your ensemble is working correctly.
It tests every component and tells you exactly what to fix.

Run: python check_ensemble.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd

sys.path.append(".")

print("=" * 60)
print("  ENSEMBLE HEALTH CHECK")
print("=" * 60)

all_ok = True

# ── Check 1: Required files exist ─────────────────────────────────────────────
print("\n[1/6] Checking required files...")

required_files = {
    "data/nifty50_labeled.csv": "Z-score labeled price data",
    "models/scaler.pkl": "Feature scaler",
    "models/feature_cols.pkl": "Feature column list",
    "models/stacking_ensemble.pkl": "Stacking ensemble model",
}

for path, desc in required_files.items():
    exists = os.path.exists(path)
    status = "✅" if exists else "❌"
    print(f"  {status}  {path:40s}  ({desc})")
    if not exists:
        all_ok = False

if not all_ok:
    print("\n  Some files are missing. Fix:")
    if not os.path.exists("data/nifty50_labeled.csv"):
        print("  → python src/data_ingestion.py")
        print("  → python src/zscore_labeling.py")
    if not os.path.exists("models/stacking_ensemble.pkl"):
        print("  → python src/stacking_ensemble.py")
    print("\n  Run those commands, then re-run this check.")
    sys.exit(1)

print("  All required files present ✅")

# ── Check 2: Load data and verify label distribution ──────────────────────────
print("\n[2/6] Checking label distribution...")
df = pd.read_csv("data/nifty50_labeled.csv", index_col=0, parse_dates=True)
counts = df["label"].value_counts()
total = len(df)

for label in ["Normal", "Bubble", "Crash"]:
    n = counts.get(label, 0)
    pct = n / total * 100
    bar = "█" * int(pct / 2)
    print(f"  {label:7s}: {n:5d}  ({pct:5.1f}%)  {bar}")

crash_count = counts.get("Crash", 0)
if crash_count < 20:
    print("  ⚠️  Very few Crash samples — ADASYN may fail. Try period='10y'")
else:
    print("  Distribution looks OK for ADASYN ✅")

# ── Check 3: Load and inspect ensemble ────────────────────────────────────────
print("\n[3/6] Loading stacking ensemble...")
try:
    # required for pickle to resolve the class
    from src.stacking_ensemble import StackingEnsemble
    with open("models/stacking_ensemble.pkl", "rb") as f:
        ensemble = pickle.load(f)
    print(f"  Ensemble type: {type(ensemble).__name__}")
    print(f"  is_fitted:     {getattr(ensemble, 'is_fitted', 'ATTR MISSING')}")

    if hasattr(ensemble, "fitted_bases"):
        print(
            f"  Base models:   {[name for name, _ in ensemble.fitted_bases]}")
    else:
        print("  ❌ fitted_bases attribute missing — old ensemble format")
        print("     Re-run: python src/stacking_ensemble.py")
        all_ok = False

    if hasattr(ensemble, "meta_model"):
        print(f"  Meta-model:    {type(ensemble.meta_model).__name__}")
    else:
        print("  ❌ meta_model missing")
        all_ok = False

except Exception as e:
    print(f"  ❌ Failed to load ensemble: {e}")
    print("     Re-run: python src/stacking_ensemble.py")
    all_ok = False

# ── Check 4: Test prediction on real data ─────────────────────────────────────
print("\n[4/6] Testing ensemble prediction...")

try:
    from sklearn.preprocessing import StandardScaler
    from src.ml_models import prepare_features

    with open("models/scaler.pkl",       "rb") as f:
        scaler = pickle.load(f)
    with open("models/feature_cols.pkl", "rb") as f:
        feat = pickle.load(f)

    df_ml, _ = prepare_features(df)
    X = df_ml[feat].values
    y = df_ml["label_numeric"].values.astype(int)
    X_sc = scaler.transform(X)

    # Test on last 100 rows
    X_sample = X_sc[-100:]
    y_sample = y[-100:]
    preds = ensemble.predict(X_sample)
    proba = ensemble.predict_proba(X_sample)

    print(f"  Tested on last 100 rows")
    print(f"  Prediction shape: {preds.shape}")
    print(f"  Proba shape:      {proba.shape}")
    print(f"  Unique predictions: {np.unique(preds, return_counts=True)}")
    print(f"  Proba sum check:  min={proba.sum(axis=1).min():.4f}  "
          f"max={proba.sum(axis=1).max():.4f}  (should be ~1.0)")

    if proba.shape[1] != 3:
        print("  ❌ Proba should have 3 columns (Normal/Bubble/Crash)")
        all_ok = False
    elif not np.allclose(proba.sum(axis=1), 1.0, atol=0.01):
        print("  ❌ Probabilities don't sum to 1 — model may be corrupt")
        all_ok = False
    else:
        print("  Prediction test passed ✅")

except Exception as e:
    print(f"  ❌ Prediction test failed: {e}")
    all_ok = False

# ── Check 5: Evaluate on held-out test set ────────────────────────────────────
print("\n[5/6] Evaluating on 20% held-out test set...")
try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, classification_report

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sc, y, test_size=0.2, random_state=42, stratify=y)

    y_pred = ensemble.predict(X_te)
    f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)

    print(classification_report(y_te, y_pred,
          target_names=["Normal", "Bubble", "Crash"],
          zero_division=0, digits=3))
    print(f"  Macro F1: {f1:.4f}", end="  ")

    if f1 >= 0.85:
        print("✅ Meets project requirement")
    elif f1 >= 0.70:
        print("⚠️  Below target of 0.85 — see fixes below")
    else:
        print("❌ Too low — check ADASYN + data size")

    # Per-class recall
    from sklearn.metrics import recall_score
    recalls = recall_score(y_te, y_pred, average=None, zero_division=0)
    labels = ["Normal", "Bubble", "Crash"]
    for lbl, rec in zip(labels, recalls):
        icon = "✅" if rec >= 0.7 else "⚠️ "
        print(f"  {icon} {lbl:7s} recall: {rec:.3f}")

except Exception as e:
    print(f"  ❌ Evaluation failed: {e}")
    all_ok = False

# ── Check 6: ADASYN test ─────────────────────────────────────────────────────
print("\n[6/6] Testing ADASYN on current data...")
try:
    from imblearn.over_sampling import ADASYN
    from sklearn.model_selection import train_test_split

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sc, y, test_size=0.2, random_state=42, stratify=y)

    adasyn = ADASYN(sampling_strategy="not majority",
                    random_state=42, n_neighbors=5)
    X_res, y_res = adasyn.fit_resample(X_tr, y_tr)
    unique, counts = np.unique(y_res, return_counts=True)
    label_names = {0: "Normal", 1: "Bubble", 2: "Crash"}

    print("  ADASYN works! Balanced distribution:")
    for u, c in zip(unique, counts):
        print(f"    {label_names[u]:7s}: {c}")

except Exception as e:
    print(f"  ⚠️  ADASYN issue: {e}")
    print("  This is usually OK — SMOTE fallback will be used")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
if all_ok:
    print("  ✅ ALL CHECKS PASSED — your ensemble is working correctly!")
    print("     Run: streamlit run app.py")
else:
    print("  ❌ SOME CHECKS FAILED — follow the fix instructions above")
    print("\n  Quick fix sequence:")
    print("    python src/data_ingestion.py")
    print("    python src/zscore_labeling.py")
    print("    python src/ml_models.py")
    print("    python src/stacking_ensemble.py")
    print("    python check_ensemble.py   ← run this again to verify")
print("="*60)
