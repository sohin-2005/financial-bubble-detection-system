"""
ML MODELS MODULE  —  with ADASYN fully integrated
===================================================
WHY ADASYN:
  Your data: 86% Normal, 9.4% Bubble, 4.5% Crash.
  Without ADASYN the model just predicts "Normal" always → 86% accuracy but useless.
  ADASYN creates synthetic minority samples so model learns Bubble and Crash properly.

HOW ADASYN WORKS:
  For each minority sample (Bubble/Crash):
    1. Count how many of its K nearest neighbours are Normal (majority)
    2. Assign more weight to samples surrounded by many Normal neighbours (hard cases)
    3. Generate MORE synthetic samples near those hard cases
  Result: model focuses on hardest-to-learn regions.

Run standalone: python src/ml_models.py
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import xgboost as xgb
from imblearn.over_sampling import ADASYN, SMOTE

warnings.filterwarnings("ignore")

ALL_FEATURES = [
    # ── Technical / Statistical ──────────────────────────────
    "zscore", "rolling_mean", "rolling_std",
    "log_return", "roc", "bb_width", "atr",
    "rsi", "macd", "macd_signal", "macd_diff",
    # ── VIX-based historical sentiment (10 years) ────────────
    "vix", "vix_zscore", "vix_sentiment", "vix_percentile",
    "price_momentum_sent", "rsi_sentiment",
    "composite_sentiment", "sentiment_momentum", "fear_spike",
    # ── NewsAPI sentiment (last 30 days only) ─────────────────
    "avg_polarity", "avg_positive", "avg_negative",
    "pct_positive", "pct_negative", "news_count",
]


# ── 1. Feature prep ───────────────────────────────────────────────────────────

def prepare_features(df):
    available = [c for c in ALL_FEATURES if c in df.columns]
    df_ml = df[available + ["label_numeric"]].copy()
    before = len(df_ml)
    df_ml.dropna(inplace=True)
    dropped = before - len(df_ml)
    if dropped:
        print(f"  Dropped {dropped} rows with NaN")
    print(f"  samples={len(df_ml)}  features={len(available)}")
    print(f"  {available}")
    return df_ml, available


# ── 2. ADASYN ─────────────────────────────────────────────────────────────────

def apply_adasyn(X_train, y_train, strategy="adasyn", random_state=42):
    """
    Fix class imbalance on TRAINING data only.
    Never apply to test data — that would fake your evaluation results.
    """
    label_names = {0: "Normal", 1: "Bubble", 2: "Crash"}

    print("\n" + "─"*50)
    print("  ADASYN — fixing class imbalance")
    print("─"*50)

    unique, counts = np.unique(y_train, return_counts=True)
    print("\n  BEFORE:")
    for u, c in zip(unique, counts):
        print(f"    {label_names.get(u, u):7s}: {c:5d}  {'█'*(c//20)}")

    try:
        sampler = (ADASYN(sampling_strategy="not majority",
                          random_state=random_state, n_neighbors=5)
                   if strategy == "adasyn"
                   else SMOTE(sampling_strategy="not majority",
                              random_state=random_state, k_neighbors=5))

        X_res, y_res = sampler.fit_resample(X_train, y_train)

        unique_r, counts_r = np.unique(y_res, return_counts=True)
        print(f"\n  AFTER {strategy.upper()}:")
        for u, c in zip(unique_r, counts_r):
            print(f"    {label_names.get(u, u):7s}: {c:5d}  {'█'*(c//20)}")
        print(f"\n  Added {len(X_res)-len(X_train)} synthetic samples")
        print(f"  Total training rows: {len(X_res)}")
        return X_res, y_res

    except ValueError as e:
        print(f"  ADASYN failed: {e}  → trying SMOTE k=3")
        try:
            X_res, y_res = SMOTE(sampling_strategy="not majority",
                                 random_state=random_state,
                                 k_neighbors=3).fit_resample(X_train, y_train)
            print(f"  SMOTE OK. Training rows: {len(X_res)}")
            return X_res, y_res
        except Exception as e2:
            print(f"  SMOTE failed too: {e2}  → using class_weight=balanced")
            return X_train, y_train


# ── 3. Train models ───────────────────────────────────────────────────────────

def train_random_forest(X_train, y_train):
    print("\n  Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        min_samples_split=10, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    print("  RF trained")
    return rf


def train_xgboost(X_train, y_train):
    print("\n  Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="mlogloss", use_label_encoder=False,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("  XGBoost trained")
    return model


# ── 4. Evaluation ─────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name, show_confusion=True):
    """
    Precision = of days predicted Bubble, how many were actually Bubble?
    Recall    = of all actual Bubble days, how many did we catch?
    F1        = harmonic mean of precision and recall
    Macro F1  = average F1 across all 3 classes  ← use this in your report
    """
    y_pred = model.predict(X_test)

    print(f"\n{'='*55}\n  {model_name}\n{'='*55}")
    print(classification_report(y_test, y_pred,
          target_names=["Normal", "Bubble", "Crash"],
          zero_division=0, digits=4))

    if show_confusion:
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(pd.DataFrame(cm,
              index=["Act Normal", "Act Bubble", "Act Crash"],
              columns=["Pred Normal", "Pred Bubble", "Pred Crash"]))

        if cm.shape == (3, 3):
            br = cm[1, 1]/cm[1, :].sum() if cm[1, :].sum() > 0 else 0
            cr = cm[2, 2]/cm[2, :].sum() if cm[2, :].sum() > 0 else 0
            print(f"\n  Bubble detection rate: {br*100:.1f}%")
            print(f"  Crash  detection rate: {cr*100:.1f}%")
            if br < 0.5:
                print("  LOW bubble recall — more ADASYN or lower Z threshold")
            if cr < 0.5:
                print("  LOW crash recall  — crash class still underrepresented")

    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    status = ("✅ meets target" if f1 >= 0.85
              else "⚠️  below target" if f1 >= 0.70
              else "❌ too low")
    print(f"\n  Macro F1: {f1:.4f}  {status}")
    return f1


# ── 5. Save / Load ────────────────────────────────────────────────────────────

def save_model(obj, filename):
    os.makedirs("models", exist_ok=True)
    with open(f"models/{filename}", "wb") as f:
        pickle.dump(obj, f)
    print(f"  Saved → models/{filename}")


def load_model(filename):
    path = f"models/{filename}"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run main.py first.")
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.append(".")

    # Use sentiment-enriched data if available, else fall back
    sentiment_path = "data/nifty50_with_sentiment.csv"
    labeled_path = "data/nifty50_labeled.csv"

    if os.path.exists(sentiment_path):
        data_path = sentiment_path
        print("  📊 Using sentiment-enriched dataset")
    elif os.path.exists(labeled_path):
        data_path = labeled_path
        print("  ⚠️  Sentiment data not found — using labeled data only")
        print("     Run: python3 src/historical_sentiment.py  to add VIX sentiment")
    else:
        print("Run data_ingestion.py and zscore_labeling.py first")
        sys.exit(1)

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df_ml, feat = prepare_features(df)
    X = df_ml[feat].values
    y = df_ml["label_numeric"].values.astype(int)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sc, y, test_size=0.2, random_state=42, stratify=y)

    X_tr, y_tr = apply_adasyn(X_tr, y_tr)    # ← ADASYN only on train

    rf = train_random_forest(X_tr, y_tr)
    xgb_m = train_xgboost(X_tr, y_tr)

    rf_f1 = evaluate_model(rf,    X_te, y_te, "Random Forest")
    xgb_f1 = evaluate_model(xgb_m, X_te, y_te, "XGBoost")

    save_model(scaler, "scaler.pkl")
    save_model(feat,   "feature_cols.pkl")
    save_model(rf,     "random_forest.pkl")
    save_model(xgb_m,  "xgboost.pkl")

    print(f"\nDone. RF={rf_f1:.4f}  XGB={xgb_f1:.4f}")
    print("Next: python src/stacking_ensemble.py")
