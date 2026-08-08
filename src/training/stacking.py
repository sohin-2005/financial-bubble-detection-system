import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ================= CONFIG =================
FEATURES_RF = [
    "zscore_value",
    "log_return",
    "psy_12",
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "realized_vol_10d",
    "realized_vol_63d",
    "vol_ratio",
    "drawdown",
    "trend_21d",
    "price_accel",
    "ret_skew_21d",
    "ret_kurt_21d",
    "vol_zscore",
    "gdp_growth",
    "cpi_inflation",
    "repo_rate",
]

FEATURES_XGB = [
    "zscore_value",
    "log_return",
    "psy_12",
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "gdp_growth",
    "cpi_inflation",
    "repo_rate",
    "daily_sentiment_index",
]

CLASSES = np.array(["Bubble", "Crash", "Normal"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
EXPORTS_DIR = DATA_DIR / "exports"
MODELS_DIR = DATA_DIR / "models"

# ================= HELPERS =================


def proba_fixed_order(clf, X):
    proba = np.zeros((len(X), len(CLASSES)))
    clf_proba = clf.predict_proba(X)

    for j, cls in enumerate(clf.classes_):
        proba[:, int(cls)] = clf_proba[:, j]

    return proba


def build_meta_features(rf_probs, xgb_probs):
    return np.hstack([rf_probs, xgb_probs])


def compute_technical_features(df):
    """Compute missing technical features."""
    df = df.sort_values("date").reset_index(drop=True)
    if "psy_12" not in df.columns:
        up_days = (df["log_return"].fillna(0) > 0).astype(float)
        df["psy_12"] = up_days.rolling(12, min_periods=1).mean() * 100.0
    if "realized_vol_10d" not in df.columns:
        df["realized_vol_10d"] = df["log_return"].rolling(
            10, min_periods=1).std() * np.sqrt(252)
    if "realized_vol_63d" not in df.columns:
        df["realized_vol_63d"] = df["log_return"].rolling(
            63, min_periods=1).std() * np.sqrt(252)
    if "vol_ratio" not in df.columns:
        df["vol_ratio"] = df["realized_vol_10d"] / \
            (df["realized_vol_63d"] + 1e-6)
    if "drawdown" not in df.columns:
        cum_returns = (1 + df["log_return"].fillna(0)).cumprod()
        running_max = cum_returns.expanding().max()
        df["drawdown"] = (cum_returns - running_max) / (running_max + 1e-6)
    if "trend_21d" not in df.columns:
        df["trend_21d"] = df["log_return"].rolling(21, min_periods=1).sum()
    if "price_accel" not in df.columns:
        df["price_accel"] = df["log_return"].rolling(5, min_periods=1).mean()
    if "ret_skew_21d" not in df.columns:
        df["ret_skew_21d"] = df["log_return"].rolling(21, min_periods=1).skew()
    if "ret_kurt_21d" not in df.columns:
        df["ret_kurt_21d"] = df["log_return"].rolling(21, min_periods=1).apply(
            lambda x: x.kurtosis() if len(x) > 1 else 0, raw=False)
    if "vol_zscore" not in df.columns:
        vol_63_mean = df["realized_vol_63d"].rolling(63, min_periods=1).mean()
        vol_63_std = df["realized_vol_63d"].rolling(63, min_periods=1).std()
        df["vol_zscore"] = (df["realized_vol_63d"] -
                            vol_63_mean) / (vol_63_std + 1e-6)
    return df.fillna(0.0)


# ================= LOAD DATA =================
def load_data():
    train_df = pd.read_csv(EXPORTS_DIR / "NIFTY50_train.csv")
    val_df = pd.read_csv(EXPORTS_DIR / "NIFTY50_validation.csv")
    test_df = pd.read_csv(EXPORTS_DIR / "NIFTY50_test.csv")

    sent_train = pd.read_csv(EXPORTS_DIR / "nifty50_daily_sentiment.csv")
    sent_val = pd.read_csv(EXPORTS_DIR / "nifty50_validation_sentiment.csv")
    sent_test = pd.read_csv(EXPORTS_DIR / "nifty50_test_sentiment.csv")

    for df in [train_df, val_df, test_df, sent_train, sent_val, sent_test]:
        df["date"] = pd.to_datetime(df["date"])

    train_df = train_df.merge(sent_train, on="date", how="left")
    val_df = val_df.merge(sent_val, on="date", how="left")
    test_df = test_df.merge(sent_test, on="date", how="left")

    for df in [train_df, val_df, test_df]:
        if "polarity_score" in df.columns:
            df.rename(
                columns={"polarity_score": "daily_sentiment_index"}, inplace=True)
        df["daily_sentiment_index"] = df["daily_sentiment_index"].fillna(0.0)

    train_df = compute_technical_features(train_df)
    val_df = compute_technical_features(val_df)
    test_df = compute_technical_features(test_df)

    all_features = list(set(FEATURES_RF + FEATURES_XGB))
    train_df = train_df.dropna(subset=all_features + ["label"])
    val_df = val_df.dropna(subset=all_features + ["label"])
    test_df = test_df.dropna(subset=all_features + ["label"])

    return train_df, val_df, test_df


# ================= MAIN =================
if __name__ == "__main__":
    print("=" * 60)
    print("K-FOLD STACKING (LOGISTIC REGRESSION)")
    print("=" * 60)

    train_df, val_df, test_df = load_data()

    X_rf = train_df[FEATURES_RF].values
    X_rf_test = test_df[FEATURES_RF].values
    X_xgb = train_df[FEATURES_XGB].values
    X_xgb_test = test_df[FEATURES_XGB].values

    le = joblib.load(MODELS_DIR / "label_encoder.pkl")
    y = le.transform(train_df["label"])
    y_test = le.transform(test_df["label"])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    meta_features = []
    meta_labels = []

    print("\nGenerating meta features...")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_rf)):
        print(f"  Fold {fold+1}")

        X_rf_train, X_rf_val = X_rf[train_idx], X_rf[val_idx]
        X_xgb_train, X_xgb_val = X_xgb[train_idx], X_xgb[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        rf = joblib.load(MODELS_DIR / "rf_model.pkl")
        xgb = joblib.load(MODELS_DIR / "xgb_model.pkl")

        if hasattr(xgb, 'early_stopping_rounds'):
            xgb.early_stopping_rounds = None

        rf.fit(X_rf_train, y_train)
        xgb.fit(X_xgb_train, y_train)

        rf_probs = proba_fixed_order(rf, X_rf_val)
        xgb_probs = proba_fixed_order(xgb, X_xgb_val)

        meta = build_meta_features(rf_probs, xgb_probs)
        meta_features.append(meta)
        meta_labels.append(y_val)

    meta_X = np.vstack(meta_features)
    meta_y = np.concatenate(meta_labels)

    print(f"\nMeta dataset shape: {meta_X.shape}")

    # ================= META MODEL =================
    scaler = StandardScaler()
    meta_scaled = scaler.fit_transform(meta_X)

    meta_model = LogisticRegression(max_iter=2000)
    meta_model.fit(meta_scaled, meta_y)

    print("Meta model trained ✅")

    # ================= TEST =================
    rf_model = joblib.load(MODELS_DIR / "rf_model.pkl")
    xgb_model = joblib.load(MODELS_DIR / "xgb_model.pkl")

    rf_probs = proba_fixed_order(rf_model, X_rf_test)
    xgb_probs = proba_fixed_order(xgb_model, X_xgb_test)

    meta_test = build_meta_features(rf_probs, xgb_probs)
    meta_test_scaled = scaler.transform(meta_test)

    preds = meta_model.predict(meta_test_scaled)

    f1 = f1_score(y_test, preds, average="macro")

    print(f"\nK-Fold Stacking Test F1: {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=le.classes_))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, preds)
    print(pd.DataFrame(
        cm,
        index=[f"Actual {c}" for c in le.classes_],
        columns=[f"Pred {c}" for c in le.classes_]
    ))

    # ================= SAVE MODELS =================
    print("\n" + "=" * 60)
    print("SAVING MODELS...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(meta_model, MODELS_DIR / "stacking_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "stacking_scaler.pkl")
    stats = {
        "f1_score": float(f1),
        "n_features_rf": len(FEATURES_RF),
        "n_features_xgb": len(FEATURES_XGB),
        "meta_features": 6,
        "test_samples": len(y_test),
    }
    joblib.dump(stats, MODELS_DIR / "stacking_stats.pkl")
    print("✅ Models saved to data/models/")
    print(f"   - stacking_model.pkl")
    print(f"   - stacking_scaler.pkl")
    print(f"   - stacking_stats.pkl (F1: {f1:.4f})")
    print("=" * 60)
