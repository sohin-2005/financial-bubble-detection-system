import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import ADASYN
import joblib
import os

FEATURES = [
    "zscore_value",
    # Short term — daily signals
    "log_return",
    "psy_12",
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    # REMOVED: daily_sentiment_index — constant zero, zero variance, hurts trees
    # Regime detection features
    "realized_vol_10d",
    "realized_vol_63d",
    "vol_ratio",
    "drawdown",
    "trend_21d",
    "price_accel",
    "ret_skew_21d",
    "ret_kurt_21d",
    "vol_zscore",
    # Long term — macro signals
    "gdp_growth",
    "cpi_inflation",
    "repo_rate",
]

OUTPUT_DIR = "../../data/models"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    print("[1/5] Loading data...")

    train_df = pd.read_csv("../../data/exports/NIFTY50_train.csv")
    val_df = pd.read_csv("../../data/exports/NIFTY50_validation.csv")
    test_df = pd.read_csv("../../data/exports/NIFTY50_test.csv")

    try:
        sent_train = pd.read_csv(
            "../../data/exports/nifty50_daily_sentiment.csv")
        sent_val = pd.read_csv(
            "../../data/exports/nifty50_validation_sentiment.csv")
        sent_test = pd.read_csv(
            "../../data/exports/nifty50_test_sentiment.csv")
        has_sentiment = True
    except FileNotFoundError:
        print("  Warning: Sentiment files not found — using default daily_sentiment_index = 0.0")
        has_sentiment = False

    for df in [train_df, val_df, test_df]:
        df["date"] = pd.to_datetime(df["date"])

    if has_sentiment:
        for df in [sent_train, sent_val, sent_test]:
            df["date"] = pd.to_datetime(df["date"])
        train_df = train_df.merge(sent_train, on="date", how="left")
        val_df = val_df.merge(sent_val, on="date", how="left")
        test_df = test_df.merge(sent_test, on="date", how="left")

    for df in [train_df, val_df, test_df]:
        if "polarity_score" in df.columns:
            df.rename(
                columns={"polarity_score": "daily_sentiment_index"}, inplace=True)

    for df in [train_df, val_df, test_df]:
        if "daily_sentiment_index" not in df.columns:
            df["daily_sentiment_index"] = 0.0
        else:
            df["daily_sentiment_index"] = df["daily_sentiment_index"].fillna(
                0.0)

    # Compute missing regime features if not present in exports
    def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
        close_col = "close" if "close" in df.columns else "Close" if "Close" in df.columns else None
        if close_col is None:
            return df

        close = df[close_col]
        if "log_return" not in df.columns:
            df["log_return"] = np.log(close / close.shift(1))

        if "psy_12" not in df.columns:
            up_days = (close.diff() > 0).astype(float)
            df["psy_12"] = up_days.rolling(12).mean() * 100.0

        if "realized_vol_10d" not in df.columns:
            df["realized_vol_10d"] = df["log_return"].rolling(
                10).std() * np.sqrt(252)
        if "realized_vol_63d" not in df.columns:
            df["realized_vol_63d"] = df["log_return"].rolling(
                63).std() * np.sqrt(252)
        if "vol_ratio" not in df.columns:
            df["vol_ratio"] = df["realized_vol_10d"] / \
                (df["realized_vol_63d"] + 1e-8)

        if "drawdown" not in df.columns:
            rolling_max = close.rolling(252, min_periods=1).max()
            df["drawdown"] = (close - rolling_max) / rolling_max

        if "trend_21d" not in df.columns:
            df["trend_21d"] = (
                (close - close.rolling(21).mean())
                / (close.rolling(21).std() + 1e-8)
            )

        if "price_accel" not in df.columns:
            ret_21d = np.log(close / close.shift(21))
            df["price_accel"] = ret_21d - ret_21d.shift(21)

        if "ret_skew_21d" not in df.columns:
            df["ret_skew_21d"] = df["log_return"].rolling(21).skew()
        if "ret_kurt_21d" not in df.columns:
            df["ret_kurt_21d"] = df["log_return"].rolling(21).kurt()

        if "vol_zscore" not in df.columns:
            vol = df["realized_vol_10d"]
            df["vol_zscore"] = (vol - vol.rolling(252).mean()) / \
                (vol.rolling(252).std() + 1e-8)

        return df

    train_df = add_regime_features(train_df)
    val_df = add_regime_features(val_df)
    test_df = add_regime_features(test_df)

    train_df = train_df.dropna(subset=FEATURES + ["label"])
    val_df = val_df.dropna(subset=FEATURES + ["label"])
    test_df = test_df.dropna(subset=FEATURES + ["label"])

    print(f"  Train rows: {len(train_df)}")
    print(f"  Val rows:   {len(val_df)}")
    print(f"  Test rows:  {len(test_df)}")

    return train_df, val_df, test_df


def encode_labels(train_df, val_df, test_df):
    print("\n[2/5] Encoding labels...")

    encoder_path = f"{OUTPUT_DIR}/label_encoder.pkl"
    if os.path.exists(encoder_path):
        le = joblib.load(encoder_path)
    else:
        le = LabelEncoder()
        le.fit(["Bubble", "Crash", "Normal"])
        joblib.dump(le, encoder_path)

    y_train = le.transform(train_df["label"])
    y_val = le.transform(val_df["label"])
    y_test = le.transform(test_df["label"])

    print(
        f"  Train distribution: {pd.Series(y_train).value_counts().to_dict()}")
    return y_train, y_val, y_test, le


def apply_adasyn(X_train, y_train):
    print("\n[3/5] Applying SMOTE oversampling...")

    from imblearn.over_sampling import SMOTE

    for label, count in zip(*np.unique(y_train, return_counts=True)):
        print(f"  Before — Class {label}: {count}")

    # Safe k: minority class must have at least k+1 samples
    min_class_count = int(min(
        count for count in np.unique(y_train, return_counts=True)[1]
    ))
    k = max(1, min(5, min_class_count - 1))
    print(f"  Using k_neighbors={k}")

    smote = SMOTE(
        sampling_strategy="not majority",  # upsample Bubble + Crash to match Normal
        k_neighbors=k,
        random_state=42
    )

    try:
        X_res, y_res = smote.fit_resample(X_train, y_train)
    except Exception as e:
        print(f"  SMOTE failed: {e} — using original data")
        return X_train, y_train

    for label, count in zip(*np.unique(y_res, return_counts=True)):
        print(f"  After  — Class {label}: {count}")

    return X_res, y_res


def train_random_forest(X_train, y_train, X_val, y_val):
    print("\n[4/5] Training Random Forest...")

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    val_f1 = f1_score(y_val, val_pred, average="macro")
    print(f"  Validation F1: {val_f1:.4f}")

    return model


def evaluate(model, X_val, y_val, X_test, y_test, le):
    print("\n[5/5] Evaluating...")

    val_pred = model.predict(X_val)
    val_f1 = f1_score(y_val, val_pred, average="macro")

    test_pred = model.predict(X_test)
    test_f1 = f1_score(y_test, test_pred, average="macro")

    print(f"\n  Validation F1: {val_f1:.4f}")
    print(f"  Test F1:       {test_f1:.4f}")

    if test_f1 >= 0.85:
        print("  ✅ Meets NFR1 requirement")
    else:
        print("  ❌ Below NFR1 requirement")

    print("\n  Per-class breakdown (Test):")
    print(classification_report(y_test, test_pred, target_names=le.classes_))

    print("  Confusion Matrix (Test):")
    cm = confusion_matrix(y_test, test_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual {c}" for c in le.classes_],
        columns=[f"Predicted {c}" for c in le.classes_]
    )
    print(cm_df)

    return val_f1, test_f1


if __name__ == "__main__":
    print("=" * 60)
    print("RANDOM FOREST TRAINING PIPELINE")
    print("=" * 60)

    train_df, val_df, test_df = load_data()

    X_train = train_df[FEATURES].values
    X_val = val_df[FEATURES].values
    X_test = test_df[FEATURES].values

    y_train, y_val, y_test, le = encode_labels(train_df, val_df, test_df)

    X_train_res, y_train_res = apply_adasyn(X_train, y_train)

    model = train_random_forest(X_train_res, y_train_res, X_val, y_val)

    val_f1, test_f1 = evaluate(model, X_val, y_val, X_test, y_test, le)

    joblib.dump(model, f"{OUTPUT_DIR}/rf_model.pkl")
    print(f"\n  Saved: rf_model.pkl")

    import json
    stats = {
        "model": "RandomForest",
        "val_f1": round(val_f1, 4),
        "test_f1": round(test_f1, 4),
        "features": FEATURES,
    }
    with open(f"{OUTPUT_DIR}/rf_model_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 60)
    print("RANDOM FOREST TRAINING COMPLETE")
    print("=" * 60)
