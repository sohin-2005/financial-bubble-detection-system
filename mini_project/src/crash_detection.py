"""
Market Crash Detection System
Using MACD signals + Random Forest + XGBoost ensemble
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb

warnings.filterwarnings("ignore")


def compute_macd(close: pd.Series,
                 fast: int = 12,
                 slow: int = 26,
                 signal: int = 9) -> pd.DataFrame:
    """Compute MACD line, signal line, histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    })


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    macd_df = compute_macd(close)
    df = df.copy()
    df = pd.concat([df, macd_df], axis=1)

    df["histogram_slope"] = df["histogram"].diff(3) / 3
    df["histogram_accel"] = df["histogram_slope"].diff()

    prev_hist = df["histogram"].shift(1)
    df["crossover"] = 0
    df.loc[(prev_hist >= 0) & (df["histogram"] < 0), "crossover"] = -1
    df.loc[(prev_hist <= 0) & (df["histogram"] > 0), "crossover"] = 1

    df["macd_signal_gap"] = df["macd"] - df["signal"]

    in_bear = (df["macd"] < 0).astype(int)
    groups = (in_bear != in_bear.shift()).cumsum()
    df["bear_zone_duration"] = in_bear.groupby(groups).cumsum()

    price_high = close.rolling(5).max()
    macd_high = df["macd"].rolling(5).max()
    df["bearish_divergence"] = (
        (close >= price_high) & (df["macd"] < macd_high)
    ).astype(int)

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    df["volatility_10"] = close.pct_change().rolling(10).std()

    df["return_20d"] = close.pct_change(20).shift(-20)
    df["crash_label"] = (df["return_20d"] < -0.10).astype(int)

    return df.dropna()


FEATURE_COLS = [
    "macd", "signal", "histogram",
    "histogram_slope", "histogram_accel",
    "crossover", "macd_signal_gap",
    "bear_zone_duration", "bearish_divergence",
    "rsi_14", "volatility_10",
]


def prepare_data(df: pd.DataFrame):
    X = df[FEATURE_COLS]
    y = df["crash_label"]
    return train_test_split(X, y, test_size=0.2, shuffle=False)


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        max_features="sqrt",
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf


def train_xgboost(X_train, y_train) -> xgb.XGBClassifier:
    scale = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb_model = xgb.XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        verbose=False,
    )
    return xgb_model


def ensemble_predict(rf: RandomForestClassifier,
                     xgb_model: xgb.XGBClassifier,
                     X,
                     rf_weight: float = 0.45,
                     xgb_weight: float = 0.55) -> np.ndarray:
    rf_prob = rf.predict_proba(X)[:, 1]
    xgb_prob = xgb_model.predict_proba(X)[:, 1]
    return rf_weight * rf_prob + xgb_weight * xgb_prob


def classify(prob: np.ndarray, threshold: float = 0.55) -> np.ndarray:
    return (prob >= threshold).astype(int)


def evaluate(y_true, probs, threshold=0.55):
    preds = classify(probs, threshold)
    print("=" * 55)
    print(f"  Crash Detection Report  (threshold={threshold})")
    print("=" * 55)
    print(classification_report(y_true, preds,
                                target_names=["No crash", "Crash"],
                                digits=3))
    auc = roc_auc_score(y_true, probs)
    print(f"  ROC-AUC: {auc:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    print("Crash detection module ready. Import functions from src.crash_detection.")
