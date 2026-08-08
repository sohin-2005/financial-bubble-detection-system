"""
Market Crash Detection System
Using MACD signals + Random Forest + XGBoost ensemble
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. MACD FEATURE EXTRACTION
# ─────────────────────────────────────────────

def compute_macd(close: pd.Series,
                 fast: int = 12,
                 slow: int = 26,
                 signal: int = 9) -> pd.DataFrame:
    """Compute MACD line, signal line, histogram."""
    ema_fast   = close.ewm(span=fast, adjust=False).mean()
    ema_slow   = close.ewm(span=slow, adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return pd.DataFrame({
        "macd":      macd_line,
        "signal":    signal_line,
        "histogram": histogram,
    })


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer crash-predictive features from OHLCV + MACD.

    Features:
      - macd, signal, histogram (raw)
      - histogram_slope       : 3-day slope of histogram
      - histogram_accel       : change in slope (acceleration)
      - crossover             : -1 bearish cross, +1 bullish cross, 0 none
      - macd_signal_gap       : MACD minus signal (signed distance)
      - bear_zone_duration    : consecutive days MACD < 0
      - bearish_divergence    : price higher high but MACD lower high (5-day window)
      - rsi_14                : RSI for confluence
      - volatility_10         : 10-day rolling std of returns
      - return_5d             : 5-day forward return (used to build label)
    """
    close = df["close"]
    macd_df = compute_macd(close)
    df = df.copy()
    df = pd.concat([df, macd_df], axis=1)

    # Histogram slope & acceleration
    df["histogram_slope"] = df["histogram"].diff(3) / 3
    df["histogram_accel"] = df["histogram_slope"].diff()

    # Bearish/bullish crossover flag
    prev_hist = df["histogram"].shift(1)
    df["crossover"] = 0
    df.loc[(prev_hist >= 0) & (df["histogram"] < 0), "crossover"] = -1  # bearish
    df.loc[(prev_hist <= 0) & (df["histogram"] > 0), "crossover"] =  1  # bullish

    # MACD-Signal gap
    df["macd_signal_gap"] = df["macd"] - df["signal"]

    # Bear zone duration (MACD below zero)
    in_bear = (df["macd"] < 0).astype(int)
    groups  = (in_bear != in_bear.shift()).cumsum()
    df["bear_zone_duration"] = in_bear.groupby(groups).cumsum()

    # Bearish divergence: price makes 5-day high but MACD does not
    price_high = close.rolling(5).max()
    macd_high  = df["macd"].rolling(5).max()
    df["bearish_divergence"] = (
        (close >= price_high) & (df["macd"] < macd_high)
    ).astype(int)

    # RSI-14
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Volatility
    df["volatility_10"] = close.pct_change().rolling(10).std()

    # Target: did price drop > 10% in next 20 trading days?
    df["return_20d"] = close.pct_change(20).shift(-20)
    df["crash_label"] = (df["return_20d"] < -0.10).astype(int)

    return df.dropna()


# ─────────────────────────────────────────────
# 2. DATASET SPLIT
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# 3. RANDOM FOREST MODEL
# ─────────────────────────────────────────────

def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """
    Random Forest:
      - n_estimators=300   : 300 bagged decision trees
      - max_features='sqrt': random subset per split (feature randomisation)
      - class_weight       : compensate for crash rarity
      - max_depth=8        : prevents overfitting to noise
    """
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


def rf_feature_importance(rf: RandomForestClassifier,
                           feature_names: list) -> pd.Series:
    return pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)


# ─────────────────────────────────────────────
# 4. XGBOOST MODEL
# ─────────────────────────────────────────────

def train_xgboost(X_train, y_train) -> xgb.XGBClassifier:
    """
    XGBoost:
      - n_estimators=400   : sequential boosted trees
      - learning_rate=0.05 : shrinkage (slow, robust)
      - max_depth=5        : shallow trees to reduce overfitting
      - subsample=0.8      : row sampling per tree
      - colsample_bytree   : column sampling per tree
      - reg_alpha/lambda   : L1 + L2 regularization
      - scale_pos_weight   : handles class imbalance
    """
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


# ─────────────────────────────────────────────
# 5. ENSEMBLE
# ─────────────────────────────────────────────

def ensemble_predict(rf: RandomForestClassifier,
                     xgb_model: xgb.XGBClassifier,
                     X,
                     rf_weight: float = 0.45,
                     xgb_weight: float = 0.55) -> np.ndarray:
    """
    Weighted average of RF + XGBoost crash probabilities.
    XGBoost gets slightly higher weight: it better captures
    sequential momentum decay patterns in MACD data.
    """
    rf_prob  = rf.predict_proba(X)[:, 1]
    xgb_prob = xgb_model.predict_proba(X)[:, 1]
    return rf_weight * rf_prob + xgb_weight * xgb_prob


def classify(prob: np.ndarray, threshold: float = 0.55) -> np.ndarray:
    return (prob >= threshold).astype(int)


# ─────────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# 7. LIVE SCORING — single latest row
# ─────────────────────────────────────────────

def score_latest(rf, xgb_model, df: pd.DataFrame) -> dict:
    """
    Score the most recent candle and explain which MACD
    features contributed most to the crash signal.
    """
    latest = df[FEATURE_COLS].iloc[[-1]]
    prob   = ensemble_predict(rf, xgb_model, latest)[0]

    rf_imp  = rf_feature_importance(rf, FEATURE_COLS)
    top_drivers = rf_imp.head(3).index.tolist()

    return {
        "crash_probability": round(float(prob), 4),
        "alert":             prob >= 0.55,
        "risk_level":        "HIGH" if prob > 0.70 else ("MEDIUM" if prob > 0.45 else "LOW"),
        "top_macd_drivers":  top_drivers,
        "latest_signals": {
            "macd":               round(float(latest["macd"].iloc[0]), 4),
            "histogram":          round(float(latest["histogram"].iloc[0]), 4),
            "histogram_slope":    round(float(latest["histogram_slope"].iloc[0]), 4),
            "bearish_divergence": int(latest["bearish_divergence"].iloc[0]),
            "crossover":          int(latest["crossover"].iloc[0]),
            "bear_zone_days":     int(latest["bear_zone_duration"].iloc[0]),
        },
    }


# ─────────────────────────────────────────────
# 8. DEMO — synthetic data
# ─────────────────────────────────────────────

def generate_demo_data(n: int = 1200, seed: int = 0) -> pd.DataFrame:
    """Synthetic price series with embedded crash episodes."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0003, 0.012, n)

    # Inject 6 crash episodes
    for start in [150, 320, 500, 700, 850, 1050]:
        returns[start:start+25] = rng.normal(-0.025, 0.018, 25)

    close = pd.Series(np.cumprod(1 + returns) * 100, name="close")
    return pd.DataFrame({"close": close})


if __name__ == "__main__":
    # ── Build dataset ──────────────────────────────
    print("Building feature dataset...")
    raw   = generate_demo_data()
    data  = build_features(raw)
    print(f"  Samples: {len(data)}  |  Crash days: {data['crash_label'].sum()}")

    X_train, X_test, y_train, y_test = prepare_data(data)

    # ── Train models ───────────────────────────────
    print("\nTraining Random Forest...")
    rf_model = train_random_forest(X_train, y_train)

    print("Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)

    # ── Evaluate ───────────────────────────────────
    probs = ensemble_predict(rf_model, xgb_model, X_test)
    evaluate(y_test, probs)

    # ── Feature importance ─────────────────────────
    print("\nTop MACD feature importances (Random Forest):")
    fi = rf_feature_importance(rf_model, FEATURE_COLS)
    for feat, imp in fi.items():
        bar = "█" * int(imp * 60)
        print(f"  {feat:<25} {bar} {imp:.3f}")

    # ── Score latest candle ────────────────────────
    print("\nScoring latest candle...")
    result = score_latest(rf_model, xgb_model, data)
    print(f"\n  Crash probability : {result['crash_probability']:.2%}")
    print(f"  Risk level        : {result['risk_level']}")
    print(f"  Alert triggered   : {result['alert']}")
    print(f"  Top MACD drivers  : {', '.join(result['top_macd_drivers'])}")
    print(f"\n  Latest MACD signals:")
    for k, v in result["latest_signals"].items():
        print(f"    {k:<25} {v}")