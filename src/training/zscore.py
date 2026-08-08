"""
zscore.py — PSY Bubble Detection + Technical Indicators
Replaces old Z-score threshold labeling with PSY/BSADF procedure.
Technical indicators (RSI, MACD) remain as features — unchanged.
"""

import pandas as pd
import numpy as np
import ta
from database import get_engine


# ══════════════════════════════════════════════════════════════════
# PSY LABELING — replaces old Z-score threshold
# ══════════════════════════════════════════════════════════════════

def compute_bsadf(prices: pd.Series,
                  min_window: int = 157,   # changed from 20
                  max_lag:    int = 4) -> pd.Series:
    """
    Compute BSADF (Backward Sup ADF) statistic for each day.
    This is the core of the PSY procedure.

    For each day t:
      - Looks at all sub-windows ending at t
      - Runs ADF regression on each sub-window
      - Takes the supremum (maximum) ADF statistic → BSADF score

    BSADF score interpretation:
      High positive → explosive upward growth  → Bubble
      Low negative  → explosive downward crash → Crash
      Near zero     → normal random walk       → Normal

    Args:
        prices:     pd.Series of log prices (daily closing)
        min_window: minimum sub-window length (default 20 days)

    Returns:
        pd.Series of BSADF scores, same index as prices
    """

    n = len(prices)
    bsadf_scores = np.full(n, np.nan)
    for t in range(min_window, n):
        max_adf = -np.inf

        for start in range(0, t - min_window + 1):
            window = prices.iloc[start:t+1].values

            if len(window) < max_lag + 5:
                continue

            y = window
            dy = np.diff(y)
            y_lag = y[:-1]

            # Choose best lag by BIC
            best_bic = np.inf
            best_t = np.nan

            for k in range(max_lag + 1):
                try:
                    # FIX 1: constant only, NO trend term
                    # FIX 2: include k lagged differences
                    if k == 0:
                        X = np.column_stack([
                            np.ones(len(y_lag)),
                            y_lag
                        ])
                        y_dep = dy
                    else:
                        lag_diffs = np.column_stack([
                            dy[k-i:-i] if i > 0 else dy[k:]
                            for i in range(1, k+1)
                        ])
                        X = np.column_stack([
                            np.ones(len(y_lag[k:])),
                            y_lag[k:],
                            lag_diffs
                        ])
                        y_dep = dy[k:]

                    # OLS
                    XtX = X.T @ X
                    if np.linalg.det(XtX) == 0:
                        continue
                    beta = np.linalg.inv(XtX) @ X.T @ y_dep
                    residuals = y_dep - X @ beta
                    n_obs = len(y_dep)
                    n_params = X.shape[1]
                    sigma_sq = np.sum(residuals**2) / (n_obs - n_params)

                    # BIC to select lag
                    bic = n_obs * np.log(sigma_sq) + n_params * np.log(n_obs)

                    if bic < best_bic:
                        best_bic = bic
                        se_beta = np.sqrt(
                            sigma_sq * np.linalg.inv(XtX)[1, 1]
                        )
                        if se_beta > 0:
                            best_t = beta[1] / se_beta

                except Exception:
                    continue

            if not np.isnan(best_t) and best_t > max_adf:
                max_adf = best_t

        if max_adf != -np.inf:
            bsadf_scores[t] = max_adf

    return pd.Series(bsadf_scores, index=prices.index)


def apply_hybrid_labels(df: pd.DataFrame,
                        upper_pct:      float = 0.95,
                        crash_zscore:   float = -2.0,
                        bubble_zscore:  float = 2.0,   # ← NEW
                        rolling_window: int = 504) -> pd.DataFrame:

    print("  Applying hybrid PSY + Z-score labeling...")

    # Rolling threshold for PSY Bubble
    upper_bound = df["bsadf_score"].rolling(
        window=rolling_window, min_periods=252
    ).quantile(upper_pct)

    df["label"] = "Normal"

    # Z-score for Crash
    df["rolling_mean"] = df["close"].rolling(window=20).mean()
    df["rolling_std"] = df["close"].rolling(window=20).std()
    df["zscore_value"] = (
        (df["close"] - df["rolling_mean"]) / df["rolling_std"]
    )

    df.loc[
        (df["zscore_value"] < crash_zscore) &
        (df["label"] == "Normal"),
        "label"
    ] = "Crash"

    # PSY Bubble — only when price going UP
    df.loc[
        (df["bsadf_score"] >= upper_bound) &
        (df["log_return"] > 0) &
        (df["label"] == "Normal"),   # don't override Crash
        "label"
    ] = "Bubble"

    # Z-score Bubble — catches gradual bubbles PSY misses
    df.loc[
        (df["zscore_value"] > bubble_zscore) &
        (df["label"] == "Normal"),   # don't override PSY Bubble or Crash
        "label"
    ] = "Bubble"

    dist = df["label"].value_counts()
    total = len(df.dropna(subset=["bsadf_score"]))

    print(f"\n  Hybrid Label Distribution:")
    for label in ["Bubble", "Crash", "Normal"]:
        count = dist.get(label, 0)
        pct = 100 * count / total if total > 0 else 0
        print(f"    {label:8s}: {count:4d} days ({pct:.1f}%)")

    return df


def apply_psy_labels(df: pd.DataFrame,
                     rolling_window: int = 504,  # 2 trading years
                     upper_pct:      float = 0.95,
                     lower_pct:      float = 0.05,
                     min_window:     int = 157
                     ) -> pd.DataFrame:

    print("  Computing PSY/BSADF scores (corrected implementation)...")
    print(f"  Min window: {min_window} days (PSY r0 rule)")
    print(f"  Lag augmentation: up to k=4")
    print(f"  Rolling threshold window: {rolling_window} days")
    print("  (This will take 20-40 minutes...)")

    log_prices = np.log(df["close"])
    df["bsadf_score"] = compute_bsadf(
        log_prices,
        min_window=min_window,
        max_lag=4
    )

    # Rolling quantile thresholds — no look-ahead bias
    upper_bound = df["bsadf_score"].rolling(
        window=rolling_window,
        min_periods=252
    ).quantile(upper_pct)

    lower_bound = df["bsadf_score"].rolling(
        window=rolling_window,
        min_periods=252
    ).quantile(lower_pct)

    df["label"] = "Normal"
    df.loc[df["bsadf_score"] > upper_bound, "label"] = "Bubble"
    df.loc[df["bsadf_score"] < lower_bound, "label"] = "Crash"

    dist = df["label"].value_counts()
    total = len(df.dropna(subset=["bsadf_score"]))

    print(f"\n  PSY Label Distribution:")
    for label in ["Bubble", "Crash", "Normal"]:
        count = dist.get(label, 0)
        pct = 100 * count / total if total > 0 else 0
        print(f"    {label:8s}: {count:4d} days ({pct:.1f}%)")
    return df


# ══════════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS — updated with 8 new regime-detection features
# ══════════════════════════════════════════════════════════════════

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RSI, MACD, log return + 8 regime-detection features.
    All use only past data — no lookahead.
    """

    # --- Original features (unchanged) ---
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    df["rsi"] = ta.momentum.RSIIndicator(
        close=df["close"], window=14
    ).rsi()

    macd_indicator = ta.trend.MACD(
        close=df["close"], window_slow=26, window_fast=12, window_sign=9
    )
    df["macd"] = macd_indicator.macd()
    df["macd_signal"] = macd_indicator.macd_signal()
    df["macd_hist"] = macd_indicator.macd_diff()

    # --- NEW: Volatility regime features ---
    df["realized_vol_10d"] = df["log_return"].rolling(10).std() * np.sqrt(252)
    df["realized_vol_63d"] = df["log_return"].rolling(63).std() * np.sqrt(252)
    # Ratio > 1 means short-term vol spike (crash signal)
    df["vol_ratio"] = df["realized_vol_10d"] / (df["realized_vol_63d"] + 1e-8)

    # --- NEW: Drawdown (critical for crash detection) ---
    rolling_max = df["close"].rolling(252, min_periods=1).max()
    df["drawdown"] = (df["close"] - rolling_max) / rolling_max

    # --- NEW: Trend strength (bubble = sustained uptrend) ---
    df["trend_21d"] = (
        (df["close"] - df["close"].rolling(21).mean())
        / (df["close"].rolling(21).std() + 1e-8)
    )

    # --- NEW: Price acceleration (bubble blowoffs show positive accel) ---
    ret_21d = np.log(df["close"] / df["close"].shift(21))
    df["price_accel"] = ret_21d - ret_21d.shift(21)

    # --- NEW: Return distribution shape ---
    df["ret_skew_21d"] = df["log_return"].rolling(21).skew()
    df["ret_kurt_21d"] = df["log_return"].rolling(21).kurt()

    # --- NEW: Volume anomaly (volume spikes precede crashes) ---
    if "volume" in df.columns:
        df["vol_zscore"] = (
            (df["volume"] - df["volume"].rolling(21).mean())
            / (df["volume"].rolling(21).std() + 1e-8)
        )
    else:
        df["vol_zscore"] = 0.0

    return df


# ══════════════════════════════════════════════════════════════════
# MAIN PIPELINE FUNCTION
# ══════════════════════════════════════════════════════════════════
def run_zscore_pipeline(ticker_id: str,
                        engine):
    """
    Main function called by pipeline.py.

    Reads price data from DB, computes:
      1. Technical indicators (RSI, MACD, log_return + new features) — features
      2. PSY/BSADF labels (Bubble/Crash/Normal)                      — labels

    Saves results to bubble_analysis table.

    Args:
        ticker_id: e.g. 'NIFTY50' or 'SENSEX'
        engine:    SQLAlchemy engine from database.py

    Returns:
        DataFrame with all features and PSY labels
    """

    print(f"\n[PSY Pipeline] {ticker_id}")

    # Load price data from database
    query = f"""
        SELECT date, open, high, low, close, volume
        FROM market_data
        WHERE ticker_id = '{ticker_id}'
        ORDER BY date
    """
    df = pd.read_sql(query, engine)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    print(f"  Loaded {len(df)} rows from database")
    print(f"  Date range: {df['date'].min().date()} → "
          f"{df['date'].max().date()}")

    # Step A — compute technical indicators (features)
    print("\n  Computing technical indicators...")
    df = compute_technical_indicators(df)
    print("  RSI, MACD, log_return + 8 regime features computed ✅")

    # Step B — compute PSY labels (replaces Z-score threshold)
    # Replace apply_psy_labels call with:
    # NEW — compute bsadf FIRST, then apply hybrid labels
    print("\n  Computing PSY/BSADF scores...")
    print("  Min window: 157 days")
    print("  Lag augmentation: up to k=4")
    print("  (This will take 20-40 minutes...)")

    log_prices = np.log(df["close"])
    df["bsadf_score"] = compute_bsadf(
        log_prices,
        min_window=157,
        max_lag=4
    )

    # Now apply hybrid labels
    df = apply_hybrid_labels(
        df,
        upper_pct=0.95,
        crash_zscore=-2.0,
        rolling_window=504
    )

    # ── UPDATED: Drop rows missing any feature or label ───────────
    df = df.dropna(subset=[
        "log_return", "rsi", "macd",
        "macd_signal", "macd_hist", "bsadf_score",
        "realized_vol_10d", "realized_vol_63d", "vol_ratio",
        "drawdown", "trend_21d", "price_accel",
        "ret_skew_21d", "ret_kurt_21d",
    ])
    print(f"\n  Final rows after dropping NaN: {len(df)}")

    # Save to bubble_analysis table
    df["ticker_id"] = ticker_id

    # ── UPDATED: save_cols includes all new features ──────────────
    save_cols = [
        "date", "ticker_id", "close",
        "log_return", "rsi", "macd", "macd_signal", "macd_hist",
        "realized_vol_10d", "realized_vol_63d", "vol_ratio",
        "drawdown", "trend_21d", "price_accel",
        "ret_skew_21d", "ret_kurt_21d", "vol_zscore",
        "bsadf_score", "label"
    ]

    df[save_cols].to_sql(
        "bubble_analysis",
        engine,
        if_exists="replace",
        index=False
    )

    print(f"  Saved to bubble_analysis table ✅")

    return df