"""
HISTORICAL SENTIMENT MODULE
=============================
Creates sentiment features for the full 10-year training period.

Since NewsAPI only provides last 30 days of news, we use:

  1. India VIX (Fear Index)  — ^INDIAVIX via yfinance (2008 → present)
     - HIGH VIX = Fear = Negative sentiment (crash signal)
     - LOW  VIX = Greed = Positive sentiment (bubble risk)

  2. Price-Derived Sentiment — momentum + breadth proxies
     - Strong upward momentum → positive sentiment
     - Sharp drawdowns        → negative sentiment

This gives us a FULL 10-year sentiment proxy aligned with price data.

Run alone to test:
    python src/historical_sentiment.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")


# ─────────────────────────────────────────────────────────────────────────────
# INDIA VIX  (primary sentiment proxy)
# ─────────────────────────────────────────────────────────────────────────────

def download_india_vix(period: str = "max") -> pd.DataFrame:
    """
    Download India VIX — the official NSE fear/greed index.

    VIX meaning:
      < 15  → Very calm / greedy market  (bubble risk ↑)
      15-20 → Normal
      20-25 → Elevated fear
      > 25  → High fear / panic           (crash risk ↑)
      > 40  → Extreme panic (2008, COVID)
    """
    print("  ↓ Downloading India VIX (^INDIAVIX)...", end=" ")
    try:
        raw = yf.download("^INDIAVIX", period=period,
                          progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df = raw[["Close"]].copy()
        df.columns = ["vix"]
        df.index = pd.to_datetime(df.index).normalize()
        df.dropna(inplace=True)
        print(
            f"OK  ({len(df)} rows, {df.index[0].date()} → {df.index[-1].date()})")
        return df
    except Exception as e:
        print(f"FAILED ({e})")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE SENTIMENT FEATURES FROM VIX + PRICE
# ─────────────────────────────────────────────────────────────────────────────

def compute_historical_sentiment(price_df: pd.DataFrame,
                                 vix_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build daily sentiment features by combining:
      - India VIX (fear index)
      - Price-derived momentum signals

    Returns a DataFrame aligned to price_df index with columns:
      vix                 — raw VIX value
      vix_zscore          — how extreme is current VIX vs 90-day history
      vix_sentiment       — [-1, +1] inverted VIX (high VIX = -1 = fear)
      vix_percentile      — rolling 252-day percentile rank of VIX
      price_momentum_sent — sentiment from price momentum
      composite_sentiment — weighted combination of all above
    """

    df = price_df.copy()
    df.index = pd.to_datetime(df.index).normalize()

    # ── 1. Merge VIX ─────────────────────────────────────────────────────────
    if not vix_df.empty:
        df = df.join(vix_df[["vix"]], how="left")
        df["vix"] = df["vix"].ffill()          # fill weekends/holidays
    else:
        # Fallback: estimate VIX from price volatility
        df["vix"] = df["log_return"].rolling(21).std() * np.sqrt(252) * 100
        print("  ⚠️  Using volatility-based VIX proxy (real VIX unavailable)")

    # ── 2. VIX-based features ─────────────────────────────────────────────────
    # Z-score of VIX (90-day window)
    vix_mean = df["vix"].rolling(90, min_periods=20).mean()
    vix_std = df["vix"].rolling(90, min_periods=20).std()
    df["vix_zscore"] = (df["vix"] - vix_mean) / (vix_std + 1e-9)

    # Invert VIX → sentiment: high VIX = fear = -1, low VIX = greed = +1
    vix_min = df["vix"].rolling(252, min_periods=60).min()
    vix_max = df["vix"].rolling(252, min_periods=60).max()
    vix_norm = (df["vix"] - vix_min) / (vix_max - vix_min + 1e-9)
    df["vix_sentiment"] = 1 - 2 * vix_norm     # maps [0,1] → [+1,-1]

    # Rolling percentile rank (0=lowest fear ever, 1=highest fear ever)
    df["vix_percentile"] = df["vix"].rolling(
        252, min_periods=60).rank(pct=True)

    # ── 3. Price-derived sentiment ────────────────────────────────────────────
    # 5-day momentum normalized to [-1, +1]
    mom_5d = df["Close"].pct_change(5)
    mom_20d = df["Close"].pct_change(20)

    # Clip extreme values, then scale
    df["price_momentum_sent"] = (
        0.6 * np.tanh(mom_5d / 0.05) +    # 5-day momentum (weighted more)
        0.4 * np.tanh(mom_20d / 0.10)       # 20-day momentum
    )

    # RSI → sentiment: RSI>70 = greed (+1), RSI<30 = fear (-1)
    if "rsi" in df.columns:
        df["rsi_sentiment"] = (df["rsi"] - 50) / 50   # maps [0,100] → [-1,+1]
    else:
        df["rsi_sentiment"] = 0.0

    # ── 4. Composite sentiment ────────────────────────────────────────────────
    # Weighted combination
    df["composite_sentiment"] = (
        0.40 * df["vix_sentiment"] +   # VIX is most reliable
        0.30 * df["price_momentum_sent"] +   # price momentum
        0.20 * df["rsi_sentiment"] +   # overbought/oversold
        0.10 * (-df["vix_zscore"].clip(-3, 3) / 3)  # VIX spike signal
    )

    # ── 5. Sentiment momentum (3-day rolling mean) ────────────────────────────
    df["sentiment_momentum"] = df["composite_sentiment"].rolling(
        3, min_periods=1).mean()

    # ── 6. Fear spike flag ────────────────────────────────────────────────────
    # 1 if VIX jumped > 20% in last 5 days (crash precursor)
    df["fear_spike"] = (df["vix"].pct_change(5) > 0.20).astype(float)

    sentiment_cols = [
        "vix", "vix_zscore", "vix_sentiment", "vix_percentile",
        "price_momentum_sent", "rsi_sentiment",
        "composite_sentiment", "sentiment_momentum", "fear_spike"
    ]

    result = df[sentiment_cols].copy()
    result.dropna(how="all", inplace=True)

    print(f"  ✅ Historical sentiment computed: {len(result)} rows, "
          f"{result.index[0].date()} → {result.index[-1].date()}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MERGE WITH LABELED DATA
# ─────────────────────────────────────────────────────────────────────────────

def merge_historical_sentiment(labeled_df: pd.DataFrame,
                               sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """Merge sentiment features into labeled price DataFrame."""
    labeled_df = labeled_df.copy()
    labeled_df.index = pd.to_datetime(labeled_df.index).normalize()

    merged = labeled_df.join(sentiment_df, how="left")

    # Fill any remaining NaN with neutral values
    fill_vals = {
        "vix":                  merged["vix"].median() if "vix" in merged else 20.0,
        "vix_zscore":           0.0,
        "vix_sentiment":        0.0,
        "vix_percentile":       0.5,
        "price_momentum_sent":  0.0,
        "rsi_sentiment":        0.0,
        "composite_sentiment":  0.0,
        "sentiment_momentum":   0.0,
        "fear_spike":           0.0,
    }
    merged.fillna(fill_vals, inplace=True)

    rows_with_vix = merged["vix"].notna().sum()
    print(f"  ✅ Merged: {len(merged)} rows  |  VIX coverage: {rows_with_vix} rows "
          f"({rows_with_vix/len(merged)*100:.1f}%)")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────

def save_historical_sentiment(df: pd.DataFrame,
                              filename: str = "nifty50_with_sentiment.csv") -> None:
    os.makedirs("data", exist_ok=True)
    df.to_csv(f"data/{filename}")
    print(f"  💾 Saved → data/{filename}")


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.data_ingestion import download_ticker

    print("=" * 55)
    print("  HISTORICAL SENTIMENT BUILDER")
    print("=" * 55)

    # 1. Load labeled data
    print("\n[1/4] Loading labeled price data...")
    labeled = pd.read_csv("data/nifty50_labeled.csv",
                          index_col=0, parse_dates=True)
    print(f"  Loaded {len(labeled)} rows  "
          f"({labeled.index[0].date()} → {labeled.index[-1].date()})")

    # 2. Download India VIX
    print("\n[2/4] Downloading India VIX...")
    vix = download_india_vix(period="max")

    # 3. Compute sentiment features
    print("\n[3/4] Computing sentiment features...")
    sentiment = compute_historical_sentiment(labeled, vix)

    print("\n  Sentiment feature stats:")
    print(sentiment[["vix", "vix_sentiment", "composite_sentiment",
                     "fear_spike"]].describe().round(3).to_string())

    # 4. Merge & save
    print("\n[4/4] Merging & saving...")
    final = merge_historical_sentiment(labeled, sentiment)
    save_historical_sentiment(final)

    # Show label-wise sentiment
    print("\n  Average sentiment by label:")
    print(final.groupby("label")["composite_sentiment"].agg(
        ["mean", "std", "count"]).round(3))

    print("\n  Average VIX by label:")
    print(final.groupby("label")["vix"].agg(["mean", "min", "max"]).round(2))

    print("\n✅ Done. Now run:")
    print("   python3 src/ml_models.py")
    print("   python3 src/stacking_ensemble.py")
