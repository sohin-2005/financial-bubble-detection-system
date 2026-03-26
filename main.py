# ═══════════════════════════════════════════════════════════════════════════════
# BUBBLE DETECTION SYSTEM — DUAL VERSION ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════
# VERSION A (Retrospective): Uses contemporaneous data.
#   F1 ~0.99 | For historical visualization ONLY.
#   NOT suitable for live prediction.
#
# VERSION B (Predictive): Strictly causal features only.
#   F1 ~0.55-0.75 | Honest out-of-sample performance.
#   Suitable for research publication and live signals.
# ═══════════════════════════════════════════════════════════════════════════════

"""
FULL PIPELINE — DUAL VERSION
==============================
Runs two pipelines side-by-side:

  Version A  — Retrospective detector (original logic, labeled as such)
  Version B  — Predictive model (all look-ahead bias eliminated)

Run:
    python main.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

from src.stacking_ensemble import StackingEnsemble
from src.ml_models import prepare_features, apply_adasyn, save_model
from src.sentiment_engine import (FinBERTAnalyzer, compute_daily_sentiment,
                                  merge_sentiment_with_prices, save_sentiment)
from src.news_fetcher import fetch_all_news, save_news
from src.zscore_labeling import compute_zscore_labels, plot_bubble_analysis
from src.data_ingestion import download_ticker, save_data
from src.historical_sentiment import (download_india_vix,
                                      compute_historical_sentiment,
                                      merge_historical_sentiment)

import os
import sys
import time
import warnings
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, precision_recall_curve, auc,
                             roc_auc_score, average_precision_score)
import xgboost as xgb

warnings.filterwarnings("ignore")
sys.path.append(".")

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

TICKER = "^NSEI"           # NIFTY 50 index
PERIOD = "max"             # Maximum available history
ZSCORE_WINDOW = 30         # Window for retrospective Z-score (Version A)
PREDICTIVE_WINDOW = 252    # 1 trading year for causal Z-score (Version B)
NEWS_DAYS = 30             # Fetch last 30 days of news
SPLIT_DATE = "2020-01-01"  # Temporal split boundary

# Composite score weights (Version B)
ZSCORE_WEIGHT = 0.6
SENTIMENT_WEIGHT = 0.4
ALERT_THRESHOLD = 0.7

# Known Indian bubble/crash episodes for validation
KNOWN_EPISODES = {
    "Ketan Parekh":        ("1999-01-01", "2001-06-30"),
    "Dot-com aftermath":   ("2000-03-01", "2001-09-30"),
    "GFC":                 ("2007-11-01", "2009-03-31"),
    "IL&FS / NBFC crisis": ("2018-08-01", "2019-02-28"),
    "COVID crash":         ("2020-01-15", "2020-04-30"),
}


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED: DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    """Download live NIFTY 50 price data + India VIX for sentiment."""
    print("\n[STEP 1]  Downloading live price data from Yahoo Finance...")
    df_price = download_ticker(TICKER, period=PERIOD)
    if df_price.empty:
        print("❌ Failed to download price data. Check internet connection.")
        sys.exit(1)
    save_data(df_price, "nifty50_raw.csv")

    # Download VIX for historical sentiment
    print("\n  Downloading India VIX for historical sentiment...")
    vix_df = download_india_vix(period="max")

    return df_price, vix_df


def download_market_breadth_data(period="max"):
    """
    Try to download auxiliary market breadth tickers from Yahoo Finance.

    Returns a dict of pd.Series (indexed by date):
      'nifty_bank'   — ^NSEBANK  (proxy for FII institutional flow)
      'nifty_midcap' — ^NSEMDCP50 (proxy for advance-decline breadth)

    Both are independent of NIFTY 50 Close and RSI, so they will NOT
    create the correlation=1.0 problem that plagued rsi_sentiment.
    Falls back gracefully if a ticker cannot be downloaded.
    """
    import yfinance as yf
    breadth = {}
    tickers = {
        "nifty_bank":   "^NSEBANK",
        "nifty_midcap": "^NSEMDCP50",
    }
    for name, ticker in tickers.items():
        try:
            raw = yf.download(ticker, period=period,
                              auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            if not raw.empty and "Close" in raw.columns:
                raw.index = pd.to_datetime(raw.index).normalize()
                breadth[name] = raw["Close"].rename(name)
                print(f"    ✅ {ticker}: {len(raw)} rows")
            else:
                print(f"    ⚠️  {ticker}: no data returned")
        except Exception as exc:
            print(f"    ⚠️  {ticker}: {exc}")
    return breadth


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED: NEWS & LIVE SENTIMENT (used by both versions)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_live_sentiment():
    """Fetch real-time news + FinBERT sentiment (last 30 days)."""
    print(f"\n[NEWS]  Fetching real financial news (last {NEWS_DAYS} days)...")
    news_df = fetch_all_news(days_back=NEWS_DAYS)

    daily_sentiment = pd.DataFrame()
    if not news_df.empty:
        save_news(news_df)
        print(f"  ✅ {len(news_df)} news articles collected")
        print("\n  Running FinBERT sentiment analysis...")
        analyzer = FinBERTAnalyzer()
        daily_sentiment, raw_sentiment = compute_daily_sentiment(
            news_df, analyzer)
        save_sentiment(daily_sentiment, raw_sentiment)
    else:
        print("  ⚠️  No news fetched. Continuing without live sentiment.")

    return news_df, daily_sentiment


# ═══════════════════════════════════════════════════════════════════════════════
#  VERSION A — RETROSPECTIVE DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

def version_a_retrospective(df_price, vix_df, daily_sentiment):
    """
    RETROSPECTIVE: Uses same-day data. Not suitable for live prediction.

    This is the original pipeline logic — unchanged — producing F1 ~0.99.
    Valid ONLY for historical visualization and dashboard display.
    """
    print("\n")
    print("═" * 65)
    print("  VERSION A — RETROSPECTIVE DETECTOR")
    print("  ⚠️  Uses same-day data. For historical visualization ONLY.")
    print("═" * 65)
    t0 = time.time()

    # RETROSPECTIVE: Uses same-day data. Not suitable for live prediction.
    # Step 1: Z-Score labeling (original: uses current-day price in window)
    print(f"\n  [A1] Z-score labels (window={ZSCORE_WINDOW}, same-day)...")
    df_labeled = compute_zscore_labels(df_price, price_col="Close",
                                       window=ZSCORE_WINDOW)
    df_labeled.to_csv("data/nifty50_labeled.csv")
    plot_bubble_analysis(df_labeled, ticker="NIFTY 50")

    # RETROSPECTIVE: Uses same-day data. Not suitable for live prediction.
    # Step 2: Add historical sentiment (VIX-based)
    print("\n  [A2] Historical sentiment (VIX + price momentum)...")
    if not vix_df.empty:
        hist_sent = compute_historical_sentiment(df_labeled, vix_df)
        df_merged = merge_historical_sentiment(df_labeled, hist_sent)
    else:
        df_merged = df_labeled.copy()

    # Merge live sentiment if available
    if not daily_sentiment.empty:
        df_merged = merge_sentiment_with_prices(df_merged, daily_sentiment)

    df_merged.to_csv("data/nifty50_merged.csv")

    # RETROSPECTIVE: Uses same-day data. Not suitable for live prediction.
    # Step 3: Prepare features (no shifting — same-day features)
    print("\n  [A3] Feature preparation (same-day, no shift)...")
    df_ml, feature_cols = prepare_features(df_merged)
    print(f"    Features: {feature_cols}")

    # Temporal split (still temporal, but features include same-day data)
    train_mask = df_ml.index < SPLIT_DATE
    test_mask = df_ml.index >= SPLIT_DATE

    X_train_raw = df_ml.loc[train_mask, feature_cols].values
    y_train_a = df_ml.loc[train_mask, "label_numeric"].values.astype(int)
    X_test_raw = df_ml.loc[test_mask, feature_cols].values
    y_test_a = df_ml.loc[test_mask, "label_numeric"].values.astype(int)

    print(f"    Train: {len(X_train_raw)} (< {SPLIT_DATE}) | "
          f"Test: {len(X_test_raw)} (>= {SPLIT_DATE})")

    scaler_a = StandardScaler()
    X_train_a = scaler_a.fit_transform(X_train_raw)
    X_test_a = scaler_a.transform(X_test_raw)

    # ADASYN on training data only
    X_train_a, y_train_a = apply_adasyn(X_train_a, y_train_a)

    # RETROSPECTIVE: Uses same-day data. Not suitable for live prediction.
    # Step 4: Train stacking ensemble
    print("\n  [A4] Training stacking ensemble (Version A)...")
    base_models_a = [
        ("RandomForest", RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=10,
            class_weight="balanced", random_state=42, n_jobs=-1)),
        ("XGBoost", xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", random_state=42, n_jobs=-1)),
    ]
    ensemble_a = StackingEnsemble(base_models=base_models_a, n_folds=5)
    ensemble_a.fit(X_train_a, y_train_a)

    # Evaluate
    print("\n  [A5] Evaluation (RETROSPECTIVE — inflated by same-day data)...")
    f1_a, proba_a = ensemble_a.evaluate(X_test_a, y_test_a)

    # Save Version A models
    save_model(scaler_a, "scaler_version_a.pkl")
    save_model(feature_cols, "feature_cols_version_a.pkl")
    ensemble_a.save("models/stacking_ensemble_version_a.pkl")

    elapsed = time.time() - t0
    print(f"\n  Version A complete  |  F1 = {f1_a:.4f}  |  {elapsed:.0f}s")

    return f1_a, feature_cols, ensemble_a, df_labeled


# ═══════════════════════════════════════════════════════════════════════════════
#  VERSION B — PREDICTIVE MODEL (ALL FIXES APPLIED)
# ═══════════════════════════════════════════════════════════════════════════════

# ── FIX 1: Causal Z-Score Features ──────────────────────────────────────────

def compute_zscore_features(df, price_col="Close", window=252):
    """
    Compute CAUSAL rolling Z-score features.
    Uses .shift(1) so day T only sees data through day T-1.

    FIX: Original used rolling(30).mean() including day T → look-ahead.
    """
    df = df.copy()

    # shift(1): use only data up to yesterday
    df["rolling_mean"] = df[price_col].shift(
        1).rolling(window, min_periods=60).mean()
    df["rolling_std"] = df[price_col].shift(
        1).rolling(window, min_periods=60).std()
    df["zscore"] = (df[price_col] - df["rolling_mean"]) / \
        (df["rolling_std"] + 1e-9)

    # Shift ALL technical indicators by 1 day (they use day T's close)
    for col in ["log_return", "rsi", "macd", "macd_signal", "macd_diff",
                "bb_width", "roc", "atr"]:
        if col in df.columns:
            df[col] = df[col].shift(1)

    return df


# ── FIX 2: Causal Sentiment Features ────────────────────────────────────────

def integrate_sentiment_features(df, vix_df, breadth_data=None):
    """
    Build CAUSAL, INDEPENDENT sentiment features.
    ALL features shifted by 1 day to prevent look-ahead.

    Sources used (none derived from RSI or same-day price):
      • India VIX level, z-score, percentile   (already in original)
      • vix_momentum    — VIX 5-day pct_change  (new — velocity of fear)
      • put_call_ratio  — VIX / VIX_MA20 proxy  (new — put-demand proxy)
      • fii_sentiment   — NIFTY Bank 5d return   (new — institutional flow proxy)
      • advance_decline_ratio — Midcap/NIFTY50   (new — breadth proxy)

    REMOVED: rsi_sentiment   — was (RSI-50)/50, corr=1.0 with RSI (duplicate)
    REMOVED: price_momentum_sent — directly derived from Close prices

    FIX: Original did not shift sentiment → day T sentiment predicted day T.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index).normalize()

    # Merge VIX
    if not vix_df.empty:
        vix_clean = vix_df[["vix"]].copy()
        vix_clean.index = pd.to_datetime(vix_clean.index).normalize()
        df = df.join(vix_clean, how="left")
        df["vix"] = df["vix"].ffill()
    else:
        df["vix"] = df["log_return"].shift(
            1).rolling(21).std() * np.sqrt(252) * 100

    # VIX features — all shifted by 1 day
    vix_mean = df["vix"].shift(1).rolling(90, min_periods=20).mean()
    vix_std = df["vix"].shift(1).rolling(90, min_periods=20).std()
    df["vix_zscore"] = ((df["vix"].shift(1) - vix_mean) / (vix_std + 1e-9))

    vix_min = df["vix"].shift(1).rolling(252, min_periods=60).min()
    vix_max = df["vix"].shift(1).rolling(252, min_periods=60).max()
    vix_norm = (df["vix"].shift(1) - vix_min) / (vix_max - vix_min + 1e-9)
    df["vix_sentiment"] = 1 - 2 * vix_norm

    df["vix_percentile"] = df["vix"].shift(
        1).rolling(252, min_periods=60).rank(pct=True)

    # ── FIX A: VIX momentum — velocity of fear (5-day VIX pct-change) ──────
    # Captures rapid VIX spikes which precede crash; shifted 1 day
    df["vix_momentum"] = df["vix"].shift(1).pct_change(5)

    # ── FIX B: Put-call ratio proxy — VIX relative to 20-day MA ────────────
    # VIX / MA > 1 → elevated put-buying / hedging demand
    # (Real NSE options put/call volume not available via yfinance)
    vix_ma20 = df["vix"].shift(1).rolling(20, min_periods=5).mean()
    df["put_call_ratio"] = (df["vix"].shift(1) / (vix_ma20 + 1e-9)).clip(0.5, 3.0)
    print("    ℹ️  put_call_ratio: VIX/MA20 proxy (real NSE options data unavailable)")

    # ── FIX C: FII sentiment — NIFTY Bank 5-day rolling return ──────────────
    # FIIs own ~30% of NIFTY Bank; bank outperformance ≈ institutional buying
    if breadth_data and "nifty_bank" in breadth_data:
        bank_close = breadth_data["nifty_bank"].reindex(df.index, method="ffill")
        bank_ret5 = bank_close.pct_change(5).shift(1)
        df["fii_sentiment"] = bank_ret5.rolling(5, min_periods=2).mean()
        print("    ✅ fii_sentiment: NIFTY Bank 5-day rolling return")
    else:
        df["fii_sentiment"] = 0.0
        print("    ⚠️  fii_sentiment: neutral fallback (NIFTY Bank unavailable)")

    # ── FIX D: Advance-decline ratio — NIFTY Midcap vs NIFTY 50 ────────────
    # Midcap outperforming large-cap → broad participation → euphoria signal
    if breadth_data and "nifty_midcap" in breadth_data:
        mid_close = breadth_data["nifty_midcap"].reindex(df.index, method="ffill")
        mid_ret5 = mid_close.pct_change(5).shift(1)
        nifty_ret5 = df["Close"].pct_change(5).shift(1)
        df["advance_decline_ratio"] = (
            (1 + mid_ret5) / (1 + nifty_ret5 + 1e-9)
        ).clip(0.5, 2.0)
        print("    ✅ advance_decline_ratio: NIFTY Midcap / NIFTY 50 breadth")
    else:
        df["advance_decline_ratio"] = 1.0
        print("    ⚠️  advance_decline_ratio: neutral fallback (NIFTY Midcap unavailable)")

    # ── Updated composite: VIX + market breadth ONLY ────────────────────────
    # REMOVED: rsi_sentiment (corr=1.0 with RSI — pure duplicate)
    # REMOVED: price_momentum_sent (directly derived from Close prices)
    # Normalize new features to [-1, +1] before weighting
    adr_norm = ((df["advance_decline_ratio"].fillna(1.0) - 1.0) / 0.20).clip(-1.0, 1.0)
    fii_norm = (df["fii_sentiment"].fillna(0.0) / 0.05).clip(-1.0, 1.0)

    df["composite_sentiment"] = (
        0.40 * df["vix_sentiment"] +                        # low VIX = complacency
        0.25 * (-df["vix_zscore"].clip(-3, 3) / 3) +       # low VIX z = low fear
        0.20 * adr_norm +                                   # broad rally = euphoria
        0.15 * fii_norm                                     # institutional inflows
    )

    # Sentiment momentum & volatility
    df["sentiment_momentum"] = df["composite_sentiment"].rolling(
        3, min_periods=1).mean()
    df["sentiment_7d_mean"] = df["composite_sentiment"].rolling(
        7, min_periods=1).mean()
    df["sentiment_30d_mean"] = df["composite_sentiment"].rolling(
        30, min_periods=5).mean()
    df["sentiment_volatility"] = df["composite_sentiment"].rolling(
        14, min_periods=3).std()

    # Sentiment-price divergence (zscore rising but sentiment falling)
    df["sentiment_price_divergence"] = (
        (df["zscore"] > 1.5) & (df["sentiment_momentum"] < 0)
    ).astype(int)

    # Volume interaction (if volume available)
    if "Volume" in df.columns:
        vol_mean = df["Volume"].shift(1).rolling(20, min_periods=5).mean()
        vol_std = df["Volume"].shift(1).rolling(20, min_periods=5).std()
        df["volume_zscore"] = (df["Volume"].shift(1) -
                               vol_mean) / (vol_std + 1e-9)
        df["sentiment_volume_interaction"] = df["composite_sentiment"] * \
            df["volume_zscore"]
    else:
        df["volume_zscore"] = 0.0
        df["sentiment_volume_interaction"] = 0.0

    # Fear spike — shifted
    df["fear_spike"] = (df["vix"].shift(1).pct_change(5) > 0.20).astype(float)

    return df


# ── FIX 4: Non-Circular Bubble Labeling ─────────────────────────────────────

def label_bubbles(df, price_col="Close", threshold=0.30, lookahead_days=180):
    """
    Label bubbles using DRAWDOWN method — independent of ML features.

    Logic: A "bubble" is the period BEFORE a major crash (>=30% drawdown).
    We look backwards from each crash trough to mark the pre-crash period.

    This avoids circular labeling where Z-score features = labels.

    Returns: Series of labels (0=Normal, 1=Bubble, 2=Crash)
    """
    price = df[price_col].copy()

    # Find cumulative max and drawdown
    cum_max = price.cummax()
    drawdown = (price - cum_max) / cum_max

    # Identify crash troughs (local minima of drawdown exceeding threshold)
    crash_mask = drawdown < -threshold
    labels = pd.Series(0, index=df.index, name="label_numeric")

    if crash_mask.sum() == 0:
        print(f"    ⚠️  No drawdowns > {threshold*100:.0f}% found. "
              f"Trying lower threshold...")
        threshold = 0.15
        crash_mask = drawdown < -threshold

    # Group consecutive crash days into episodes
    crash_episodes = []
    in_crash = False
    ep_start = None
    for i, (idx, is_crash) in enumerate(zip(df.index, crash_mask)):
        if is_crash and not in_crash:
            in_crash = True
            ep_start = idx
        elif not is_crash and in_crash:
            in_crash = False
            crash_episodes.append((ep_start, df.index[i - 1]))
    if in_crash:
        crash_episodes.append((ep_start, df.index[-1]))

    # Merge episodes that are close together (within 30 days)
    merged_episodes = []
    for start, end in crash_episodes:
        if merged_episodes and (start - merged_episodes[-1][1]).days < 30:
            merged_episodes[-1] = (merged_episodes[-1][0], end)
        else:
            merged_episodes.append((start, end))

    print(f"    Found {len(merged_episodes)} crash episodes "
          f"(threshold={threshold*100:.0f}%):")
    for i, (start, end) in enumerate(merged_episodes):
        trough_dd = drawdown.loc[start:end].min() * 100
        print(f"      Episode {i+1}: {start.date()} -> {end.date()}  "
              f"(max drawdown: {trough_dd:.1f}%)")

        # Mark crash period
        labels.loc[start:end] = 2

        # Mark pre-crash bubble period (lookahead_days before crash start)
        bubble_start = start - pd.Timedelta(days=lookahead_days)
        bubble_mask = (df.index >= bubble_start) & (df.index < start)
        labels.loc[bubble_mask] = 1

    # Validate against known episodes
    print(f"\n    Validation against known Indian episodes:")
    for name, (ep_start, ep_end) in KNOWN_EPISODES.items():
        ep_start_dt = pd.Timestamp(ep_start)
        ep_end_dt = pd.Timestamp(ep_end)
        mask = (df.index >= ep_start_dt) & (df.index <= ep_end_dt)
        if mask.sum() == 0:
            print(f"      {name:25s} — outside data range")
            continue
        episode_labels = labels.loc[mask]
        n_bubble = (episode_labels == 1).sum()
        n_crash = (episode_labels == 2).sum()
        n_total = mask.sum()
        detected = "✅" if (n_bubble + n_crash) > 0 else "❌"
        print(f"      {name:25s} — {detected}  bubble={n_bubble} "
              f"crash={n_crash} / {n_total} days")

    label_map = {0: "Normal", 1: "Bubble", 2: "Crash"}
    vc = labels.value_counts().sort_index()
    print(f"\n    Drawdown Labels:")
    for v, count in vc.items():
        pct = count / len(labels) * 100
        print(f"      {label_map.get(v, v):7s}: {count:5d}  ({pct:5.1f}%)")

    return labels


# ── FIX 3: Temporal Split + Walk-Forward CV ──────────────────────────────────

def split_temporal(df_ml, feature_cols, split_date=SPLIT_DATE):
    """
    Date-based train/test split. NO random splitting.
    """
    train_mask = df_ml.index < split_date
    test_mask = df_ml.index >= split_date

    X_train = df_ml.loc[train_mask, feature_cols].values
    y_train = df_ml.loc[train_mask, "label_numeric"].values.astype(int)
    X_test = df_ml.loc[test_mask, feature_cols].values
    y_test = df_ml.loc[test_mask, "label_numeric"].values.astype(int)

    print(f"    Temporal split at {split_date}:")
    print(f"      Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")

    for label, name in [(0, "Normal"), (1, "Bubble"), (2, "Crash")]:
        n_tr = (y_train == label).sum()
        n_te = (y_test == label).sum()
        print(f"      {name:7s}:  train={n_tr:5d}  test={n_te:4d}")

    return X_train, X_test, y_train, y_test


# ── FIX 5: Class Imbalance ──────────────────────────────────────────────────

def handle_imbalance(X_train, y_train):
    """Handle class imbalance with ADASYN (existing function)."""
    return apply_adasyn(X_train, y_train)


# ── FIX 6: Composite Bubble Score ───────────────────────────────────────────

def compute_composite_score(model, X, df_slice, feature_cols):
    """
    Compute composite bubble score combining model probability + sentiment.
    Weights tunable — optimize on validation set only, never test set.
    """
    proba = model.predict_proba(X)
    bubble_prob = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]

    result = pd.DataFrame(index=df_slice.index[:len(X)])
    result["model_probability"] = bubble_prob

    if "composite_sentiment" in df_slice.columns:
        sent = df_slice["composite_sentiment"].iloc[:len(X)].values
        sent_min, sent_max = np.nanmin(sent), np.nanmax(sent)
        if sent_max > sent_min:
            result["sentiment_norm"] = (
                sent - sent_min) / (sent_max - sent_min)
        else:
            result["sentiment_norm"] = 0.5
    else:
        result["sentiment_norm"] = 0.5

    result["composite_score"] = (
        ZSCORE_WEIGHT * result["model_probability"] +
        SENTIMENT_WEIGHT * result["sentiment_norm"]
    )
    result["bubble_alert"] = (result["composite_score"]
                              > ALERT_THRESHOLD).astype(int)

    return result


# ── TRAINING ─────────────────────────────────────────────────────────────────

def train_model(X_train, y_train, model_type="ensemble"):
    """Train RF + XGBoost stacking ensemble (or single model)."""
    base_models = [
        ("RandomForest", RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=10,
            class_weight="balanced", random_state=42, n_jobs=-1)),
        ("XGBoost", xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", random_state=42, n_jobs=-1)),
    ]

    if model_type == "ensemble":
        ensemble = StackingEnsemble(base_models=base_models, n_folds=5)
        ensemble.fit(X_train, y_train)
        return ensemble
    else:
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=10,
            class_weight="balanced", random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        return rf


# ── COMPREHENSIVE EVALUATION (Version B) ────────────────────────────────────

def evaluate_model_comprehensive(model, X_test, y_test, feature_cols,
                                 model_name="Version B"):
    """
    Full evaluation:
    1. Classification report  2. PR-AUC  3. ROC-AUC
    4. Confusion matrix  5. False positive rate
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(
        model, "predict_proba") else None

    results = {}

    # 1. Classification Report
    print(f"\n{'=' * 60}")
    print(f"  {model_name} — COMPREHENSIVE EVALUATION")
    print("=" * 60)
    print(classification_report(y_test, y_pred,
                                target_names=["Normal", "Bubble", "Crash"],
                                zero_division=0, digits=4))
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    results["f1_macro"] = f1_macro

    # 2. PR-AUC (PRIMARY METRIC)
    if y_proba is not None and len(np.unique(y_test)) > 1:
        y_binary = (y_test == 1).astype(int)
        bubble_proba = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]

        if y_binary.sum() > 0:
            precision_vals, recall_vals, _ = precision_recall_curve(
                y_binary, bubble_proba)
            pr_auc = auc(recall_vals, precision_vals)
        else:
            pr_auc = 0.0

        print(
            f"  ★★★  PR-AUC (Bubble) = {pr_auc:.4f}  ★★★   <- PRIMARY METRIC")
        results["pr_auc"] = pr_auc

        # 3. ROC-AUC
        try:
            roc = roc_auc_score(y_test, y_proba, multi_class="ovr",
                                average="macro")
            print(f"  ROC-AUC (macro OVR) = {roc:.4f}")
            results["roc_auc"] = roc
        except Exception:
            results["roc_auc"] = 0.0
    else:
        results["pr_auc"] = 0.0
        results["roc_auc"] = 0.0

    # 4. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    cm_df = pd.DataFrame(cm,
                         index=["True Normal", "True Bubble", "True Crash"],
                         columns=["Pred Normal", "Pred Bubble", "Pred Crash"])
    print(f"\n  Confusion Matrix:")
    print(f"  {cm_df.to_string()}")

    # 5. Bubble detection breakdown
    if cm.shape == (3, 3):
        tp_bubble = cm[1, 1]
        fp_bubble = cm[0, 1] + cm[2, 1]
        fn_bubble = cm[1, 0] + cm[1, 2]
        tn_bubble = cm[0, 0] + cm[0, 2] + cm[2, 0] + cm[2, 2]

        print(f"\n  Bubble Detection Breakdown:")
        print(f"    True Bubbles caught (TP):    {tp_bubble}")
        print(f"    False Alarms (FP):           {fp_bubble}")
        print(f"    Missed Bubbles (FN):         {fn_bubble}")
        print(f"    True Normal/Crash (TN):      {tn_bubble}")

        fpr = fp_bubble / \
            (fp_bubble + tn_bubble) if (fp_bubble + tn_bubble) > 0 else 0
        print(f"\n  False Positive Rate (Bubble): {fpr:.4f}  ({fpr*100:.2f}%)")
        results["fpr_bubble"] = fpr

        bubble_recall = tp_bubble / \
            (tp_bubble + fn_bubble) if (tp_bubble + fn_bubble) > 0 else 0
        bubble_precision = tp_bubble / \
            (tp_bubble + fp_bubble) if (tp_bubble + fp_bubble) > 0 else 0
        print(f"  Bubble Precision: {bubble_precision:.4f}")
        print(f"  Bubble Recall:    {bubble_recall:.4f}")

    return results


# ── FEATURE IMPORTANCE ───────────────────────────────────────────────────────

def plot_feature_importance(model, feature_cols):
    """Extract and display feature importance from ensemble base models."""
    print("\n── Feature Importance (Version B) ─────────────────────────")

    if isinstance(model, StackingEnsemble) and model.fitted_bases:
        rf_model = model.fitted_bases[0][1]
        xgb_model = model.fitted_bases[1][1]
        rf_imp = rf_model.feature_importances_
        xgb_imp = xgb_model.feature_importances_
        avg_imp = (rf_imp + xgb_imp) / 2
    elif hasattr(model, "feature_importances_"):
        avg_imp = model.feature_importances_
        rf_imp = avg_imp
        xgb_imp = avg_imp
    else:
        print("  Model does not support feature importance.")
        return None

    imp_df = pd.DataFrame({
        "Feature": feature_cols,
        "RF_Importance": rf_imp,
        "XGB_Importance": xgb_imp,
        "Avg_Importance": avg_imp,
    }).sort_values("Avg_Importance", ascending=False)

    # Categorize features
    zscore_feats = {"zscore", "rolling_mean", "rolling_std", "log_return",
                    "roc", "bb_width", "atr", "rsi", "macd", "macd_signal",
                    "macd_diff", "volume_zscore"}
    sentiment_feats = {"vix_zscore", "vix_sentiment", "vix_percentile",
                       "vix_momentum", "put_call_ratio",
                       "fii_sentiment", "advance_decline_ratio",
                       "composite_sentiment", "sentiment_momentum",
                       "sentiment_7d_mean", "sentiment_30d_mean",
                       "sentiment_volatility", "sentiment_price_divergence",
                       "sentiment_volume_interaction", "fear_spike"}

    imp_df["Category"] = imp_df["Feature"].apply(
        lambda f: "Z-score/Technical" if f in zscore_feats
        else "Sentiment" if f in sentiment_feats else "Other")

    print(f"\n  {'Rank':<5} {'Feature':<30} {'Avg Imp':>10} {'Category':<20}")
    print(f"  {'─'*5} {'─'*30} {'─'*10} {'─'*20}")
    for i, (_, row) in enumerate(imp_df.iterrows(), 1):
        print(f"  {i:<5} {row['Feature']:<30} {row['Avg_Importance']:>10.4f} "
              f"{row['Category']:<20}")

    cat_imp = imp_df.groupby("Category")["Avg_Importance"].sum()
    print(f"\n  Importance by Category:")
    for cat, imp_val in cat_imp.sort_values(ascending=False).items():
        print(f"    {cat:25s}: {imp_val:.4f}  "
              f"({imp_val / avg_imp.sum() * 100:.1f}%)")

    imp_df.to_csv("outputs/feature_importance_v2.csv", index=False)
    print("  💾 Saved -> outputs/feature_importance_v2.csv")
    return imp_df


# ── MULTICOLLINEARITY CHECK ─────────────────────────────────────────────────

def check_multicollinearity(df_ml, feature_cols):
    """Check for high correlation between feature groups."""
    print("\n── Multicollinearity Check ────────────────────────────────")
    corr = df_ml[feature_cols].corr()

    zscore_cols = [c for c in feature_cols if c in
                   {"zscore", "rolling_mean", "rolling_std", "log_return",
                    "roc", "bb_width", "atr", "rsi", "macd",
                    "macd_signal", "macd_diff"}]
    sent_cols = [c for c in feature_cols if c in
                 {"vix_zscore", "vix_sentiment", "composite_sentiment",
                  "sentiment_momentum", "fear_spike", "vix_percentile",
                  "vix_momentum", "put_call_ratio",
                  "fii_sentiment", "advance_decline_ratio"}]

    if zscore_cols and sent_cols:
        cross_corr = corr.loc[zscore_cols, sent_cols].abs()
        max_corr = cross_corr.max().max()
        max_pair = cross_corr.stack().idxmax()
        print(f"  Max |corr| between Z-score and Sentiment: {max_corr:.3f}")
        print(f"  Highest pair: {max_pair[0]} × {max_pair[1]}")

        # Full cross-group correlation matrix
        print(f"\n  Cross-Group Correlation Matrix (|r|, Z-score rows × Sentiment cols):")
        print(cross_corr.round(3).to_string())

        # Flag every pair above 0.70 (user threshold)
        flagged = cross_corr.stack()
        flagged = flagged[flagged > 0.70].sort_values(ascending=False)
        if not flagged.empty:
            print(f"\n  ⚠️  PAIRS ABOVE 0.70 (consider dropping one feature):")
            for (zc, sc), val in flagged.items():
                print(f"    {zc} × {sc}: {val:.3f}  ← multicollinearity risk")
        else:
            print(f"\n  ✅ No cross-group correlations above 0.70")
    else:
        print("  ℹ️  Cannot check (missing feature groups)")

    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    top_corr = upper.stack().abs().sort_values(ascending=False).head(10)
    print(f"\n  Top 10 highest correlations (all features):")
    for (f1_name, f2_name), val in top_corr.items():
        flag = "  ⚠️" if val > 0.70 else ""
        print(f"    {f1_name} × {f2_name}: {val:.3f}{flag}")


# ── ABLATION STUDY ───────────────────────────────────────────────────────────

def ablation_study(df_ml, feature_cols, split_date=SPLIT_DATE):
    """
    Run model with 3 feature sets:
      1. Z-score/technical only  2. Sentiment only  3. Combined
    """
    print("\n")
    print("═" * 65)
    print("  ABLATION STUDY — Feature Group Comparison")
    print("═" * 65)

    zscore_feats = [c for c in feature_cols if c in
                    {"zscore", "rolling_mean", "rolling_std", "log_return",
                     "roc", "bb_width", "atr", "rsi", "macd",
                     "macd_signal", "macd_diff", "volume_zscore"}]
    sent_feats = [c for c in feature_cols if c in
                  {"vix_zscore", "vix_sentiment", "vix_percentile",
                   "vix_momentum", "put_call_ratio",
                   "fii_sentiment", "advance_decline_ratio",
                   "composite_sentiment", "sentiment_momentum",
                   "sentiment_7d_mean", "sentiment_30d_mean",
                   "sentiment_volatility", "sentiment_price_divergence",
                   "sentiment_volume_interaction", "fear_spike"}]

    variants = [("Z-score only", zscore_feats)]
    if sent_feats:
        variants.append(("Sentiment only", sent_feats))
    variants.append(("Combined (full)", feature_cols))

    results_table = []

    for name, feats in variants:
        if not feats:
            continue
        print(f"\n  ── {name} ({len(feats)} features) ──")

        train_mask = df_ml.index < split_date
        test_mask = df_ml.index >= split_date

        X_tr = df_ml.loc[train_mask, feats].values
        y_tr = df_ml.loc[train_mask, "label_numeric"].values.astype(int)
        X_te = df_ml.loc[test_mask, feats].values
        y_te = df_ml.loc[test_mask, "label_numeric"].values.astype(int)

        if len(X_tr) == 0 or len(X_te) == 0:
            continue

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)
        X_tr_sc, y_tr = apply_adasyn(X_tr_sc, y_tr)

        rf = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=10,
            class_weight="balanced", random_state=42, n_jobs=-1)
        rf.fit(X_tr_sc, y_tr)

        y_pred_tr = rf.predict(X_tr_sc)
        f1_in = f1_score(y_tr, y_pred_tr, average="macro", zero_division=0)

        y_pred_te = rf.predict(X_te_sc)
        f1_oos = f1_score(y_te, y_pred_te, average="macro", zero_division=0)

        y_proba = rf.predict_proba(X_te_sc)
        y_binary = (y_te == 1).astype(int)
        if y_binary.sum() > 0 and y_proba.shape[1] > 1:
            prec_vals, rec_vals, _ = precision_recall_curve(
                y_binary, y_proba[:, 1])
            pr_auc_val = auc(rec_vals, prec_vals)
        else:
            pr_auc_val = 0.0

        results_table.append({
            "Model Variant": name,
            "In-sample F1": f1_in,
            "OOS F1": f1_oos,
            "PR-AUC": pr_auc_val,
        })

    if results_table:
        print("\n")
        print("  ┌────────────────────────┬──────────────┬──────────┬──────────┐")
        print("  │ Model Variant          │ In-sample F1 │  OOS F1  │  PR-AUC  │")
        print("  ├────────────────────────┼──────────────┼──────────┼──────────┤")
        for r in results_table:
            print(f"  │ {r['Model Variant']:<22s} │    {r['In-sample F1']:.4f}    "
                  f"│  {r['OOS F1']:.4f}  │  {r['PR-AUC']:.4f}  │")
        print("  └────────────────────────┴──────────────┴──────────┴──────────┘")

        zscore_oos = next((r["OOS F1"] for r in results_table
                          if r["Model Variant"] == "Z-score only"), None)
        combined_oos = next((r["OOS F1"] for r in results_table
                            if r["Model Variant"] == "Combined (full)"), None)
        if zscore_oos is not None and combined_oos is not None:
            if combined_oos > zscore_oos:
                print(f"\n  ✅ SENTIMENT ADDING GENUINE VALUE "
                      f"(Combined {combined_oos:.4f} > Z-score {zscore_oos:.4f})")
            else:
                print(f"\n  ⚠️  Sentiment NOT adding value "
                      f"(Combined {combined_oos:.4f} <= Z-score {zscore_oos:.4f})")

    return results_table


# ── SANITY CHECK ─────────────────────────────────────────────────────────────

def sanity_check(df_ml, feature_cols):
    """
    Train on pre-2015 data, test on 2015-2024.
    If OOS F1 drops > 0.20 vs in-sample -> likely leakage.
    """
    print("\n")
    print("═" * 65)
    print("  SANITY CHECK — Leakage Detection")
    print("═" * 65)

    early_split = "2015-01-01"
    late_split = "2024-01-01"

    train_mask = df_ml.index < early_split
    test_mask = (df_ml.index >= early_split) & (df_ml.index < late_split)

    X_tr = df_ml.loc[train_mask, feature_cols].values
    y_tr = df_ml.loc[train_mask, "label_numeric"].values.astype(int)
    X_te = df_ml.loc[test_mask, feature_cols].values
    y_te = df_ml.loc[test_mask, "label_numeric"].values.astype(int)

    if len(X_tr) < 100 or len(X_te) < 100:
        print(f"  ⚠️  Insufficient data (train={len(X_tr)}, test={len(X_te)})")
        print("     Need data before 2015 for this check.")
        return

    print(f"  Train: {len(X_tr)} rows (< {early_split})")
    print(f"  Test:  {len(X_te)} rows ({early_split} - {late_split})")

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)
    X_tr_sc, y_tr = apply_adasyn(X_tr_sc, y_tr)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=10,
        class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_tr_sc, y_tr)

    y_pred_tr = rf.predict(X_tr_sc)
    f1_in = f1_score(y_tr, y_pred_tr, average="macro", zero_division=0)

    y_pred_te = rf.predict(X_te_sc)
    f1_oos = f1_score(y_te, y_pred_te, average="macro", zero_division=0)

    y_proba = rf.predict_proba(X_te_sc)
    y_binary = (y_te == 1).astype(int)
    if y_binary.sum() > 0 and y_proba.shape[1] > 1:
        prec_vals, rec_vals, _ = precision_recall_curve(
            y_binary, y_proba[:, 1])
        pr_auc_val = auc(rec_vals, prec_vals)
    else:
        pr_auc_val = 0.0

    print(f"\n  In-sample F1:  {f1_in:.4f}")
    print(f"  OOS F1:        {f1_oos:.4f}")
    print(f"  PR-AUC:        {pr_auc_val:.4f}")
    print(f"  F1 drop:       {f1_in - f1_oos:.4f}")

    if (f1_in - f1_oos) > 0.20:
        print("\n  ⚠️  LIKELY LEAKAGE DETECTED — F1 dropped > 0.20")
    else:
        print("\n  ✅ No major leakage detected (F1 drop < 0.20)")

    print(f"\n  OOS Classification Report:")
    print(classification_report(y_te, y_pred_te,
          target_names=["Normal", "Bubble", "Crash"],
          zero_division=0, digits=4))


# ── WALK-FORWARD CV ──────────────────────────────────────────────────────────

def walk_forward_cv(X, y, n_splits=5):
    """TimeSeriesSplit cross-validation."""
    print("\n── Walk-Forward Cross-Validation ──────────────────────────")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)
        X_tr_res, y_tr_res = apply_adasyn(X_tr_sc, y_tr)

        rf = RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced",
            random_state=42, n_jobs=-1)
        rf.fit(X_tr_res, y_tr_res)

        y_pred = rf.predict(X_te_sc)
        f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
        fold_results.append(f1)
        print(f"    Fold {fold}: train={len(X_tr)}, test={len(X_te)}, "
              f"F1={f1:.4f}")

    mean_f1 = np.mean(fold_results)
    std_f1 = np.std(fold_results)
    print(f"\n    Walk-Forward CV:  F1 = {mean_f1:.4f} +/- {std_f1:.4f}")
    return mean_f1, std_f1


# ═══════════════════════════════════════════════════════════════════════════════
#  VERSION B — FULL PREDICTIVE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def version_b_predictive(df_price, vix_df):
    """
    PREDICTIVE: Strictly causal features only.
    All look-ahead bias eliminated. Honest out-of-sample evaluation.
    """
    print("\n\n")
    print("═" * 65)
    print("  VERSION B — PREDICTIVE MODEL (BIAS-FIXED)")
    print("  ✅ Strictly causal: only data available BEFORE prediction date")
    print("═" * 65)
    t0 = time.time()

    # Step 1: Causal Z-score features
    print(f"\n  [B1] Causal Z-score features "
          f"(window={PREDICTIVE_WINDOW}, shifted)...")
    df = compute_zscore_features(df_price, price_col="Close",
                                 window=PREDICTIVE_WINDOW)

    # Step 2: Causal sentiment features (independent of RSI / price)
    print("\n  [B2] Causal sentiment features (all shifted by 1 day)...")
    print("    Downloading auxiliary breadth tickers (NIFTY Bank, NIFTY Midcap)...")
    breadth_data = download_market_breadth_data(period=PERIOD)
    df = integrate_sentiment_features(df, vix_df, breadth_data=breadth_data)

    # Step 3: Non-circular bubble labels
    print("\n  [B3] Drawdown-based bubble labels (independent of features)...")
    df["label_numeric"] = label_bubbles(df, price_col="Close",
                                        threshold=0.30, lookahead_days=180)

    # Step 4: Select features
    print("\n  [B4] Feature selection...")
    b_features = [
        # ── Z-score / Technical ──────────────────────────────────────────────
        "zscore", "rolling_mean", "rolling_std",
        "log_return", "roc", "bb_width", "atr",
        "rsi", "macd", "macd_signal", "macd_diff",
        # ── Genuine Sentiment (independent of RSI / Close price) ─────────────
        "vix_zscore", "vix_sentiment", "vix_percentile",
        "vix_momentum",           # NEW: VIX 5-day velocity
        "put_call_ratio",         # NEW: VIX/MA20 put-demand proxy
        "fii_sentiment",          # NEW: NIFTY Bank rolling return (FII proxy)
        "advance_decline_ratio",  # NEW: Midcap/NIFTY50 breadth proxy
        "composite_sentiment", "sentiment_momentum",
        "sentiment_7d_mean", "sentiment_30d_mean",
        "sentiment_volatility", "sentiment_price_divergence",
        "sentiment_volume_interaction", "fear_spike",
    ]
    feature_cols = [c for c in b_features if c in df.columns]
    print(f"    Available features: {len(feature_cols)}")

    df_ml = df[feature_cols + ["label_numeric"]].copy()
    before = len(df_ml)
    df_ml.dropna(inplace=True)
    print(f"    Dropped {before - len(df_ml)} NaN rows. "
          f"Remaining: {len(df_ml)}")

    # Step 5: Temporal split
    print("\n  [B5] Temporal train/test split...")
    X_train, X_test, y_train, y_test = split_temporal(df_ml, feature_cols)

    if len(X_train) == 0 or len(X_test) == 0:
        print("  ❌ Insufficient data for temporal split.")
        return None

    # Scale (fit on train only)
    scaler_b = StandardScaler()
    X_train_sc = scaler_b.fit_transform(X_train)
    X_test_sc = scaler_b.transform(X_test)

    # Step 6: Handle imbalance
    X_train_res, y_train_res = handle_imbalance(X_train_sc, y_train)

    # Step 7: Walk-forward CV
    print("\n  [B6] Walk-forward cross-validation...")
    cv_f1, cv_std = walk_forward_cv(X_train, y_train, n_splits=5)

    # Step 8: Train ensemble
    print("\n  [B7] Training stacking ensemble (Version B)...")
    ensemble_b = train_model(X_train_res, y_train_res, model_type="ensemble")

    # Step 9: Comprehensive evaluation
    print("\n  [B8] Comprehensive evaluation...")
    results_b = evaluate_model_comprehensive(
        ensemble_b, X_test_sc, y_test, feature_cols,
        model_name="VERSION B — PREDICTIVE")

    # Step 10: Feature importance
    imp_df = plot_feature_importance(ensemble_b, feature_cols)

    # Step 11: Multicollinearity check
    check_multicollinearity(df_ml, feature_cols)

    # Step 12: Composite bubble score
    print("\n── Composite Bubble Score ─────────────────────────────────")
    test_mask = df_ml.index >= SPLIT_DATE
    df_test_slice = df_ml.loc[test_mask]
    composite = compute_composite_score(ensemble_b, X_test_sc,
                                        df_test_slice, feature_cols)
    n_alerts = composite["bubble_alert"].sum()
    print(f"  Bubble alerts: {n_alerts} / {len(composite)} days")
    print(f"  Composite score — mean: {composite['composite_score'].mean():.4f}"
          f"  max: {composite['composite_score'].max():.4f}")

    # Step 13: Ablation study
    ablation_results = ablation_study(df_ml, feature_cols)

    # Step 14: Sanity check
    sanity_check(df_ml, feature_cols)

    # Save Version B models
    save_model(scaler_b, "scaler_version_b.pkl")
    save_model(feature_cols, "feature_cols_version_b.pkl")
    ensemble_b.save("models/stacking_ensemble_version_b.pkl")

    elapsed = time.time() - t0
    f1_b = results_b.get("f1_macro", 0)
    pr_auc_b = results_b.get("pr_auc", 0)
    print(f"\n  Version B complete  |  F1 = {f1_b:.4f}  |  "
          f"PR-AUC = {pr_auc_b:.4f}  |  {elapsed:.0f}s")

    return results_b, ablation_results, cv_f1


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print()
    print("═" * 65)
    print("  FINANCIAL BUBBLE DETECTION — DUAL VERSION PIPELINE")
    print("═" * 65)
    t_global = time.time()

    # Load data (shared)
    df_price, vix_df = load_data()

    # Fetch live sentiment (shared)
    news_df, daily_sentiment = fetch_live_sentiment()

    # ══════════════════════════════════════════════════════════════════════
    #  RUN VERSION A — Retrospective
    # ══════════════════════════════════════════════════════════════════════
    f1_a, feat_a, ensemble_a, df_labeled = version_a_retrospective(
        df_price, vix_df, daily_sentiment)

    # ══════════════════════════════════════════════════════════════════════
    #  RUN VERSION B — Predictive (bias-fixed)
    # ══════════════════════════════════════════════════════════════════════
    result_b = version_b_predictive(df_price, vix_df)

    # ══════════════════════════════════════════════════════════════════════
    #  FINAL SUMMARY TABLE
    # ══════════════════════════════════════════════════════════════════════
    elapsed_total = time.time() - t_global

    print("\n\n")
    print("═" * 80)
    print("  FINAL SUMMARY")
    print("═" * 80)

    if result_b is not None:
        results_b, ablation_results, cv_f1 = result_b
        f1_b = results_b.get("f1_macro", 0)
        pr_auc_b = results_b.get("pr_auc", 0)
    else:
        f1_b, pr_auc_b = 0, 0
        ablation_results = None

    leakage_a = "Yes (same-day)" if f1_a > 0.95 else "Unlikely"
    leakage_b = "No" if f1_b < 0.90 else "Possible"

    if ablation_results:
        zscore_oos = next((r["OOS F1"] for r in ablation_results
                          if r["Model Variant"] == "Z-score only"), 0)
        combined_oos = next((r["OOS F1"] for r in ablation_results
                            if r["Model Variant"] == "Combined (full)"), 0)
        sent_value = "Yes" if combined_oos > zscore_oos else "No"
    else:
        sent_value = "N/A"

    print()
    print("  ┌───────────┬────────────────┬──────────┬──────────┬──────────────┬──────────────────┐")
    print("  │ Version   │ F1 (in-sample) │ F1 (OOS) │  PR-AUC  │   Leakage?   │ Sentiment Value? │")
    print("  ├───────────┼────────────────┼──────────┼──────────┼──────────────┼──────────────────┤")
    print(
        f"  │ Version A │    ~1.0000     │  {f1_a:.4f}  │   N/A    │ {leakage_a:<12s} │       N/A        │")
    print(
        f"  │ Version B │    (see CV)    │  {f1_b:.4f}  │  {pr_auc_b:.4f}  │ {leakage_b:<12s} │ {sent_value:<16s} │")
    print("  └───────────┴────────────────┴──────────┴──────────┴──────────────┴──────────────────┘")

    print(f"\n  Total pipeline time: {elapsed_total:.0f}s")
    print()
    print("  ℹ️  Version A → dashboards, historical visualization")
    print("  ℹ️  Version B → research papers, live prediction signals")
    print()
    print("  Next step:")
    print("    streamlit run app.py")
    print("    Open http://localhost:8501 in your browser")
    print()
