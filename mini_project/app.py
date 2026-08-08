"""
REAL-TIME STREAMLIT DASHBOARD
================================
Full real-time dashboard with:
  • Live NIFTY 50 / NIFTY 500 data from Yahoo Finance
  • Live news from RSS feeds (no key) + NewsAPI / GNews (optional)
  • FinBERT sentiment on real headlines
  • Z-score bubble labeling
  • Stacking ensemble crash probability
  • Auto-refresh every 5 minutes

HOW TO RUN:
    streamlit run app.py

FIRST TIME SETUP:
    1. Copy .env.example to .env
    2. (Optional) Add your free API keys to .env
    3. Run:  pip install -r requirements.txt
    4. Run:  streamlit run app.py
"""

import os
import sys
import time
import json
import pickle
import warnings
import shutil
import importlib
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")
sys.path.append(".")

ROOT_DIR = Path(__file__).resolve().parent
FBDS_SRC = ROOT_DIR / "financial-bubble-detection-system" / "src"
if FBDS_SRC.exists() and str(FBDS_SRC) not in sys.path:
    sys.path.append(str(FBDS_SRC))

# Ensure financial-bubble-detection-system/src is importable for macro rebuilds
FBDS_SRC = Path(__file__).resolve().parent / \
    "financial-bubble-detection-system" / "src"
if FBDS_SRC.exists() and str(FBDS_SRC) not in sys.path:
    sys.path.append(str(FBDS_SRC))

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Sans:wght@300;400;500&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
    /* Base */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif !important;
        color: #c8cad4;
    }

    /* All h1 headings — Playfair Display like the reference */
    h1 {
        font-family: 'Playfair Display', serif !important;
        font-weight: 700 !important;
        font-size: 42px !important;
        color: #f0f2f8 !important;
        letter-spacing: -0.5px !important;
        line-height: 1.15 !important;
    }

    /* h2 subheadings — DM Sans medium */
    h2, h3 {
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        color: #e0e2ec !important;
        letter-spacing: 0.3px !important;
    }

    /* Body paragraphs */
    p, li, label {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 14px !important;
        line-height: 1.65 !important;
        color: #8a8d9a !important;
    }
    /* Do NOT apply global color to span — it overrides ticker colors */

    /* Metric values — monospace for numbers */
    [data-testid="metric-container"] > div:first-child {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 26px !important;
        font-weight: 700 !important;
    }

    /* Tab labels */
    button[data-baseweb="tab"] {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 13px !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        color: #555 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #e8eaf0 !important;
        font-weight: 500 !important;
    }

    /* Sidebar labels */
    .sidebar .sidebar-content {
        font-family: 'DM Sans', sans-serif !important;
    }

    /* Section headers — uppercase tracked labels */
    .section-label {
        font-family: 'DM Sans', sans-serif;
        font-size: 10px;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #444;
        margin-bottom: 12px;
    }

    /* Dividers */
    hr { border-color: #1a1f2e !important; }

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    .block-container { padding-top: 0.5rem !important; }
    header[data-testid="stHeader"] { height: 0px; }
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "data" / "models"

print("MODEL_DIR =", MODEL_DIR)
print("Exists =", MODEL_DIR.exists())

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NSE Market Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Hardcode period, window, news_days
period = "3y"
window = 30
news_days = 7

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Dark card */
.stat-card {
    background: #1a1d2e;
    border: 1px solid #2d3250;
    border-radius: 12px;
    padding: 18px 14px;
    text-align: center;
    margin-bottom: 8px;
}
.stat-value { font-size: 1.7rem; font-weight: 700; margin: 0; }
.stat-label { font-size: 0.78rem; color: #a0a8c0; margin: 0; }

/* Alert banners */
.banner-bubble {
    background: linear-gradient(135deg,#c0392b,#e74c3c);
    color:#fff; padding:12px 20px; border-radius:10px;
    font-size:1.1rem; font-weight:700; text-align:center;
}
.banner-crash {
    background: linear-gradient(135deg,#1a5276,#2980b9);
    color:#fff; padding:12px 20px; border-radius:10px;
    font-size:1.1rem; font-weight:700; text-align:center;
}
.banner-normal {
    background: linear-gradient(135deg,#1e8449,#27ae60);
    color:#fff; padding:12px 20px; border-radius:10px;
    font-size:1.1rem; font-weight:700; text-align:center;
}
.news-card {
    background:#1a1d2e; border-left: 4px solid #3498db;
    padding:10px 14px; border-radius:6px; margin-bottom:8px;
}
.news-pos { border-left-color: #27ae60; }
.news-neg { border-left-color: #e74c3c; }
.badge-pos { background:#1e8449; color:#fff; padding:2px 8px; border-radius:12px; font-size:0.7rem; }
.badge-neg { background:#c0392b; color:#fff; padding:2px 8px; border-radius:12px; font-size:0.7rem; }
.badge-neu { background:#555;    color:#fff; padding:2px 8px; border-radius:12px; font-size:0.7rem; }

/* Ticker tape */
.ticker-track {
        display: flex;
    animation: ticker-scroll 70s linear infinite;
        white-space: nowrap;
        width: max-content;
}

@keyframes ticker-scroll {
    0% { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CACHED DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_data(ttl=300)   # refresh every 5 minutes
def load_price_data(ticker: str, period: str, window: int) -> pd.DataFrame:
    """Download + label price data. Cached for 5 min."""
    from src.data_ingestion import download_ticker
    from src.zscore_labeling import compute_zscore_labels

    df = download_ticker(ticker, period=period)
    if df.empty:
        return df
    df = compute_zscore_labels(df, price_col="Close", window=window)
    return df


@st.cache_data(ttl=600)   # refresh every 10 minutes
def load_live_news(days_back: int) -> pd.DataFrame:
    """Fetch real news. Cached for 10 min."""
    from src.news_fetcher import fetch_all_news
    return fetch_all_news(days_back=days_back)


@st.cache_resource        # load model only once per session
def load_finbert():
    """Load FinBERT model once and reuse."""
    from src.sentiment_engine import FinBERTAnalyzer
    return FinBERTAnalyzer()


@st.cache_data(ttl=600)
def compute_live_sentiment(days_back: int):
    """Fetch news + run FinBERT. Cached 10 min."""
    from src.sentiment_engine import compute_daily_sentiment

    news = load_live_news(days_back=days_back)
    if news.empty:
        return pd.DataFrame(), pd.DataFrame()

    analyzer = load_finbert()
    daily, raw = compute_daily_sentiment(news, analyzer)
    return daily, raw


@st.cache_resource
def load_ensemble():
    """Load pre-trained stacking ensemble if it exists."""
    import joblib

    candidates = [
        Path("models/stacking_ensemble.pkl"),
        MODEL_DIR / "stacking_model.pkl",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return None
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


@st.cache_resource
def load_scaler_and_features():
    import joblib

    scaler_candidates = [
        Path("models/scaler.pkl"),
        MODEL_DIR / "stacking_scaler.pkl",
    ]
    scaler_path = next((p for p in scaler_candidates if p.exists()), None)
    if scaler_path is None:
        return None, None

    try:
        scaler = joblib.load(scaler_path)
    except Exception:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    feature_cols = None
    feature_path = Path("models/feature_cols.pkl")
    if feature_path.exists():
        try:
            feature_cols = joblib.load(feature_path)
        except Exception:
            with open(feature_path, "rb") as f:
                feature_cols = pickle.load(f)

    return scaler, feature_cols


@st.cache_resource
def load_trained_models():
    import joblib

    rf_path = MODEL_DIR / "rf_model.pkl"
    xgb_path = MODEL_DIR / "xgb_model.pkl"

    rf_model = None
    xgb_model = None

    if rf_path.exists():
        try:
            rf_model = joblib.load(rf_path)
        except Exception:
            with open(rf_path, "rb") as f:
                rf_model = pickle.load(f)
    if xgb_path.exists():
        try:
            xgb_model = joblib.load(xgb_path)
        except Exception:
            with open(xgb_path, "rb") as f:
                xgb_model = pickle.load(f)

    return rf_model, xgb_model


@st.cache_resource
def load_label_encoder():
    import joblib

    candidates = [
        Path("models/label_encoder.pkl"),
        MODEL_DIR / "label_encoder.pkl",
    ]

    for path in candidates:
        if path.exists():
            try:
                return joblib.load(path)
            except Exception:
                with open(path, "rb") as f:
                    return pickle.load(f)
    return None


@st.cache_data(ttl=120)
def load_live_nifty() -> dict:
    from src.data_ingestion import download_ticker

    df = download_ticker("^NSEI", period="5d")
    if df.empty:
        return {"price": None, "change": None, "pct": None}

    df = df.dropna(subset=["Close"])
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    change = float(latest["Close"] - prev["Close"])
    pct = float(change / prev["Close"] * 100) if prev["Close"] else 0.0
    return {
        "price": float(latest["Close"]),
        "change": change,
        "pct": pct,
    }


@st.cache_data(ttl=120)
def load_live_quotes(tickers: list[str]) -> pd.DataFrame:
    import yfinance as yf

    data = yf.download(
        tickers=tickers,
        period="5d",
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    rows = []
    for t in tickers:
        try:
            df_t = data[t] if isinstance(data.columns, pd.MultiIndex) else data
            df_t = df_t.dropna(subset=["Close"])
            if df_t.empty:
                continue
            latest = df_t.iloc[-1]
            prev = df_t.iloc[-2] if len(df_t) > 1 else latest
            change = float(latest["Close"] - prev["Close"])
            pct = float(change / prev["Close"] * 100) if prev["Close"] else 0.0
            rows.append({
                "ticker": t,
                "price": float(latest["Close"]),
                "pct": pct,
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


@st.cache_data(ttl=600)
def load_sector_history(symbol: str) -> pd.DataFrame:
    import yfinance as yf

    ticker_obj = yf.Ticker(symbol)
    hist = ticker_obj.history(period="1mo")
    return hist.dropna(subset=["Close"]) if not hist.empty else hist


def ensure_macro_fresh(max_age_hours: int = 24) -> Path | None:
    """Rebuild macro_daily.csv if missing or stale.

    Prefers live rebuild via financial-bubble-detection-system/src/amal/macro.py.
    Copies the freshly built file into the app root for fast reads.
    """
    candidate_paths = [
        ROOT_DIR / "macro_daily.csv",
        ROOT_DIR / "financial-bubble-detection-system" /
        "data" / "exports" / "macro_daily.csv",
    ]

    now = datetime.utcnow()
    for p in candidate_paths:
        if p.exists():
            age_hours = (
                now - datetime.utcfromtimestamp(p.stat().st_mtime)).total_seconds() / 3600.0
            if age_hours <= max_age_hours:
                return p

    try:
        macro_mod = importlib.import_module("amal.macro")
    except Exception as exc:
        print(f"[MACRO] Could not import macro builder: {exc}")
        return None

    try:
        end_date = datetime.utcnow().strftime("%Y-%m-%d")
        macro_mod.build_macro_daily("2008-01-01", end_date)
    except Exception as exc:
        print(f"[MACRO] Rebuild failed: {exc}")
        return None

    built_path = (Path(macro_mod.__file__).resolve().parent /
                  macro_mod.OUTPUT_DIR / "macro_daily.csv").resolve()
    if built_path.exists():
        try:
            shutil.copy(built_path, candidate_paths[0])
            return built_path
        except Exception as exc:
            print(f"[MACRO] Copy to app root failed: {exc}")
            return built_path
    return None


@st.cache_data(ttl=86400)
def load_macro_latest(target_date=None) -> dict:
    ensure_macro_fresh()
    paths = [
        ROOT_DIR / "macro_daily.csv",
        ROOT_DIR / "financial-bubble-detection-system" /
        "data" / "exports" / "macro_daily.csv",
    ]
    macro_df = pd.DataFrame()
    source_path = None
    for p in paths:
        if p.exists():
            macro_df = pd.read_csv(p, parse_dates=["date"])
            source_path = p
            break
    if macro_df.empty or "date" not in macro_df.columns:
        return {}

    macro_df = macro_df.sort_values("date")
    if target_date is not None:
        t = pd.to_datetime(target_date).normalize()
        row = macro_df[macro_df["date"] <= t].tail(1)
        if row.empty:
            row = macro_df.tail(1)
    else:
        row = macro_df.tail(1)

    rec = row.iloc[0].to_dict()
    last_dt = pd.to_datetime(rec.get("date")).date() if "date" in rec else None
    mtime = source_path.stat().st_mtime if source_path else None
    return {
        "gdp_growth": float(rec.get("gdp_growth", 0.0)),
        "cpi_inflation": float(rec.get("cpi_inflation", 0.0)),
        "repo_rate": float(rec.get("repo_rate", 0.0)),
        "_macro_last_date": str(last_dt) if last_dt else None,
        "_macro_file_mtime": mtime,
        "_macro_source": str(source_path) if source_path else None,
    }


@st.cache_resource
def load_model_stats():
    stats = {}
    rf_path = MODEL_DIR / "rf_model_stats.json"
    xgb_path = MODEL_DIR / "xgb_model_stats.json"

    if rf_path.exists():
        with open(rf_path, "r") as f:
            stats["rf"] = json.load(f)
    if xgb_path.exists():
        with open(xgb_path, "r") as f:
            stats["xgb"] = json.load(f)

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# CRASH PROBABILITY  (ensemble or simple heuristic if no model yet)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_crash_probability(
    latest_row: pd.Series,
    ensemble=None,
    scaler=None,
    feature_cols=None,
) -> float:
    """
    Return probability of crash(0–1).

    If ensemble is trained → use it.
    Otherwise → heuristic based on PSY + RSI + sentiment.
    """
    # ── Heuristic (no model trained yet) ──────────────────────────────────
    if ensemble is None:
        rsi = latest_row.get("rsi", 50)
        pol = latest_row.get("avg_polarity", 0)
        psy = latest_row.get("psy_12", 50)

        # RSI > 50 = more risk
        rsi_risk = max(0, (rsi - 50) / 50)
        # PSY > 50 = more risk (crowded bullishness)
        psy_risk = max(0, (psy - 50) / 50)
        # Negative sentiment → crash risk
        sent_risk = max(0, -pol)

        prob = 0.4 * rsi_risk + 0.4 * psy_risk + 0.2 * sent_risk
        return float(np.clip(prob, 0, 1))

    # ── Ensemble prediction ────────────────────────────────────────────────
    try:
        available = [c for c in feature_cols if c in latest_row.index]
        x = latest_row[available].values.reshape(1, -1)
        x_scaled = scaler.transform(x)
        proba = ensemble.predict_proba(x_scaled)
        return float(proba[0, 2])   # index 2 = Crash class
    except Exception:
        return 0.0


def apply_model_predictions(
    df: pd.DataFrame,
    rf_model,
    xgb_model,
    ensemble,
    scaler,
    label_encoder,
    macro_vals: dict,
    sentiment_fallback: float,
):
    """
    Compute day-level bubble/crash labels using trained models.

    Returns updated df plus latest probabilities (crash, bubble, normal).
    """
    if label_encoder is None or (rf_model is None and xgb_model is None):
        return df, None, None, None

    class_order = list(label_encoder.classes_)
    work = df.copy()

    # Sentiment fallback
    if "daily_sentiment_index" not in work.columns:
        work["daily_sentiment_index"] = float(sentiment_fallback)
    else:
        work["daily_sentiment_index"] = work["daily_sentiment_index"].fillna(
            float(sentiment_fallback)
        )

    # Macro features default to latest snapshot
    for col in ["gdp_growth", "cpi_inflation", "repo_rate"]:
        default_val = float(macro_vals.get(col, 0.0))
        if col not in work.columns:
            work[col] = default_val
        else:
            work[col] = work[col].fillna(default_val)

    # Technical feature fill-ins (align with training pipelines)
    if "Close" in work.columns:
        close_s = work["Close"].astype(float)
        work["log_return"] = np.log(close_s / close_s.shift(1))

        if "rsi" not in work.columns:
            delta = close_s.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            work["rsi"] = 100 - (100 / (1 + rs))

        if "macd" not in work.columns or "macd_signal" not in work.columns:
            ema_fast = close_s.ewm(span=12, adjust=False).mean()
            ema_slow = close_s.ewm(span=26, adjust=False).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=9, adjust=False).mean()
            work["macd"] = macd
            work["macd_signal"] = macd_signal
            work["macd_hist"] = macd - macd_signal

        if "psy_12" not in work.columns:
            up_days = (close_s.diff() > 0).astype(float)
            work["psy_12"] = up_days.rolling(window=12).mean() * 100.0

        ret_s = work["log_return"].astype(float)
        work["realized_vol_10d"] = ret_s.rolling(
            10, min_periods=1).std() * np.sqrt(252)
        work["realized_vol_63d"] = ret_s.rolling(
            63, min_periods=1).std() * np.sqrt(252)
        work["vol_ratio"] = work["realized_vol_10d"] / \
            (work["realized_vol_63d"] + 1e-6)

        cum_returns = (1 + ret_s.fillna(0)).cumprod()
        running_max = cum_returns.expanding().max()
        work["drawdown"] = (cum_returns - running_max) / (running_max + 1e-6)

        work["trend_21d"] = ret_s.rolling(21, min_periods=1).sum()
        work["price_accel"] = ret_s.rolling(5, min_periods=1).mean()
        work["ret_skew_21d"] = ret_s.rolling(21, min_periods=1).skew()
        work["ret_kurt_21d"] = ret_s.rolling(21, min_periods=1).kurt()

        vol_63_mean = work["realized_vol_63d"].rolling(
            63, min_periods=1).mean()
        vol_63_std = work["realized_vol_63d"].rolling(63, min_periods=1).std()
        work["vol_zscore"] = (
            work["realized_vol_63d"] - vol_63_mean
        ) / (vol_63_std + 1e-6)

    xgb_features = [
        "zscore_value",
        "log_return",
        "psy_12",
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "daily_sentiment_index",
        "gdp_growth",
        "cpi_inflation",
        "repo_rate",
    ]

    xgb_features_legacy = [
        "zscore_value",
        "log_return",
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "daily_sentiment_index",
        "gdp_growth",
        "cpi_inflation",
        "repo_rate",
    ]

    rf_features = [
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

    def reorder_probs(model_classes, prob_matrix):
        ordered = np.zeros((prob_matrix.shape[0], len(class_order)))
        for j, cls_code in enumerate(list(model_classes)):
            idx = int(cls_code)
            if 0 <= idx < len(class_order):
                ordered[:, idx] = prob_matrix[:, j]
        return ordered

    rf_probs = None
    xgb_probs = None

    if xgb_model is not None:
        try:
            feats = xgb_features_legacy if (
                hasattr(xgb_model, "n_features_in_")
                and int(xgb_model.n_features_in_)
                == len(xgb_features_legacy)
            ) else xgb_features

            xgb_X = (
                work.reindex(columns=feats)
                .astype(float)
                .fillna(0.0)
                .values
            )
            xgb_raw = xgb_model.predict_proba(xgb_X)
            xgb_probs = reorder_probs(xgb_model.classes_, xgb_raw)
        except Exception as exc:
            print(f"[DEBUG] XGB full-series prediction failed: {exc}")

    if rf_model is not None:
        try:
            feats = rf_features if (
                hasattr(rf_model, "n_features_in_")
                and int(rf_model.n_features_in_) == len(rf_features)
            ) else xgb_features

            rf_X = (
                work.reindex(columns=feats)
                .astype(float)
                .fillna(0.0)
                .values
            )
            rf_raw = rf_model.predict_proba(rf_X)
            rf_probs = reorder_probs(rf_model.classes_, rf_raw)
        except Exception as exc:
            print(f"[DEBUG] RF full-series prediction failed: {exc}")

    final_probs = None
    if (
        ensemble is not None
        and scaler is not None
        and rf_probs is not None
        and xgb_probs is not None
    ):
        try:
            meta = np.hstack([rf_probs, xgb_probs])
            meta_scaled = scaler.transform(meta)
            ens_raw = ensemble.predict_proba(meta_scaled)
            final_probs = reorder_probs(ensemble.classes_, ens_raw)
        except Exception as exc:
            print(f"[DEBUG] Ensemble meta prediction failed: {exc}")

    if final_probs is None:
        base = [p for p in [rf_probs, xgb_probs] if p is not None]
        if not base:
            return df, None, None, None
        final_probs = np.mean(base, axis=0)

    pred_indices = np.argmax(final_probs, axis=1)
    prob_threshold = 0.7
    # last N days: much stricter; older history uses milder gates
    recent_strict_window = 180
    labels = []
    for row_idx, idx in enumerate(pred_indices):
        top_prob = float(final_probs[row_idx, idx])
        # Require higher confidence; apply larger tie-break gap to avoid spurious flips.
        pb = float(final_probs[row_idx, class_order.index("Bubble")])
        pc = float(final_probs[row_idx, class_order.index("Crash")])
        pn = float(final_probs[row_idx, class_order.index("Normal")])

        label = class_order[idx]

        # Identify how far from the end we are (0 = most recent)
        days_from_end = len(pred_indices) - row_idx - 1
        in_recent_window = days_from_end < recent_strict_window

        # Enforce class-specific minimums and gap vs. runner-up
        if label == "Crash":
            runner_up = max(pb, pn)
            if in_recent_window:
                # Extremely strict near-present
                if top_prob < 0.97 or (top_prob - runner_up) < 0.30:
                    label = "Normal"
            else:
                # Milder for older history
                if top_prob < 0.85 or (top_prob - runner_up) < 0.15:
                    label = "Normal"
        elif label == "Bubble":
            runner_up = max(pc, pn)
            if in_recent_window:
                if top_prob < 0.90 or (top_prob - runner_up) < 0.20:
                    label = "Normal"
            else:
                if top_prob < max(runner_up + 0.08, 0.78):
                    label = "Normal"
        else:
            # Normal if confidence is low for others
            if top_prob < 0.5:
                label = "Normal"
        if in_recent_window and label == "Crash":
            label = "High Risk"
        labels.append(label)

    df = df.copy()
    if "label" in df.columns:
        df["label_heuristic"] = df["label"]
    df["label"] = labels

    df["prob_bubble"] = final_probs[:, class_order.index("Bubble")]
    df["prob_crash"] = final_probs[:, class_order.index("Crash")]
    df["prob_normal"] = final_probs[:, class_order.index("Normal")]

    return (
        df,
        float(df["prob_crash"].iloc[-1]),
        float(df["prob_bubble"].iloc[-1]),
        float(df["prob_normal"].iloc[-1]),
    )


# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def build_price_chart(df: pd.DataFrame, ticker_label: str) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f"{ticker_label} — Price & Bubble Zones",
            "Volume"
        ),
        row_heights=[0.75, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.04,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        name="OHLC",
        increasing_line_color="#27ae60",
        decreasing_line_color="#e74c3c",
    ), row=1, col=1)

    # Rolling mean
    fig.add_trace(go.Scatter(
        x=df.index, y=df["rolling_mean"],
        name="Rolling Mean",
        line=dict(color="orange", width=1.5, dash="dash"),
    ), row=1, col=1)

    # Shade bubble / crash regions
    _shade_regions(fig, df, row=1)

    # Volume bars
    vol_colors = [
        "#27ae60" if c >= o else "#e74c3c"
        for c, o in zip(df["Close"], df["Open"])
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        name="Volume", marker_color=vol_colors, opacity=0.6,
    ), row=2, col=1)

    fig.update_layout(
        height=680, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True, xaxis_rangeslider_visible=False,
        margin=dict(l=40, r=40, t=40, b=20),
    )
    return fig


def _shade_regions(fig, df, row):
    """Add colored background rectangles for Bubble/Crash periods."""
    label_colors = {
        "Bubble": "rgba(231,76,60,0.12)",
        "Crash": "rgba(41,128,185,0.12)",
        "High Risk": "rgba(41,128,185,0.12)",
    }
    for label, color in label_colors.items():
        in_region = False
        start_date = None
        mask = df["label"] == label
        for i, (date, is_label) in enumerate(zip(df.index, mask)):
            if is_label and not in_region:
                in_region = True
                start_date = date
            elif not is_label and in_region:
                in_region = False
                fig.add_vrect(
                    x0=start_date, x1=df.index[i - 1],
                    fillcolor=color, line_width=0, row=row, col=1,
                )
        if in_region:
            fig.add_vrect(
                x0=start_date, x1=df.index[-1],
                fillcolor=color, line_width=0, row=row, col=1,
            )


def build_sentiment_chart(daily_sent: pd.DataFrame) -> go.Figure:
    if daily_sent.empty:
        return go.Figure()

    recent = daily_sent.tail(60)
    colors = [
        "#27ae60" if v > 0.05 else "#e74c3c" if v < -0.05 else "#f39c12"
        for v in recent["avg_polarity"]
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=recent.index, y=recent["avg_polarity"],
        marker_color=colors, name="Daily Polarity",
    ))
    fig.add_trace(go.Scatter(
        x=recent.index, y=recent["sentiment_momentum"],
        mode="lines", name="3-day MA",
        line=dict(color="white", width=2),
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(
        title="Daily Sentiment Index (FinBERT)",
        height=300, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=40, b=20),
    )
    return fig


def build_gauge(crash_prob: float) -> go.Figure:
    pct = round(crash_prob * 100, 1)
    color = (
        "#e74c3c" if pct > 60
        else "#f39c12" if pct > 35
        else "#27ae60"
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        title={"text": "Crash Probability %", "font": {"size": 16}},
        number={"suffix": "%", "font": {"size": 28, "color": color}},
        gauge={
            "axis":  {"range": [0, 100]},
            "bar":   {"color": color},
            "steps": [
                {"range": [0, 35],  "color": "rgba(39,174,96,0.15)"},
                {"range": [35, 60], "color": "rgba(243,156,18,0.15)"},
                {"range": [60, 100], "color": "rgba(231,76,60,0.15)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.8, "value": 60,
            },
        },
    ))
    fig.update_layout(
        height=220, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=10),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

TICKER_OPTIONS = {
    "NIFTY 50 (Index)": "^NSEI",
}

# ─────────────────────────────────────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

if "app_loaded" not in st.session_state:
    st.session_state["app_loaded"] = False

if not st.session_state["app_loaded"]:
    _load_ph = st.empty()
    _load_ph.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Bebas+Neue&display=swap');
@keyframes draw-line {
  0%   { stroke-dashoffset: 900; opacity:0.2; }
  100% { stroke-dashoffset: 0;   opacity:1; }
}
@keyframes draw-bar {
  0%   { transform:scaleY(0); opacity:0; }
  100% { transform:scaleY(1); opacity:1; }
}
@keyframes fade-up {
  0%   { opacity:0; transform:translateY(14px); }
  100% { opacity:1; transform:translateY(0); }
}
@keyframes progress-run {
  0%  { width:0%; }
  25% { width:28%; }
  55% { width:61%; }
  80% { width:84%; }
  100%{ width:100%; }
}
@keyframes blink { 50%{ opacity:0; } }
@keyframes ticker-load {
  0%   { transform:translateX(0); }
  100% { transform:translateX(-50%); }
}
@keyframes pulse-glow {
  0%,100% { opacity:0.4; }
  50%     { opacity:1; }
}
.lr { position:fixed; inset:0; background:#06080f;
  display:flex; flex-direction:column;
  align-items:center; justify-content:center;
  z-index:9999; overflow:hidden;
  font-family:'JetBrains Mono',monospace; }
.bg-grid { position:absolute; inset:0;
  background-image:
    linear-gradient(rgba(255,255,255,0.018) 1px, transparent 1px),
    linear-gradient(90deg,rgba(255,255,255,0.018) 1px,transparent 1px);
  background-size:44px 44px; }
.corner { position:absolute; width:22px; height:22px; }
.tl { top:22px; left:22px;
  border-top:1px solid #22c55e; border-left:1px solid #22c55e; }
.tr { top:22px; right:22px;
  border-top:1px solid #22c55e; border-right:1px solid #22c55e; }
.bl { bottom:40px; left:22px;
  border-bottom:1px solid #22c55e; border-left:1px solid #22c55e; }
.br { bottom:40px; right:22px;
  border-bottom:1px solid #22c55e; border-right:1px solid #22c55e; }
.chart-wrap { position:relative; width:460px; height:150px;
  margin-bottom:36px; }
.chart-line { stroke-dasharray:900; stroke-dashoffset:900;
  animation:draw-line 2.4s cubic-bezier(0.4,0,0.2,1) forwards; }
.bar-row { position:absolute; bottom:0; left:0; right:0;
  display:flex; align-items:flex-end; gap:5px; height:100%; }
.cb { flex:1; border-radius:2px 2px 0 0;
  transform-origin:bottom; animation:draw-bar 0.35s ease forwards;
  opacity:0; }
.cb.u { background:#22c55e; }
.cb.d { background:#ef4444; }
.load-title { font-family:'Bebas Neue',sans-serif; font-size:54px;
  color:#f0f2f8; letter-spacing:5px; margin-bottom:4px;
  animation:fade-up 0.6s ease 0.3s both; }
.load-sub { font-size:10px; letter-spacing:4px; color:#2d3748;
  text-transform:uppercase; margin-bottom:28px;
  animation:fade-up 0.6s ease 0.5s both; }
.pills { display:flex; gap:10px; margin-bottom:28px;
  animation:fade-up 0.6s ease 0.9s both; opacity:0; }
.pill { background:#0d0f18; border:1px solid #1a1f2e;
  border-radius:6px; padding:5px 14px; font-size:10px;
  letter-spacing:1px; }
.pill .v { font-weight:700; font-size:12px; }
.prog-wrap { width:340px; height:2px; background:#0f1117;
  border-radius:2px; overflow:hidden; margin-bottom:14px;
  animation:fade-up 0.4s ease 0.7s both; opacity:0; }
.prog-fill { height:100%; border-radius:2px;
  background:linear-gradient(90deg,#22c55e,#f59e0b,#ef4444);
  animation:progress-run 3s ease-in-out 0.7s both; }
.load-status { font-size:10px; letter-spacing:2px;
  color:#374151; min-height:16px;
  animation:fade-up 0.4s ease 0.9s both; }
.cur { animation:blink 1s step-end infinite; color:#22c55e; }
.scan-line { position:absolute; left:0; right:0; height:1px;
  background:linear-gradient(90deg,transparent,#22c55e33,transparent);
  animation:scan 3s linear infinite; top:0; }
@keyframes scan {
  0%   { top:0%; opacity:0; }
  10%  { opacity:1; }
  90%  { opacity:1; }
  100% { top:100%; opacity:0; }
}
.load-ticker { position:absolute; bottom:0; left:0; right:0;
  height:26px; background:#0a0c10;
  border-top:1px solid #0f1117; overflow:hidden;
  display:flex; align-items:center; }
.lt-track { white-space:nowrap;
  animation:ticker-load 10s linear infinite;
  font-size:9px; letter-spacing:2px; color:#1f2937; }
</style>
<div class="lr">
  <div class="bg-grid"></div>
  <div class="scan-line"></div>
  <div class="corner tl"></div><div class="corner tr"></div>
  <div class="corner bl"></div><div class="corner br"></div>

  <div class="chart-wrap">
    <svg width="460" height="150"
         style="position:absolute;top:0;left:0;z-index:2">
      <defs>
        <linearGradient id="lg" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%"   stop-color="#22c55e"/>
          <stop offset="45%"  stop-color="#f59e0b"/>
          <stop offset="100%" stop-color="#ef4444"/>
        </linearGradient>
      </defs>
      <polyline class="chart-line"
        points="0,130 35,112 65,120 95,88 125,98
                155,68 185,78 215,50 245,62
                275,38 305,54 335,24 365,40
                400,18 435,28 460,14"
        fill="none" stroke="url(#lg)" stroke-width="2.5"
        stroke-linecap="round" stroke-linejoin="round"/>
      <polyline class="chart-line"
        points="0,130 35,112 65,120 95,88 125,98
                155,68 185,78 215,50 245,62
                275,38 305,54 335,24 365,40
                400,18 435,28 460,14"
        fill="none" stroke="url(#lg)" stroke-width="10"
        stroke-linecap="round" stroke-linejoin="round"
        opacity="0.1"/>
    </svg>
    <div class="bar-row">
      <div class="cb u" style="height:28%;animation-delay:.05s"></div>
      <div class="cb d" style="height:20%;animation-delay:.12s"></div>
      <div class="cb u" style="height:38%;animation-delay:.19s"></div>
      <div class="cb u" style="height:48%;animation-delay:.26s"></div>
      <div class="cb d" style="height:32%;animation-delay:.33s"></div>
      <div class="cb u" style="height:58%;animation-delay:.40s"></div>
      <div class="cb d" style="height:42%;animation-delay:.47s"></div>
      <div class="cb u" style="height:66%;animation-delay:.54s"></div>
      <div class="cb u" style="height:75%;animation-delay:.61s"></div>
      <div class="cb d" style="height:55%;animation-delay:.68s"></div>
      <div class="cb u" style="height:82%;animation-delay:.75s"></div>
      <div class="cb u" style="height:90%;animation-delay:.82s"></div>
      <div class="cb d" style="height:70%;animation-delay:.89s"></div>
      <div class="cb u" style="height:95%;animation-delay:.96s"></div>
      <div class="cb u" style="height:100%;animation-delay:1.03s"></div>
    </div>
  </div>

  <div class="load-title">NSE MARKET INTELLIGENCE</div>
  <div class="load-sub">Initializing live data streams</div>

  <div class="pills">
    <div class="pill">
      <span style="color:#374151">NIFTY 50</span>&nbsp;
      <span class="v" style="color:#22c55e">▲ LIVE</span>
    </div>
    <div class="pill">
      <span style="color:#374151">FINBERT</span>&nbsp;
      <span class="v" style="color:#f59e0b">LOADING</span>
    </div>
    <div class="pill">
      <span style="color:#374151">ML MODEL</span>&nbsp;
      <span class="v" style="color:#a855f7">READY</span>
    </div>
    <div class="pill">
      <span style="color:#374151">RF + XGB</span>&nbsp;
      <span class="v" style="color:#38bdf8">ARMED</span>
    </div>
  </div>

  <div class="prog-wrap">
    <div class="prog-fill"></div>
  </div>
  <div class="load-status">
    FETCHING MARKET DATA<span class="cur">_</span>
  </div>

  <div class="load-ticker">
    <div class="lt-track">
      &nbsp;&nbsp;&nbsp;
      NIFTY 50 &nbsp;◈&nbsp; SENSEX &nbsp;◈&nbsp;
      NIFTY BANK &nbsp;◈&nbsp; NIFTY IT &nbsp;◈&nbsp;
      RELIANCE &nbsp;◈&nbsp; TCS &nbsp;◈&nbsp;
      HDFC BANK &nbsp;◈&nbsp; INFOSYS &nbsp;◈&nbsp;
      ICICI BANK &nbsp;◈&nbsp; BAJAJ FINANCE &nbsp;◈&nbsp;
      INITIALIZING BUBBLE DETECTION ENGINE
      &nbsp;&nbsp;&nbsp;
      NIFTY 50 &nbsp;◈&nbsp; SENSEX &nbsp;◈&nbsp;
      NIFTY BANK &nbsp;◈&nbsp; NIFTY IT &nbsp;◈&nbsp;
      RELIANCE &nbsp;◈&nbsp; TCS &nbsp;◈&nbsp;
      HDFC BANK &nbsp;◈&nbsp; INFOSYS &nbsp;◈&nbsp;
      ICICI BANK &nbsp;◈&nbsp; BAJAJ FINANCE &nbsp;◈&nbsp;
      INITIALIZING BUBBLE DETECTION ENGINE
      &nbsp;&nbsp;&nbsp;
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
    import time as _time
    _time.sleep(2.8)
    st.session_state["app_loaded"] = True
    _load_ph.empty()

nifty = load_live_nifty()
header = st.container()
with header:
    index_map = {
        "^NSEI": "NIFTY 50",
        "^NSEBANK": "NIFTY BANK",
        "^BSESN": "SENSEX",
        "^CNXIT": "NIFTY IT",
        "^NSEMDCP50": "NIFTY MIDCAP",
    }
    stock_tickers = [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
        "ICICIBANK.NS", "SBIN.NS", "ITC.NS", "LT.NS",
        "HINDUNILVR.NS", "KOTAKBANK.NS",
        "AXISBANK.NS", "BAJFINANCE.NS", "BHARTIARTL.NS", "HCLTECH.NS",
        "HDFCLIFE.NS", "SUNPHARMA.NS", "MARUTI.NS", "WIPRO.NS",
        "ASIANPAINT.NS", "ULTRACEMCO.NS", "TITAN.NS", "NTPC.NS",
        "POWERGRID.NS", "TATAMOTORS.NS", "INDUSINDBK.NS", "JSWSTEEL.NS",
        "BAJAJFINSV.NS", "ADANIENT.NS", "ADANIPORTS.NS", "COALINDIA.NS",
        "GRASIM.NS", "HINDALCO.NS", "DRREDDY.NS", "CIPLA.NS",
        "DIVISLAB.NS", "EICHERMOT.NS", "BRITANNIA.NS", "NESTLEIND.NS",
        "BPCL.NS", "ONGC.NS", "SBIINS.NS", "APOLLOHOSP.NS",
        "HEROMOTOCO.NS", "UPL.NS", "SHRIRAMFIN.NS"
    ]
    tickers = list(index_map.keys()) + stock_tickers
    quotes = load_live_quotes(tickers)

    rows_by_ticker = {
        row["ticker"]: row for _, row in quotes.iterrows()
    } if not quotes.empty else {}

    stock_data = []
    for sym, label_txt in index_map.items():
        row = rows_by_ticker.get(sym)
        if row is not None:
            stock_data.append((label_txt, row["price"], row["pct"]))
        elif sym == "^NSEI" and nifty["price"] is not None:
            stock_data.append((label_txt, nifty["price"], nifty["pct"]))

    for sym in stock_tickers:
        row = rows_by_ticker.get(sym)
        if row is None:
            continue
        symbol = sym.replace(".NS", "")
        stock_data.append((symbol, row["price"], row["pct"]))

    if stock_data:
        market_status = st.session_state.get("market_status", "SAFE")

        if market_status == "CRASH":
            pos_css = "#22863a"
            neg_css = "#ff3b30"
        elif market_status == "BUBBLE":
            pos_css = "#fb923c"
            neg_css = "#ef4444"
        elif market_status == "WARNING":
            pos_css = "#facc15"
            neg_css = "#f87171"
        else:
            pos_css = "#4ade80"
            neg_css = "#f87171"

        def get_ticker_colors(chg, market_status):
            if market_status == "CRASH":
                pos_color = "#22863a"
                neg_color = "#ff3b30"
            elif market_status == "BUBBLE":
                pos_color = "#fb923c"
                neg_color = "#ef4444"
            elif market_status == "WARNING":
                pos_color = "#facc15"
                neg_color = "#f87171"
            else:
                pos_color = "#4ade80"
                neg_color = "#f87171"

            if chg is None or (isinstance(chg, float) and np.isnan(chg)):
                return pos_color

            return pos_color if chg >= 0 else neg_color

        items = []
        for sym, price, chg in stock_data:
            safe_chg = 0.0 if chg is None or (
                isinstance(chg, float) and np.isnan(chg)) else float(chg)
            arrow = "▲" if safe_chg >= 0 else "▼"
            color = get_ticker_colors(chg, market_status)
            chg_str = f"{abs(safe_chg):.2f}%"
            dir_class = "tk-pos" if safe_chg >= 0 else "tk-neg"
            items.append(
                f'<span style="display:inline-flex; align-items:center; '
                f'gap:5px; margin:0 18px; '
                f'font-family:\'JetBrains Mono\',monospace; '
                f'font-size:11px; white-space:nowrap; line-height:1;">'

                f'<span class="{dir_class}" style="color:{color} !important; '
                f'font-size:10px; line-height:1; '
                f'display:inline-block; '
                f'-webkit-text-fill-color:{color} !important;">'
                f'{arrow}</span>'

                f'<span class="{dir_class}" style="color:{color} !important; '
                f'font-weight:700; '
                f'-webkit-text-fill-color:{color} !important;">'
                f'{chg_str}</span>'

                f'<span class="tk-sym" style="color:#d1d5db !important; '
                f'font-weight:600; letter-spacing:0.5px; '
                f'-webkit-text-fill-color:#d1d5db !important;">'
                f'{sym}</span>'

                f'<span class="tk-price" style="color:#6b7280 !important; '
                f'-webkit-text-fill-color:#6b7280 !important;">'
                f'{price:,.2f}</span>'

                f'</span>'
                f'<span class="tk-sep" style="color:#2a2a2a; margin:0 2px;">|</span>'
            )

        ticker_html = "".join(items)
        st.markdown(f"""
<style>
@keyframes ticker-scroll {{
    0%   {{ transform: translateX(0); }}
    100% {{ transform: translateX(-50%); }}
}}
.ticker-outer {{
    background: #0a0a0a;
    border-bottom: 1px solid #1c1c1e;
    height: 34px;
    overflow: hidden;
    display: flex;
    align-items: center;
    width: 100%;
    margin-bottom: 4px;
}}
.ticker-label {{
    background: #141414;
    border-right: 1px solid #2a2a2a;
    color: #4b5563;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    letter-spacing: 2px;
    padding: 0 14px;
    height: 100%;
    display: flex;
    align-items: center;
    white-space: nowrap;
    flex-shrink: 0;
    text-transform: uppercase;
}}
.ticker-track {{
    display: inline-flex;
    animation: ticker-scroll 120s linear infinite;
    align-items: center;
    white-space: nowrap;
    will-change: transform;
    transform: translate3d(0, 0, 0);
    backface-visibility: hidden;
}}
.ticker-track:hover {{ animation-play-state: paused; }}

.ticker-outer .tk-pos {{
    color: {pos_css} !important;
    -webkit-text-fill-color: {pos_css} !important;
}}
.ticker-outer .tk-neg {{
    color: {neg_css} !important;
    -webkit-text-fill-color: {neg_css} !important;
}}
.ticker-outer .tk-sym {{
    color: #d1d5db !important;
    -webkit-text-fill-color: #d1d5db !important;
}}
.ticker-outer .tk-price {{
    color: #6b7280 !important;
    -webkit-text-fill-color: #6b7280 !important;
}}
.ticker-outer .tk-sep {{
    color: #2a2a2a !important;
    -webkit-text-fill-color: #2a2a2a !important;
}}
</style>

<div class="ticker-outer">
    <div class="ticker-label">NSE LIVE</div>
    <div style="overflow:hidden; flex:1; height:100%; 
        display:flex; align-items:center;">
        <div class="ticker-track">
            {ticker_html}{ticker_html}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

cc1, cc2, cc3, cc4 = st.columns([3, 1, 1, 1])
with cc1:
    ticker_label = st.selectbox(
        "s", list(TICKER_OPTIONS.keys()),
        label_visibility="collapsed"
    )
    ticker = TICKER_OPTIONS[ticker_label]
with cc2:
    use_sentiment = st.toggle("FinBERT", value=True)
with cc3:
    auto_refresh = st.toggle("Auto-Refresh", value=False)
with cc4:
    if st.button("↺  Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.markdown("""
<style>
div[data-testid="stSelectbox"] > div > div {
    background: #0f1117 !important;
    border: 1px solid #1e2130 !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    color: #e8eaf0 !important;
}
div[data-testid="stToggle"] label p {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: #4b5563 !important;
}
div[data-testid="stButton"] > button {
    background: #0f1117 !important;
    border: 1px solid #1e2130 !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 2px !important;
    color: #4b5563 !important;
    text-transform: uppercase !important;
    transition: all 0.2s !important;
}
div[data-testid="stButton"] > button:hover {
    border-color: #22c55e !important;
    color: #22c55e !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<h1 style="font-family:'Playfair Display',serif; font-weight:700;
  font-size:44px; color:#f0f2f8; letter-spacing:-0.5px; 
  line-height:1.1; margin:8px 0 20px 0;">
  NSE Market <span style="color:#666; font-style:italic; 
  font-weight:400;">Intelligence</span>
</h1>
<p style="font-family:'DM Sans',sans-serif; font-size:12px; color:#666; margin:0 0 10px 0;">
    {ticker} &nbsp;&nbsp;•&nbsp;&nbsp; Last updated: {datetime.now().strftime('%H:%M:%S')}
</p>
""", unsafe_allow_html=True)

# ── Load price data ────────────────────────────────────────────────────────
_loading_ph = st.empty()
_loading_ph.markdown("""
<style>
@keyframes mini-draw-line {
  0%   { stroke-dashoffset: 450; opacity:0.2; }
  100% { stroke-dashoffset: 0;   opacity:1; }
}
@keyframes mini-draw-bar {
  0%   { transform:scaleY(0); opacity:0; }
  100% { transform:scaleY(1); opacity:1; }
}
@keyframes mini-fade-up {
  0%   { opacity:0; transform:translateY(8px); }
  100% { opacity:1; transform:translateY(0); }
}
@keyframes mini-progress {
  0%  { width:0%; }
  50% { width:70%; }
  100%{ width:100%; }
}
.mini-lr { 
  display:flex; flex-direction:column;
  align-items:center; justify-content:center;
  padding:28px 0;
  background: linear-gradient(180deg, rgba(15,17,23,0.8), rgba(15,17,23,0.4));
  border-radius:12px;
  border:1px solid #1e2130;
}
.mini-chart { position:relative; width:280px; height:80px; margin-bottom:16px; }
.mini-line { stroke-dasharray:450; stroke-dashoffset:450;
  animation:mini-draw-line 1.6s cubic-bezier(0.4,0,0.2,1) forwards; }
.mini-bars { position:absolute; bottom:0; left:0; right:0;
  display:flex; align-items:flex-end; gap:3px; height:100%; }
.mini-bar { flex:1; border-radius:1px 1px 0 0;
  transform-origin:bottom; animation:mini-draw-bar 0.25s ease forwards;
  opacity:0; }
.mini-bar.u { background:#22c55e; }
.mini-bar.d { background:#ef4444; }
.mini-title { font-family:'JetBrains Mono',monospace; font-size:11px;
  color:#f0f2f8; letter-spacing:2px; margin-bottom:8px;
  animation:mini-fade-up 0.4s ease 0.2s both; }
.mini-prog { width:200px; height:1.5px; background:#0f1117;
  border-radius:1px; overflow:hidden; margin-top:12px;
  animation:mini-fade-up 0.3s ease 0.4s both; opacity:0; }
.mini-fill { height:100%; border-radius:1px;
  background:linear-gradient(90deg,#22c55e,#f59e0b,#ef4444);
  animation:mini-progress 2s ease-in-out 0.4s both; }
.mini-status { font-size:8px; letter-spacing:1px;
  color:#4b5563; margin-top:8px;
  animation:mini-fade-up 0.3s ease 0.5s both; opacity:0; }
</style>
<div class="mini-lr">
  <div class="mini-title">⧗ Loading market data...</div>
  
  <div class="mini-chart">
    <svg width="280" height="80" style="position:absolute;top:0;left:0;z-index:2">
      <defs>
        <linearGradient id="mini-lg" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%"   stop-color="#22c55e"/>
          <stop offset="50%"  stop-color="#f59e0b"/>
          <stop offset="100%" stop-color="#ef4444"/>
        </linearGradient>
      </defs>
      <polyline class="mini-line"
        points="0,70 20,60 40,65 60,50 80,58 100,40 120,48 140,30 160,38 180,22 200,32 220,18 240,25 260,14 280,20"
        fill="none" stroke="url(#mini-lg)" stroke-width="1.5"
        stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    <div class="mini-bars">
      <div class="mini-bar u" style="height:25%;animation-delay:.03s"></div>
      <div class="mini-bar d" style="height:18%;animation-delay:.08s"></div>
      <div class="mini-bar u" style="height:32%;animation-delay:.13s"></div>
      <div class="mini-bar u" style="height:40%;animation-delay:.18s"></div>
      <div class="mini-bar d" style="height:28%;animation-delay:.23s"></div>
      <div class="mini-bar u" style="height:48%;animation-delay:.28s"></div>
      <div class="mini-bar d" style="height:35%;animation-delay:.33s"></div>
      <div class="mini-bar u" style="height:55%;animation-delay:.38s"></div>
      <div class="mini-bar u" style="height:62%;animation-delay:.43s"></div>
      <div class="mini-bar d" style="height:45%;animation-delay:.48s"></div>
    </div>
  </div>
  
  <div class="mini-prog">
    <div class="mini-fill"></div>
  </div>
  <div class="mini-status">FETCHING {ticker} SNAPSHOT...</div>
</div>
""", unsafe_allow_html=True)

df = load_price_data(ticker, period, window)
_loading_ph.empty()

if df.empty:
    st.error(
        f"❌ Could not fetch data for `{ticker}`. Check your internet connection.")
    st.stop()

# ── Load sentiment ─────────────────────────────────────────────────────────
daily_sent = pd.DataFrame()
raw_sent = pd.DataFrame()

if use_sentiment:
    _sent_ph = st.empty()
    _sent_ph.markdown("""
<style>
@keyframes mini-draw-line {
  0%   { stroke-dashoffset: 450; opacity:0.2; }
  100% { stroke-dashoffset: 0;   opacity:1; }
}
@keyframes mini-draw-bar {
  0%   { transform:scaleY(0); opacity:0; }
  100% { transform:scaleY(1); opacity:1; }
}
@keyframes mini-fade-up {
  0%   { opacity:0; transform:translateY(8px); }
  100% { opacity:1; transform:translateY(0); }
}
@keyframes mini-progress {
  0%  { width:0%; }
  50% { width:70%; }
  100%{ width:100%; }
}
.mini-lr { 
  display:flex; flex-direction:column;
  align-items:center; justify-content:center;
  padding:28px 0;
  background: linear-gradient(180deg, rgba(15,17,23,0.8), rgba(15,17,23,0.4));
  border-radius:12px;
  border:1px solid #1e2130;
}
.mini-chart { position:relative; width:280px; height:80px; margin-bottom:16px; }
.mini-line { stroke-dasharray:450; stroke-dashoffset:450;
  animation:mini-draw-line 1.6s cubic-bezier(0.4,0,0.2,1) forwards; }
.mini-bars { position:absolute; bottom:0; left:0; right:0;
  display:flex; align-items:flex-end; gap:3px; height:100%; }
.mini-bar { flex:1; border-radius:1px 1px 0 0;
  transform-origin:bottom; animation:mini-draw-bar 0.25s ease forwards;
  opacity:0; }
.mini-bar.u { background:#22c55e; }
.mini-bar.d { background:#ef4444; }
.mini-title { font-family:'JetBrains Mono',monospace; font-size:11px;
  color:#f0f2f8; letter-spacing:2px; margin-bottom:8px;
  animation:mini-fade-up 0.4s ease 0.2s both; }
.mini-prog { width:200px; height:1.5px; background:#0f1117;
  border-radius:1px; overflow:hidden; margin-top:12px;
  animation:mini-fade-up 0.3s ease 0.4s both; opacity:0; }
.mini-fill { height:100%; border-radius:1px;
  background:linear-gradient(90deg,#22c55e,#f59e0b,#ef4444);
  animation:mini-progress 2s ease-in-out 0.4s both; }
.mini-status { font-size:8px; letter-spacing:1px;
  color:#4b5563; margin-top:8px;
  animation:mini-fade-up 0.3s ease 0.5s both; opacity:0; }
</style>
<div class="mini-lr">
  <div class="mini-title">⧗ Analyzing sentiment...</div>
  
  <div class="mini-chart">
    <svg width="280" height="80" style="position:absolute;top:0;left:0;z-index:2">
      <defs>
        <linearGradient id="mini-lg" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%"   stop-color="#22c55e"/>
          <stop offset="50%"  stop-color="#f59e0b"/>
          <stop offset="100%" stop-color="#ef4444"/>
        </linearGradient>
      </defs>
      <polyline class="mini-line"
        points="0,70 20,60 40,65 60,50 80,58 100,40 120,48 140,30 160,38 180,22 200,32 220,18 240,25 260,14 280,20"
        fill="none" stroke="url(#mini-lg)" stroke-width="1.5"
        stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    <div class="mini-bars">
      <div class="mini-bar u" style="height:25%;animation-delay:.03s"></div>
      <div class="mini-bar d" style="height:18%;animation-delay:.08s"></div>
      <div class="mini-bar u" style="height:32%;animation-delay:.13s"></div>
      <div class="mini-bar u" style="height:40%;animation-delay:.18s"></div>
      <div class="mini-bar d" style="height:28%;animation-delay:.23s"></div>
      <div class="mini-bar u" style="height:48%;animation-delay:.28s"></div>
      <div class="mini-bar d" style="height:35%;animation-delay:.33s"></div>
      <div class="mini-bar u" style="height:55%;animation-delay:.38s"></div>
      <div class="mini-bar u" style="height:62%;animation-delay:.43s"></div>
      <div class="mini-bar d" style="height:45%;animation-delay:.48s"></div>
    </div>
  </div>
  
  <div class="mini-prog">
    <div class="mini-fill"></div>
  </div>
  <div class="mini-status">FINBERT + NEWS PROCESSING...</div>
</div>
""", unsafe_allow_html=True)

    try:
        daily_sent, raw_sent = compute_live_sentiment(days_back=news_days)
        # Merge into price df
        from src.sentiment_engine import merge_sentiment_with_prices
        df = merge_sentiment_with_prices(df, daily_sent)
        _sent_ph.empty()
    except Exception as exc:
        import traceback
        st.exception(exc)
        st.code(traceback.format_exc())
            
    if _sent_ph:
        _sent_ph.empty()
else:
    try:
        from src.news_fetcher import fetch_all_news
        raw_sent = fetch_all_news(days_back=news_days)
        if not raw_sent.empty:
            raw_sent["label"] = "neutral"
            raw_sent["polarity"] = 0.0
    except Exception:
        pass

# ── Load models ────────────────────────────────────────────────────────────
ensemble, scaler, feature_cols = None, None, None
rf_model, xgb_model = None, None
try:
    ensemble = load_ensemble()
    scaler, feature_cols = load_scaler_and_features()
except Exception:
    pass

try:
    rf_model, xgb_model = load_trained_models()
except Exception:
    pass

label_encoder = None
try:
    label_encoder = load_label_encoder()
except Exception:
    pass

# ── Latest row values ─────────────────────────────────────────────────────
latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else latest
crash_prob = estimate_crash_probability(latest, ensemble, scaler, feature_cols)
bubble_prob = 0.0
normal_prob = max(0.0, 1.0 - crash_prob)

macro_vals = load_macro_latest(
    latest.name if hasattr(latest, "name") else None)


def resolve_sentiment_index(
    latest_row: pd.Series,
    daily_sent_df: pd.DataFrame
) -> float:
    # Try all known column names in priority order
    for col in ["daily_sentiment_index", "avg_polarity",
                "sentiment_score", "polarity", "sentiment"]:
        val = latest_row.get(col, None)
        if val is not None and pd.notna(val) and float(val) != 0.0:
            print(f"[DEBUG] Sentiment from latest_row['{col}'] = {val}")
            return float(val)

    # Fallback: read directly from daily_sent_df
    if not daily_sent_df.empty:
        for col in ["avg_polarity", "daily_sentiment_index",
                    "sentiment_score", "polarity"]:
            if col in daily_sent_df.columns:
                val = daily_sent_df[col].dropna()
                if not val.empty and val.iloc[-1] != 0.0:
                    print(f"[DEBUG] Sentiment from daily_sent['{col}'] "
                          f"= {val.iloc[-1]}")
                    return float(val.iloc[-1])

    # Last fallback: check raw_sent if available in scope
    print("[DEBUG] Sentiment resolved to 0.0 — check FinBERT output cols:")
    if not daily_sent_df.empty:
        print(f"  daily_sent columns: {daily_sent_df.columns.tolist()}")
    print(f"  latest_row columns: {latest_row.index.tolist()}")
    return 0.0


sentiment_index = resolve_sentiment_index(latest, daily_sent)

try:
    df, model_crash, model_bubble, model_normal = apply_model_predictions(
        df,
        rf_model,
        xgb_model,
        ensemble,
        scaler,
        label_encoder,
        macro_vals,
        sentiment_index,
    )
    if model_crash is not None:
        crash_prob = model_crash
    if model_bubble is not None:
        bubble_prob = model_bubble
    if model_normal is not None:
        normal_prob = model_normal
    elif model_crash is not None or model_bubble is not None:
        normal_prob = max(0.0, 1.0 - crash_prob - bubble_prob)
except Exception as exc:
    print(f"[DEBUG] Model label application failed: {exc}")

if crash_prob >= 0.60 and crash_prob >= bubble_prob:
    market_status = "CRASH"
elif bubble_prob >= 0.60 and bubble_prob > crash_prob:
    market_status = "BUBBLE"
elif max(crash_prob, bubble_prob) >= 0.35:
    market_status = "WARNING"
else:
    market_status = "SAFE"

crash_risk = int(round(crash_prob * 100))

st.session_state["market_status"] = market_status

status_color_map = {
    "CRASH":   {
        "border": "#e8372a",
        "glow": "rgba(232,55,42,0.25)",
        "dot": "#e8372a",
        "bg": "linear-gradient(135deg, #1a0a0a 0%, #2d0f0f 100%)",
        "shadow": "0 8px 40px rgba(232,55,42,0.2)",
    },
    "WARNING": {
        "border": "#f59e0b",
        "glow": "rgba(245,158,11,0.25)",
        "dot": "#f59e0b",
        "bg": "linear-gradient(135deg, #1a1200 0%, #2d2000 100%)",
        "shadow": "0 8px 40px rgba(245,158,11,0.2)",
    },
    "SAFE":    {
        "border": "#1db954",
        "glow": "rgba(29,185,84,0.25)",
        "dot": "#1db954",
        "bg": "linear-gradient(135deg, #041a0e 0%, #082d18 100%)",
        "shadow": "0 8px 40px rgba(29,185,84,0.2)",
    },
    "BUBBLE":  {
        "border": "#a855f7",
        "glow": "rgba(168,85,247,0.25)",
        "dot": "#a855f7",
        "bg": "linear-gradient(135deg, #120a1a 0%, #1e0f2d 100%)",
        "shadow": "none",
    },
}
c = status_color_map.get(market_status, status_color_map["SAFE"])

st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
@keyframes pulse-dot {{
    0%, 100% {{ box-shadow: 0 0 0 0 {c['glow']}; }}
    50%       {{ box-shadow: 0 0 0 10px transparent; }}
}}
@keyframes blink {{ 50% {{ opacity: 0; }} }}
.status-card {{
    background: {c['bg']};
    border: 1px solid {c['border']};
    border-left: 4px solid {c['border']};
    border-radius: 12px;
    padding: 20px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 12px 0;
    box-shadow: {c['shadow']};
}}
.status-dot {{
    width: 16px; height: 16px;
    border-radius: 50%;
    background: {c['dot']};
    animation: pulse-dot 1.6s ease-in-out infinite;
    flex-shrink: 0;
}}
.status-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 3px;
    color: #888;
    text-transform: uppercase;
    margin-bottom: 2px;
}}
.status-text {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 52px;
    line-height: 1;
    color: {c['border']};
    letter-spacing: 2px;
}}
.status-risk-pill {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 700;
    color: {c['border']};
    background: {c['bg']};
    border: 1px solid {c['border']};
    border-radius: 999px;
    padding: 6px 16px;
}}
</style>
<div class="status-card">
    <div style="display:flex; align-items:center; gap:16px;">
        <div class="status-dot"></div>
        <div>
            <div class="status-label">Market Status</div>
            <div class="status-text">{market_status}</div>
        </div>
    </div>
    <div class="status-risk-pill">Risk Level: {crash_risk}%</div>
</div>
""", unsafe_allow_html=True)
st.markdown("")
c1, c2, c3, c4, c5, c6 = st.columns(6)


def mcard(value, label, accent, value_color, bg):
    return f"""
<style>
.mcard {{
    border: 1px solid #1e2130;
    border-radius: 10px;
    padding: 16px 18px;
    text-align: center;
}}
.mcard-val {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 22px;
    font-weight: 700;
    color: {value_color};
}}
.mcard-label {{
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #555;
    margin-top: 4px;
}}
</style>
<div class="mcard" style="background:{bg}; border-top:2px solid {accent};">
    <div class="mcard-val">{value}</div>
    <div class="mcard-label">{label}</div>
</div>
"""


stats = load_model_stats()

xgb_stats = stats.get("xgb") or {}
xgb_f1 = xgb_stats.get("test_f1", 0.0)
delta = latest["Close"] - prev["Close"]
pct = delta / prev["Close"] * 100
rsi = latest.get("rsi", 0)
pol = float(sentiment_index)

nifty_price = nifty.get("price") if isinstance(nifty, dict) else None
nifty_pct = nifty.get("pct") if isinstance(nifty, dict) else None

if nifty_price is not None and nifty_pct is not None:
    nifty_card_val = f"{nifty_price:,.2f}"
    nifty_card_color = "#1db954" if nifty_pct >= 0 else "#e8372a"
    nifty_card_bg = (
        "linear-gradient(160deg, #041a0e, #0a0c14)"
        if nifty_pct >= 0
        else "linear-gradient(160deg, #1a0808, #0a0c14)"
    )
else:
    nifty_card_val = "N/A"
    nifty_card_color = "#334155"
    nifty_card_bg = "linear-gradient(160deg, #0d0f1a, #0a0c14)"

with c1:
    st.markdown(mcard(
        f"{xgb_f1:.3f}",
        "XGBOOST F1",
        "#334155",
        "#9ca3af",
        "linear-gradient(160deg, #0d0f1a, #0a0c14)",
    ), unsafe_allow_html=True)
with c2:
    dc_color = "#1db954" if pct >= 0 else "#e8372a"
    dc_bg = "linear-gradient(160deg, #041a0e, #0a0c14)" if pct >= 0 else "linear-gradient(160deg, #1a0808, #0a0c14)"
    st.markdown(mcard(
        f"{pct:+.2f}%",
        "DAY CHANGE",
        dc_color,
        dc_color,
        dc_bg,
    ), unsafe_allow_html=True)
with c3:
    rf_stats = stats.get("rf") or {}
    rf_f1 = rf_stats.get("test_f1", 0.0)
    st.markdown(mcard(
        f"{rf_f1:.3f}",
        "RANDOM FOREST F1",
        "#334155",
        "#9ca3af",
        "linear-gradient(160deg, #0d0f1a, #0a0c14)",
    ), unsafe_allow_html=True)
with c4:
    rsi_color = "#e8372a" if rsi > 70 else "#1db954" if rsi < 30 else "#334155"
    rsi_bg = "linear-gradient(160deg, #1a0808, #0a0c14)" if rsi > 70 else "linear-gradient(160deg, #041a0e, #0a0c14)" if rsi < 30 else "linear-gradient(160deg, #0d0f1a, #0a0c14)"
    st.markdown(mcard(
        f"{rsi:.0f}",
        "RSI (14)",
        rsi_color,
        rsi_color,
        rsi_bg,
    ), unsafe_allow_html=True)
with c5:
    pol_color = "#e8372a" if pol < 0 else "#1db954" if pol > 0 else "#334155"
    pol_bg = "linear-gradient(160deg, #1a0808, #0a0c14)" if pol < 0 else "linear-gradient(160deg, #041a0e, #0a0c14)" if pol > 0 else "linear-gradient(160deg, #0d0f1a, #0a0c14)"
    st.markdown(mcard(
        f"{pol:+.3f}",
        "SENTIMENT",
        pol_color,
        pol_color,
        pol_bg,
    ), unsafe_allow_html=True)
with c6:
    st.markdown(mcard(
        nifty_card_val,
        "NIFTY 50 LIVE",
        nifty_card_color,
        nifty_card_color,
        nifty_card_bg,
    ), unsafe_allow_html=True)

st.markdown("")
macd_val = float(latest.get("macd", 0.0))
macd_hist = float(latest.get("macd_hist", 0.0))
sent_idx = float(sentiment_index)
gdp_growth = float(macro_vals.get("gdp_growth", 0.0))
cpi_inflation = float(macro_vals.get("cpi_inflation", 0.0))
repo_rate = float(macro_vals.get("repo_rate", 0.0))
macro_last_date = macro_vals.get("_macro_last_date")
macro_mtime = macro_vals.get("_macro_file_mtime")
macro_source = macro_vals.get("_macro_source")

freshness = None
if macro_mtime:
    mtime_dt = datetime.utcfromtimestamp(macro_mtime)
    age_hours = (datetime.utcnow() - mtime_dt).total_seconds() / 3600.0
    freshness = f"Macro updated {mtime_dt:%Y-%m-%d %H:%M} UTC (\u2248{age_hours:.1f}h ago)"
elif macro_last_date:
    freshness = f"Macro snapshot date: {macro_last_date}"
source_label = f"Source: {macro_source}" if macro_source else None

# Optional: show freshness only in debug mode
if os.getenv("SHOW_MACRO_FRESHNESS", "0") == "1" and freshness:
    st.markdown(
        f"<div style='font-size:13px;color:#94a3b8;margin-bottom:4px;'>[macro] {freshness}"
        + (f" • {source_label}" if source_label else "")
        + "</div>",
        unsafe_allow_html=True,
    )

rsi_color = "#f59e0b" if rsi > 70 else "#38bdf8" if rsi >= 50 else "#94a3b8"
macd_color = "#38bdf8" if macd_val >= 0 else "#f59e0b"
macd_hist_color = "#38bdf8" if macd_hist >= 0 else "#f59e0b"
sent_color = "#38bdf8" if sent_idx > 0.05 else "#f59e0b" if sent_idx < -0.05 else "#94a3b8"
gdp_color = "#38bdf8" if gdp_growth >= 0 else "#f59e0b"
cpi_color = "#38bdf8" if cpi_inflation <= 3 else "#f59e0b" if cpi_inflation <= 5 else "#ef4444"
repo_color = "#f59e0b" if repo_rate > 0 else "#38bdf8"

nifty_snap_value = nifty_card_val
nifty_snap_color = nifty_card_color

snapshot_rows = [
    ("NIFTY 50 Live", nifty_snap_value, nifty_snap_color),
    ("RSI (14)", f"{rsi:.0f}", rsi_color),
    ("MACD", f"{macd_val:+.4f}", macd_color),
    ("MACD Histogram", f"{macd_hist:+.4f}", macd_hist_color),
    ("Sentiment Index", f"{sent_idx:+.3f}", sent_color),
    ("GDP Growth", f"{gdp_growth:+.2f}%", gdp_color),
    ("CPI Inflation", f"{cpi_inflation:+.2f}%", cpi_color),
    ("Repo Rate", f"{repo_rate:+.2f}%", repo_color),
]

snapshot_html = """
<style>
.snapshot-panel {
    background: #0f1117;
    border: 1px solid #1e2130;
    border-radius: 12px;
    padding: 18px 20px;
    margin-top: 6px;
}
.snapshot-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #8b93a7;
    margin-bottom: 12px;
}
.snapshot-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px 16px;
}
.snap-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 12px;
    background: #0b0d13;
    border: 1px solid #1b1f2d;
    border-radius: 10px;
}
.snap-label {
    color: #cbd5e1;
    font-size: 12px;
}
.snap-value {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 12px;
    padding: 4px 8px;
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.08);
}
@media (max-width: 1200px) {
    .snapshot-grid { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 700px) {
    .snapshot-grid { grid-template-columns: 1fr; }
}
</style>
<div class="snapshot-panel">
  <div class="snapshot-title">Live Feature Snapshot</div>
  <div class="snapshot-grid">
"""

for label, value, color in snapshot_rows:
    snapshot_html += (
        f"<div class='snap-row'>"
        f"<div class='snap-label'>{label}</div>"
        f"<div class='snap-value' style='color:{color}; border-color:{color}33;'>{value}</div>"
        f"</div>"
    )

snapshot_html += """
  </div>
</div>
"""

st.markdown(snapshot_html, unsafe_allow_html=True)

st.markdown("")

feature_categories = {
    "Z-Score": ("TECHNICAL", "#f59e0b"),
    "RSI": ("TECHNICAL", "#f59e0b"),
    "MACD": ("TECHNICAL", "#f59e0b"),
    "MACD Hist": ("TECHNICAL", "#f59e0b"),
    "Sentiment": ("SENTIMENT", "#2dd4bf"),
    "GDP Growth": ("MACRO", "#a855f7"),
    "CPI Inflation": ("MACRO", "#a855f7"),
    "Repo Rate": ("MACRO", "#a855f7"),
    "BSADF": ("TECHNICAL", "#f59e0b"),
}

feature_label_map = {
    "zscore_value": "Z-Score",
    "zscore": "Z-Score",
    "rsi": "RSI",
    "macd": "MACD",
    "macd_hist": "MACD Hist",
    "macd_diff": "MACD Hist",
    "daily_sentiment_index": "Sentiment",
    "avg_polarity": "Sentiment",
    "gdp_growth": "GDP Growth",
    "cpi_inflation": "CPI Inflation",
    "repo_rate": "Repo Rate",
    "bsadf": "BSADF",
}


def build_feature_importance(model):
    if model is None or not hasattr(model, "feature_importances_"):
        return {}

    names = getattr(model, "feature_names_in_", None)
    if names is None:
        names = feature_cols or []

    importances = getattr(model, "feature_importances_", [])
    if not len(names) or not len(importances):
        return {}

    size = min(len(names), len(importances))
    raw = {}
    for name, imp in zip(names[:size], importances[:size]):
        label = feature_label_map.get(str(name))
        if not label:
            continue
        raw[label] = raw.get(label, 0.0) + float(imp)

    total = sum(raw.values())
    if total <= 0:
        return {}

    return {k: (v / total) * 100 for k, v in raw.items()}


importance_dict = build_feature_importance(
    xgb_model) or build_feature_importance(rf_model)
if importance_dict:
    rows_html = ""
    for feat, pct in sorted(importance_dict.items(), key=lambda x: -x[1]):
        cat, color = feature_categories.get(feat, ("TECHNICAL", "#f59e0b"))
        rows_html += f"""
        <div style="display:flex; align-items:center; 
            padding:10px 0; border-bottom:1px solid #12151f;">
            <div style="width:130px; font-family:'DM Sans',sans-serif; 
                font-size:13px; color:#9ca3af;">{feat}</div>
            <div style="flex:1; background:#1e2235; border-radius:4px; 
                height:8px; margin:0 16px; overflow:hidden;">
                <div style="width:{pct}%; height:100%; 
                    background:{color}; border-radius:4px;
                    box-shadow: 0 0 8px {color}55;"></div>
            </div>
            <div style="width:42px; text-align:right; 
                font-family:'JetBrains Mono',monospace; 
                font-size:13px; font-weight:700; color:#e8eaf0;">
                {pct:.0f}%
            </div>
        </div>
        """

    legend_html = """
    <div style="display:flex; gap:20px; margin-bottom:16px; justify-content:flex-end;">
        <span style="font-size:11px; color:#f59e0b; letter-spacing:1px">● TECHNICAL</span>
        <span style="font-size:11px; color:#a855f7; letter-spacing:1px">● MACRO</span>
        <span style="font-size:11px; color:#2dd4bf; letter-spacing:1px">● SENTIMENT</span>
    </div>
"""

    st.markdown(f"""
<div style="background:#0d0f1a; border:1px solid #1a1f2e; 
    border-radius:14px; padding:24px 28px; margin:16px 0;">
    <div style="display:flex; justify-content:space-between; 
        align-items:flex-start; margin-bottom:8px;">
        <div style="font-family:'Playfair Display',serif; font-size:18px; 
            font-weight:700; color:#f0f2f8; max-width:260px; line-height:1.3;">
            Feature Contribution to Today's Prediction
        </div>
        {legend_html}
    </div>
    {rows_html}
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab_price, tab_news, tab_stats = st.tabs(
    [
        "∷  Price Analysis",
        "⌖  Live News",
        "⊟  Statistics",
    ]
)

# ── Tab 1: Price Analysis ──────────────────────────────────────────────────
with tab_price:
    col_chart, col_gauge = st.columns([3, 1])

    with col_chart:
        st.plotly_chart(build_price_chart(df, ticker_label),
                        use_container_width=True)
        st.markdown("""
    Legend: 
    - 🔴 Red zones = Bubble (hybrid ensemble)
    - 🔵 Blue zones = Crash/High Risk (hybrid ensemble)
    - 🟠 Dashed = Rolling mean
""")

    with col_gauge:
        st.plotly_chart(build_gauge(crash_prob), use_container_width=True)
        st.markdown("")

    # Sentiment chart
    if not daily_sent.empty:
        st.plotly_chart(build_sentiment_chart(
            daily_sent), use_container_width=True)
    else:
        if use_sentiment:
            st.info("No sentiment data yet. Ensure internet is connected.")
        else:
            st.info(
                "Enable 'Live FinBERT Sentiment' in the sidebar to see this chart.")

# ── Tab 2: Live News ───────────────────────────────────────────────────────
with tab_news:
    if not raw_sent.empty:
        st.markdown(
            f"### 📰 Live Financial News  ({len(raw_sent)} articles, last {news_days} days)")

        # Filter controls
        f1, f2, f3 = st.columns(3)
        with f1:
            filter_sent = st.multiselect(
                "Filter by sentiment", ["positive", "negative", "neutral"],
                default=["positive", "negative", "neutral"]
            )
        with f2:
            min_pol = st.slider("Min |polarity|", 0.0, 1.0, 0.0, 0.05)
        with f3:
            sort_by = st.selectbox(
                "Sort by", ["Date (newest)", "Polarity (highest)", "Polarity (lowest)"])

        filtered = raw_sent[raw_sent["label"].isin(filter_sent)].copy()
        filtered = filtered[filtered["polarity"].abs() >= min_pol]

        if sort_by == "Date (newest)":
            filtered = filtered.sort_values("date", ascending=False)
        elif sort_by == "Polarity (highest)":
            filtered = filtered.sort_values("polarity", ascending=False)
        else:
            filtered = filtered.sort_values("polarity", ascending=True)

        st.markdown(f"Showing **{len(filtered)}** articles")

        for _, row in filtered.head(30).iterrows():
            badge_cls = (
                f"badge-{'pos' if row['label'] == 'positive' else 'neg' if row['label'] == 'negative' else 'neu'}"
            )
            card_cls = (
                f"news-card {'news-pos' if row['label'] == 'positive' else 'news-neg' if row['label'] == 'negative' else ''}"
            )
            headline = row.get("headline", row.get("title", ""))
            st.markdown(f"""
<div class="{card_cls}">
  <span class="{badge_cls}">{row['label'].upper()}</span>
  &nbsp;<small>{row['date']}</small>
  &nbsp;&nbsp;<small style="color:#888">{row.get('source', '')}</small>
  <br/>
  <b>{headline}</b>
  <br/>
  <small>Polarity: <b style="color:{'#27ae60' if row['polarity'] > 0 else '#e74c3c'}">{row['polarity']:+.3f}</b></small>
</div>
""", unsafe_allow_html=True)

        # Export
        csv = filtered.to_csv(index=False)
        st.download_button("⬇️ Download News CSV", csv,
                           file_name="live_news_sentiment.csv", mime="text/csv")

    else:
        st.info(
            "📰 News will appear here once FinBERT sentiment is enabled "
            "and internet is connected."
        )
        st.markdown("""
** To enable live news: **
1. Toggle '🤖 Live FinBERT Sentiment' in the sidebar
2. (Optional) Add free API keys to `.env` for more articles:
   ```
   NEWS_API_KEY= your_newsapi_key   # https://newsapi.org/register
   GNEWS_API_KEY= your_gnews_key    # https://gnews.io/
   ```
""")

# ── Tab 3: Statistics ──────────────────────────────────────────────────────
with tab_stats:
    col_left, col_right = st.columns(2)

    if "label" not in df.columns:
        st.warning("No label column available to compute statistics.")
    else:
        with col_left:
            st.markdown("#### Label Distribution")
            label_df = df["label"].value_counts().reset_index()
            label_df.columns = ["Label", "Days"]
            label_df["% of Time"] = (label_df["Days"] / len(df) * 100).round(1)

            color_map = {
                "Normal": "#27ae60",
                "Bubble": "#e74c3c",
                "Crash": "#2980b9",
                "High Risk": "#1e3a8a",
            }
            order_labels = ["Normal", "Bubble", "High Risk", "Crash"]
            label_df = label_df[label_df["Label"].isin(order_labels)].copy()
            label_df["Label"] = pd.Categorical(
                label_df["Label"], order_labels, ordered=True)
            label_df = label_df.sort_values("Label")

            pie_colors = [color_map.get(str(lbl), "#555")
                          for lbl in label_df["Label"]]

            fig_pie = go.Figure(go.Pie(
                labels=label_df["Label"],
                values=label_df["Days"],
                marker_colors=pie_colors,
                hole=0.4,
            ))
            fig_pie.update_layout(
                height=280, template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=20, b=0),
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            st.dataframe(label_df, hide_index=True, use_container_width=True)

        with col_right:
            st.markdown("#### Recent Data (last 20 days)")
            sentiment_col = None
            for cand in ["avg_polarity", "daily_sentiment_index", "sentiment", "polarity"]:
                if cand in df.columns:
                    sentiment_col = cand
                    break

            cols = ["Close", "label"]
            if "rsi" in df.columns:
                cols.append("rsi")
            if sentiment_col:
                cols.append(sentiment_col)

            recent = df[cols].tail(20).copy()
            if "Date" in df.columns:
                recent.index = pd.to_datetime(
                    df.loc[recent.index, "Date"]).dt.strftime("%Y-%m-%d")
            else:
                recent.index = pd.to_datetime(recent.index, errors="coerce")
                recent.index = recent.index.strftime("%Y-%m-%d")
            recent = recent.round(3)

            rename_map = {
                "Close": "Price",
                "label": "Label",
                "rsi": "RSI",
            }
            if sentiment_col:
                rename_map[sentiment_col] = "Sentiment"
            recent.rename(columns=rename_map, inplace=True)

            def color_label(val):
                colors = {"Bubble": "background-color:#5c1010",
                          "Crash":  "background-color:#0a2a4a",
                          "High Risk": "background-color:#0a2a4a",
                          "Normal": "background-color:#0a3020"}
                return colors.get(val, "")

            styled = recent.style.map(color_label, subset=["Label"])
            st.dataframe(styled, use_container_width=True)

    # Download
    csv = df.to_csv()
    st.download_button(
        "⬇️ Download Full Analysis (CSV)", csv,
        file_name=f"{ticker.replace('^', '')}_bubble_analysis.csv",
        mime="text/csv",
    )

# ─────────────────────────────────────────────────────────────────────────────
# AUTO REFRESH
# ─────────────────────────────────────────────────────────────────────────────

if auto_refresh:
    time.sleep(300)
    st.rerun()
