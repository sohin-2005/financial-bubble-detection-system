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
import pickle
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
sys.path.append(".")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Real-Time Bubble Detector",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CACHED DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)   # refresh every 5 minutes
def load_price_data(ticker: str, period: str, window: int) -> pd.DataFrame:
    """Download + label price data. Cached for 5 min."""
    from src.data_ingestion   import download_ticker
    from src.zscore_labeling  import compute_zscore_labels

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
    path = "models/stacking_ensemble.pkl"
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_scaler_and_features():
    if not os.path.exists("models/scaler.pkl"):
        return None, None
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return scaler, feature_cols


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
    Return probability of crash (0–1).

    If ensemble is trained → use it.
    Otherwise → heuristic based on Z-score + RSI + sentiment.
    """
    # ── Heuristic (no model trained yet) ──────────────────────────────────
    if ensemble is None:
        z   = latest_row.get("zscore", 0)
        rsi = latest_row.get("rsi", 50)
        pol = latest_row.get("avg_polarity", 0)

        # Higher Z = more bubble → higher crash risk
        z_risk  = max(0, min(1, (z - 1.0) / 2.0))
        # RSI > 70 = overbought → risk
        rsi_risk = max(0, (rsi - 50) / 50)
        # Negative sentiment → crash risk
        sent_risk = max(0, -pol)

        prob = 0.5 * z_risk + 0.3 * rsi_risk + 0.2 * sent_risk
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


# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def build_price_chart(df: pd.DataFrame, ticker_label: str) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            f"{ticker_label} — Price & Bubble Zones",
            "Z-Score",
            "Volume"
        ),
        row_heights=[0.6, 0.25, 0.15],
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

    # Z-score line + thresholds
    fig.add_trace(go.Scatter(
        x=df.index, y=df["zscore"],
        name="Z-Score",
        line=dict(color="#9b59b6", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(155,89,182,0.08)",
    ), row=2, col=1)

    for level, color, label in [
        (2, "rgba(231,76,60,0.8)", "+2 Bubble"),
        (-2, "rgba(41,128,185,0.8)", "−2 Crash"),
    ]:
        fig.add_hline(
            y=level, line_dash="dash", line_color=color,
            annotation_text=label, annotation_position="right",
            row=2, col=1,
        )

    # Volume bars
    vol_colors = [
        "#27ae60" if c >= o else "#e74c3c"
        for c, o in zip(df["Close"], df["Open"])
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        name="Volume", marker_color=vol_colors, opacity=0.6,
    ), row=3, col=1)

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
    label_colors = {"Bubble": "rgba(231,76,60,0.12)", "Crash": "rgba(41,128,185,0.12)"}
    for label, color in label_colors.items():
        in_region  = False
        start_date = None
        mask       = df["label"] == label
        for i, (date, is_label) in enumerate(zip(df.index, mask)):
            if is_label and not in_region:
                in_region  = True
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
                {"range": [60, 100],"color": "rgba(231,76,60,0.15)"},
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
    "NIFTY 50 (Index)":  "^NSEI",
    "SENSEX (Index)":    "^BSESN",
    "NIFTY Bank":        "^NSEBANK",
    "Reliance":          "RELIANCE.NS",
    "TCS":               "TCS.NS",
    "HDFC Bank":         "HDFCBANK.NS",
    "Infosys":           "INFY.NS",
    "ICICI Bank":        "ICICIBANK.NS",
    "SBI":               "SBIN.NS",
    "Bajaj Finance":     "BAJFINANCE.NS",
    "Adani Enterprises": "ADANIENT.NS",
}

with st.sidebar:
    st.markdown("## 📡 Real-Time Controls")
    st.markdown("---")

    ticker_label = st.selectbox("📊 Index / Stock", list(TICKER_OPTIONS.keys()))
    ticker       = TICKER_OPTIONS[ticker_label]

    period  = st.selectbox("📅 Historical Period",
                            ["1y", "2y", "3y", "5y", "10y"], index=2)
    window  = st.slider("🔢 Z-Score Window (days)", 10, 90, 30, 5)
    news_days = st.slider("📰 News Lookback (days)", 1, 30, 7)

    st.markdown("---")
    use_sentiment = st.toggle("🤖 Live FinBERT Sentiment", value=True,
                               help="Fetches real news + runs FinBERT (takes ~30s first time)")

    auto_refresh = st.toggle("🔄 Auto-Refresh (5 min)", value=False)

    st.markdown("---")
    st.markdown("**Data Sources:**")
    st.markdown("- 📈 Yahoo Finance (live)")
    st.markdown("- 📰 RSS Feeds (free)")
    st.markdown("- 🔑 NewsAPI *(optional)*")
    st.markdown("- 🔑 GNews *(optional)*")

    st.markdown("---")
    if st.button("🔍 Analyze Now", type="primary", use_container_width=True):
        st.cache_data.clear()   # force fresh data

    last_update = st.empty()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("# 📡 Real-Time Stock Market Bubble Detector")
st.markdown(
    "*Live data · FinBERT sentiment · Z-Score labeling · Stacking Ensemble*  "
    f"&nbsp;&nbsp;`{ticker}` &nbsp; Last updated: `{datetime.now().strftime('%H:%M:%S')}`"
)

# ── Load price data ────────────────────────────────────────────────────────
with st.spinner(f"📥 Fetching live price data for {ticker}..."):
    df = load_price_data(ticker, period, window)

if df.empty:
    st.error(f"❌ Could not fetch data for `{ticker}`. Check your internet connection.")
    st.stop()

# ── Load sentiment ─────────────────────────────────────────────────────────
daily_sent = pd.DataFrame()
raw_sent   = pd.DataFrame()

if use_sentiment:
    with st.spinner("📰 Fetching live news & running FinBERT..."):
        try:
            daily_sent, raw_sent = compute_live_sentiment(days_back=news_days)
            # Merge into price df
            from src.sentiment_engine import merge_sentiment_with_prices
            df = merge_sentiment_with_prices(df, daily_sent)
        except Exception as exc:
            st.warning(f"Sentiment unavailable: {exc}")

# ── Load models ────────────────────────────────────────────────────────────
ensemble, scaler, feature_cols = None, None, None
try:
    ensemble              = load_ensemble()
    scaler, feature_cols  = load_scaler_and_features()
except Exception:
    pass

# ── Latest row values ─────────────────────────────────────────────────────
latest       = df.iloc[-1]
prev         = df.iloc[-2] if len(df) > 1 else latest
crash_prob   = estimate_crash_probability(latest, ensemble, scaler, feature_cols)

# ─────────────────────────────────────────────────────────────────────────────
# STATUS BANNER
# ─────────────────────────────────────────────────────────────────────────────

label = latest.get("label", "Normal")
zscore = latest.get("zscore", 0.0)

banner_class = {
    "Bubble": "banner-bubble",
    "Crash":  "banner-crash",
    "Normal": "banner-normal",
}[label]

icons = {"Bubble": "🔴", "Crash": "🔵", "Normal": "🟢"}
st.markdown(
    f'<div class="{banner_class}">'
    f'{icons[label]} MARKET STATUS: <b>{label.upper()}</b>'
    f' &nbsp;|&nbsp; Z-Score: <b>{zscore:.2f}</b>'
    f' &nbsp;|&nbsp; Crash Risk: <b>{crash_prob*100:.0f}%</b>'
    f'</div>',
    unsafe_allow_html=True,
)
st.markdown("")

# ─────────────────────────────────────────────────────────────────────────────
# METRIC CARDS ROW
# ─────────────────────────────────────────────────────────────────────────────

c1, c2, c3, c4, c5, c6 = st.columns(6)

def metric_html(value, label, color="#ffffff"):
    return (
        f'<div class="stat-card">'
        f'<p class="stat-value" style="color:{color}">{value}</p>'
        f'<p class="stat-label">{label}</p></div>'
    )

with c1:
    st.markdown(metric_html(f"₹{latest['Close']:,.0f}", "Current Price"), unsafe_allow_html=True)
with c2:
    delta = latest["Close"] - prev["Close"]
    pct   = delta / prev["Close"] * 100
    col   = "#27ae60" if delta >= 0 else "#e74c3c"
    st.markdown(metric_html(f"{pct:+.2f}%", "Day Change", color=col), unsafe_allow_html=True)
with c3:
    z_col = "#e74c3c" if zscore > 2 else "#2980b9" if zscore < -2 else "#27ae60"
    st.markdown(metric_html(f"{zscore:.2f}σ", "Z-Score", color=z_col), unsafe_allow_html=True)
with c4:
    rsi = latest.get("rsi", 0)
    rsi_col = "#e74c3c" if rsi > 70 else "#2980b9" if rsi < 30 else "#f0f0f0"
    st.markdown(metric_html(f"{rsi:.0f}", "RSI (14)", color=rsi_col), unsafe_allow_html=True)
with c5:
    pol = latest.get("avg_polarity", 0)
    pol_col = "#27ae60" if pol > 0.05 else "#e74c3c" if pol < -0.05 else "#f39c12"
    st.markdown(metric_html(f"{pol:+.3f}", "Sentiment", color=pol_col), unsafe_allow_html=True)
with c6:
    crash_col = "#e74c3c" if crash_prob > 0.6 else "#f39c12" if crash_prob > 0.35 else "#27ae60"
    st.markdown(metric_html(f"{crash_prob*100:.0f}%", "Crash Risk", color=crash_col), unsafe_allow_html=True)

st.markdown("")

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab_price, tab_news, tab_stats, tab_model = st.tabs(
    ["📈 Price Analysis", "📰 Live News", "📊 Statistics", "🤖 Model Info"]
)

# ── Tab 1: Price Analysis ──────────────────────────────────────────────────
with tab_price:
    col_chart, col_gauge = st.columns([3, 1])

    with col_chart:
        st.plotly_chart(build_price_chart(df, ticker_label), use_container_width=True)

    with col_gauge:
        st.plotly_chart(build_gauge(crash_prob), use_container_width=True)
        st.markdown("")

        # Legend
        st.markdown("""
**Legend:**
- 🔴 Red zones = Bubble (Z > +2)
- 🔵 Blue zones = Crash (Z < −2)
- 🟠 Dashed = Rolling mean
""")

    # Sentiment chart
    if not daily_sent.empty:
        st.plotly_chart(build_sentiment_chart(daily_sent), use_container_width=True)
    else:
        if use_sentiment:
            st.info("No sentiment data yet. Ensure internet is connected.")
        else:
            st.info("Enable 'Live FinBERT Sentiment' in the sidebar to see this chart.")

# ── Tab 2: Live News ───────────────────────────────────────────────────────
with tab_news:
    if not raw_sent.empty:
        st.markdown(f"### 📰 Live Financial News  ({len(raw_sent)} articles, last {news_days} days)")

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
            sort_by = st.selectbox("Sort by", ["Date (newest)", "Polarity (highest)", "Polarity (lowest)"])

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
            badge_cls = f"badge-{'pos' if row['label']=='positive' else 'neg' if row['label']=='negative' else 'neu'}"
            card_cls  = f"news-card {'news-pos' if row['label']=='positive' else 'news-neg' if row['label']=='negative' else ''}"
            st.markdown(f"""
<div class="{card_cls}">
  <span class="{badge_cls}">{row['label'].upper()}</span>
  &nbsp;<small>{row['date']}</small>
  &nbsp;&nbsp;<small style="color:#888">{row.get('source','')}</small>
  <br/>
  <b>{row['headline']}</b>
  <br/>
  <small>Polarity: <b style="color:{'#27ae60' if row['polarity']>0 else '#e74c3c'}">{row['polarity']:+.3f}</b></small>
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
**To enable live news:**
1. Toggle '🤖 Live FinBERT Sentiment' in the sidebar
2. (Optional) Add free API keys to `.env` for more articles:
   ```
   NEWS_API_KEY=your_newsapi_key   # https://newsapi.org/register
   GNEWS_API_KEY=your_gnews_key    # https://gnews.io/
   ```
""")

# ── Tab 3: Statistics ──────────────────────────────────────────────────────
with tab_stats:
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Label Distribution")
        label_df = df["label"].value_counts().reset_index()
        label_df.columns = ["Label", "Days"]
        label_df["% of Time"] = (label_df["Days"] / len(df) * 100).round(1)

        fig_pie = go.Figure(go.Pie(
            labels=label_df["Label"],
            values=label_df["Days"],
            marker_colors=["#27ae60", "#e74c3c", "#2980b9"],
            hole=0.4,
        ))
        fig_pie.update_layout(
            height=280, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=20,b=0),
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.dataframe(label_df, hide_index=True, use_container_width=True)

    with col_right:
        st.markdown("#### Recent Data (last 20 days)")
        recent = df[["Close", "zscore", "label", "rsi", "avg_polarity"]].tail(20).copy()
        recent.index = recent.index.strftime("%Y-%m-%d")
        recent = recent.round(3)
        recent.rename(columns={
            "Close": "Price", "zscore": "Z-Score",
            "label": "Label",  "rsi": "RSI",
            "avg_polarity": "Sentiment",
        }, inplace=True)

        def color_label(val):
            colors = {"Bubble": "background-color:#5c1010",
                      "Crash":  "background-color:#0a2a4a",
                      "Normal": "background-color:#0a3020"}
            return colors.get(val, "")

        styled = recent.style.map(color_label, subset=["Label"])
        st.dataframe(styled, use_container_width=True)

    st.markdown("#### Z-Score Distribution")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=df["zscore"], nbinsx=60, name="Z-Score",
        marker_color="#9b59b6", opacity=0.8,
    ))
    for x, color, name in [(2, "red", "+2"), (-2, "blue", "-2")]:
        fig_hist.add_vline(x=x, line_dash="dash", line_color=color,
                           annotation_text=name)
    fig_hist.update_layout(
        height=250, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40,r=40,t=20,b=20),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Download
    csv = df.to_csv()
    st.download_button(
        "⬇️ Download Full Analysis (CSV)", csv,
        file_name=f"{ticker.replace('^','')}_bubble_analysis.csv",
        mime="text/csv",
    )

# ── Tab 4: Model Info ──────────────────────────────────────────────────────
with tab_model:
    st.markdown("#### Stacking Ensemble Status")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        if ensemble:
            st.success("✅ Trained stacking ensemble loaded from `models/`")
        else:
            st.warning("⚠️ No trained model found. Run `python main.py` to train.")
            st.markdown("""
**Training steps:**
```bash
# 1. Download data + compute labels
python src/data_ingestion.py
python src/zscore_labeling.py

# 2. Train full pipeline  
python main.py

# 3. Restart dashboard
streamlit run app.py
```
""")

    with col_m2:
        st.markdown("""
**Architecture:**
```
Input Features
    ↓
┌──────────────┐  ┌──────────────┐
│ Random Forest│  │   XGBoost    │  ← Base models
└──────┬───────┘  └──────┬───────┘
       └────────┬─────────┘
                ↓
     ┌──────────────────────┐
     │  Logistic Regression │  ← Meta-model
     └──────────────────────┘
                ↓
     P(Normal) P(Bubble) P(Crash)
```
""")

    st.markdown("#### Feature List")
    features = [
        ("zscore", "Rolling Z-score of close price"),
        ("rolling_mean", "Rolling mean of close price"),
        ("rolling_std", "Rolling std dev of close price"),
        ("log_return", "Daily log return"),
        ("rsi", "Relative Strength Index (14-day)"),
        ("macd", "MACD line"),
        ("macd_signal", "MACD signal line"),
        ("macd_diff", "MACD histogram"),
        ("bb_width", "Bollinger Band width"),
        ("roc", "Rate of change (14-day)"),
        ("avg_polarity", "FinBERT daily sentiment polarity"),
        ("avg_positive", "FinBERT positive probability"),
        ("avg_negative", "FinBERT negative probability"),
        ("sentiment_momentum", "3-day rolling sentiment mean"),
    ]
    st.dataframe(
        pd.DataFrame(features, columns=["Feature", "Description"]),
        hide_index=True, use_container_width=True
    )

# ─────────────────────────────────────────────────────────────────────────────
# AUTO REFRESH
# ─────────────────────────────────────────────────────────────────────────────

last_update.markdown(
    f"<small style='color:#888'>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small>",
    unsafe_allow_html=True,
)

if auto_refresh:
    time.sleep(300)
    st.rerun()