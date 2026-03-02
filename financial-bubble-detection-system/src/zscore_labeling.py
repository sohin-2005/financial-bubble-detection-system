"""
Z-SCORE BUBBLE LABELING MODULE
================================
Computes rolling Z-scores and labels each trading day as:
    Bubble  (Z > +2)
    Crash   (Z < −2)
    Normal  (−2 ≤ Z ≤ +2)

Also generates an interactive Plotly chart saved to outputs/.

Standalone test:
    python src/zscore_labeling.py
"""

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


LABEL_MAP = {"Normal": 0, "Bubble": 1, "Crash": 2}


def compute_zscore_labels(
    df: pd.DataFrame,
    price_col: str = "Close",
    window: int = 30,
    bubble_threshold: float = 2.0,
    crash_threshold: float = -2.0,
) -> pd.DataFrame:
    """
    Compute rolling Z-scores and assign Bubble / Crash / Normal labels.

    Formula:
        Z_t = (P_t − μ_t) / σ_t

    where μ_t and σ_t are the rolling mean and std over `window` days.

    Parameters
    ----------
    df                : DataFrame with at minimum a Close price column.
    price_col         : Column name to compute Z-score on.
    window            : Rolling window size in trading days.
    bubble_threshold  : Z-score above which day is labelled "Bubble".
    crash_threshold   : Z-score below which day is labelled "Crash".

    Returns
    -------
    df with extra columns:
        rolling_mean  rolling_std  zscore  label  label_numeric
    """
    df = df.copy()

    df["rolling_mean"] = df[price_col].rolling(window=window).mean()
    df["rolling_std"] = df[price_col].rolling(window=window).std()

    df["zscore"] = (df[price_col] - df["rolling_mean"]) / df["rolling_std"]

    def _label(z):
        if z > bubble_threshold:
            return "Bubble"
        elif z < crash_threshold:
            return "Crash"
        else:
            return "Normal"

    df["label"] = df["zscore"].apply(_label)
    df["label_numeric"] = df["label"].map(LABEL_MAP)

    df.dropna(subset=["zscore"], inplace=True)

    # Summary
    vc = df["label"].value_counts()
    print(f"\n  Z-Score Labels  (window={window}, thresh=±{bubble_threshold})")
    for lbl in ["Normal", "Bubble", "Crash"]:
        n = vc.get(lbl, 0)
        pct = n / len(df) * 100
        print(f"    {lbl:7s}: {n:4d}  ({pct:5.1f}%)")

    return df


def plot_bubble_analysis(df: pd.DataFrame, ticker: str = "NIFTY 50") -> go.Figure:
    """
    Create and save an interactive HTML chart:
        Row 1 – Price + Rolling Mean  (shaded Bubble / Crash regions)
        Row 2 – Z-Score with ±2 threshold lines
    """
    os.makedirs("outputs", exist_ok=True)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f"{ticker}  —  Price with Bubble / Crash Zones",
            "Rolling Z-Score",
        ),
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
    )

    # ── Price line ────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        name="Close Price",
        line=dict(color="#2196F3", width=1.5),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["rolling_mean"],
        name="Rolling Mean",
        line=dict(color="orange", width=1.5, dash="dash"),
        opacity=0.8,
    ), row=1, col=1)

    # ── Shade Bubble / Crash regions ─────────────────────────────────────
    for label, color in [("Bubble", "rgba(231,76,60,0.15)"),
                         ("Crash",  "rgba(41,128,185,0.15)")]:
        mask = df["label"] == label
        in_region = False
        start = None
        for i, (idx, is_lbl) in enumerate(zip(df.index, mask)):
            if is_lbl and not in_region:
                in_region, start = True, idx
            elif not is_lbl and in_region:
                in_region = False
                fig.add_vrect(x0=start, x1=df.index[i - 1],
                              fillcolor=color, line_width=0, row=1, col=1)
        if in_region:
            fig.add_vrect(x0=start, x1=df.index[-1],
                          fillcolor=color, line_width=0, row=1, col=1)

    # ── Z-Score ───────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=df["zscore"],
        name="Z-Score",
        line=dict(color="#9b59b6", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(155,89,182,0.08)",
    ), row=2, col=1)

    for level, color, label in [
        (2,  "rgba(231,76,60,0.9)",  "Bubble threshold +2"),
        (-2, "rgba(41,128,185,0.9)", "Crash threshold −2"),
    ]:
        fig.add_hline(y=level, line_dash="dash", line_color=color,
                      annotation_text=label, annotation_position="right",
                      row=2, col=1)

    fig.update_layout(
        title=f"Bubble Detection — {ticker}",
        height=650,
        template="plotly_white",
        showlegend=True,
    )

    out_path = "outputs/bubble_analysis.html"
    fig.write_html(out_path)
    print(f"  📊 Chart saved → {out_path}")
    return fig


# ── Standalone test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.data_ingestion import download_ticker, save_data

    df = download_ticker("^NSEI", period="10y")
    if df.empty:
        print("Could not download data. Check internet connection.")
        sys.exit(1)

    df_labeled = compute_zscore_labels(df, window=30)
    save_data(df_labeled, "nifty50_labeled.csv")
    plot_bubble_analysis(df_labeled, ticker="NIFTY 50")

    print("\nSample (last 5 rows):")
    print(df_labeled[["Close", "rolling_mean", "zscore", "label"]].tail(5))
