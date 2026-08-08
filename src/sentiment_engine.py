"""
SENTIMENT ENGINE
================
Scores financial headlines with FinBERT (ProsusAI/finbert) and aggregates them
into a daily sentiment index that can be joined onto a price series.

    FinBERTAnalyzer            wraps the HuggingFace sentiment pipeline
    compute_daily_sentiment    headlines -> (daily index, scored headlines)
    merge_sentiment_with_prices  joins the daily index onto an OHLCV frame
"""

from __future__ import annotations

import pandas as pd
from transformers import pipeline


class FinBERTAnalyzer:
    """Lazy wrapper around the FinBERT sentiment-analysis pipeline."""
    def __init__(self):
        self._pipe = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
        )

    def analyze(self, texts: list[str]):
        return self._pipe(texts)


def _score_label(label: str, score: float) -> float:
    label = label.lower()
    if "positive" in label:
        return score
    if "negative" in label:
        return -score
    return 0.0


def compute_daily_sentiment(news_df: pd.DataFrame, analyzer: FinBERTAnalyzer):
    if news_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    if "headline" in news_df.columns:
        texts = news_df["headline"].fillna("").tolist()
    else:
        texts = news_df["title"].fillna("").tolist()
    results = analyzer.analyze(texts)

    scored = news_df.copy()
    if "headline" not in scored.columns:
        scored["headline"] = scored["title"]
    scored["label"] = [r["label"] for r in results]
    scored["score"] = [r["score"] for r in results]
    scored["polarity"] = [
        _score_label(r["label"], r["score"]) for r in results
    ]

    daily = (
        scored.groupby(scored["date"].dt.normalize())
        .agg(avg_polarity=("polarity", "mean"), count=("polarity", "size"))
        .reset_index()
        .rename(columns={"date": "date"})
    )

    daily["sentiment_momentum"] = (
        daily["avg_polarity"].rolling(3).mean().fillna(0.0)
    )

    return daily, scored


def merge_sentiment_with_prices(price_df: pd.DataFrame, daily_sent: pd.DataFrame):
    if price_df.index.name in ("Date", None):
        df = price_df.copy()
        if "Date" in df.columns:
            df["date"] = pd.to_datetime(df["Date"])
        else:
            df["date"] = pd.to_datetime(df.index)
    else:
        df = price_df.copy()
        df["date"] = pd.to_datetime(df.index)

    if daily_sent.empty:
        df["avg_polarity"] = 0.0
        return df

    daily_sent = daily_sent.copy()
    daily_sent["date"] = pd.to_datetime(daily_sent["date"], errors="coerce")
    daily_sent["date"] = daily_sent["date"].dt.tz_localize(None)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["date"] = df["date"].dt.tz_localize(None)
    merged = df.merge(daily_sent, on="date", how="left")
    merged["avg_polarity"] = merged["avg_polarity"].fillna(0.0)
    return merged
