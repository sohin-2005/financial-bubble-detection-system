"""
REAL-TIME SENTIMENT ENGINE
============================
Processes LIVE news headlines through FinBERT.

Pipeline:
    fetch_all_news()          ← src/news_fetcher.py
         ↓
    FinBERTSentimentAnalyzer  ← this file
         ↓
    Daily Sentiment Index     (one polarity score per trading day)
         ↓
    Merged with price data    (for ML features)

Two processing modes:
    FAST  – headline only (3-5 words → very quick, <0.1s per item)
    FULL  – headline + summary (more context, ~0.3s per item)

Run standalone:
    python src/sentiment_engine.py
"""

import os
import time
import warnings
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
from typing import Optional

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# FINBERT ANALYZER
# ─────────────────────────────────────────────────────────────────────────────

class FinBERTAnalyzer:
    """
    Wrapper around ProsusAI/finbert for financial sentiment.

    FinBERT output labels → index mapping:
        0 = positive
        1 = negative
        2 = neutral

    Polarity formula (from Atsiwo 2025):
        polarity = P(positive) − P(negative)
        Range: −1.0  (very negative)  …  +1.0  (very positive)
    """

    MODEL_NAME = "ProsusAI/finbert"

    def __init__(self, device: str = "auto", batch_size: int = 16):
        """
        Load FinBERT.  First run downloads ~440 MB from HuggingFace.

        Parameters
        ----------
        device     : "cpu", "cuda", or "auto" (uses GPU if available)
        batch_size : How many headlines to process in one GPU/CPU batch
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device     = device
        self.batch_size = batch_size

        print(f"Loading FinBERT on {device}  (first run ~440 MB download)...")
        t0 = time.time()
        self.tokenizer = BertTokenizer.from_pretrained(self.MODEL_NAME)
        self.model     = BertForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self.model.to(device)
        self.model.eval()
        print(f"✅ FinBERT ready  ({time.time()-t0:.1f}s)")

    # ── Single text ────────────────────────────────────────────────────────

    def analyze(self, text: str) -> dict:
        """
        Analyze one piece of text.

        Returns
        -------
        {
          "positive": 0.82,
          "negative": 0.05,
          "neutral":  0.13,
          "polarity": 0.77,          # positive − negative
          "label":    "positive"
        }
        """
        if not text or pd.isna(text):
            return {"positive": 0.33, "negative": 0.33, "neutral": 0.34,
                    "polarity": 0.0, "label": "neutral"}

        inputs = self.tokenizer(
            str(text),
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = softmax(logits, dim=1).squeeze().cpu().numpy()
        pos, neg, neu = float(probs[0]), float(probs[1]), float(probs[2])
        polarity = pos - neg
        label    = ["positive", "negative", "neutral"][int(np.argmax(probs))]

        return {
            "positive": round(pos, 4),
            "negative": round(neg, 4),
            "neutral":  round(neu, 4),
            "polarity": round(polarity, 4),
            "label":    label,
        }

    # ── Batch processing  ─────────────────────────────────────────────────

    def analyze_batch(self, texts: list[str]) -> list[dict]:
        """
        Process a list of texts in mini-batches for efficiency.

        Significantly faster than calling analyze() one-by-one on GPU.
        """
        results = []
        total   = len(texts)

        for start in range(0, total, self.batch_size):
            batch = texts[start : start + self.batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits

            probs_batch = softmax(logits, dim=1).cpu().numpy()

            for probs in probs_batch:
                pos, neg, neu = float(probs[0]), float(probs[1]), float(probs[2])
                results.append({
                    "positive": round(pos, 4),
                    "negative": round(neg, 4),
                    "neutral":  round(neu, 4),
                    "polarity": round(pos - neg, 4),
                    "label":    ["positive", "negative", "neutral"][int(np.argmax(probs))],
                })

            done = min(start + self.batch_size, total)
            print(f"  FinBERT: {done}/{total} headlines processed", end="\r")

        print()
        return results


# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE DAILY SENTIMENT INDEX  (main entry point)
# ─────────────────────────────────────────────────────────────────────────────

def compute_daily_sentiment(
    news_df: pd.DataFrame,
    analyzer: FinBERTAnalyzer,
    text_mode: str = "headline",   # "headline" | "headline+summary"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run FinBERT on every headline, then aggregate per trading day.

    Parameters
    ----------
    news_df   : DataFrame from news_fetcher.fetch_all_news()
                Must have columns: date, headline[, summary]
    analyzer  : A loaded FinBERTAnalyzer instance.
    text_mode : What text to feed to FinBERT.

    Returns
    -------
    daily_df  : One row per date with aggregate sentiment scores
    raw_df    : One row per article with individual FinBERT scores
    """
    if news_df.empty:
        print("⚠️  No news to process")
        return pd.DataFrame(), pd.DataFrame()

    df = news_df.copy()

    # Build the text to analyze
    if text_mode == "headline+summary" and "summary" in df.columns:
        df["_text"] = df["headline"].fillna("") + ". " + df["summary"].fillna("")
    else:
        df["_text"] = df["headline"].fillna("")

    texts = df["_text"].tolist()

    print(f"\nRunning FinBERT on {len(texts)} articles  (mode={text_mode})...")
    t0      = time.time()
    results = analyzer.analyze_batch(texts)
    print(f"✅ Sentiment computed in {time.time()-t0:.1f}s")

    # Attach scores back to the DataFrame
    scores_df = pd.DataFrame(results)
    raw_df    = pd.concat(
        [df[["date", "source", "headline", "relevance_score"]].reset_index(drop=True),
         scores_df],
        axis=1,
    )

    # ── Daily aggregation ────────────────────────────────────────────────
    raw_df["date"] = pd.to_datetime(raw_df["date"])

    daily = (
        raw_df
        .groupby("date")
        .agg(
            avg_polarity       = ("polarity",  "mean"),
            std_polarity       = ("polarity",  "std"),
            avg_positive       = ("positive",  "mean"),
            avg_negative       = ("negative",  "mean"),
            avg_neutral        = ("neutral",   "mean"),
            pct_positive       = ("label",     lambda x: (x == "positive").mean()),
            pct_negative       = ("label",     lambda x: (x == "negative").mean()),
            pct_neutral        = ("label",     lambda x: (x == "neutral").mean()),
            news_count         = ("headline",  "count"),
        )
        .reset_index()
    )
    daily.set_index("date", inplace=True)
    daily.fillna(0, inplace=True)

    # Sentiment momentum: 3-day rolling mean
    daily["sentiment_momentum"] = daily["avg_polarity"].rolling(3, min_periods=1).mean()

    print(f"✅ Daily sentiment computed for {len(daily)} days")
    print(daily[["avg_polarity", "pct_positive", "pct_negative", "news_count"]].tail(5))

    return daily, raw_df


# ─────────────────────────────────────────────────────────────────────────────
# MERGE SENTIMENT WITH PRICE DATA
# ─────────────────────────────────────────────────────────────────────────────

def merge_sentiment_with_prices(
    price_df: pd.DataFrame,
    daily_sentiment: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join price data with daily sentiment scores.

    Days with no news → sentiment columns filled with 0 (neutral).
    Also forward-fills weekend/holiday gaps.

    Parameters
    ----------
    price_df         : DataFrame from data_ingestion.download_ticker()
                       (must have a DatetimeIndex)
    daily_sentiment  : DataFrame from compute_daily_sentiment()

    Returns
    -------
    Merged DataFrame ready for ML feature preparation.
    """
    price_df = price_df.copy()
    price_df.index = pd.to_datetime(price_df.index).normalize()

    daily_sentiment = daily_sentiment.copy()
    daily_sentiment.index = pd.to_datetime(daily_sentiment.index).normalize()

    sentiment_cols = [
        "avg_polarity", "std_polarity",
        "avg_positive", "avg_negative",
        "pct_positive", "pct_negative",
        "news_count", "sentiment_momentum",
    ]
    available_cols = [c for c in sentiment_cols if c in daily_sentiment.columns]

    merged = price_df.join(daily_sentiment[available_cols], how="left")

    # Fill missing sentiment with neutral values
    fill_values = {
        "avg_polarity":       0.0,
        "std_polarity":       0.0,
        "avg_positive":       0.33,
        "avg_negative":       0.33,
        "pct_positive":       0.33,
        "pct_negative":       0.33,
        "news_count":         0,
        "sentiment_momentum": 0.0,
    }
    for col in available_cols:
        if col in fill_values:
            merged[col] = merged[col].fillna(fill_values[col])

    print(f"✅ Merged {len(price_df)} price rows with sentiment data")
    print(f"   Rows with news: {(merged['news_count'] > 0).sum() if 'news_count' in merged else 'N/A'}")

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# SAVE / LOAD
# ─────────────────────────────────────────────────────────────────────────────

def save_sentiment(
    daily_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    prefix: str = "nifty50",
) -> None:
    os.makedirs("data", exist_ok=True)
    daily_df.to_csv(f"data/{prefix}_daily_sentiment.csv")
    raw_df.to_csv(f"data/{prefix}_raw_sentiment.csv", index=False)
    print(f"💾 Saved daily → data/{prefix}_daily_sentiment.csv")
    print(f"💾 Saved raw   → data/{prefix}_raw_sentiment.csv")


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.news_fetcher import fetch_all_news

    # Step 1: Fetch real news
    news_df = fetch_all_news(days_back=7)

    if news_df.empty:
        # Fallback: demo headlines for testing when internet isn't available
        print("\nNo live news fetched. Using demo data for FinBERT test...")
        demo = [
            "Sensex surges 1000 points as FII buying resumes",
            "NIFTY crashes 500 points amid global recession fears",
            "RBI holds interest rates, markets react positively",
            "Adani stocks plunge on fresh controversy",
            "Indian GDP growth beats expectations at 7.5 percent",
            "Markets flat ahead of Union Budget announcement",
            "Tech stocks rally on strong quarterly earnings",
            "Rupee hits all-time low against US dollar",
            "SEBI introduces new F&O regulations to curb speculation",
            "Mutual fund inflows hit record high in January",
        ]
        news_df = pd.DataFrame({
            "date":             pd.date_range("2024-01-01", periods=10).strftime("%Y-%m-%d"),
            "source":           ["Demo"] * 10,
            "headline":         demo,
            "summary":          [""] * 10,
            "relevance_score":  [3] * 10,
        })

    # Step 2: Load FinBERT and analyze
    analyzer = FinBERTAnalyzer()
    daily_sent, raw_sent = compute_daily_sentiment(news_df, analyzer)

    # Step 3: Show results
    print("\n── Daily Sentiment Summary ──────────────────────────────")
    print(daily_sent[["avg_polarity", "pct_positive", "pct_negative", "news_count"]])

    print("\n── Headline-Level Scores (top 5) ────────────────────────")
    print(raw_sent[["date", "headline", "polarity", "label"]].head())

    # Save
    save_sentiment(daily_sent, raw_sent)
    print("\n✅ sentiment_engine.py → DONE")