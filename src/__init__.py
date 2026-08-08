"""Runtime modules used by the dashboard (app.py).

    data_ingestion    price download from Yahoo Finance
    zscore_labeling   technical indicators + Z-score/BSADF bubble labels
    news_fetcher      headlines from NewsAPI / GNews / RSS
    sentiment_engine  FinBERT scoring and daily sentiment aggregation

The offline training pipeline lives in `src.training`.
"""
