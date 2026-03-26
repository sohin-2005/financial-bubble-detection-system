"""
FULL REAL-TIME PIPELINE
========================
Runs the complete system end-to-end with LIVE data.

What happens:
  1. Download live NIFTY 50 price data from Yahoo Finance
  2. Compute Z-score labels (Bubble / Crash / Normal)
  3. Fetch real financial news from RSS feeds + optional NewsAPI
  4. Run FinBERT sentiment on real headlines
  5. Merge price + sentiment features
  6. Train stacking ensemble (RF + XGBoost + LR)
  7. Evaluate and save models → ready for the dashboard

Run:
    python main.py
"""

from src.stacking_ensemble import StackingEnsemble
from src.ml_models import prepare_features, apply_adasyn, evaluate_model, save_model
from src.sentiment_engine import FinBERTAnalyzer, compute_daily_sentiment, \
    merge_sentiment_with_prices, save_sentiment
from src.news_fetcher import fetch_all_news, save_news
from src.zscore_labeling import compute_zscore_labels, plot_bubble_analysis
from src.data_ingestion import download_ticker, save_data
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

warnings.filterwarnings("ignore")
sys.path.append(".")


os.makedirs("data",    exist_ok=True)
os.makedirs("models",  exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

TICKER = "^NSEI"      # NIFTY 50 index
PERIOD = "max"         # Maximum available history (from ~2000 onwards)
ZSCORE_WINDOW = 30           # 30-day rolling window
NEWS_DAYS = 30           # Fetch last 30 days of news for training

# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 60)
print("  FINANCIAL BUBBLE DETECTION — REAL-TIME PIPELINE")
print("=" * 60)
t_start = time.time()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Live Price Data
# ─────────────────────────────────────────────────────────────────────────────

print("\n[1/6]  Downloading live price data from Yahoo Finance...")
df_price = download_ticker(TICKER, period=PERIOD)

if df_price.empty:
    print("❌ Failed to download price data. Check internet connection.")
    sys.exit(1)

save_data(df_price, "nifty50_raw.csv")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Z-Score Labeling
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n[2/6]  Computing Z-score labels (window={ZSCORE_WINDOW} days)...")
df_labeled = compute_zscore_labels(
    df_price, price_col="Close", window=ZSCORE_WINDOW)
df_labeled.to_csv("data/nifty50_labeled.csv")
plot_bubble_analysis(df_labeled, ticker="NIFTY 50")

print(f"\n  Label distribution:")
print(df_labeled["label"].value_counts().to_string())

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Live News
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n[3/6]  Fetching real financial news (last {NEWS_DAYS} days)...")
news_df = fetch_all_news(days_back=NEWS_DAYS)

if not news_df.empty:
    save_news(news_df)
    print(f"  ✅ {len(news_df)} news articles collected")
else:
    print("  ⚠️  No news fetched (internet may be unavailable).")
    print("       Continuing without sentiment features...")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: FinBERT Sentiment
# ─────────────────────────────────────────────────────────────────────────────

daily_sentiment = pd.DataFrame()

if not news_df.empty:
    print("\n[4/6]  Running FinBERT sentiment analysis...")
    analyzer = FinBERTAnalyzer()
    daily_sentiment, raw_sentiment = compute_daily_sentiment(news_df, analyzer)
    save_sentiment(daily_sentiment, raw_sentiment)
else:
    print("\n[4/6]  Skipping sentiment (no news data)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Merge & Prepare ML Features
# ─────────────────────────────────────────────────────────────────────────────

print("\n[5/6]  Preparing ML features...")

if not daily_sentiment.empty:
    df_merged = merge_sentiment_with_prices(df_labeled, daily_sentiment)
else:
    df_merged = df_labeled.copy()

df_merged.to_csv("data/nifty50_merged.csv")

df_ml, feature_cols = prepare_features(df_merged)
print(f"  Features used: {feature_cols}")

# Temporal train/test split to prevent look-ahead bias
# Train on pre-2020 data, test on 2020+ (out-of-sample)
SPLIT_DATE = "2020-01-01"
# df_ml has date as index (inherited from df_merged)
train_mask = df_ml.index < SPLIT_DATE
test_mask = df_ml.index >= SPLIT_DATE

X_train_raw = df_ml.loc[train_mask, feature_cols].values
y_train = df_ml.loc[train_mask, "label_numeric"].values.astype(int)
X_test_raw = df_ml.loc[test_mask, feature_cols].values
y_test = df_ml.loc[test_mask, "label_numeric"].values.astype(int)

print(f"  Train: {len(X_train_raw)} rows (< {SPLIT_DATE}) | Test: {len(X_test_raw)} rows (>= {SPLIT_DATE})")

# Scale features (important for Logistic Regression meta-model)
# Fit scaler ONLY on training data to prevent data leakage
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# Handle class imbalance with ADASYN (proper function with SMOTE fallback)
X_train, y_train = apply_adasyn(X_train, y_train)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Train Stacking Ensemble
# ─────────────────────────────────────────────────────────────────────────────

print("\n[6/6]  Training stacking ensemble...")

base_models = [
    ("RandomForest", RandomForestClassifier(
        n_estimators=200, max_depth=10,
        min_samples_split=10, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )),
    ("XGBoost", xgb.XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", random_state=42, n_jobs=-1,
    )),
]

ensemble = StackingEnsemble(base_models=base_models, n_folds=5)
ensemble.fit(X_train, y_train)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

print("\n── Feature Importance Analysis ──────────────────────────────")

# Extract feature importance from RandomForest base model
# First base model is RandomForest
rf_model = ensemble.fitted_bases[0][1]
rf_importance = rf_model.feature_importances_

# Extract feature importance from XGBoost base model
xgb_model = ensemble.fitted_bases[1][1]  # Second base model is XGBoost
xgb_importance = xgb_model.feature_importances_

# Average importance across both models
avg_importance = (rf_importance + xgb_importance) / 2

# Create DataFrame and sort
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'RF_Importance': rf_importance,
    'XGB_Importance': xgb_importance,
    'Avg_Importance': avg_importance
}).sort_values('Avg_Importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(importance_df.head(10).to_string(index=False))
print(f"\nTotal features: {len(feature_cols)}")

# Save feature importance
importance_df.to_csv("outputs/feature_importance.csv", index=False)
print("  💾 Feature importance saved → outputs/feature_importance.csv")

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────────────────────────────────────

print("\n── Evaluation ───────────────────────────────────────────────")
f1, proba = ensemble.evaluate(X_test, y_test)

# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────

save_model(scaler,       "scaler.pkl")
save_model(feature_cols, "feature_cols.pkl")
ensemble.save("models/stacking_ensemble.pkl")

elapsed = time.time() - t_start

print()
print("=" * 60)
print(f"  ✅ PIPELINE COMPLETE  |  F1 = {f1:.4f}  |  {elapsed:.0f}s")
print("=" * 60)
print()
print("  Next step:")
print("    streamlit run app.py")
print("    Open http://localhost:8501 in your browser")
print()
