# test_xgb_predictions.py
import pandas as pd
import numpy as np
import joblib

# Load model and encoder
model = joblib.load("../../data/models/xgb_model.pkl")
le = joblib.load("../../data/models/label_encoder.pkl")

# Load full dataset
train = pd.read_csv("../../data/exports/NIFTY50_train.csv")
val = pd.read_csv("../../data/exports/NIFTY50_validation.csv")
test = pd.read_csv("../../data/exports/NIFTY50_test.csv")
full = pd.concat([train, val, test]).reset_index(drop=True)
full["date"] = pd.to_datetime(full["date"])

# Load sentiment
s1 = pd.read_csv("../../data/exports/nifty50_daily_sentiment.csv")
s2 = pd.read_csv("../../data/exports/nifty50_validation_sentiment.csv")
s3 = pd.read_csv("../../data/exports/nifty50_test_sentiment.csv")
sent = pd.concat([s1, s2, s3]).drop_duplicates(subset=["date"])
sent["date"] = pd.to_datetime(sent["date"])

full = full.merge(sent, on="date", how="left")
if "polarity_score" in full.columns:
    full.rename(
        columns={"polarity_score": "daily_sentiment_index"}, inplace=True)
full["daily_sentiment_index"] = full["daily_sentiment_index"].fillna(0.0)

FEATURES = [
    "zscore_value", "log_return", "psy_12", "rsi",
    "macd", "macd_signal", "macd_hist",
    "daily_sentiment_index",
    "gdp_growth", "cpi_inflation", "repo_rate"
]

if "psy_12" not in full.columns:
    close_col = "close" if "close" in full.columns else "Close" if "Close" in full.columns else None
    if close_col is not None:
        up_days = (full[close_col].diff() > 0).astype(float)
        full["psy_12"] = up_days.rolling(12).mean() * 100.0

full = full.dropna(subset=FEATURES)

# ── Known events with expected labels ────────────────────────────
known_events = [
    # Date           Expected  Event description
    ("2008-10-24", "Crash",  "GFC — Black Friday, biggest single day crash"),
    ("2020-03-23", "Crash",  "COVID crash — circuit breaker hit"),
    ("2020-03-12", "Crash",  "COVID panic selling"),
    ("2022-06-17", "Crash",  "Global selloff — FII outflows"),
    ("2021-10-19", "Bubble", "Post-COVID all time high"),
    ("2021-01-21", "Bubble", "Bull run peak"),
    ("2021-07-15", "Bubble", "Strong bull market"),
    ("2019-06-05", "Normal", "Regular trading day"),
    ("2019-09-12", "Normal", "Regular trading day"),
    ("2023-07-20", "Normal", "Stable market"),
    ("2017-04-10", "Normal", "Normal day mid-bull"),
    ("2015-08-24", "Crash",  "Black Monday — China crash ripple"),
]

print("=" * 75)
print("XGBOOST PREDICTION TEST ON KNOWN MARKET EVENTS")
print("=" * 75)
print(f"{'Date':<12} {'Expected':<10} {'Predicted':<10} "
      f"{'Bub%':<8} {'Cra%':<8} {'Nor%':<8} {'✓/✗':<5} Event")
print("-" * 75)

correct = 0
total = 0

for date_str, expected, event in known_events:

    # Find closest date in dataset
    target = pd.Timestamp(date_str)
    full["diff"] = abs(full["date"] - target)
    row = full.nsmallest(1, "diff")

    if len(row) == 0:
        print(f"{date_str:<12} — no data found")
        continue

    actual_date = str(row["date"].values[0])[:10]
    X = row[FEATURES].values

    # Get probabilities
    probs = model.predict_proba(X)[0]
    pred_enc = model.predict(X)[0]
    predicted = le.inverse_transform([pred_enc])[0]

    # Map probabilities to class names
    prob_dict = dict(zip(le.classes_, probs))
    bub_pct = prob_dict.get("Bubble", 0)
    cra_pct = prob_dict.get("Crash",  0)
    nor_pct = prob_dict.get("Normal", 0)

    is_correct = "✅" if predicted == expected else "❌"
    if predicted == expected:
        correct += 1
    total += 1

    print(f"{actual_date:<12} {expected:<10} {predicted:<10} "
          f"{bub_pct:<8.1%} {cra_pct:<8.1%} {nor_pct:<8.1%} "
          f"{is_correct:<5} {event}")

print("-" * 75)
print(f"\nScore: {correct}/{total} correct ({100*correct/total:.0f}%)")

# ── Also show actual labels from dataset ─────────────────────────
print("\n" + "=" * 75)
print("ACTUAL LABELS IN DATASET FOR SAME DATES")
print("=" * 75)
print(f"{'Date':<12} {'Actual Label':<15} {'Z-Score':<10} {'BSADF':<10}")
print("-" * 75)

for date_str, expected, event in known_events:
    target = pd.Timestamp(date_str)
    full["diff"] = abs(full["date"] - target)
    row = full.nsmallest(1, "diff")

    if len(row) == 0:
        continue

    actual_date = str(row["date"].values[0])[:10]
    actual_label = row["label"].values[0]
    zscore = row["zscore_value"].values[0] if "zscore_value" in row else "N/A"
    bsadf = row["bsadf_score"].values[0] if "bsadf_score" in row else "N/A"

    match = "✅" if actual_label == expected else "⚠️"

    print(f"{actual_date:<12} {actual_label:<15} "
          f"{zscore:<10.3f} "
          f"{float(bsadf):<10.3f} {match} {event}")

print("\nNote: ⚠️ means the LABEL itself disagrees with expected")
print("      This means the labeling method may have missed this event")
print("      The model is predicting what it was trained on")
