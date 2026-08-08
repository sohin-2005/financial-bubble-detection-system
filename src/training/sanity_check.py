from sklearn.metrics import f1_score
from stacking import predict_stacked
import pandas as pd
import numpy as np
import joblib
import sys
sys.path.append('.')


# ── Load all models ───────────────────────────────────────────────
OUTPUT_DIR = "../../data/models"

rf_model = joblib.load(f"{OUTPUT_DIR}/rf_model.pkl")
xgb_model = joblib.load(f"{OUTPUT_DIR}/xgb_model.pkl")
lr_model = joblib.load(f"{OUTPUT_DIR}/stacking_model.pkl")
scaler = joblib.load(f"{OUTPUT_DIR}/stacking_scaler.pkl")
le = joblib.load(f"{OUTPUT_DIR}/label_encoder.pkl")

print("All models loaded:")
print("  rf_model.pkl      ✅")
print("  xgb_model.pkl     ✅")
print("  stacking_model.pkl ✅")

# ── Load data ─────────────────────────────────────────────────────
df = pd.read_csv("../../data/exports/NIFTY50_train.csv")
val = pd.read_csv("../../data/exports/NIFTY50_validation.csv")
test = pd.read_csv("../../data/exports/NIFTY50_test.csv")
full = pd.concat([df, val, test]).reset_index(drop=True)
full["date"] = pd.to_datetime(full["date"])

sent = pd.read_csv("../../data/exports/nifty50_daily_sentiment.csv")
sent_val = pd.read_csv("../../data/exports/nifty50_validation_sentiment.csv")
sent_tst = pd.read_csv("../../data/exports/nifty50_test_sentiment.csv")
sent_all = pd.concat([sent, sent_val, sent_tst]
                     ).drop_duplicates(subset=["date"])
sent_all["date"] = pd.to_datetime(sent_all["date"])

full = full.merge(sent_all, on="date", how="left")
if "polarity_score" in full.columns:
    full.rename(
        columns={"polarity_score": "daily_sentiment_index"}, inplace=True)
full["daily_sentiment_index"] = full["daily_sentiment_index"].fillna(0.0)

FEATURES = [
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

full = full.dropna(subset=FEATURES)


# ════════════════════════════════════════════════════════════════
# TEST 1 — SANITY CHECK ON KNOWN EVENTS
# ════════════════════════════════════════════════════════════════
known_events = [
    ("2008-10-10", "Crash",  "Global Financial Crisis peak crash"),
    ("2009-03-09", "Crash",  "GFC bottom — worst point"),
    ("2020-03-23", "Crash",  "COVID crash bottom"),
    ("2020-03-12", "Crash",  "COVID panic selling"),
    ("2021-10-19", "Bubble", "Post-COVID all time high bubble"),
    ("2021-01-21", "Bubble", "Bull run peak"),
    ("2022-06-17", "Crash",  "Global selloff"),
    ("2019-06-05", "Normal", "Regular trading day"),
    ("2019-09-12", "Normal", "Regular trading day"),
    ("2023-07-20", "Normal", "Regular trading day"),
]

print("\n" + "=" * 65)
print("TEST 1 — SANITY CHECK (STACKING ENSEMBLE)")
print("=" * 65)
print(f"{'Date':<12} {'Expected':<10} {'RF':<10} {'XGB':<10} {'STACK':<10} {'OK'}")
print("-" * 65)

correct = 0
total = 0

for date_str, expected, event in known_events:

    row = full[full["date"] == date_str]
    if len(row) == 0:
        full["diff"] = abs(full["date"] - pd.Timestamp(date_str))
        row = full.nsmallest(1, "diff")

    actual_date = str(row["date"].values[0])[:10]
    X = row[FEATURES].values

    # Individual predictions
    rf_pred = le.inverse_transform(rf_model.predict(X))[0]
    xgb_pred = le.inverse_transform(xgb_model.predict(X))[0]

    # Stacking prediction
    stack_pred, stack_probs = predict_stacked(
        rf_model, xgb_model, lr_model, scaler, X
    )
    stack_label = le.inverse_transform(stack_pred)[0]

    is_correct = "✅" if stack_label == expected else "❌"
    if stack_label == expected:
        correct += 1
    total += 1

    print(f"{actual_date:<12} {expected:<10} {rf_pred:<10} "
          f"{xgb_pred:<10} {stack_label:<10} {is_correct}  {event}")

print("-" * 65)
print(f"Stacking score: {correct}/{total} ({100*correct/total:.0f}%)")

if correct == total:
    print("✅ Perfect — stacking model knows all known events")
elif correct >= 8:
    print("✅ Strong result — minor misses are explainable")
elif correct >= 7:
    print("⚠️  Acceptable — investigate remaining misses")
else:
    print("❌ Too many misses — model needs work")


# ════════════════════════════════════════════════════════════════
# TEST 2 — COMPARE ALL THREE MODELS
# ════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("TEST 2 — COMPARE RF vs XGB vs STACKING on Test Set")
print("=" * 65)

test_full = full[full["date"] >= "2023-01-01"].copy()
test_full = test_full.dropna(subset=FEATURES)
y_true = le.transform(test_full["label"].values)
X_test = test_full[FEATURES].values

rf_preds,  _ = rf_model.predict(X_test), None
xgb_preds = xgb_model.predict(X_test)
stack_preds, _ = predict_stacked(rf_model, xgb_model, lr_model, scaler, X_test)

rf_f1 = f1_score(y_true, rf_model.predict(X_test), average="macro")
xgb_f1 = f1_score(y_true, xgb_preds,               average="macro")
stack_f1 = f1_score(y_true, stack_preds,              average="macro")

print(f"\n  Random Forest F1:  {rf_f1:.4f}")
print(f"  XGBoost F1:        {xgb_f1:.4f}")
print(f"  Stacking F1:       {stack_f1:.4f}")

best = max(rf_f1, xgb_f1, stack_f1)
if stack_f1 == best:
    print("\n  ✅ Stacking is the best model — use this for dashboard")
elif stack_f1 >= 0.85:
    print("\n  ✅ Stacking meets NFR1 — acceptable for dashboard")
else:
    print("\n  ❌ Stacking underperforms — investigate")


# ════════════════════════════════════════════════════════════════
# TEST 3 — PREDICTION DISTRIBUTION
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("TEST 3 — PREDICTION DISTRIBUTION (2023-2024)")
print("=" * 65)

stack_labels = le.inverse_transform(stack_preds)
pred_dist = pd.Series(stack_labels).value_counts()
actual_dist = test_full["label"].value_counts()

print(f"\n{'Label':<10} {'Actual':<10} {'Predicted':<12} {'Diff'}")
print("-" * 40)
for label in ["Bubble", "Crash", "Normal"]:
    actual = actual_dist.get(label, 0)
    predicted = pred_dist.get(label, 0)
    diff = predicted - actual
    sign = "+" if diff > 0 else ""
    flag = "⚠️" if abs(diff) > 20 else "✅"
    print(f"{label:<10} {actual:<10} {predicted:<12} {sign}{diff} {flag}")

print("\n" + "=" * 65)
print("FINAL SUMMARY")
print("=" * 65)
print(f"  Random Forest F1:     {rf_f1:.4f}")
print(f"  XGBoost F1:           {xgb_f1:.4f}")
print(f"  Stacking Ensemble F1: {stack_f1:.4f}")
print(f"  Sanity Check:         {correct}/{total}")
print(f"\n  Model to use for dashboard: stacking_model.pkl")
print("=" * 65)
