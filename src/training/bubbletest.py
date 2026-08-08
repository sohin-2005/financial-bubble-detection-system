"""
bubble_stress_test.py
Compares XGBoost vs Stacking on:
  - Real historical market events
  - Artificial synthetic extreme cases
Focus: Bubble detection primarily
"""

import pandas as pd
import numpy as np
import joblib

FEATURES = [
    "zscore_value", "log_return", "rsi",
    "macd", "macd_signal", "macd_hist",
    "daily_sentiment_index",
    "gdp_growth", "cpi_inflation", "repo_rate"
]

CLASSES = np.array(["Bubble", "Crash", "Normal"])

# ── Load models ───────────────────────────────────────────────────
xgb_model      = joblib.load("../../data/models/xgb_model.pkl")
rf_model       = joblib.load("../../data/models/rf_model.pkl")
stacking_model = joblib.load("../../data/models/stacking_model.pkl")
stacking_scaler= joblib.load("../../data/models/stacking_scaler.pkl")
le             = joblib.load("../../data/models/label_encoder.pkl")


def proba_fixed_order(clf, X) -> np.ndarray:
    proba     = np.zeros((len(X), len(CLASSES)))
    clf_proba = clf.predict_proba(X)
    for j, cls in enumerate(clf.classes_):
        class_idx = int(cls)
        if 0 <= class_idx < len(CLASSES):
            proba[:, class_idx] = clf_proba[:, j]
    return proba


def entropy(p, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def build_meta_features(rf_probs, xgb_probs):
    diff_bubble = (rf_probs[:, 0] - xgb_probs[:, 0]).reshape(-1, 1)
    diff_crash  = (rf_probs[:, 1] - xgb_probs[:, 1]).reshape(-1, 1)
    rf_conf     = rf_probs.max(axis=1).reshape(-1, 1)
    xgb_conf    = xgb_probs.max(axis=1).reshape(-1, 1)
    max_conf    = np.maximum(rf_conf, xgb_conf)
    H_rf        = entropy(rf_probs).reshape(-1, 1)
    H_xgb       = entropy(xgb_probs).reshape(-1, 1)
    return np.hstack([rf_probs, xgb_probs, diff_bubble, diff_crash,
                      rf_conf, xgb_conf, max_conf, H_rf, H_xgb])


def predict_xgb(X):
    probs = xgb_model.predict_proba(X)[0]
    pred  = le.inverse_transform([xgb_model.predict(X)[0]])[0]
    return pred, dict(zip(le.classes_, probs))


def predict_stack(X):
    rf_p   = proba_fixed_order(rf_model,  X)
    xgb_p  = proba_fixed_order(xgb_model, X)
    meta   = build_meta_features(rf_p, xgb_p)
    meta_s = stacking_scaler.transform(meta)
    probs  = stacking_model.predict_proba(meta_s)[0]
    pred   = le.inverse_transform([stacking_model.predict(meta_s)[0]])[0]
    return pred, dict(zip(le.classes_, probs))


# ── Load data ─────────────────────────────────────────────────────
train = pd.read_csv("../../data/exports/NIFTY50_train.csv")
val   = pd.read_csv("../../data/exports/NIFTY50_validation.csv")
test  = pd.read_csv("../../data/exports/NIFTY50_test.csv")
full  = pd.concat([train, val, test]).reset_index(drop=True)
full["date"] = pd.to_datetime(full["date"])

s1   = pd.read_csv("../../data/exports/nifty50_daily_sentiment.csv")
s2   = pd.read_csv("../../data/exports/nifty50_validation_sentiment.csv")
s3   = pd.read_csv("../../data/exports/nifty50_test_sentiment.csv")
sent = pd.concat([s1, s2, s3]).drop_duplicates(subset=["date"])
sent["date"] = pd.to_datetime(sent["date"])

full = full.merge(sent, on="date", how="left")
if "polarity_score" in full.columns:
    full.rename(columns={"polarity_score": "daily_sentiment_index"}, inplace=True)
full["daily_sentiment_index"] = full["daily_sentiment_index"].fillna(0.0)
full = full.dropna(subset=FEATURES)


def get_row(date_str):
    full["diff"] = abs(full["date"] - pd.Timestamp(date_str))
    row = full.nsmallest(1, "diff")
    if len(row) == 0:
        return None, None, None
    return row[FEATURES].values.reshape(1, -1), \
           str(row["date"].values[0])[:10], \
           row["label"].values[0]


# ══════════════════════════════════════════════════════════════════
# PART 1 — REAL HISTORICAL EVENTS
# ══════════════════════════════════════════════════════════════════

real_events = [
    ("2021-07-15", "Bubble", "NIFTY all time high — bull run"),
    ("2021-09-15", "Bubble", "Bull market continuation"),
    ("2021-10-19", "Bubble", "Post-COVID peak NIFTY 18600"),
    ("2021-11-18", "Bubble", "Late 2021 elevated prices"),
    ("2017-07-19", "Bubble", "2017 bull run peak"),
    ("2019-06-03", "Bubble", "Pre-election rally peak"),
    ("2023-12-15", "Bubble", "2023 year-end rally"),
    ("2024-09-26", "Bubble", "2024 bull market"),
    ("2008-10-24", "Crash",  "GFC Black Friday crash"),
    ("2020-03-12", "Crash",  "COVID panic selling"),
    ("2022-06-17", "Crash",  "Global selloff FII outflows"),
    ("2015-08-24", "Crash",  "Black Monday China ripple"),
    ("2019-09-12", "Normal", "Regular trading day"),
    ("2017-04-10", "Normal", "Normal day mid-bull"),
    ("2023-07-05", "Normal", "Stable summer trading"),
]

print("=" * 80)
print("PART 1 — REAL HISTORICAL EVENTS")
print("=" * 80)
print(f"{'Date':<12} {'Expect':<9} {'XGB':<9} {'Stack':<9} "
      f"{'XGB✓':<6} {'Stk✓':<6} Event")
print("-" * 80)

xgb_r = stk_r = 0

for date_str, expected, event in real_events:
    X, actual_date, actual_label = get_row(date_str)
    if X is None:
        continue

    xgb_pred,  xgb_p  = predict_xgb(X)
    stk_pred,  stk_p  = predict_stack(X)

    xgb_ok = "✅" if xgb_pred == expected else "❌"
    stk_ok = "✅" if stk_pred == expected else "❌"

    label_warn = "" if actual_label == expected else f" ⚠️label={actual_label}"

    if xgb_pred == expected: xgb_r += 1
    if stk_pred == expected: stk_r += 1

    print(f"{actual_date:<12} {expected:<9} {xgb_pred:<9} {stk_pred:<9} "
          f"{xgb_ok:<6} {stk_ok:<6} {event}{label_warn}")

n_real = len(real_events)
print("-" * 80)
print(f"XGBoost: {xgb_r}/{n_real} ({100*xgb_r/n_real:.0f}%)   "
      f"Stacking: {stk_r}/{n_real} ({100*stk_r/n_real:.0f}%)")


# ══════════════════════════════════════════════════════════════════
# PART 2 — SYNTHETIC CASES
# ══════════════════════════════════════════════════════════════════

synthetic_cases = [
    # EXTREME BUBBLES
    ("Textbook Bubble",        "Bubble",
     3.8, 0.030, 85.0, 600, 450, 150, 0.90, 9.0, 5.5, 4.0),
    ("Strong Bubble",          "Bubble",
     2.8, 0.018, 78.0, 420, 330,  90, 0.70, 8.2, 5.0, 4.5),
    ("Moderate Bubble",        "Bubble",
     2.1, 0.009, 71.0, 200, 160,  40, 0.45, 7.0, 4.8, 5.0),
    ("2021-style Gradual",     "Bubble",
     1.9, 0.006, 68.0, 180, 140,  40, 0.55, 9.1, 5.1, 4.0),
    ("Bubble cheap money",     "Bubble",
     2.5, 0.012, 73.0, 310, 240,  70, 0.60, 7.5, 4.0, 3.5),
    ("Bubble negative news",   "Bubble",
     2.9, 0.014, 76.0, 390, 300,  90,-0.25, 6.5, 6.0, 5.5),
    ("Tech bubble high CPI",   "Bubble",
     3.2, 0.022, 80.0, 480, 370, 110, 0.80, 5.5, 8.0, 4.5),
    ("Borderline Bubble",      "Bubble",
     2.0, 0.005, 65.0, 120,  95,  25, 0.30, 6.8, 4.5, 6.0),
    # CRASHES
    ("Textbook Crash",         "Crash",
    -3.8,-0.090, 15.0,-700,-520,-180,-0.95,-6.0, 3.0, 8.5),
    ("Moderate Crash",         "Crash",
    -2.3,-0.040, 25.0,-350,-270, -80,-0.65, 2.0, 7.0, 7.0),
    # NORMALS
    ("Perfect Normal",         "Normal",
     0.2, 0.001, 51.0,  15,  12,   3, 0.02, 6.5, 4.0, 6.5),
    ("Slightly elevated",      "Normal",
     1.2, 0.004, 58.0,  80,  65,  15, 0.18, 6.0, 4.3, 6.0),
    # EDGE CASES
    ("High RSI neutral zscore","Normal",
     0.5, 0.003, 72.0,  30,  25,   5, 0.10, 6.0, 4.5, 6.5),
    ("Bubble zscore neg return","Bubble",
     2.4,-0.008, 68.0, 250, 200,  50,-0.15, 7.0, 5.0, 5.5),
]

print("\n\n" + "=" * 80)
print("PART 2 — SYNTHETIC CASES")
print("=" * 80)
print(f"{'Expect':<9} {'XGB':<9} {'Stack':<9} "
      f"{'XGB✓':<6} {'Stk✓':<6} Case")
print("-" * 80)

xgb_s = stk_s = 0

for row in synthetic_cases:
    name    = row[0]
    expected= row[1]
    vals    = row[2:]

    X = np.array([list(vals)]).reshape(1, -1)

    xgb_pred, xgb_p = predict_xgb(X)
    stk_pred, stk_p = predict_stack(X)

    xgb_ok = "✅" if xgb_pred == expected else "❌"
    stk_ok = "✅" if stk_pred == expected else "❌"

    if xgb_pred == expected: xgb_s += 1
    if stk_pred == expected: stk_s += 1

    print(f"{expected:<9} {xgb_pred:<9} {stk_pred:<9} "
          f"{xgb_ok:<6} {stk_ok:<6} {name}")

n_syn = len(synthetic_cases)
print("-" * 80)
print(f"XGBoost: {xgb_s}/{n_syn} ({100*xgb_s/n_syn:.0f}%)   "
      f"Stacking: {stk_s}/{n_syn} ({100*stk_s/n_syn:.0f}%)")


# ══════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

xgb_total = xgb_r + xgb_s
stk_total = stk_r + stk_s
n_total   = n_real + n_syn

print(f"\n{'Model':<20} {'Real':<15} {'Synthetic':<15} {'Overall':<15}")
print("-" * 60)
print(f"{'XGBoost':<20} "
      f"{xgb_r}/{n_real} ({100*xgb_r/n_real:.0f}%)  "
      f"{xgb_s}/{n_syn} ({100*xgb_s/n_syn:.0f}%)     "
      f"{xgb_total}/{n_total} ({100*xgb_total/n_total:.0f}%)")
print(f"{'Stacking':<20} "
      f"{stk_r}/{n_real} ({100*stk_r/n_real:.0f}%)  "
      f"{stk_s}/{n_syn} ({100*stk_s/n_syn:.0f}%)     "
      f"{stk_total}/{n_total} ({100*stk_total/n_total:.0f}%)")

print(f"\nConclusion:")
if xgb_total > stk_total:
    print(f"  XGBoost performs better on this test")
    print(f"  Final model: xgb_model.pkl")
elif stk_total > xgb_total:
    print(f"  Stacking performs better on this test")
    print(f"  Final model: stacking_model.pkl")
else:
    print(f"  Both models perform equally")
    print(f"  Use XGBoost — simpler and same score")