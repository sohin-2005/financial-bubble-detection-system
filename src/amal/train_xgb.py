import pandas as pd
import numpy as np
import xgboost as xgb
from imblearn.over_sampling import ADASYN
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# ── Config ────────────────────────────────────────────────────────
FEATURES = [
    "zscore_value",
    # Short term — daily signals
    "log_return",
    "psy_12",
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "daily_sentiment_index",
    # Long term — macro signals
    "gdp_growth",
    "cpi_inflation",
    "repo_rate",
]

OUTPUT_DIR = "../../data/models"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Step 1: Load Data ─────────────────────────────────────────────
def load_data():

    print("[1/6] Loading data...")

    # Load Amal's price/feature CSVs
    train_df = pd.read_csv("../../data/exports/NIFTY50_train.csv")
    val_df = pd.read_csv("../../data/exports/NIFTY50_validation.csv")
    test_df = pd.read_csv("../../data/exports/NIFTY50_test.csv")

    # Load Arjun's sentiment CSVs
    # ⚠️ Check the exact filename Arjun gave you
    sent_train = pd.read_csv("../../data/exports/nifty50_daily_sentiment.csv")
    sent_val = pd.read_csv(
        "../../data/exports/nifty50_validation_sentiment.csv")
    sent_test = pd.read_csv("../../data/exports/nifty50_test_sentiment.csv")

    # Convert dates
    for df in [train_df, val_df, test_df, sent_train, sent_val, sent_test]:
        df["date"] = pd.to_datetime(df["date"])

    # Merge sentiment
    train_df = train_df.merge(sent_train, on="date", how="left")
    val_df = val_df.merge(sent_val,   on="date", how="left")
    test_df = test_df.merge(sent_test,   on="date", how="left")

    # ⚠️ If Arjun's column is called polarity_score, rename it
    for df in [train_df, val_df, test_df]:
        if "polarity_score" in df.columns:
            df.rename(
                columns={"polarity_score": "daily_sentiment_index"}, inplace=True)

    # Fill any remaining missing sentiment with neutral
    train_df["daily_sentiment_index"] = train_df["daily_sentiment_index"].fillna(
        0.0)
    val_df["daily_sentiment_index"] = val_df["daily_sentiment_index"].fillna(
        0.0)
    test_df["daily_sentiment_index"] = test_df["daily_sentiment_index"].fillna(
        0.0)

    def add_psy_feature(df: pd.DataFrame) -> pd.DataFrame:
        close_col = "close" if "close" in df.columns else "Close" if "Close" in df.columns else None
        if close_col is None:
            return df
        if "psy_12" not in df.columns:
            up_days = (df[close_col].diff() > 0).astype(float)
            df["psy_12"] = up_days.rolling(12).mean() * 100.0
        return df

    train_df = add_psy_feature(train_df)
    val_df = add_psy_feature(val_df)
    test_df = add_psy_feature(test_df)

    # Drop missing values
    before = len(train_df)
    train_df = train_df.dropna(subset=FEATURES + ["label"])
    val_df = val_df.dropna(subset=FEATURES + ["label"])
    test_df = test_df.dropna(subset=FEATURES + ["label"])
    after = len(train_df)

    print(
        f"  Train rows: {len(train_df)} ({before - after} dropped for missing values)")
    print(f"  Val rows:   {len(val_df)}")
    print(f"  Test rows:  {len(test_df)}")

    return train_df, val_df, test_df


# ── Step 2: Encode Labels ─────────────────────────────────────────
def encode_labels(train_df, val_df, test_df):
    """
    XGBoost needs numeric labels, not strings.
    Bubble=0, Crash=1, Normal=2
    """
    print("\n[2/6] Encoding labels...")

    le = LabelEncoder()
    le.fit(["Bubble", "Crash", "Normal"])

    y_train = le.transform(train_df["label"])
    y_val = le.transform(val_df["label"])
    y_test = le.transform(test_df["label"])

    print(
        f"  Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    print(
        f"  Train distribution: {pd.Series(y_train).value_counts().to_dict()}")

    # Save encoder for later use in live predictions
    joblib.dump(le, f"{OUTPUT_DIR}/label_encoder.pkl")

    return y_train, y_val, y_test, le


# ── Step 3: Apply ADASYN ──────────────────────────────────────────
def apply_adasyn(X_train, y_train):
    """
    Balance the class distribution on TRAINING data only.
    Never apply to val or test.
    """
    print("\n[3/6] Applying ADASYN oversampling...")

    print(f"  Before ADASYN:")
    for label, count in zip(*np.unique(y_train, return_counts=True)):
        print(f"    Class {label}: {count} samples")

    adasyn = ADASYN(random_state=42, n_neighbors=5,  sampling_strategy={
        0: 800,   # Bubble — create up to 800 samples (was 152)
        1: 800,   # Crash  — create up to 800 samples (was 142)
        # Normal (2) not listed — leave at 2851
    })

    try:
        X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
    except Exception as e:
        print(f"  ADASYN failed: {e}")
        print("  Falling back to original data...")
        return X_train, y_train

    print(f"\n  After ADASYN:")
    for label, count in zip(*np.unique(y_resampled, return_counts=True)):
        print(f"    Class {label}: {count} samples")

    return X_resampled, y_resampled


# ── Step 4: Train XGBoost ─────────────────────────────────────────
def train_xgboost(X_train, y_train, X_val, y_val):
    """
    Train XGBoost classifier with tuned hyperparameters.
    """
    print("\n[4/6] Training XGBoost...")

    model = xgb.XGBClassifier(

        # Core parameters
        n_estimators=500,     # number of trees
        max_depth=6,       # how deep each tree grows
        learning_rate=0.05,    # how fast model learns — lower = more careful

        # Regularization — prevents overfitting
        subsample=0.8,     # use 80% of rows per tree
        colsample_bytree=0.8,     # use 80% of features per tree
        min_child_weight=3,       # minimum samples needed to split
        gamma=0.1,     # minimum loss reduction to split
        reg_alpha=0.1,     # L1 regularization
        reg_lambda=1.0,     # L2 regularization

        # Multi-class setup
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        early_stopping_rounds=50,
        # Reproducibility
        random_state=42,


        # Speed
        n_jobs=-1,      # use all CPU cores
        verbosity=0,
    )

    # Train with early stopping on validation set
    # Stops training if val score doesn't improve for 50 rounds
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    print(f"\n  Best iteration: {model.best_iteration}")

    return model


# ── Step 5: Evaluate ──────────────────────────────────────────────
def evaluate(model, X_val, y_val, X_test, y_test, le):

    print("\n[5/6] Evaluating model...")

    # Validation
    val_pred = model.predict(X_val)
    val_f1 = f1_score(y_val, val_pred, average="macro")

    # Test — the honest final score
    test_pred = model.predict(X_test)
    test_f1 = f1_score(y_test, test_pred, average="macro")

    print(f"\n  Validation F1 (macro): {val_f1:.4f}")
    print(f"  Test F1 (macro):       {test_f1:.4f}")

    if test_f1 >= 0.85:
        print("  ✅ Meets NFR1 requirement (F1 >= 0.85)")
    else:
        print("  ❌ Below NFR1 requirement — needs improvement")

    # Detailed report per class
    print("\n  Per-class breakdown (Test set):")
    print(classification_report(
        y_test, test_pred,
        target_names=le.classes_
    ))

    # Confusion matrix
    print("  Confusion Matrix (Test set):")
    cm = confusion_matrix(y_test, test_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual {c}" for c in le.classes_],
        columns=[f"Predicted {c}" for c in le.classes_]
    )
    print(cm_df)

    return val_f1, test_f1


# ── Step 6: Save Model ────────────────────────────────────────────
def save_model(model, val_f1, test_f1):

    print("\n[6/6] Saving model...")

    path = f"{OUTPUT_DIR}/xgb_model.pkl"
    joblib.dump(model, path)

    # Save performance stats
    stats = {
        "model":     "XGBoost",
        "val_f1":    round(val_f1,  4),
        "test_f1":   round(test_f1, 4),
        "features":  FEATURES,
        "trained_on": "NIFTY50 2008-2020",
        "tested_on":  "NIFTY50 2023-2024",
    }

    import json
    with open(f"{OUTPUT_DIR}/xgb_model_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  Saved model to {path}")
    print(f"  Saved stats to xgb_model_stats.json")


def walk_forward_validation(df, features):
    """
    Test model across multiple time windows.
    Most honest evaluation for financial data.
    """

    windows = [
        ("2019-01-01", "2019-01-01", "2020-12-31"),
        ("2020-01-01", "2020-01-01", "2021-12-31"),
        ("2021-01-01", "2021-01-01", "2022-12-31"),
        ("2022-01-01", "2022-01-01", "2023-12-31"),
        ("2023-01-01", "2023-01-01", "2024-12-31"),
    ]

    f1_scores = []

    for train_end, test_start, test_end in windows:

        train = df[df["date"] < pd.Timestamp(train_end)]
        test = df[(df["date"] >= pd.Timestamp(test_start)) &
                  (df["date"] <= pd.Timestamp(test_end))]

        if len(test) == 0 or len(train) == 0:
            print(f"  Skipping {test_start}→{test_end} — not enough data")
            continue

        # Check both classes exist in train
        if len(np.unique(train["label"].values)) < 3:
            print(
                f"  Skipping {test_start}→{test_end} — missing classes in train")
            continue

        le_wf = LabelEncoder()
        le_wf.fit(["Bubble", "Crash", "Normal"])

        X_tr = train[features].values
        y_tr = le_wf.transform(train["label"].values)
        X_te = test[features].values
        y_te = le_wf.transform(test["label"].values)

        # Quick model
        m = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            random_state=42,
            verbosity=0
        )
        m.fit(X_tr, y_tr)
        preds = m.predict(X_te)

        f1 = f1_score(y_te, preds, average="macro")
        f1_scores.append(f1)
        print(f"  Train: 2008→{train_end[:4]}  "
              f"Test: {test_start[:4]}→{test_end[:4]}  "
              f"Rows: {len(test)}  F1: {f1:.4f}")

    print(f"\n  Average F1 across all windows: "
          f"{np.mean(f1_scores):.4f}")
    print(f"  Min F1: {min(f1_scores):.4f}")
    print(f"  Max F1: {max(f1_scores):.4f}")

    return f1_scores


# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 60)
    print("XGBOOST TRAINING PIPELINE")
    print("=" * 60)

    # Load
    train_df, val_df, test_df = load_data()

    # Features
    X_train = train_df[FEATURES].values
    X_val = val_df[FEATURES].values
    X_test = test_df[FEATURES].values

    # Encode labels
    y_train, y_val, y_test, le = encode_labels(train_df, val_df, test_df)

    # ADASYN on train only
    X_train_res, y_train_res = apply_adasyn(X_train, y_train)

    # Train
    model = train_xgboost(X_train_res, y_train_res, X_val, y_val)

    # Evaluate
    val_f1, test_f1 = evaluate(model, X_val, y_val, X_test, y_test, le)

    # Save
    save_model(model, val_f1, test_f1)

    print("\n" + "=" * 60)
    print("WALK FORWARD VALIDATION")
    print("=" * 60)

    # Load full dataset for walk forward
    # Combine train + val + test back into one dataframe
    full_df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
    full_df["date"] = pd.to_datetime(full_df["date"])

    walk_forward_validation(full_df, FEATURES)

    print("\n" + "=" * 60)
    print("XGBOOST TRAINING COMPLETE")
    print("=" * 60)
