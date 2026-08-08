import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import glob

def relabel_csv(ticker_id: str):
    
    # Find latest labeled CSV
    files = glob.glob(f"../../data/exports/{ticker_id}_labeled_*.csv")
    if not files:
        print(f"No CSV found for {ticker_id}")
        return
    
    latest = max(files)
    print(f"\nLoading: {latest}")
    
    df = pd.read_csv(latest)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    print(f"  Loaded {len(df)} rows")
    
    if "bsadf_score" not in df.columns:
        print("  No bsadf_score column — cannot relabel")
        return
    
    # Rolling threshold for PSY Bubble
    upper_bound = df["bsadf_score"].rolling(
        window=504, min_periods=252
    ).quantile(0.95)
    
    # Compute zscore if not present
    if "zscore_value" not in df.columns:
        df["rolling_mean"] = df["close"].rolling(window=20).mean()
        df["rolling_std"]  = df["close"].rolling(window=20).std()
        df["zscore_value"] = (
            (df["close"] - df["rolling_mean"]) / df["rolling_std"]
        )
    
    # Apply three-rule labeling
    df["label"] = "Normal"
    
    # Rule 1: Z-score Crash
    df.loc[
        (df["zscore_value"] < -2.0) &
        (df["label"] == "Normal"),
        "label"
    ] = "Crash"
    
    # Rule 2: PSY Bubble — price going up only
    df.loc[
        (df["bsadf_score"] >= upper_bound) &
        (df["log_return"]  >  0) &
        (df["label"] == "Normal"),
        "label"
    ] = "Bubble"
    
    # Rule 3: Z-score Bubble — catches gradual bubbles PSY misses
    df.loc[
        (df["zscore_value"] > 1.8) &
        (df["label"] == "Normal"),
        "label"
    ] = "Bubble"
    
    # Print distribution
    dist  = df["label"].value_counts()
    total = len(df)
    print(f"\n  New Label Distribution:")
    for label in ["Bubble", "Crash", "Normal"]:
        count = dist.get(label, 0)
        pct   = 100 * count / total if total > 0 else 0
        print(f"    {label:8s}: {count:4d} days ({pct:.1f}%)")
    
    # Save new CSV with timestamp
    exports_dir = Path(__file__).resolve().parent.parent.parent / "data" / "exports"
    timestamp   = datetime.now().strftime("%Y%m%d%H%M")
    path        = exports_dir / f"{ticker_id}_labeled_{timestamp}.csv"
    df.to_csv(path, index=False)
    print(f"\n  Saved: {path}")
    
    return df


if __name__ == "__main__":
    relabel_csv("NIFTY50")
    relabel_csv("SENSEX")
    
    print("\nDone! Now run:")
    print("  python split.py")
    print("  python train_xgb.py")
    print("  python xgbpredict.py")