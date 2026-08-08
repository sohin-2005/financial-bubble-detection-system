# relabel.py — just fix labels without rerunning PSY
import pandas as pd
import numpy as np
from database import get_engine
from pathlib import Path
from datetime import datetime

def relabel_with_direction(ticker_id: str, engine,
                           upper_pct:    float = 0.95,
                           crash_zscore: float = -2.0,
                           bubble_zscore: float = 2.0,
                           rolling_window: int = 504):
    """
    Reload existing BSADF scores from DB and relabel
    with direction fix. No PSY recomputation.
    """
    print(f"\nRelabeling {ticker_id}...")
    
    # Load existing BSADF + price data from DB
    query = f"""
        SELECT b.date, b.close, b.bsadf_score,
               b.log_return, b.rsi, b.macd,
               b.macd_signal, b.macd_hist
        FROM bubble_analysis b
        WHERE b.ticker_id = '{ticker_id}'
        ORDER BY b.date
    """
    df = pd.read_sql(query, engine)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    print(f"  Loaded {len(df)} rows with existing BSADF scores")
    if df.empty:
        print(f"  No rows found for {ticker_id}; skipping.")
        return df
    
    # Rolling threshold for Bubble
    upper_bound = df["bsadf_score"].rolling(
        window=rolling_window, min_periods=252
    ).quantile(upper_pct)
    
    # Compute Z-score fields if they are not already present.
    if "rolling_mean" not in df.columns:
        df["rolling_mean"] = df["close"].rolling(window=20).mean()
    if "rolling_std" not in df.columns:
        df["rolling_std"] = df["close"].rolling(window=20).std()
    if "zscore_value" not in df.columns:
        df["zscore_value"] = (df["close"] - df["rolling_mean"]) / df["rolling_std"]

    df["label"] = "Normal"

    # Z-score Crash
    df.loc[
        (df["zscore_value"] < crash_zscore) &
        (df["label"] == "Normal"),
        "label"
    ] = "Crash"

    # PSY Bubble — price must be going UP
    df.loc[
        (df["bsadf_score"] >= upper_bound) &
        (df["log_return"]  >  0) &
        (df["label"] == "Normal"),
        "label"
    ] = "Bubble"

    # Z-score Bubble — catches what PSY misses
    df.loc[
        (df["zscore_value"] > bubble_zscore) &
        (df["label"] == "Normal"),
        "label"
    ] = "Bubble"
    
    # Print distribution
    dist  = df["label"].value_counts()
    total = len(df.dropna(subset=["bsadf_score"]))
    print(f"\n  New Label Distribution:")
    for label in ["Bubble", "Crash", "Normal"]:
        count = dist.get(label, 0)
        pct   = 100 * count / total if total > 0 else 0
        print(f"    {label:8s}: {count:4d} days ({pct:.1f}%)")
    
    # Save back to DB
    df["ticker_id"] = ticker_id
    save_cols = [
        "date", "ticker_id", "close",
        "log_return", "rsi", "macd",
        "macd_signal", "macd_hist",
        "bsadf_score", "zscore_value",
        "rolling_mean", "rolling_std", "label"
    ]
    df[save_cols].to_sql(
        "bubble_analysis", engine,
        if_exists="replace", index=False
    )
    print(f"  Saved updated labels to DB ✅")
    
    # Export CSV with macro
    from macro import build_macro_daily
    
    
    macro_df = build_macro_daily("2008-01-01", "2024-12-31")
    macro_df["date"] = pd.to_datetime(macro_df["date"])
    
    df = pd.merge_asof(
        df.sort_values("date"),
        macro_df.sort_values("date"),
        on="date", direction="backward"
    )
    
    exports_dir = Path(__file__).resolve().parent.parent.parent / "data" / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d")
    path = exports_dir / f"{ticker_id}_labeled_{timestamp}.csv"
    df.to_csv(path, index=False)
    print(f"  Exported: {path}")
    
    return df


if __name__ == "__main__":
    engine = get_engine()
    
    for ticker_id in ["NIFTY50", "SENSEX"]:
        relabel_with_direction(ticker_id, engine)
    
    print("\nDone! Now run:")
    print("  python split.py")
    print("  python train_rf.py")
    print("  python train_xgb.py")
    print("  python xgbpredict.py")