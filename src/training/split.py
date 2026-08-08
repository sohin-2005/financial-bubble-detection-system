import pandas as pd
import glob
import os


def split_and_export(csv_path: str, ticker_id: str):
    """
    Split labeled data chronologically into train, validation, test sets.
    Saves three CSV files ready for Jerome.
    """
    
    # Load labeled CSV
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    print(f"\nLoaded {ticker_id}: {len(df)} rows")
    print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    
    # Chronological split — NEVER random
    train = df[df["date"] <  "2021-01-01"]
    val   = df[(df["date"] >= "2021-01-01") & (df["date"] < "2023-01-01")]
    test  = df[df["date"] >= "2023-01-01"]
    
    # Print summary
    print(f"\n=== {ticker_id} SPLIT SUMMARY ===")
    for name, split in [("Train", train), ("Validation", val), ("Test", test)]:
        print(f"\n{name}: {len(split)} rows "
              f"({split['date'].min().date()} → {split['date'].max().date()})")
        print(f"  Bubble : {(split['label']=='Bubble').sum():4d} days "
              f"({100*(split['label']=='Bubble').mean():.1f}%)")
        print(f"  Crash  : {(split['label']=='Crash').sum():4d} days "
              f"({100*(split['label']=='Crash').mean():.1f}%)")
        print(f"  Normal : {(split['label']=='Normal').sum():4d} days "
              f"({100*(split['label']=='Normal').mean():.1f}%)")
    
    # Save files
    output_dir = "../../data/exports"
    
    train.to_csv(f"{output_dir}/{ticker_id}_train.csv",      index=False)
    val.to_csv(  f"{output_dir}/{ticker_id}_validation.csv", index=False)
    test.to_csv( f"{output_dir}/{ticker_id}_test.csv",       index=False)
    
    print(f"\n  Saved: {ticker_id}_train.csv")
    print(f"  Saved: {ticker_id}_validation.csv")
    print(f"  Saved: {ticker_id}_test.csv")
    
    return train, val, test


if __name__ == "__main__":
    
    output_dir = "../../data/exports"
    
    # All tickers to process
    tickers = ["NIFTY50", "SENSEX"]
    
    print("=" * 50)
    print("SPLITTING DATA FOR ALL TICKERS")
    print("=" * 50)
    
    for ticker_id in tickers:
        
        # Find latest labeled CSV for this ticker
        files = glob.glob(f"{output_dir}/{ticker_id}_labeled_*.csv")
        
        if not files:
            print(f"\nNo labeled CSV found for {ticker_id}.")
            print(f"Run pipeline.py for {ticker_id} first.")
            continue
        
        latest = max(files)
        print(f"\nUsing: {latest}")
        split_and_export(latest, ticker_id)
    
    print("\n" + "=" * 50)
    print("ALL FILES READY — Send these 6 files to Jerome:")
    print("=" * 50)
    print("  NIFTY50_train.csv")
    print("  NIFTY50_validation.csv")
    print("  NIFTY50_test.csv")
    print("  SENSEX_train.csv")
    print("  SENSEX_validation.csv")
    print("  SENSEX_test.csv")