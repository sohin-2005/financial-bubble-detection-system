"""
Amal's master pipeline.
Run this file to execute the entire data ingestion and labeling pipeline.

Usage:
    python pipeline.py
    python pipeline.py --ticker ^NSEI --start 2019-01-01 --end 2024-12-31
"""
import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path
from macro import build_macro_daily
from database import get_engine, create_tables
from ingestion import download_ohlcv, save_to_market_data
from zscore import run_zscore_pipeline


def export_csv(df: pd.DataFrame, ticker_id: str,
               start_date: str, end_date: str):
    """
    Export labeled data with macro features to CSV.
    """
    from macro import build_macro_daily
    from pathlib import Path

    # Build daily macro features
    macro_df = build_macro_daily(start_date, end_date)
    macro_df["date"] = pd.to_datetime(macro_df["date"])
    df["date"] = pd.to_datetime(df["date"])

    # Merge macro into price+label data
    df = pd.merge_asof(
        df.sort_values("date"),
        macro_df.sort_values("date"),
        on="date",
        direction="backward"
    )

    # Export
    exports_dir = Path(__file__).resolve(
    ).parent.parent.parent / "data" / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d")
    path = exports_dir / f"{ticker_id}_labeled_{timestamp}.csv"

    df.to_csv(path, index=False)
    print(f"  Exported: {path}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Rows: {len(df)}")

    return df


def run_pipeline(
    ticker:     str = "^NSEI",
    ticker_id:  str = "NIFTY50",
    start:      str = "2019-01-01",
    end:        str = "2024-12-31",
    window:     int = 20,
    export_csv_flag: bool = True
):
    print("=" * 60)
    print(f"BUBBLE DETECTION PIPELINE")
    print(f"Ticker:  {ticker} ({ticker_id})")
    print(f"Period:  {start} → {end}")
    print(f"Window:  {window} trading days")
    print("=" * 60)

    # Step 1: Database setup
    print("\n[1/5] Setting up database...")
    engine = get_engine()
    create_tables(engine)

    # Step 2: Download OHLCV data
    print("\n[2/5] Downloading market data...")
    df = download_ohlcv(ticker, start=start, end=end)
    save_to_market_data(df, ticker_id, engine)

    # Step 3 + 4: PSY labeling + technical indicators
    # Replaces old compute_zscore + apply_labels
    print("\n[3/5] Running PSY bubble detection + technical indicators...")
    from zscore import run_zscore_pipeline
    df = run_zscore_pipeline(
        ticker_id=ticker_id,
        engine=engine
    )

    # Step 5: Export CSV with macro features
    if export_csv_flag:
        print("\n[5/5] Exporting CSV with macro features...")
        export_csv(df, ticker_id, start, end)

    print("\nPipeline complete!")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Amal's data pipeline")
    parser.add_argument("--ticker",    default="^NSEI",
                        help="Yahoo Finance ticker")
    parser.add_argument("--ticker_id", default="NIFTY50",
                        help="Internal ticker ID")
    parser.add_argument("--start",     default="2019-01-01",
                        help="Start date YYYY-MM-DD")
    parser.add_argument("--end",       default="2024-12-31",
                        help="End date YYYY-MM-DD")
    parser.add_argument("--window",    default=20,
                        type=int, help="Rolling window size")
    args = parser.parse_args()

    run_pipeline(
        ticker=args.ticker,
        ticker_id=args.ticker_id,
        start=args.start,
        end=args.end,
        window=args.window,
    )


def get_live_features(ticker: str = "^NSEI") -> dict:
    """
    Get today's features ready for Jerome's model to predict on.
    Sohin calls this via FastAPI endpoint.
    """
    from ingestion import fetch_today

    print(f"Fetching live data for {ticker}...")
    features = fetch_today(ticker)

    print(f"\nLive Features for {features['date']}:")
    print(f"  Close Price : {features['close']:.2f}")
    print(f"  BSADF Score : {features.get('bsadf_score', float('nan')):.3f}")
    print(f"  PSY (12)    : {features.get('psy_12', float('nan')):.2f}")
    print(f"  RSI         : {features['rsi']:.2f}")
    print(f"  MACD        : {features['macd']:.2f}")
    print(f"  Label       : {features['label']}")

    return features
