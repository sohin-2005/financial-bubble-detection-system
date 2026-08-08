import yfinance as yf
import pandas as pd
import numpy as np


# Ticker symbols for Indian indices
TICKERS = {
    "NIFTY50": "^NSEI",
    "SENSEX":  "^BSESN",
}


def download_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download daily OHLCV data for a ticker from Yahoo Finance.

    Args:
        ticker: Yahoo Finance ticker symbol e.g. "^NSEI"
        start:  Start date string "YYYY-MM-DD"
        end:    End date string "YYYY-MM-DD"

    Returns:
        Cleaned DataFrame with columns: date, open, high, low, close, volume, log_return
    """
    print(f"Downloading data for {ticker} from {start} to {end}...")

    # Download from Yahoo Finance
    raw = yf.download(ticker, start=start, end=end, progress=False)

    if raw.empty:
        raise ValueError(
            f"No data returned for ticker {ticker}. Check the symbol.")

    # Flatten multi-level columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Rename columns to lowercase
    df = raw.rename(columns={
        "Open":   "open",
        "High":   "high",
        "Low":    "low",
        "Close":  "close",
        "Volume": "volume",
    })

    # Keep only what we need
    df = df[["open", "high", "low", "close", "volume"]].copy()

    # Reset index so 'date' becomes a column
    df = df.reset_index()
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    elif "index" in df.columns:
        df = df.rename(columns={"index": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # Forward-fill missing values (public holidays etc.)
    df = df.ffill()

    # Drop any remaining nulls
    df = df.dropna(subset=["close"])

    # Calculate log return: log(today_close / yesterday_close)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Drop first row (no log return for it)
    df = df.dropna(subset=["log_return"])

    print(f"Downloaded {len(df)} rows for {ticker}.")
    return df


def save_to_market_data(df: pd.DataFrame, ticker_id: str, engine) -> int:
    """
    Save cleaned OHLCV dataframe to the market_data table.

    Returns:
        Number of rows inserted
    """
    from sqlalchemy import text

    rows_inserted = 0

    with engine.connect() as conn:
        for _, row in df.iterrows():
            conn.execute(text("""
                INSERT INTO market_data 
                    (ticker_id, date, open, high, low, close, volume, log_return)
                VALUES 
                    (:ticker_id, :date, :open, :high, :low, :close, :volume, :log_return)
                ON CONFLICT (ticker_id, date) DO UPDATE SET
                    close      = EXCLUDED.close,
                    log_return = EXCLUDED.log_return
            """), {
                "ticker_id":  ticker_id,
                "date":       str(row["date"]),
                "open":       float(row["open"]),
                "high":       float(row["high"]),
                "low":        float(row["low"]),
                "close":      float(row["close"]),
                "volume":     int(row["volume"]),
                "log_return": float(row["log_return"]),
            })
            rows_inserted += 1

        conn.commit()

    print(f"Saved {rows_inserted} rows to market_data for {ticker_id}.")
    return rows_inserted


if __name__ == "__main__":
    from database import get_engine, create_tables

    engine = get_engine()
    create_tables(engine)

    df = download_ohlcv("^NSEI", start="2019-01-01", end="2024-12-31")
    save_to_market_data(df, "NIFTY50", engine)

    print(df.tail())


def fetch_today(ticker: str) -> dict:
    """
    Fetch today's latest price data for live prediction.
    Returns a single row of features.
    """
    import yfinance as yf
    from datetime import datetime, timedelta

    # Fetch enough history for PSY/BSADF + rolling thresholds
    # (needs >252 rows for rolling quantile and ~157 for BSADF window)
    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=900)).strftime("%Y-%m-%d")

    df = yf.download(ticker, start=start, end=end,
                     auto_adjust=True, ignore_tz=True, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={
        "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume"
    })

    df = df.reset_index()
    df = df.rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df = df.dropna(subset=["log_return"])

    # PSY (Psychological Line) feature used by model training
    up_days = (df["close"].diff() > 0).astype(float)
    df["psy_12"] = up_days.rolling(12).mean() * 100.0

    # Main bubble detection path: zscore.py PSY/BSADF hybrid labeling
    from zscore import compute_technical_indicators, compute_bsadf, apply_hybrid_labels

    df = compute_technical_indicators(df)
    log_prices = np.log(df["close"])
    df["bsadf_score"] = compute_bsadf(log_prices, min_window=157, max_lag=4)
    df = apply_hybrid_labels(
        df,
        upper_pct=0.95,
        crash_zscore=-2.0,
        bubble_zscore=2.0,
        rolling_window=504,
    )

    df = df.dropna(subset=["rsi", "macd", "macd_signal", "macd_hist"])

    # Return only today — the last row
    today = df.iloc[-1]

    return {
        "date":         str(today["date"]),
        "close":        float(today["close"]),
        "bsadf_score":  float(today["bsadf_score"])
        if pd.notna(today.get("bsadf_score", np.nan)) else np.nan,
        "psy_12":       float(today["psy_12"])
        if pd.notna(today.get("psy_12", np.nan)) else np.nan,
        "zscore_value": float(today["zscore_value"])
        if pd.notna(today.get("zscore_value", np.nan)) else np.nan,
        "rsi":          float(today["rsi"]),
        "macd":         float(today["macd"]),
        "macd_signal":  float(today["macd_signal"]),
        "macd_hist":    float(today["macd_hist"]),
        "label":        str(today["label"]),
        "log_return":   float(today["log_return"]),
    }


if __name__ == "__main__":
    result = fetch_today("^NSEI")
    print(result)
