"""
REAL-TIME DATA INGESTION MODULE
=================================
Fetches live NIFTY 50 and NIFTY 500 stock data using yfinance.
No hardcoded data — everything is pulled fresh from Yahoo Finance.

NIFTY 50  → India's top 50 companies (benchmark index)
NIFTY 500 → India's top 500 companies (broad market)

Run this file alone to test:
    python src/data_ingestion.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import os
import time
from datetime import datetime, timedelta
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# TICKER LISTS
# ─────────────────────────────────────────────────────────────────────────────

# NIFTY 50 constituent tickers (Yahoo Finance format: add .NS suffix)
NIFTY50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "BAJFINANCE.NS", "LT.NS", "HCLTECH.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "MARUTI.NS", "NESTLEIND.NS", "WIPRO.NS", "ULTRACEMCO.NS", "TITAN.NS",
    "ADANIENT.NS", "SUNPHARMA.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS",
    "TATAMOTORS.NS", "M&M.NS", "TECHM.NS", "DIVISLAB.NS", "DRREDDY.NS",
    "CIPLA.NS", "BAJAJFINSV.NS", "COALINDIA.NS", "INDUSINDBK.NS", "GRASIM.NS",
    "BPCL.NS", "TATACONSUM.NS", "HINDALCO.NS", "HEROMOTOCO.NS", "EICHERMOT.NS",
    "APOLLOHOSP.NS", "ADANIPORTS.NS", "BRITANNIA.NS", "JSWSTEEL.NS", "TATASTEEL.NS",
    "SBILIFE.NS", "HDFCLIFE.NS", "UPL.NS", "SHRIRAMFIN.NS", "BAJAJ-AUTO.NS"
]

# Main indices
INDICES = {
    "NIFTY 50":    "^NSEI",
    "SENSEX":      "^BSESN",
    "NIFTY BANK":  "^NSEBANK",
    "NIFTY IT":    "^CNXIT",
    "NIFTY PHARMA":"^CNXPHARMA",
}

# ─────────────────────────────────────────────────────────────────────────────
# SINGLE TICKER DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

def download_ticker(
    ticker: str,
    period: str = "5y",          # "1y", "2y", "5y", "10y", "max"
    start: Optional[str] = None, # "2018-01-01"  (overrides period if set)
    end: Optional[str] = None,   # "2024-01-01"
    add_indicators: bool = True,
) -> pd.DataFrame:
    """
    Download OHLCV data for ONE ticker and compute technical indicators.

    Parameters
    ----------
    ticker   : Yahoo Finance symbol  e.g. "^NSEI", "RELIANCE.NS"
    period   : Shorthand period string used when start/end are not supplied.
    start    : ISO date string for start of range.
    end      : ISO date string for end of range (defaults to today).
    add_indicators : Whether to compute RSI, MACD, Bollinger, etc.

    Returns
    -------
    pd.DataFrame  with columns:
        Open  High  Low  Close  Volume  log_return
        rsi  macd  macd_signal  macd_diff  bb_width  roc  (if add_indicators)
    """
    print(f"  ↓ Fetching {ticker} ...", end=" ", flush=True)

    kwargs = dict(progress=False, auto_adjust=True)
    if start:
        kwargs["start"] = start
        kwargs["end"]   = end or datetime.today().strftime("%Y-%m-%d")
    else:
        kwargs["period"] = period

    try:
        raw = yf.download(ticker, **kwargs)
    except Exception as exc:
        print(f"FAILED ({exc})")
        return pd.DataFrame()

    if raw.empty:
        print("EMPTY – check ticker symbol")
        return pd.DataFrame()

    # Flatten MultiIndex columns (yfinance ≥ 0.2 may return them)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    if add_indicators:
        _add_technical_indicators(df)

    df.dropna(inplace=True)
    print(f"OK  ({len(df)} rows,  {df.index[0].date()} → {df.index[-1].date()})")
    return df


def _add_technical_indicators(df: pd.DataFrame) -> None:
    """
    Adds log_return + common technical indicators IN PLACE.

    Why each indicator matters for bubble detection:
    ─────────────────────────────────────────────────
    log_return  : Daily price change (log scale) – better stats than % return
    rsi         : 0-100; >70 = overbought (bubble signal); <30 = oversold
    macd        : Momentum; rising MACD during high Z-score confirms bubble
    bb_width    : Bollinger Band Width – measures volatility expansion
    roc         : Rate of change over 14 days
    """
    # Log return: ln(P_t / P_{t-1})
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    # RSI (14-day)
    df["rsi"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()

    # MACD (fast=12, slow=26, signal=9 — industry defaults)
    macd_ind = ta.trend.MACD(df["Close"], window_fast=12, window_slow=26, window_sign=9)
    df["macd"]        = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_diff"]   = macd_ind.macd_diff()

    # Bollinger Band Width
    bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
    df["bb_width"] = bb.bollinger_wband()

    # Rate of Change (14-day)
    df["roc"] = ta.momentum.ROCIndicator(df["Close"], window=14).roc()

    # Average True Range – volatility proxy
    df["atr"] = ta.volatility.AverageTrueRange(
        df["High"], df["Low"], df["Close"], window=14
    ).average_true_range()


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-TICKER BATCH DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

def download_nifty50(period: str = "5y") -> dict[str, pd.DataFrame]:
    """
    Download all NIFTY 50 constituents.

    Returns
    -------
    dict  { "RELIANCE.NS": DataFrame, "TCS.NS": DataFrame, ... }
    """
    print(f"\n{'='*55}")
    print(f"  Downloading NIFTY 50 constituents  (period={period})")
    print(f"{'='*55}")

    results = {}
    failed  = []

    for i, ticker in enumerate(NIFTY50_TICKERS, 1):
        print(f"  [{i:02d}/{len(NIFTY50_TICKERS)}]", end=" ")
        df = download_ticker(ticker, period=period)
        if df.empty:
            failed.append(ticker)
        else:
            results[ticker] = df
        time.sleep(0.3)   # polite rate-limit pause

    print(f"\n✅ Success: {len(results)}  |  ❌ Failed: {len(failed)}")
    if failed:
        print(f"   Failed tickers: {failed}")
    return results


def download_indices(period: str = "5y") -> dict[str, pd.DataFrame]:
    """Download all main Indian indices (NIFTY 50, SENSEX, …)."""
    print(f"\n{'='*55}")
    print(f"  Downloading Indian Indices  (period={period})")
    print(f"{'='*55}")

    results = {}
    for name, ticker in INDICES.items():
        print(f"  {name:15s}", end=" ")
        df = download_ticker(ticker, period=period)
        if not df.empty:
            results[name] = df
        time.sleep(0.3)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# LIVE / INTRADAY PRICE
# ─────────────────────────────────────────────────────────────────────────────

def get_live_price(ticker: str) -> dict:
    """
    Fetch the most recent price for a ticker using yfinance's fast_info.

    Returns
    -------
    dict with keys: ticker, price, prev_close, change_pct, market_cap, timestamp
    """
    try:
        t    = yf.Ticker(ticker)
        info = t.fast_info          # lightweight – no heavy API call

        price      = info.last_price
        prev_close = info.previous_close
        change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0.0

        return {
            "ticker":     ticker,
            "price":      round(price, 2),
            "prev_close": round(prev_close, 2),
            "change_pct": round(change_pct, 2),
            "market_cap": getattr(info, "market_cap", None),
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception as exc:
        return {"ticker": ticker, "error": str(exc)}


def get_live_prices_batch(tickers: list[str]) -> pd.DataFrame:
    """
    Fetch live prices for multiple tickers at once.
    Uses yfinance multi-download (one HTTP call) for speed.
    """
    print(f"Fetching live prices for {len(tickers)} tickers...")
    raw = yf.download(
        tickers, period="2d", progress=False, auto_adjust=True, group_by="ticker"
    )

    rows = []
    for ticker in tickers:
        try:
            if len(tickers) == 1:
                closes = raw["Close"]
            else:
                closes = raw[ticker]["Close"]

            closes = closes.dropna()
            if len(closes) < 2:
                continue

            price      = closes.iloc[-1]
            prev_close = closes.iloc[-2]
            change_pct = (price - prev_close) / prev_close * 100

            rows.append({
                "ticker":     ticker,
                "price":      round(price, 2),
                "prev_close": round(prev_close, 2),
                "change_pct": round(change_pct, 2),
            })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    print(f"✅ Got prices for {len(df)} tickers")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SAVE / LOAD HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def save_data(df: pd.DataFrame, filename: str, folder: str = "data") -> None:
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    df.to_csv(path)
    print(f"   💾 Saved → {path}")


def load_data(filename: str, folder: str = "data") -> pd.DataFrame:
    path = os.path.join(folder, filename)
    df   = pd.read_csv(path, index_col=0, parse_dates=True)
    print(f"   📂 Loaded {len(df)} rows from {path}")
    return df


def save_all_stocks(stock_dict: dict[str, pd.DataFrame], folder: str = "data/stocks") -> None:
    """Save each stock DataFrame to its own CSV."""
    os.makedirs(folder, exist_ok=True)
    for ticker, df in stock_dict.items():
        safe_name = ticker.replace(".", "_").replace("^", "IDX_")
        df.to_csv(os.path.join(folder, f"{safe_name}.csv"))
    print(f"✅ Saved {len(stock_dict)} stock files to {folder}/")


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    # 1.  Single index
    print("\n── Test 1: Single Index ──────────────────────────────────")
    nifty = download_ticker("^NSEI", period="5y")
    if not nifty.empty:
        save_data(nifty, "nifty50_index.csv")
        print(nifty.tail(3))

    # 2.  Live price
    print("\n── Test 2: Live price ────────────────────────────────────")
    live = get_live_price("^NSEI")
    print(live)

    # 3.  All indices
    print("\n── Test 3: All Indices ───────────────────────────────────")
    indices = download_indices(period="3y")
    for name, df in indices.items():
        save_data(df, f"{name.replace(' ', '_')}.csv")

    # 4.  Uncomment to download ALL NIFTY 50 stocks (~3 min)
    # print("\n── Test 4: NIFTY 50 Constituents ────────────────────────")
    # stocks = download_nifty50(period="5y")
    # save_all_stocks(stocks)

    print("\n✅ data_ingestion.py  →  ALL TESTS PASSED")