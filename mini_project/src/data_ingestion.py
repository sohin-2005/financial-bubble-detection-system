import pandas as pd
import yfinance as yf


def download_ticker(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance and return a tidy DataFrame.
    """
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename_axis("Date").reset_index()
    return df
