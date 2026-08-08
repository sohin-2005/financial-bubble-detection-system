import yfinance as yf
df = yf.download("^NSEI", start="2008-01-01", end="2009-12-31", auto_adjust=True, ignore_tz=True)
print(f"Rows: {len(df)}")
print(f"Earliest: {df.index.min()}")
print(df.head())