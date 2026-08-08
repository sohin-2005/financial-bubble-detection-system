import pandas as pd
import requests
import io
import os
from pathlib import Path

# Resolve exports directory relative to repo root
# __file__ = <repo>/src/training/macro.py  ->  parents[2] = <repo>
OUTPUT_DIR = (Path(__file__).resolve().parents[2] /
              "data" / "exports").resolve()


def fetch_gdp() -> pd.DataFrame:
    """
    India GDP growth rate — quarterly
    Source: World Bank API (free, no key needed)
    """
    print("  Fetching India GDP...")

    url = (
        "https://api.worldbank.org/v2/country/IN/indicator/"
        "NY.GDP.MKTP.KD.ZG?format=json&per_page=100&mrv=100"
    )

    try:
        response = requests.get(url, timeout=30)
        data = response.json()
        records = data[1]

        rows = []
        for record in records:
            if record["value"] is not None:
                rows.append({
                    "date":       str(record["date"]) + "-01-01",
                    "gdp_growth": float(record["value"])
                })

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        print(f"  GDP: {len(df)} annual records fetched")
        return df

    except Exception as e:
        print(f"  GDP fetch failed: {e}")
        print("  Using fallback GDP data...")
        return _fallback_gdp()


def _fallback_gdp() -> pd.DataFrame:
    """
    Hardcoded India annual GDP growth rates
    Source: World Bank / RBI annual reports
    Use this if API fails
    """
    data = {
        "date": [
            "2008-01-01", "2009-01-01", "2010-01-01",
            "2011-01-01", "2012-01-01", "2013-01-01",
            "2014-01-01", "2015-01-01", "2016-01-01",
            "2017-01-01", "2018-01-01", "2019-01-01",
            "2020-01-01", "2021-01-01", "2022-01-01",
            "2023-01-01", "2024-01-01"
        ],
        "gdp_growth": [
            3.1,   # 2008 — GFC impact
            7.9,   # 2009
            8.5,   # 2010 — recovery
            6.6,   # 2011
            5.5,   # 2012
            6.4,   # 2013
            7.4,   # 2014
            8.0,   # 2015
            8.3,   # 2016
            6.8,   # 2017
            6.5,   # 2018
            3.9,   # 2019 — slowdown
            -5.8,   # 2020 — COVID crash
            9.1,   # 2021 — recovery
            7.2,   # 2022
            8.2,   # 2023
            8.2,   # 2024 estimate
        ]
    }
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    return df


def fetch_cpi() -> pd.DataFrame:
    """
    India CPI inflation rate — annual
    Source: World Bank API
    """
    print("  Fetching India CPI...")

    url = (
        "https://api.worldbank.org/v2/country/IN/indicator/"
        "FP.CPI.TOTL.ZG?format=json&per_page=100&mrv=100"
    )

    try:
        response = requests.get(url, timeout=30)
        data = response.json()
        records = data[1]

        rows = []
        for record in records:
            if record["value"] is not None:
                rows.append({
                    "date":          str(record["date"]) + "-01-01",
                    "cpi_inflation": float(record["value"])
                })

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        print(f"  CPI: {len(df)} annual records fetched")
        return df

    except Exception as e:
        print(f"  CPI fetch failed: {e}")
        return _fallback_cpi()


def _fallback_cpi() -> pd.DataFrame:
    """Hardcoded India annual CPI inflation rates"""
    data = {
        "date": [
            "2008-01-01", "2009-01-01", "2010-01-01",
            "2011-01-01", "2012-01-01", "2013-01-01",
            "2014-01-01", "2015-01-01", "2016-01-01",
            "2017-01-01", "2018-01-01", "2019-01-01",
            "2020-01-01", "2021-01-01", "2022-01-01",
            "2023-01-01", "2024-01-01"
        ],
        "cpi_inflation": [
            8.35,   # 2008
            10.88,  # 2009
            12.11,  # 2010 — high inflation
            8.86,   # 2011
            9.30,   # 2012
            10.92,  # 2013
            6.37,   # 2014
            5.87,   # 2015
            4.94,   # 2016
            2.49,   # 2017 — low inflation
            4.86,   # 2018
            7.66,   # 2019
            6.18,   # 2020
            5.13,   # 2021
            6.70,   # 2022 — rising inflation
            5.65,   # 2023
            4.80,   # 2024
        ]
    }
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    return df


def fetch_repo_rate(start_date: str = "2008-01-01", api_key: str | None = None) -> pd.DataFrame:
    """Fetch India policy rate from FRED (series INTDSRINM193N) with fallback.

    - Primary: FRED API (requires FRED_API_KEY env or api_key argument)
    - Fallback: hardcoded RBI change log (2008–2024)
    """
    api_key = api_key or os.getenv("FRED_API_KEY")
    try:
        df_live = _fetch_repo_rate_fred(start_date=start_date, api_key=api_key)
        if not df_live.empty:
            return df_live
    except Exception as e:
        print(f"  Repo rate fetch failed: {e}")

    print("  Using fallback repo rate (hardcoded)")
    return _fallback_repo_rate()


def _fetch_repo_rate_fred(start_date: str, api_key: str | None) -> pd.DataFrame:
    if not api_key:
        raise ValueError("FRED_API_KEY missing for live repo rate fetch")

    print("  Fetching repo rate from FRED (INTDSRINM193N)...")
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "INTDSRINM193N",  # India policy rate
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    records = data.get("observations", [])

    rows = []
    for r in records:
        val = r.get("value")
        if val in (None, "."):
            continue
        rows.append({
            "date": r.get("date"),
            "repo_rate": float(val),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  Repo rate: {len(df)} observations fetched from FRED")
    return df


def _fallback_repo_rate() -> pd.DataFrame:
    """Hardcoded RBI repo rate changes (2008–2024)."""
    data = {
        "date": [
            "2008-01-01", "2008-06-11", "2008-07-25",
            "2008-10-20", "2008-11-03", "2008-12-08",
            "2009-01-02", "2009-03-04", "2009-04-21",
            "2010-03-19", "2010-04-20", "2010-07-02",
            "2010-09-16", "2010-11-02", "2011-01-25",
            "2011-03-17", "2011-05-03", "2011-06-16",
            "2011-07-26", "2011-10-25", "2012-04-17",
            "2013-01-29", "2013-03-19", "2013-05-03",
            "2014-01-28", "2014-06-03", "2015-01-15",
            "2015-03-04", "2015-09-29", "2016-04-05",
            "2016-10-04", "2017-08-02", "2018-06-06",
            "2018-08-01", "2019-02-07", "2019-04-04",
            "2019-06-06", "2019-08-07", "2019-10-04",
            "2020-03-27", "2020-05-22",
            "2022-05-04", "2022-06-08", "2022-08-05",
            "2022-09-30", "2022-12-07",
            "2023-02-08", "2023-04-06",
            "2024-10-09",
        ],
        "repo_rate": [
            7.75, 8.00, 8.50,
            8.00, 7.50, 6.50,
            5.50, 5.00, 4.75,
            5.00, 5.25, 5.50,
            5.75, 6.00, 6.25,
            6.50, 6.75, 7.00,
            7.25, 7.50, 8.00,
            7.75, 7.50, 7.25,
            8.00, 8.00, 7.75,
            7.50, 6.75, 6.50,
            6.25, 6.00, 6.25,
            6.50, 6.25, 6.00,
            5.75, 5.40, 5.15,
            4.40, 4.00,
            4.40, 4.90, 5.40,
            5.90, 6.25,
            6.50, 6.50,
            6.75,
        ]
    }

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

    print(f"  Repo rate: {len(df)} change events loaded")
    return df


def build_macro_daily(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Build a daily macro dataset by forward filling
    GDP, CPI (annual/quarterly) and repo rate
    to match daily price data frequency.
    """
    print("\n[MACRO] Building daily macro features...")

    # Fetch all three
    gdp = fetch_gdp()
    cpi = fetch_cpi()
    repo = fetch_repo_rate(start_date=start_date)

    # Create full daily date range
    daily_dates = pd.DataFrame({
        "date": pd.date_range(start=start_date, end=end_date, freq="B")
    })

    # Forward fill GDP — annual → daily
    daily_dates = pd.merge_asof(
        daily_dates, gdp,
        on="date", direction="backward"
    )

    # Forward fill CPI — annual → daily
    daily_dates = pd.merge_asof(
        daily_dates, cpi,
        on="date", direction="backward"
    )

    # Forward fill repo rate — event-based → daily
    daily_dates = pd.merge_asof(
        daily_dates, repo,
        on="date", direction="backward"
    )

    # Fill any remaining NaN at the start
    daily_dates["gdp_growth"] = daily_dates["gdp_growth"].ffill().bfill()
    daily_dates["cpi_inflation"] = daily_dates["cpi_inflation"].ffill().bfill()
    daily_dates["repo_rate"] = daily_dates["repo_rate"].ffill().bfill()

    print(f"  Daily macro rows: {len(daily_dates)}")
    print(f"  GDP range:    {daily_dates['gdp_growth'].min():.1f}% "
          f"to {daily_dates['gdp_growth'].max():.1f}%")
    print(f"  CPI range:    {daily_dates['cpi_inflation'].min():.1f}% "
          f"to {daily_dates['cpi_inflation'].max():.1f}%")
    print(f"  Repo range:   {daily_dates['repo_rate'].min():.2f}% "
          f"to {daily_dates['repo_rate'].max():.2f}%")

    # Save for reference
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    daily_dates.to_csv(
        OUTPUT_DIR / "macro_daily.csv", index=False
    )
    print("  Saved: macro_daily.csv")

    return daily_dates


if __name__ == "__main__":
    df = build_macro_daily("2008-01-01", "2024-12-31")
    print("\nSample output:")
    print(df.head(10).to_string())
