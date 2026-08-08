"""
Z-SCORE / BSADF BUBBLE LABELING
===============================
Turns an OHLCV frame into the labelled feature frame the dashboard renders:

    rolling_mean, rolling_std, zscore, zscore_value
    log_return, rsi, macd, macd_signal, macd_hist, psy_12, bsadf_score
    label  ->  "Normal" | "Bubble" | "Crash"

Labelling is hybrid: a Z-score rule catches crashes, while bubbles come from the
PSY/BSADF (Backward Sup ADF) test with a Z-score fallback.
"""

import numpy as np
import pandas as pd


def compute_bsadf(
    prices: pd.Series,
    min_window: int = 157,
    max_lag: int = 4,
) -> pd.Series:
    """Compute BSADF (Backward Sup ADF) scores for a log-price series."""
    n = len(prices)
    bsadf_scores = np.full(n, np.nan)

    for t in range(min_window, n):
        max_adf = -np.inf

        for start in range(0, t - min_window + 1):
            window = prices.iloc[start:t + 1].values
            if len(window) < max_lag + 5:
                continue

            y = window
            dy = np.diff(y)
            y_lag = y[:-1]

            best_bic = np.inf
            best_t = np.nan

            for k in range(max_lag + 1):
                try:
                    if k == 0:
                        X = np.column_stack([
                            np.ones(len(y_lag)),
                            y_lag,
                        ])
                        y_dep = dy
                    else:
                        lag_diffs = np.column_stack([
                            dy[k - i:-i] if i > 0 else dy[k:]
                            for i in range(1, k + 1)
                        ])
                        X = np.column_stack([
                            np.ones(len(y_lag[k:])),
                            y_lag[k:],
                            lag_diffs,
                        ])
                        y_dep = dy[k:]

                    XtX = X.T @ X
                    if np.linalg.det(XtX) == 0:
                        continue

                    beta = np.linalg.inv(XtX) @ X.T @ y_dep
                    residuals = y_dep - X @ beta
                    n_obs = len(y_dep)
                    n_params = X.shape[1]
                    sigma_sq = np.sum(residuals ** 2) / (n_obs - n_params)
                    bic = n_obs * np.log(sigma_sq) + n_params * np.log(n_obs)

                    if bic < best_bic:
                        best_bic = bic
                        se_beta = np.sqrt(sigma_sq * np.linalg.inv(XtX)[1, 1])
                        if se_beta > 0:
                            best_t = beta[1] / se_beta

                except Exception:
                    continue

            if not np.isnan(best_t) and best_t > max_adf:
                max_adf = best_t

        if max_adf != -np.inf:
            bsadf_scores[t] = max_adf

    return pd.Series(bsadf_scores, index=prices.index)


def compute_zscore_labels(df: pd.DataFrame, price_col: str = "Close", window: int = 20) -> pd.DataFrame:
    """
    Hybrid bubble labeling for dashboard:
    - Crash: Z-score < -2
    - Bubble: PSY/BSADF threshold + up-return direction
    - Bubble fallback: Z-score > +2
    """
    df = df.copy()
    rolling_mean = df[price_col].rolling(window=window).mean()
    rolling_std = df[price_col].rolling(window=window).std()
    zscore = (df[price_col] - rolling_mean) / rolling_std

    df["rolling_mean"] = rolling_mean
    df["rolling_std"] = rolling_std
    df["zscore"] = zscore
    df["zscore_value"] = zscore

    if "log_return" not in df.columns:
        df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))

    # RSI (14)
    delta = df[price_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD (12,26,9)
    ema_fast = df[price_col].ewm(span=12, adjust=False).mean()
    ema_slow = df[price_col].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # PSY (Psychological Line, 12-day): % of up days in recent window
    up_days = (df[price_col].diff() > 0).astype(float)
    df["psy_12"] = up_days.rolling(window=12).mean() * 100.0

    # BSADF score for PSY procedure
    log_prices = np.log(df[price_col].replace(0, np.nan)).dropna()
    bsadf = compute_bsadf(log_prices, min_window=157, max_lag=4)
    df["bsadf_score"] = bsadf.reindex(df.index)

    upper_bound = df["bsadf_score"].rolling(
        window=504,
        min_periods=252,
    ).quantile(0.95)

    df["label"] = "Normal"

    # Crash first
    df.loc[
        (df["zscore_value"] < -2.0) &
        (df["label"] == "Normal"),
        "label",
    ] = "Crash"

    # PSY/BSADF bubble with direction filter
    df.loc[
        (df["bsadf_score"] >= upper_bound) &
        (df["log_return"] > 0) &
        (df["label"] == "Normal"),
        "label",
    ] = "Bubble"

    # Bubble fallback from Z-score
    df.loc[
        (df["zscore_value"] > 2.0) &
        (df["label"] == "Normal"),
        "label",
    ] = "Bubble"

    return df
