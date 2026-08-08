# Financial Bubble Detection System

A real-time Streamlit dashboard that classifies the current state of the Indian
equity market (NIFTY 50) as **Normal**, **Bubble**, or **Crash / High Risk**, by
combining technical indicators, macroeconomic data, and FinBERT news sentiment
through a stacked Random Forest + XGBoost ensemble.

---

## Quick start

```bash
cd financial-bubble-detection-system

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env              # optional — add free API keys
streamlit run app.py
```

Run everything from the project root — `app.py`, `data/`, `models/`, and `src/`
all live here.

The dashboard runs without any API keys: news falls back to public RSS feeds and
macro indicators come from the cached `data/exports/macro_daily.csv`.

---

## Project structure

```
financial-bubble-detection-system/
├── app.py                      Streamlit dashboard — the only entry point
│
├── src/
│   ├── data_ingestion.py       Yahoo Finance OHLCV download
│   ├── zscore_labeling.py      Technical indicators + Z-score/BSADF labelling
│   ├── news_fetcher.py         NewsAPI / GNews / RSS headline collection
│   ├── sentiment_engine.py     FinBERT scoring + daily sentiment index
│   └── training/               Offline pipeline that produced the models
│       ├── pipeline.py         Orchestrates ingestion → labelling → export
│       ├── ingestion.py        Bulk OHLCV download into the database
│       ├── macro.py            GDP / CPI / repo-rate builder (also used live)
│       ├── zscore.py           Full BSADF + hybrid labelling implementation
│       ├── label.py            Label assignment and persistence
│       ├── relabel.py          Re-label an existing export
│       ├── split.py            Chronological train / validation / test split
│       ├── train_rf.py         Random Forest training
│       ├── train_xgb.py        XGBoost training
│       ├── stacking.py         Meta-learner over RF + XGB probabilities
│       ├── crash_detection.py  MACD-based crash feature experiments
│       ├── bubbletest.py       Threshold experiments
│       ├── sanity_check.py     Post-training evaluation
│       ├── xgbpredict.py       Standalone XGBoost inference
│       └── database.py         SQLAlchemy engine + table setup
│
├── data/
│   ├── raw/                    Downloaded index CSVs and raw news
│   ├── processed/              Merged / labelled / sentiment-joined frames
│   ├── exports/                Train-test splits, sentiment, macro_daily.csv
│   └── models/                 rf_model, xgb_model, label_encoder, stats JSON
│
├── models/                     Stacking ensemble + its meta scaler (+ variants)
├── notebooks/                  Exploratory analysis
├── outputs/                    Generated charts and feature-importance exports
├── reports/                    Word report and its generator script
│
├── requirements.txt
├── .env.example
└── LICENSE
```

---

## How it works

**1. Price and indicators** — `src/data_ingestion.py` pulls 3 years of daily
OHLCV data. `src/zscore_labeling.py` derives rolling mean/std, Z-score,
log returns, RSI(14), MACD(12,26,9), PSY(12), and a BSADF score, then assigns a
baseline label: crash when Z < −2, bubble when the BSADF score crosses its
rolling 95th percentile on a positive return (with a Z > +2 fallback).

**2. Sentiment** — `src/news_fetcher.py` gathers headlines from NewsAPI, GNews,
and RSS feeds. `src/sentiment_engine.py` scores each headline with FinBERT and
aggregates them into a daily polarity index plus a 3-day momentum term.

**3. Macro** — `src/training/macro.py` builds a daily series of GDP growth, CPI
inflation, and repo rate. The dashboard rebuilds it automatically when
`data/exports/macro_daily.csv` is more than 24 hours old.

**4. Prediction** — Random Forest and XGBoost each produce class probabilities.
The stacking meta-model consumes both probability vectors (scaled) and outputs
the final distribution. Labels are then gated by confidence thresholds that are
deliberately stricter for the most recent 180 days, so a single noisy day cannot
flip the headline market status.

**5. Display** — a live ticker tape, market-status banner, six metric cards, a
feature snapshot, feature-contribution bars, and three tabs (price analysis,
live news, statistics).

---

## Model artifacts

The dashboard resolves models in this order, taking the first that exists:

| Artifact       | First choice                       | Fallback                          |
|----------------|------------------------------------|-----------------------------------|
| Stacking model | `models/stacking_ensemble.pkl`     | `data/models/stacking_model.pkl`  |
| Meta scaler    | `models/scaler.pkl`                | `data/models/stacking_scaler.pkl` |
| Feature list   | `models/feature_cols.pkl`          | —                                 |
| Label encoder  | `models/label_encoder.pkl`         | `data/models/label_encoder.pkl`   |
| RF / XGB       | `data/models/rf_model.pkl`, `xgb_model.pkl` | —                        |

`models/*_version_a.pkl` and `*_version_b.pkl` are earlier training runs kept for
comparison; nothing loads them automatically.

---

## Retraining

The training scripts resolve paths relative to their own directory, so run them
from inside `src/training`:

```bash
cd src/training
python pipeline.py        # ingest + label + export
python split.py           # chronological train/val/test split
python train_rf.py
python train_xgb.py
python stacking.py        # fit the meta-learner
python sanity_check.py    # evaluate
```

---

## Configuration

Set in `.env` (all optional):

| Variable                | Purpose                                        |
|-------------------------|------------------------------------------------|
| `NEWS_API_KEY`          | NewsAPI headlines                              |
| `GNEWS_API_KEY`         | GNews headlines                                |
| `FRED_API_KEY`          | FRED macro series                              |
| `SHOW_MACRO_FRESHNESS`  | Set to `1` to show macro file age in the UI    |

Analysis parameters (`period = "3y"`, `window = 30`, `news_days = 7`) are set
near the top of `app.py`.
