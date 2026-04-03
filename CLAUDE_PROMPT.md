I am building a financial bubble detection ML system on NIFTY50 data (Python project). Please read this context and help me improve model performance.

Project goal:

- Detect 3 classes: Bubble, Crash, Normal.
- Primary metric: macro F1 score.
- Target requirement: F1 >= 0.85.

What I have done so far:

1. Environment + dependencies

- Set up virtual environment and installed requirements.
- Installed imbalanced-learn and scikit-learn.
- Installed xgboost and fixed macOS OpenMP issue with libomp.

2. Data setup

- Data files used:
  - data/exports/NIFTY50_train.csv
  - data/exports/NIFTY50_validation.csv
  - data/exports/NIFTY50_test.csv
  - data/exports/macro_daily.csv
- Sentiment files were initially missing, so neutral placeholders were created:
  - data/exports/nifty50_daily_sentiment.csv
  - data/exports/nifty50_validation_sentiment.csv
  - data/exports/nifty50_test_sentiment.csv
- Placeholder sentiment uses polarity_score = 0.0.

3. Pipeline runs completed

- Ran Random Forest training: src/amal/train_rf.py
- Ran XGBoost training: src/amal/train_xgb.py
- Ran stacking ensemble: src/amal/stacking.py

4. Small code fix already made

- Updated src/amal/train_rf.py to be robust when sentiment files are missing:
  - falls back to daily_sentiment_index = 0.0
  - creates label encoder if missing

Current results obtained:

- Random Forest:
  - Validation F1: 0.4190
  - Test F1: 0.3803
- XGBoost:
  - Validation F1: 0.3853
  - Test F1: 0.3359
- Stacking (RF + XGB + LR):
  - Validation F1: 0.4461
  - Test F1: 0.3470

Artifacts available:

- data/models/rf_model_stats.json
- data/models/xgb_model_stats.json
- data/models/stacking_stats.json
- data/models/rf_model.pkl
- data/models/xgb_model.pkl
- data/models/stacking_model.pkl

Important observation:

- Class imbalance is severe (Normal dominates), and sentiment is currently placeholder/neutral, likely hurting minority-class recall and macro F1.

What I want from you:

- Give me a practical, step-by-step improvement plan to raise macro F1 significantly.
- Prioritize actions by expected impact and effort.
- Include concrete code-level changes for this repo structure.
- Suggest better handling for class imbalance (sampling, class weights, thresholds, focal loss alternatives, etc.).
- Suggest feature engineering ideas for financial time-series and regime detection.
- Suggest time-series-safe validation strategy and hyperparameter search setup.
- Suggest how to diagnose error patterns by class and improve Bubble/Crash recall.
- Provide an execution checklist I can run in order.

Please answer in this structure:

1. Root-cause diagnosis from current metrics
2. High-impact quick wins (today)
3. Medium-term model improvements
4. Data/sentiment improvements
5. Validation + experiment tracking template
6. Exact next 10 commands/files to edit in this project
