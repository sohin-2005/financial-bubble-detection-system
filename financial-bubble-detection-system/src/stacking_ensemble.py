"""
STACKING ENSEMBLE MODULE
=========================
Combines Random Forest + XGBoost predictions using Logistic Regression as meta-model.

WHY STACKING WORKS BETTER THAN A SINGLE MODEL:
  RF and XGBoost each have different strengths and blind spots.
  RF is better at stable, noisy data.
  XGBoost is better at capturing complex patterns.
  Logistic Regression learns WHEN to trust RF vs XGBoost.

STACKING PROCESS:
  Training:
    1. Split training data into K folds (K=5)
    2. For each fold, train RF and XGBoost on K-1 folds,
       predict on the held-out fold → "out-of-fold" (OOF) predictions
    3. OOF predictions become meta-features for Logistic Regression
    4. Train Logistic Regression on those meta-features

  Prediction:
    1. RF and XGBoost predict probabilities on new data
    2. Stack those probabilities → meta-features
    3. Logistic Regression makes final prediction

WHY OUT-OF-FOLD PREDICTIONS MATTER:
  If you train RF on the SAME data and then use its predictions
  as features, the meta-model sees "cheated" predictions
  (RF will be near-perfect on training data it already saw).
  OOF predictions are made on data the model never saw,
  so meta-model learns from honest predictions.

Run standalone: python src/stacking_ensemble.py
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from imblearn.over_sampling import ADASYN, SMOTE

warnings.filterwarnings("ignore")


class StackingEnsemble:
    """
    Stacking ensemble:
      Base models : Random Forest + XGBoost
      Meta-model  : Logistic Regression

    Usage:
        ensemble = StackingEnsemble(base_models, n_folds=5)
        ensemble.fit(X_train, y_train)
        proba = ensemble.predict_proba(X_test)   # shape (n, 3)
        preds = ensemble.predict(X_test)         # shape (n,)
    """

    def __init__(self, base_models, meta_model=None, n_folds=5):
        """
        Parameters
        ----------
        base_models : list of (name, unfitted_model) tuples
        meta_model  : meta-learner (default: LogisticRegression)
        n_folds     : number of cross-validation folds for OOF
        """
        self.base_models = base_models
        self.meta_model = meta_model or LogisticRegression(
            max_iter=2000, C=1.0,
            multi_class="multinomial", solver="lbfgs",
            random_state=42,
        )
        self.n_folds = n_folds
        self.n_classes = 3           # Normal=0, Bubble=1, Crash=2
        self.fitted_bases = []          # stores models trained on full data
        self.is_fitted = False

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(self, X_train, y_train):
        print("\n" + "="*55)
        print("  TRAINING STACKING ENSEMBLE")
        print("="*55)

        # Step 1: OOF predictions for meta-features
        print("\n[Step 1/3] Generating out-of-fold predictions...")
        meta_features = self._get_oof_predictions(X_train, y_train)

        # Step 2: Train base models on FULL training data
        print("\n[Step 2/3] Training base models on full training data...")
        self.fitted_bases = []
        for name, model in self.base_models:
            print(f"  Training {name}...", end=" ", flush=True)
            model.fit(X_train, y_train)
            self.fitted_bases.append((name, model))
            print("done")

        # Step 3: Train meta-model on OOF predictions
        print("\n[Step 3/3] Training Logistic Regression meta-model...")
        self.meta_model.fit(meta_features, y_train)
        print("  Meta-model trained")

        self.is_fitted = True
        print("\n  ✅ Stacking ensemble fully trained!")
        return self

    def _get_oof_predictions(self, X, y):
        """
        Generate out-of-fold predictions using StratifiedKFold.
        Returns meta-features array: shape (n_samples, n_models * n_classes)
        """
        kfold = StratifiedKFold(n_splits=self.n_folds,
                                shuffle=True, random_state=42)
        oof_list = []   # one array per base model

        for name, model in self.base_models:
            print(f"\n  OOF for {name}:")
            oof_pred = np.zeros((len(X), self.n_classes))

            for fold, (tr_idx, val_idx) in enumerate(kfold.split(X, y)):
                X_fold_tr = X[tr_idx]
                y_fold_tr = y[tr_idx]
                X_fold_val = X[val_idx]

                # Apply ADASYN inside each fold to avoid data leakage
                try:
                    adasyn = ADASYN(sampling_strategy="not majority",
                                    random_state=fold, n_neighbors=5)
                    X_fold_tr, y_fold_tr = adasyn.fit_resample(
                        X_fold_tr, y_fold_tr)
                except Exception:
                    try:
                        smote = SMOTE(sampling_strategy="not majority",
                                      random_state=fold, k_neighbors=3)
                        X_fold_tr, y_fold_tr = smote.fit_resample(
                            X_fold_tr, y_fold_tr)
                    except Exception:
                        pass  # use as-is with class_weight=balanced

                # Deep copy then fit
                m_copy = pickle.loads(pickle.dumps(model))
                m_copy.fit(X_fold_tr, y_fold_tr)

                # Predict probabilities on held-out fold
                preds = m_copy.predict_proba(X_fold_val)

                # Handle missing classes (rare in small folds)
                if preds.shape[1] < self.n_classes:
                    full = np.zeros((len(X_fold_val), self.n_classes))
                    for i, cls in enumerate(m_copy.classes_):
                        full[:, int(cls)] = preds[:, i]
                    preds = full

                oof_pred[val_idx] = preds
                print(f"    fold {fold+1}/{self.n_folds}  "
                      f"(train={len(X_fold_tr)}, val={len(X_fold_val)})")

            oof_list.append(oof_pred)

        # Stack horizontally: (n_samples, n_models * n_classes)
        meta = np.hstack(oof_list)
        print(f"\n  Meta-features shape: {meta.shape}")
        return meta

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_proba(self, X):
        """Returns probability array of shape (n_samples, 3)."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict_proba()")

        base_preds = []
        for name, model in self.fitted_bases:
            preds = model.predict_proba(X)
            if preds.shape[1] < self.n_classes:
                full = np.zeros((len(X), self.n_classes))
                for i, cls in enumerate(model.classes_):
                    full[:, int(cls)] = preds[:, i]
                preds = full
            base_preds.append(preds)

        meta_features = np.hstack(base_preds)
        return self.meta_model.predict_proba(meta_features)

    def predict(self, X):
        """Returns class predictions (0=Normal, 1=Bubble, 2=Crash)."""
        return np.argmax(self.predict_proba(X), axis=1)

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)

        print(f"\n{'='*55}")
        print("  STACKING ENSEMBLE EVALUATION")
        print("="*55)
        print(classification_report(y_test, y_pred,
              target_names=["Normal", "Bubble", "Crash"],
              zero_division=0, digits=4))

        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(pd.DataFrame(cm,
              index=["Act Normal", "Act Bubble", "Act Crash"],
              columns=["Pred Normal", "Pred Bubble", "Pred Crash"]))

        if cm.shape == (3, 3):
            br = cm[1, 1]/cm[1, :].sum() if cm[1, :].sum() > 0 else 0
            cr = cm[2, 2]/cm[2, :].sum() if cm[2, :].sum() > 0 else 0
            print(f"\n  Bubble detection rate: {br*100:.1f}%")
            print(f"  Crash  detection rate: {cr*100:.1f}%")

        # Crash probability stats
        crash_probs = y_prob[:, 2]
        print(f"\n  Crash probability  mean={crash_probs.mean():.3f}"
              f"  max={crash_probs.max():.3f}")

        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        status = ("✅ meets target" if f1 >= 0.85
                  else "⚠️  below target" if f1 >= 0.70
                  else "❌ too low")
        print(f"\n  ★ Macro F1: {f1:.4f}  {status}")
        return f1, y_prob

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path="models/stacking_ensemble.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  Ensemble saved → {path}")

    @staticmethod
    def load(path="models/stacking_ensemble.pkl"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Run main.py first.")
        with open(path, "rb") as f:
            return pickle.load(f)


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.ml_models import prepare_features, apply_adasyn, save_model

    # Use sentiment-enriched data if available, else fall back
    if os.path.exists("data/nifty50_with_sentiment.csv"):
        data_path = "data/nifty50_with_sentiment.csv"
        print("  📊 Using sentiment-enriched dataset")
    elif os.path.exists("data/nifty50_labeled.csv"):
        data_path = "data/nifty50_labeled.csv"
        print(
            "  ⚠️  Using labeled data only (run historical_sentiment.py for VIX features)")
    else:
        print("Run data_ingestion.py and zscore_labeling.py first")
        sys.exit(1)

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df_ml, feat = prepare_features(df)
    X = df_ml[feat].values
    y = df_ml["label_numeric"].values.astype(int)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    # Split FIRST, ADASYN only on train
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sc, y, test_size=0.2, random_state=42, stratify=y)

    X_tr, y_tr = apply_adasyn(X_tr, y_tr)

    base_models = [
        ("RandomForest", RandomForestClassifier(
            n_estimators=200, max_depth=10,
            min_samples_split=10, class_weight="balanced",
            random_state=42, n_jobs=-1)),
        ("XGBoost", xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", use_label_encoder=False,
            random_state=42, n_jobs=-1)),
    ]

    ensemble = StackingEnsemble(base_models=base_models, n_folds=5)
    ensemble.fit(X_tr, y_tr)
    f1, _ = ensemble.evaluate(X_te, y_te)

    save_model(scaler, "scaler.pkl")
    save_model(feat,   "feature_cols.pkl")
    ensemble.save("models/stacking_ensemble.pkl")

    print(f"\n✅ Stacking ensemble saved. Macro F1 = {f1:.4f}")
    print("Now run: streamlit run app.py")
