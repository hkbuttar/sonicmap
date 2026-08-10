"""Gradient-boosting baseline on engineered features (MFCC/chroma/tempo/
spectral-shape statistics from features/engineered.py) — the classical
alternative to the CNN, compared head-to-head in Step 4."""

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score


def train_gbm(X_train: np.ndarray, y_train: np.ndarray, n_classes: int, seed: int = 42) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=n_classes,
        random_state=seed,
        n_jobs=-1,
        eval_metric="mlogloss",
    )
    model.fit(X_train, y_train)
    return model


def evaluate_gbm(model: xgb.XGBClassifier, X_val: np.ndarray, y_val: np.ndarray) -> dict:
    preds = model.predict(X_val)
    return {
        "accuracy": accuracy_score(y_val, preds),
        "f1_macro": f1_score(y_val, preds, average="macro"),
    }
