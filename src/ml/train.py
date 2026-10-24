"""
train.py - compare regression models, save the best one, report metrics.
Run from anywhere:  python src/ml/train.py     (or from the root: python -m src.ml.train)
"""
import sys
from pathlib import Path

# Make "src" importable even when this file is run directly from any folder
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.ml.config import (FEATURES, METRICS_PATH, MODEL_PATH, TARGET,
                           TEST_PATH, TRAIN_PATH)
from src.ml.risk import score_to_risk

SEED = 42


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=SEED),
    }

    results, fitted = {}, {}
    for name, model in candidates.items():
        pipe = make_pipeline(StandardScaler(), model)
        cv_r2 = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2").mean()
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results[name] = {
            "cv_r2": round(float(cv_r2), 3),
            "test_r2": round(float(r2_score(y_test, pred)), 3),
            "test_mae": round(float(mean_absolute_error(y_test, pred)), 2),
            "test_rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 2),
        }
        fitted[name] = pipe
        print(f"{name:20s} {results[name]}")

    best_name = max(results, key=lambda n: results[n]["cv_r2"])   # choose on CV, not on test
    best = fitted[best_name]
    print("\nBest model (by cross-validated R2):", best_name)

    # Risk classification = predicted score -> thresholds (with safety margin)
    pred = best.predict(X_test)
    y_pred_risk = [score_to_risk(s) for s in pred]
    report = classification_report(test["risk_level"], y_pred_risk, output_dict=True, zero_division=0)
    print("\nRisk classification on test set:")
    print(classification_report(test["risk_level"], y_pred_risk, zero_division=0))
    labels = ["Low", "Moderate", "High"]
    cm = confusion_matrix(test["risk_level"], y_pred_risk, labels=labels)
    print("Confusion matrix (rows=actual, cols=predicted, order Low/Moderate/High):\n", cm)

    bundle = {
        "pipeline": best,
        "features": FEATURES,
        "feature_means": X_train.mean().to_dict(),
        "model_name": best_name,
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump({"best_model": best_name, "regression": results,
                   "risk_report": report, "confusion_matrix": cm.tolist()}, f, indent=2)
    print(f"\nSaved {MODEL_PATH}\nSaved {METRICS_PATH}")


if __name__ == "__main__":
    main()
